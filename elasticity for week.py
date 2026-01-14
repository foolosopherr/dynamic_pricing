"""
Пайплайн работы кода

1) preprocess_raw(df_raw)
   - Приводим TRADE_DT к дате (normalize), режем историю по HISTORY_END.
   - Складываем продажи оффлайн+онлайн в QTY_TOTAL (важно: строки могут существовать даже при Q=0).
   - Восстанавливаем UNIT_PRICE (если SALE_PRICE нет, считаем из total/qty).
   - Считаем BUCKET по дню недели (Mon-Thu / Fri / Sat-Sun).
   - Парсим PROMO_PERIOD в диапазоны дат (PROMO_RANGE), чтобы потом считать PROMO_SHARE на интервале бакета.

2) aggregate_to_buckets(daily)
   - Агрегируем дневные строки до уровня "товар-магазин-неделя-бакет".
   - Получаем QTY (сумма), PRICE (взвешенная по продажам), PROMO_SHARE (доля дней промо в бакете),
     STOCK_END (остаток на конец бакета) и OOS_FLAG (признак OOS по STOCK_END<=0).
   - Добавляем сезонность/календарь и тренд:
     SIN_WOY, COS_WOY (неделя года), TREND_W (линейный тренд по времени),
     BUCKET_IDX (код бакета).

3) add_lag_features(bucket_df)
   - Добавляем лаги внутри (SKU, STORE, BUCKET):
     QTY_L1, PRICE_L1, PROMO_L1, STOCK_END_L1, OOS_L1.
   - Это закрывает проблему "цена сегодня -> продажи завтра/в следующем бакете".
   - Также делаем LOW_STOCK_FLAG (по STOCK_END_L1), чтобы модель могла учитывать ограничение спроса.

4) calculate_for_week(..., mode)
   - Выбираем бакеты для расчёта по режиму запуска:
     mode="monday": считаем только MON_THU
     mode="friday": считаем FRI и SAT_SUN
   - Устанавливаем cutoff (до какой даты можно смотреть факты):
     monday: cutoff = вс прошлой недели
     friday: cutoff = чт текущей недели (то есть используем свежие Mon-Thu)
   - Строим context: последняя доступная запись по SKU+STORE до cutoff.

5) REG_PRICE + промо-правило (ключевое)
   - Для каждого SKU+STORE+BUCKET считаем BASE_PRICE (последняя цена в этом бакете до cutoff).
   - Далее считаем REG_PRICE:
       a) если BASE_PRICE валиден -> REG_PRICE = BASE_PRICE
       b) иначе -> медиана цен за последние N недель (сначала в этом bucket, потом по любому bucket)
   - Если PROMO_SHARE > PROMO_THRESHOLD:
       применяем жёсткий cap: CHOSEN_PRICE <= REG_PRICE
     (реализовано через "обрезание" grid цен сверху).

6) Обучение моделей эластичности (HierDualModels)
   - Обучаем PoissonRegressor на y=QTY (сезонность+тренд+OOS+promo+lags+price features).
   - Делаем "dual" (promo/nonpromo) модели и backoff по иерархии:
     PRODUCT_STORE -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL.
   - Эластичность (приближённо) = коэффициент при LOG_PRICE_L1 (в "сырых" единицах),
     плюс локальная эластичность вокруг выбранной цены (через +/- delta).

7) Cannibalization (Cross-family) + fallback на single
   - Для семейства в магазине и бакете берём top-K SKU по продажам и учим cross-модели:
     y_sku ~ log(price всех sku в top-K) (Poisson).
   - В оптимизации по семье итеративно подбираем цены, учитывая, что изменение цены одного товара
     меняет спрос остальных (каннибализация).
   - Если cross-модель не построилась/не подходит, считаем single-оптимизацию с иерархическим backoff.

8) Оптимизация цены (grid search)
   - Строим grid цен вокруг anchor (REG_PRICE), с ограничением MIN_MULT..MAX_MULT,
     дополнительно "snap" к наблюдавшимся ценам семьи (price ladder), чтобы не предлагать странные цены.
   - Для каждого кандидата предсказываем спрос mu (и ограничиваем stock_cap_end),
     считаем objective = revenue - penalty (штраф удерживает цену около REG_PRICE),
     выбираем максимум.
   - В промо grid уже обрезан, поэтому результат гарантированно не выше REG_PRICE.

9) Выход
   - Для каждого SKU+STORE+BUCKET отдаём CHOSEN_PRICE, прогноз спроса/выручки,
     признаки эластичности (OWN_ELASTICITY_*, LOCAL_ELASTICITY), факт использования cross-модели,
     и диагностические поля REG_PRICE_SOURCE, PROMO_CAP_PRICE и т.д.

Итого: весь код описывает единый продовый пайплайн "подготовка -> агрегация -> лаги -> модели спроса/эластичности
-> учёт промо и остатков -> оптимизация цены по сетке -> результат по бакетам с нужными cutoff".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


# =========================
# Конфиг
# =========================

# Горизонт данных и бизнес-ограничения цен

# Дата конца доступной истории. Всё после неё отрезаем, чтобы модель не "видела будущее"
HISTORY_END = pd.Timestamp("2025-09-15")

# Порядок бакетов (Mon-Thu, Fri, Sat-Sun). Нужен для кодирования BUCKET_IDX
BUCKET_ORDER = ["MON_THU", "FRI", "SAT_SUN"]

# Границы мультипликатора цены относительно anchor (REG_PRICE). Вместе задают допустимый диапазон ценовых решений (guardrail бизнес-уровня).
MIN_MULT = 0.9
MAX_MULT = 1.15

# ------------------------

# Модель спроса (Poisson) и численная устойчивость

# Регуляризация модели PoissonRegressor (L2). Уменьшает переобучение и проблемы с сходимостью
POISSON_ALPHA = 1e-2

# Максимум итераций оптимизатора. Нужен, чтобы уменьшить ConvergenceWarning на сложных данных
POISSON_MAX_ITER = 3000

# Параметр shrinkage (смешивание уровней иерархии). Чем больше K, тем сильнее тянем к верхнему уровню. Вместе с иерархией даёт устойчивые прогнозы, когда у SKU мало данных.
DEFAULT_SHRINK_K = 30

# ------------------------

# Промо-логика и разделение режимов

# Порог, после которого бакет считается "промо" (PROMO_SHARE > threshold).
#  Используется:
#    1) для выбора promo/nonpromo модели (dual),
#    2) для применения промо cap (CHOSEN_PRICE <= REG_PRICE).
PROMO_THRESHOLD = 0.15

# Минимум строк, чтобы обучать отдельную promo/nonpromo модель.
# Вместе с PROMO_THRESHOLD определяет, когда мы доверяем раздельным моделям, а когда делаем backoff.
MIN_ROWS_SPLIT_MODEL = 60

# --------------------------

# Эластичность и guardrails против "всё в максимум"

# Минимально допустимая по модулю отрицательность эластичности (по log-price).
# Если модель дала слишком слабый или даже положительный наклон, принудительно используем -BETA_FLOOR,
# чтобы оптимизация не уезжала в MAX_MULT из-за "неэластичного" спроса.
BETA_FLOOR = 0.7

# Шаг для локальной эластичности вокруг выбранной цены (mu(p*(1±delta))).
# Нужен, чтобы иметь более интерпретируемую "локальную" эластичность.
LAMBDA_PRICE_PENALTY = 0.25

# ---------------------------

# Целевая функция (оптимизация)

# Сила штрафа, удерживающего цену возле REG_PRICE.
# Вместе с MIN_MULT/MAX_MULT определяет "насколько агрессивно" модель двигает цену:
# - малый штраф => чаще крайние решения,
# - большой штраф => решения ближе к REG_PRICE.
LOCAL_ELASTICITY_DELTA = 0.05

# ------------------------------

# Минимумы данных для иерархии

# Минимум строк для глобальной модели по бакету.
MIN_ROWS_GLOBAL = 600

# Минимум строк для уровня SEGMENT/CATEGORY/FAMILY.
MIN_ROWS_LEVEL = 120

# Минимум строк для модели PRODUCT_STORE.
MIN_ROWS_PS = 40

# Минимум наблюдений с QTY>0, чтобы считать эластичность "OK".
# Вместе эти пороги описывают "когда мы доверяем конкретному уровню" и когда нужно backoff.
MIN_POS_SALES_FOR_ELAST = 6

# ---------------------------------

# Регулярная цена (REG_PRICE) при пропусках BASE_PRICE

# Сколько недель смотреть назад, чтобы оценить регулярную цену медианой, если BASE_PRICE нет.
REG_PRICE_MEDIAN_WEEKS = 4

# Минимум точек цены для медианы (иначе медиана ненадёжна).
# Вместе они описывают safeguard "REG_PRICE как якорь" для промо-cap и построения grid,
# когда товары появляются/исчезают или в бакете нет недавней цены.
REG_PRICE_MIN_POINTS = 2

# ----------------------------------

# Каннибализация (cross-family)

# Сколько SKU в семье брать для cross-модели (только самые продаваемые).
CROSS_TOP_K = 7

# Глубина истории для cross-модели (недели назад)
CROSS_LOOKBACK_WEEKS = 24

# Минимум строк для обучения модели конкретного SKU внутри family-wide панели.
CROSS_MIN_ROWS_PER_SKU = 30

# Сколько итераций coordinate-descent при подборе цен в семье (цены влияют друг на друга)
CROSS_ITERS = 4

# Минимум недель в семье, чтобы вообще строить cross-модель (стабильность).
CROSS_MIN_FAMILY_WEEKS = 20

# Минимальный суммарный объём продаж семьи, чтобы cross-модель была осмысленной.
# Вместе эти параметры описывают блок "каннибализация": когда он применяется, на каких SKU,
# насколько глубоко смотрим историю и как стабильно оптимизируем.
CROSS_MIN_FAMILY_SUM_QTY = 20


# =========================
# Даты / бакеты
# =========================

def bucket_id_for_date(dt: pd.Timestamp) -> str:
    wd = pd.Timestamp(dt).weekday()
    if wd <= 3:
        return "MON_THU"
    if wd == 4:
        return "FRI"
    return "SAT_SUN"


def week_start(dt: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(dt).normalize()
    return dt - pd.Timedelta(days=dt.weekday())


def bucket_start_end(week_monday: pd.Timestamp, bucket: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    m = pd.Timestamp(week_monday).normalize()
    if bucket == "MON_THU":
        return m, m + pd.Timedelta(days=3)
    if bucket == "FRI":
        d = m + pd.Timedelta(days=4)
        return d, d
    if bucket == "SAT_SUN":
        return m + pd.Timedelta(days=5), m + pd.Timedelta(days=6)
    raise ValueError(bucket)


def decision_cutoff_for_bucket_by_mode(week_monday: pd.Timestamp, bucket: str, mode: str) -> pd.Timestamp:
    """
    Прод-логика запуска:
      - monday morning: считаем MON_THU, cutoff = воскресенье прошлой недели
      - friday morning: считаем FRI и SAT_SUN на данных Mon-Thu => cutoff = Thu для обоих
    """
    mode = mode.lower().strip()
    m = pd.Timestamp(week_monday).normalize()

    if mode == "monday":
        return m - pd.Timedelta(days=1)
    if mode == "friday":
        return m + pd.Timedelta(days=3)

    # fallback
    if bucket == "MON_THU":
        return m - pd.Timedelta(days=1)
    if bucket == "FRI":
        return m + pd.Timedelta(days=3)
    if bucket == "SAT_SUN":
        return m + pd.Timedelta(days=4)
    raise ValueError(bucket)


def _safe_log(x: float) -> float:
    return float(np.log(max(float(x), 1e-6)))


# =========================
# PROMO_PERIOD
# =========================

def parse_promo_period(s: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(s, str) or s.strip() == "" or " - " not in s:
        return None
    a, b = [p.strip() for p in s.split(" - ", 1)]
    try:
        start = pd.to_datetime(a, dayfirst=True).normalize()
        end = pd.to_datetime(b, dayfirst=True).normalize()
        if end < start:
            start, end = end, start
        return start, end
    except Exception:
        return None


def promo_share_in_range(promo_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]],
                         start: pd.Timestamp,
                         end: pd.Timestamp) -> float:
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    if start > end or not promo_ranges:
        return 0.0
    days = pd.date_range(start, end, freq="D")
    mask = np.zeros(len(days), dtype=bool)
    for a, b in promo_ranges:
        a = max(pd.Timestamp(a).normalize(), start)
        b = min(pd.Timestamp(b).normalize(), end)
        if b < a:
            continue
        mask |= (days >= a) & (days <= b)
    return float(mask.mean())


# =========================
# Preprocess daily
# =========================

def compute_unit_price(df: pd.DataFrame) -> pd.Series:
    if "SALE_PRICE" in df.columns:
        return pd.to_numeric(df["SALE_PRICE"], errors="coerce")
    total = pd.to_numeric(df.get("SALE_PRICE_TOTAL"), errors="coerce")
    qty = pd.to_numeric(df.get("SALE_QTY"), errors="coerce").fillna(0.0)
    return total / qty.clip(lower=1.0)


def preprocess_raw(df: pd.DataFrame, history_end: pd.Timestamp = HISTORY_END) -> pd.DataFrame:
    out = df.copy()
    out["TRADE_DT"] = pd.to_datetime(out["TRADE_DT"], errors="coerce").dt.normalize()
    out = out.loc[out["TRADE_DT"].notna()]
    out = out.loc[out["TRADE_DT"] <= pd.Timestamp(history_end).normalize()]

    qty_off = pd.to_numeric(out.get("SALE_QTY"), errors="coerce").fillna(0.0)
    qty_on = pd.to_numeric(out.get("SALE_QTY_ONLINE"), errors="coerce").fillna(0.0)
    out["QTY_TOTAL"] = (qty_off + qty_on).astype(float).clip(lower=0.0)

    out["UNIT_PRICE"] = compute_unit_price(out).astype(float)
    out["BUCKET"] = out["TRADE_DT"].apply(bucket_id_for_date)

    if "PROMO_PERIOD" in out.columns:
        out["PROMO_RANGE"] = out["PROMO_PERIOD"].apply(parse_promo_period)
    else:
        out["PROMO_RANGE"] = None

    if "IS_PROMO" in out.columns:
        out["IS_PROMO"] = pd.to_numeric(out["IS_PROMO"], errors="coerce").fillna(0.0).astype(int)
    else:
        out["IS_PROMO"] = 0

    keys = ["PRODUCT_CODE", "FAMILY_CODE", "CATEGORY_CODE", "SEGMENT_CODE",
            "STORE", "REGION_NAME", "STORE_TYPE", "PLACE_TYPE"]
    for k in keys:
        if k in out.columns:
            out[k] = out[k].astype(str).fillna("NA")

    for c in ["START_STOCK", "END_STOCK", "DELIVERY_QTY", "LOSS_QTY", "RETURN_QTY"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    return out


def build_promo_map(daily: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    promo_map: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if "PROMO_RANGE" not in daily.columns:
        return promo_map
    for (p, s), g in daily.groupby(["PRODUCT_CODE", "STORE"]):
        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)]
        if ranges:
            promo_map[(str(p), str(s))] = ranges
    return promo_map


# =========================
# Buckets aggregation
# =========================

def aggregate_to_buckets(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["WEEK"] = d["TRADE_DT"].apply(week_start)

    group_cols = [
        "PRODUCT_CODE", "STORE", "WEEK", "BUCKET",
        "FAMILY_CODE", "CATEGORY_CODE", "SEGMENT_CODE",
        "REGION_NAME", "STORE_TYPE", "PLACE_TYPE"
    ]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        qty = float(g["QTY_TOTAL"].sum())
        qty = max(qty, 0.0)

        pvals = g["UNIT_PRICE"].astype(float).values
        price = np.nan
        if qty > 0 and np.isfinite(pvals).any():
            w = g["QTY_TOTAL"].astype(float).values
            if np.nansum(w) > 0:
                price = float(np.nansum(pvals * w) / np.nansum(w))
        if not np.isfinite(price):
            price = float(np.nanmean(pvals)) if np.isfinite(pvals).any() else np.nan

        wk = g["WEEK"].iloc[0]
        b = g["BUCKET"].iloc[0]
        start, end = bucket_start_end(wk, b)

        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)] if "PROMO_RANGE" in g else []
        promo_share = promo_share_in_range(ranges, start, end) if ranges else float(g.get("IS_PROMO", pd.Series([0])).mean())

        stock_end = float(g["END_STOCK"].iloc[-1]) if "END_STOCK" in g else 0.0
        stock_start = float(g["START_STOCK"].iloc[0]) if "START_STOCK" in g else 0.0

        return pd.Series({
            "QTY": qty,
            "PRICE": price,
            "PROMO_SHARE": promo_share,
            "STOCK_START": stock_start,
            "STOCK_END": stock_end,
            "BUCKET_START": start,
            "BUCKET_END": end,
        })

    bucket_df = d.groupby(group_cols, as_index=False).apply(agg_group).reset_index(drop=True)

    bucket_df["WEEK_OF_YEAR"] = bucket_df["WEEK"].dt.isocalendar().week.astype(int)
    bucket_df["BUCKET_IDX"] = bucket_df["BUCKET"].map({b: i for i, b in enumerate(BUCKET_ORDER)}).astype(int)

    woy = bucket_df["WEEK_OF_YEAR"].astype(float)
    bucket_df["SIN_WOY"] = np.sin(2.0 * np.pi * woy / 52.0)
    bucket_df["COS_WOY"] = np.cos(2.0 * np.pi * woy / 52.0)

    min_week = bucket_df["WEEK"].min()
    bucket_df["TREND_W"] = ((bucket_df["WEEK"] - min_week) / np.timedelta64(1, "W")).astype(float)

    bucket_df["QTY"] = pd.to_numeric(bucket_df["QTY"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket_df["PRICE"] = pd.to_numeric(bucket_df["PRICE"], errors="coerce")
    bucket_df["STOCK_END"] = pd.to_numeric(bucket_df["STOCK_END"], errors="coerce").fillna(0.0).clip(lower=0.0)

    bucket_df["OOS_FLAG"] = (bucket_df["STOCK_END"] <= 0.0).astype(int)

    return bucket_df


# =========================
# Lags
# =========================

def add_lag_features(bucket_df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    d = bucket_df.copy().sort_values(["PRODUCT_CODE", "STORE", "BUCKET", "WEEK"])
    grp = d.groupby(["PRODUCT_CODE", "STORE", "BUCKET"], sort=False)

    for k in range(1, lags + 1):
        d[f"QTY_L{k}"] = grp["QTY"].shift(k).fillna(0.0)
        d[f"PRICE_L{k}"] = grp["PRICE"].shift(k)
        d[f"PROMO_L{k}"] = grp["PROMO_SHARE"].shift(k).fillna(0.0)
        d[f"STOCK_END_L{k}"] = grp["STOCK_END"].shift(k).fillna(0.0)
        d[f"OOS_L{k}"] = grp["OOS_FLAG"].shift(k).fillna(0).astype(int)

    d["LOG_PRICE_L1"] = d["PRICE_L1"].apply(lambda v: _safe_log(v) if np.isfinite(v) and v > 0 else 0.0)
    d["LOW_STOCK_FLAG"] = (d["STOCK_END_L1"].fillna(0.0) <= 1.0).astype(int)

    return d


# =========================
# Models
# =========================

@dataclass
class PoissonModel:
    reg: PoissonRegressor
    scaler: StandardScaler
    feature_names: List[str]
    n: int
    pos_sales: int


def fit_poisson(df: pd.DataFrame, feature_names: List[str], alpha: float = POISSON_ALPHA) -> Optional[PoissonModel]:
    g = df.copy()
    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    g["LOG_PRICE"] = g["PRICE"].astype(float).apply(_safe_log)

    y = np.clip(g["QTY"].astype(float).values, 0.0, None)
    if not np.isfinite(y).all():
        return None

    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = PoissonRegressor(alpha=max(alpha, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
    reg.fit(Xs, y)

    return PoissonModel(reg, scaler, list(feature_names), int(len(g)), int((y > 0).sum()))


def predict_mu(model: PoissonModel, x_raw: np.ndarray) -> float:
    Xs = model.scaler.transform(x_raw.reshape(1, -1))
    return max(float(model.reg.predict(Xs)[0]), 0.0)


def coef_raw(model: PoissonModel, feature: str) -> Optional[float]:
    if feature not in model.feature_names:
        return None
    j = model.feature_names.index(feature)
    std = float(model.scaler.scale_[j]) if model.scaler.scale_ is not None else 1.0
    if std <= 0:
        return None
    return float(model.reg.coef_[j]) / std


@dataclass
class DualModels:
    nonpromo: Optional[PoissonModel]
    promo: Optional[PoissonModel]


def split_promo(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nonpromo = df.loc[df["PROMO_SHARE"] <= PROMO_THRESHOLD].copy()
    promo = df.loc[df["PROMO_SHARE"] > PROMO_THRESHOLD].copy()
    return nonpromo, promo


def train_dual(df: pd.DataFrame, feature_names: List[str], alpha: float) -> DualModels:
    nonpromo_df, promo_df = split_promo(df)
    m_np = fit_poisson(nonpromo_df, feature_names, alpha=alpha) if len(nonpromo_df) >= MIN_ROWS_SPLIT_MODEL else None
    m_p = fit_poisson(promo_df, feature_names, alpha=alpha) if len(promo_df) >= MIN_ROWS_SPLIT_MODEL else None
    return DualModels(nonpromo=m_np, promo=m_p)


def choose_model_with_backoff(dm: DualModels, promo_share: float) -> Optional[PoissonModel]:
    if promo_share > PROMO_THRESHOLD:
        return dm.promo or dm.nonpromo
    return dm.nonpromo or dm.promo


@dataclass
class HierDualModels:
    global_model: Dict[str, DualModels]
    segment: Dict[Tuple[str, str], DualModels]
    category: Dict[Tuple[str, str], DualModels]
    family: Dict[Tuple[str, str], DualModels]
    product_store: Dict[Tuple[str, str, str], DualModels]


def train_hier_dual_models(bucket_df: pd.DataFrame,
                           cutoff: pd.Timestamp,
                           feature_names: List[str],
                           alpha: float = POISSON_ALPHA) -> HierDualModels:
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()

    global_models: Dict[str, DualModels] = {}
    for b, g in train.groupby("BUCKET"):
        if len(g) >= MIN_ROWS_GLOBAL:
            global_models[str(b)] = train_dual(g, feature_names, alpha=alpha)

    seg_models: Dict[Tuple[str, str], DualModels] = {}
    for (seg, b), g in train.groupby(["SEGMENT_CODE", "BUCKET"]):
        if len(g) >= MIN_ROWS_LEVEL:
            seg_models[(str(seg), str(b))] = train_dual(g, feature_names, alpha=alpha)

    cat_models: Dict[Tuple[str, str], DualModels] = {}
    for (cat, b), g in train.groupby(["CATEGORY_CODE", "BUCKET"]):
        if len(g) >= MIN_ROWS_LEVEL:
            cat_models[(str(cat), str(b))] = train_dual(g, feature_names, alpha=alpha)

    fam_models: Dict[Tuple[str, str], DualModels] = {}
    for (fam, b), g in train.groupby(["FAMILY_CODE", "BUCKET"]):
        if len(g) >= MIN_ROWS_LEVEL:
            fam_models[(str(fam), str(b))] = train_dual(g, feature_names, alpha=alpha)

    ps_models: Dict[Tuple[str, str, str], DualModels] = {}
    for (sku, store, b), g in train.groupby(["PRODUCT_CODE", "STORE", "BUCKET"]):
        if len(g) >= MIN_ROWS_PS:
            ps_models[(str(sku), str(store), str(b))] = train_dual(g, feature_names, alpha=alpha)

    return HierDualModels(global_model=global_models, segment=seg_models, category=cat_models, family=fam_models, product_store=ps_models)


def shrink_weight(n_child: int, k: int = DEFAULT_SHRINK_K) -> float:
    return float(n_child / (n_child + k)) if n_child > 0 else 0.0


# =========================
# REG_PRICE (regular price) + promo cap
# =========================

def is_valid_price(p: Optional[float]) -> bool:
    return p is not None and np.isfinite(p) and float(p) > 0


def compute_reg_price(bucket_df: pd.DataFrame,
                      sku: str,
                      store: str,
                      bucket: str,
                      cutoff: pd.Timestamp,
                      base_price: Optional[float],
                      median_weeks: int = REG_PRICE_MEDIAN_WEEKS) -> Tuple[Optional[float], str]:
    """
    REG_PRICE нужен для промо-cap и для стабильного сравнения "дорого/дёшево".

    Источники (по приоритету):
      1) BASE_BUCKET_LAST: base_price валиден (последняя цена в bucket до cutoff)
      2) MEDIAN_LAST_NW_BUCKET: медиана цены за последние N недель в этом bucket
      3) MEDIAN_LAST_NW_ANY_BUCKET: медиана цены за последние N недель по любому bucket
      4) FALLBACK_NONE: ничего нет
    """
    if is_valid_price(base_price):
        return float(base_price), "BASE_BUCKET_LAST"

    cutoff = pd.Timestamp(cutoff)
    min_week = week_start(cutoff) - pd.Timedelta(weeks=int(median_weeks))

    # 2) bucket median
    df_b = bucket_df.loc[
        (bucket_df["PRODUCT_CODE"] == sku) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET"] == bucket) &
        (bucket_df["BUCKET_END"] <= cutoff) &
        (bucket_df["WEEK"] >= min_week)
    ].copy()
    pr = pd.to_numeric(df_b["PRICE"], errors="coerce").astype(float)
    pr = pr[np.isfinite(pr) & (pr > 0)]
    if len(pr) >= REG_PRICE_MIN_POINTS:
        return float(np.median(pr)), "MEDIAN_LAST_NW_BUCKET"

    # 3) any bucket median
    df_a = bucket_df.loc[
        (bucket_df["PRODUCT_CODE"] == sku) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET_END"] <= cutoff) &
        (bucket_df["WEEK"] >= min_week)
    ].copy()
    pr = pd.to_numeric(df_a["PRICE"], errors="coerce").astype(float)
    pr = pr[np.isfinite(pr) & (pr > 0)]
    if len(pr) >= REG_PRICE_MIN_POINTS:
        return float(np.median(pr)), "MEDIAN_LAST_NW_ANY_BUCKET"

    return None, "FALLBACK_NONE"


def discount_depth(candidate_price: float, reg_price: float) -> float:
    if not is_valid_price(reg_price):
        return 0.0
    r = 1.0 - float(candidate_price) / float(reg_price)
    return float(max(r, 0.0))


def price_change_log(candidate_price: float, reg_price: float) -> float:
    if not is_valid_price(reg_price):
        return 0.0
    return float(_safe_log(candidate_price) - _safe_log(reg_price))


def make_features_for_candidate(base_row: pd.Series,
                                candidate_price: float,
                                reg_price: float,
                                bucket: str,
                                week_monday: pd.Timestamp) -> Dict[str, float]:
    woy = int(pd.Timestamp(week_monday).isocalendar().week)
    sin_woy = float(np.sin(2.0 * np.pi * woy / 52.0))
    cos_woy = float(np.cos(2.0 * np.pi * woy / 52.0))
    trend_w = float(base_row.get("TREND_W", 0.0))

    promo_share = float(base_row.get("PROMO_SHARE", 0.0))

    dd = discount_depth(candidate_price, reg_price)
    pcl = price_change_log(candidate_price, reg_price)
    promo_depth = float(dd * promo_share)

    return {
        "LOG_PRICE_L1": float(_safe_log(candidate_price)),
        "LOG_PRICE": float(_safe_log(candidate_price)),
        "DISCOUNT_DEPTH": float(dd),
        "PRICE_CHANGE_LOG": float(pcl),
        "PROMO_DEPTH": float(promo_depth),

        "PROMO_SHARE": promo_share,
        "BUCKET_IDX": float({"MON_THU": 0, "FRI": 1, "SAT_SUN": 2}[bucket]),
        "SIN_WOY": sin_woy,
        "COS_WOY": cos_woy,
        "TREND_W": trend_w,

        "LOW_STOCK_FLAG": float(base_row.get("LOW_STOCK_FLAG", 0.0)),
        "STOCK_END_L1": float(base_row.get("STOCK_END_L1", 0.0)),
        "OOS_L1": float(base_row.get("OOS_L1", 0.0)),

        "QTY_L1": float(base_row.get("QTY_L1", 0.0)),
        "PROMO_L1": float(base_row.get("PROMO_L1", 0.0)),
    }


def apply_promo_cap_to_grid(grid: np.ndarray, reg_price: Optional[float], promo_share: float) -> Tuple[np.ndarray, Optional[float]]:
    """
    Жёсткое правило:
      если PROMO_SHARE > PROMO_THRESHOLD => CHOSEN_PRICE <= REG_PRICE.
    Возвращаем (grid2, cap_price).
    """
    if grid is None or len(grid) == 0:
        return grid, None
    if promo_share > PROMO_THRESHOLD and is_valid_price(reg_price):
        cap = float(reg_price)
        grid2 = grid[grid <= cap + 1e-9]
        if len(grid2) == 0:
            return np.array([cap], dtype=float), cap
        return grid2, cap
    return grid, None


def objective(price: float, mu: float, reg_price: float, qty_scale: float) -> float:
    revenue = float(price) * float(mu)
    rel = float(price) / max(float(reg_price), 1e-6) - 1.0
    penalty_price = LAMBDA_PRICE_PENALTY * (rel * rel) * (float(reg_price) * float(qty_scale))
    return revenue - penalty_price


def predict_backoff_dual(base_row: pd.Series,
                         candidate_price: float,
                         reg_price: float,
                         bucket: str,
                         week_monday: pd.Timestamp,
                         promo_share: float,
                         feature_names: List[str],
                         hier: HierDualModels) -> Tuple[Optional[float], Optional[float], str, int]:
    feats = make_features_for_candidate(base_row, candidate_price, reg_price, bucket, week_monday)
    x = np.array([feats.get(fn, 0.0) for fn in feature_names], dtype=float)

    sku = str(base_row["PRODUCT_CODE"])
    store = str(base_row["STORE"])
    fam = str(base_row["FAMILY_CODE"])
    cat = str(base_row["CATEGORY_CODE"])
    seg = str(base_row["SEGMENT_CODE"])

    levels: List[Tuple[str, PoissonModel]] = []

    dm_ps = hier.product_store.get((sku, store, bucket))
    if dm_ps is not None:
        m = choose_model_with_backoff(dm_ps, promo_share)
        if m is not None:
            levels.append(("PRODUCT_STORE", m))

    dm_f = hier.family.get((fam, bucket))
    if dm_f is not None:
        m = choose_model_with_backoff(dm_f, promo_share)
        if m is not None:
            levels.append(("FAMILY", m))

    dm_c = hier.category.get((cat, bucket))
    if dm_c is not None:
        m = choose_model_with_backoff(dm_c, promo_share)
        if m is not None:
            levels.append(("CATEGORY", m))

    dm_s = hier.segment.get((seg, bucket))
    if dm_s is not None:
        m = choose_model_with_backoff(dm_s, promo_share)
        if m is not None:
            levels.append(("SEGMENT", m))

    dm_g = hier.global_model.get(bucket)
    if dm_g is not None:
        m = choose_model_with_backoff(dm_g, promo_share)
        if m is not None:
            levels.append(("GLOBAL", m))

    if not levels:
        return None, None, "NONE", 0

    mu = None
    beta = None
    used_level = "NONE"
    pos_sales_used = 0
    n_child = 0

    for lvl, m in levels:
        mu_m = predict_mu(m, x)
        beta_m = coef_raw(m, "LOG_PRICE_L1")

        if mu is None:
            mu = mu_m
            beta = beta_m
            used_level = lvl
            pos_sales_used = m.pos_sales
            n_child = m.n
        else:
            w = shrink_weight(n_child, DEFAULT_SHRINK_K)
            mu = w * mu + (1 - w) * mu_m
            if beta is None:
                beta = beta_m
            elif beta_m is not None:
                beta = w * beta + (1 - w) * beta_m
            n_child = max(n_child, m.n)

    return float(max(mu, 0.0)) if mu is not None else None, beta, used_level, pos_sales_used


def local_elasticity(p: float, mu_lo: float, mu_hi: float, delta: float) -> float:
    p_lo = p * (1.0 - delta)
    p_hi = p * (1.0 + delta)
    num = np.log(max(mu_hi, 1e-9)) - np.log(max(mu_lo, 1e-9))
    den = np.log(max(p_hi, 1e-9)) - np.log(max(p_lo, 1e-9))
    return float(num / den) if den != 0 else np.nan


def optimize_single(base_row: pd.Series,
                    bucket: str,
                    week_monday: pd.Timestamp,
                    promo_share: float,
                    feature_names: List[str],
                    hier: HierDualModels,
                    grid: np.ndarray,
                    reg_price: float,
                    stock_cap_end: Optional[float]) -> Optional[dict]:
    grid, promo_cap_price = apply_promo_cap_to_grid(grid, reg_price, promo_share)
    if grid is None or len(grid) == 0:
        return None

    qty_scale = float(max(base_row.get("QTY_L1", 0.0), 1.0))

    best = None
    meta = None

    for p in grid:
        mu, beta_raw, used_level, pos_sales_used = predict_backoff_dual(
            base_row, float(p), float(reg_price), bucket, week_monday, promo_share, feature_names, hier
        )
        if mu is None:
            continue

        if stock_cap_end is not None and np.isfinite(stock_cap_end):
            mu = min(mu, float(stock_cap_end))

        beta_used = beta_raw
        if beta_used is None or not np.isfinite(beta_used) or beta_used > -BETA_FLOOR:
            beta_used = -BETA_FLOOR

        # калибруем наклон возле REG_PRICE
        mu_adj = mu
        if beta_raw is not None and np.isfinite(beta_raw):
            rel_log = _safe_log(float(p)) - _safe_log(float(reg_price))
            mu_adj = float(mu) * float(np.exp((beta_used - beta_raw) * rel_log))
            mu_adj = max(mu_adj, 0.0)
            if stock_cap_end is not None and np.isfinite(stock_cap_end):
                mu_adj = min(mu_adj, float(stock_cap_end))

        score = objective(float(p), float(mu_adj), float(reg_price), qty_scale)
        rev = float(p) * float(mu_adj)

        if best is None or score > best[0]:
            best = (score, float(p), float(mu_adj), float(rev))
            meta = (beta_raw, beta_used, used_level, pos_sales_used, promo_cap_price)

    if best is None:
        return None

    _, p_star, mu_star, rev_star = best
    beta_raw, beta_used, used_level, pos_sales_used, promo_cap_price = meta

    p_lo = p_star * (1.0 - LOCAL_ELASTICITY_DELTA)
    p_hi = p_star * (1.0 + LOCAL_ELASTICITY_DELTA)

    mu_lo, _, _, _ = predict_backoff_dual(base_row, p_lo, reg_price, bucket, week_monday, promo_share, feature_names, hier)
    mu_hi, _, _, _ = predict_backoff_dual(base_row, p_hi, reg_price, bucket, week_monday, promo_share, feature_names, hier)
    mu_lo = float(mu_lo) if mu_lo is not None else 0.0
    mu_hi = float(mu_hi) if mu_hi is not None else 0.0

    if stock_cap_end is not None and np.isfinite(stock_cap_end):
        mu_lo = min(mu_lo, float(stock_cap_end))
        mu_hi = min(mu_hi, float(stock_cap_end))

    e_loc = local_elasticity(p_star, mu_lo, mu_hi, LOCAL_ELASTICITY_DELTA)

    quality = "OK" if (pos_sales_used >= MIN_POS_SALES_FOR_ELAST and reg_price > 0) else "LOW_DATA"

    dd = discount_depth(p_star, reg_price)
    pcl = price_change_log(p_star, reg_price)
    promo_depth = dd * promo_share

    return {
        "CHOSEN_PRICE": p_star,
        "PRED_QTY": mu_star,
        "PRED_REV": rev_star,
        "OWN_ELASTICITY_RAW": beta_raw,
        "OWN_ELASTICITY_USED": beta_used,
        "LOCAL_ELASTICITY": e_loc,
        "ELASTICITY_QUALITY": quality,
        "USED_LEVEL": used_level,
        "POS_SALES_USED": pos_sales_used,
        "DISCOUNT_DEPTH": dd,
        "PRICE_CHANGE_LOG": pcl,
        "PROMO_DEPTH": promo_depth,
        "PROMO_CAP_PRICE": promo_cap_price,
    }


# =========================
# Price ladder
# =========================

def collect_observed_price_ladder_by_family(bucket_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    ladders: Dict[str, np.ndarray] = {}
    for fam, g in bucket_df.groupby("FAMILY_CODE"):
        prices = pd.to_numeric(g["PRICE"], errors="coerce").astype(float).values
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) > 0:
            ladders[str(fam)] = np.unique(np.round(prices, 4))
    return ladders


def build_price_grid(anchor_price: float, observed_prices: Optional[np.ndarray], n_mult: int = 15) -> np.ndarray:
    """
    ВАЖНО: сетку строим вокруг anchor_price (обычно REG_PRICE).
    Это делает оптимизацию стабильнее, если BASE_PRICE пропал.
    """
    anchor_price = float(anchor_price)
    if not np.isfinite(anchor_price) or anchor_price <= 0:
        return np.array([], dtype=float)

    lo = anchor_price * MIN_MULT
    hi = anchor_price * MAX_MULT
    grid = anchor_price * np.linspace(MIN_MULT, MAX_MULT, n_mult)

    if observed_prices is not None and len(observed_prices) > 0:
        ladder = np.unique(observed_prices[np.isfinite(observed_prices) & (observed_prices > 0)])
        ladder = ladder[(ladder >= lo) & (ladder <= hi)]
        if len(ladder) > 0:
            grid = np.array([ladder[np.argmin(np.abs(ladder - p))] for p in grid], dtype=float)

    grid = grid[(grid >= lo) & (grid <= hi)]
    return np.sort(np.unique(np.round(grid, 4)))


# =========================
# Context helpers
# =========================

def get_base_price_and_stock_end(bucket_df: pd.DataFrame,
                                 sku: str,
                                 store: str,
                                 bucket: str,
                                 cutoff: pd.Timestamp) -> Tuple[Optional[float], Optional[float]]:
    df_b = bucket_df.loc[
        (bucket_df["PRODUCT_CODE"] == sku) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET"] == bucket) &
        (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff))
    ].sort_values("BUCKET_END")

    if len(df_b) > 0 and np.isfinite(df_b["PRICE"].iloc[-1]):
        base_p = float(df_b["PRICE"].iloc[-1])
        stock_end = float(df_b["STOCK_END"].iloc[-1]) if np.isfinite(df_b["STOCK_END"].iloc[-1]) else None
        return base_p, stock_end

    df_any = bucket_df.loc[
        (bucket_df["PRODUCT_CODE"] == sku) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff))
    ].sort_values("BUCKET_END")
    if len(df_any) == 0:
        return None, None

    base_p = float(df_any["PRICE"].iloc[-1]) if np.isfinite(df_any["PRICE"].iloc[-1]) else None
    stock_end = float(df_any["STOCK_END"].iloc[-1]) if np.isfinite(df_any["STOCK_END"].iloc[-1]) else None
    return base_p, stock_end


def build_context(bucket_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()
    recent = train.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(12)
    return recent.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(1).copy()


# =========================
# Cross-family (как в v4, но cap по REG_PRICE)
# =========================

@dataclass
class CrossFamilyModel:
    family_code: str
    store: str
    bucket: str
    family_products: List[str]
    scaler: StandardScaler
    models_by_product: Dict[str, PoissonRegressor]


def build_family_topk_for_context(bucket_df: pd.DataFrame,
                                  cutoff: pd.Timestamp,
                                  context_keys: Set[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], List[str]]:
    df = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()
    min_week = pd.Timestamp(cutoff) - pd.Timedelta(weeks=int(CROSS_LOOKBACK_WEEKS))
    df = df.loc[df["WEEK"] >= min_week]

    topk_map: Dict[Tuple[str, str, str], List[str]] = {}
    for (fam, store, bucket), g in df.groupby(["FAMILY_CODE", "STORE", "BUCKET"]):
        key = (str(fam), str(store), str(bucket))
        if key not in context_keys:
            continue
        if g["WEEK"].nunique() < CROSS_MIN_FAMILY_WEEKS:
            continue
        if float(g["QTY"].sum()) < CROSS_MIN_FAMILY_SUM_QTY:
            continue
        sku_sales = g.groupby("PRODUCT_CODE")["QTY"].sum().sort_values(ascending=False)
        skus = [str(x) for x in sku_sales.head(CROSS_TOP_K).index.tolist()]
        if len(skus) >= 2:
            topk_map[key] = skus
    return topk_map


def make_family_wide_panel(bucket_df: pd.DataFrame,
                           fam: str,
                           store: str,
                           bucket: str,
                           skus: List[str],
                           cutoff: pd.Timestamp) -> pd.DataFrame:
    df = bucket_df.loc[
        (bucket_df["FAMILY_CODE"] == fam) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET"] == bucket) &
        (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)) &
        (bucket_df["PRODUCT_CODE"].isin(skus))
    ].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df = df.groupby(["WEEK", "PRODUCT_CODE"], as_index=False).agg(QTY=("QTY", "sum"), PRICE=("PRICE", "mean"))

    qty_w = df.pivot(index="WEEK", columns="PRODUCT_CODE", values="QTY").add_prefix("Q_")
    prc_w = df.pivot(index="WEEK", columns="PRODUCT_CODE", values="PRICE").add_prefix("P_")
    wide = qty_w.join(prc_w, how="outer").sort_index()

    for sku in skus:
        pcol = f"P_{sku}"
        lcol = f"LOGP_{sku}"
        wide[lcol] = wide[pcol].apply(lambda v: _safe_log(v) if np.isfinite(v) and v > 0 else np.nan) if pcol in wide.columns else np.nan

    log_cols = [f"LOGP_{s}" for s in skus]
    wide[log_cols] = wide[log_cols].ffill()
    return wide


def train_cross_family_models(bucket_df: pd.DataFrame,
                              cutoff: pd.Timestamp,
                              topk_map: Dict[Tuple[str, str, str], List[str]]) -> Dict[Tuple[str, str, str], CrossFamilyModel]:
    out: Dict[Tuple[str, str, str], CrossFamilyModel] = {}

    for (fam, store, bucket), skus in topk_map.items():
        wide = make_family_wide_panel(bucket_df, fam, store, bucket, skus, cutoff=cutoff)
        if wide.empty:
            continue

        X_cols = [f"LOGP_{sku}" for sku in skus]
        X_all = wide[X_cols].astype(float).replace([np.inf, -np.inf], np.nan)

        scaler = StandardScaler()
        scaler.fit(X_all.fillna(0.0).values)

        models_by_product: Dict[str, PoissonRegressor] = {}
        for sku in skus:
            y_col = f"Q_{sku}"
            if y_col not in wide.columns:
                continue
            data = pd.concat([wide[y_col], X_all], axis=1).dropna(subset=[y_col, f"LOGP_{sku}"])
            if len(data) < CROSS_MIN_ROWS_PER_SKU:
                continue
            y = np.clip(data[y_col].astype(float).values, 0.0, None)
            if (y > 0).sum() < 5:
                continue
            Xs = scaler.transform(X_all.loc[data.index].fillna(0.0).values)

            reg = PoissonRegressor(alpha=max(POISSON_ALPHA, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
            reg.fit(Xs, y)
            models_by_product[sku] = reg

        if len(models_by_product) >= 2:
            out[(fam, store, bucket)] = CrossFamilyModel(fam, store, bucket, skus, scaler, models_by_product)

    return out


def predict_family_demands(cfm: CrossFamilyModel, prices: Dict[str, float]) -> Dict[str, float]:
    X_raw = np.array([_safe_log(prices[sku]) for sku in cfm.family_products], dtype=float).reshape(1, -1)
    Xs = cfm.scaler.transform(X_raw)
    out: Dict[str, float] = {}
    for sku, reg in cfm.models_by_product.items():
        out[sku] = max(float(reg.predict(Xs)[0]), 0.0)
    return out


# =========================
# Run (single + cross)
# =========================

def buckets_for_mode(mode: str) -> List[str]:
    mode = mode.lower().strip()
    if mode == "monday":
        return ["MON_THU"]
    if mode == "friday":
        return ["FRI", "SAT_SUN"]
    raise ValueError("mode must be 'monday' or 'friday'")


def calculate_for_week(bucket_df: pd.DataFrame,
                       daily_df: pd.DataFrame,
                       as_of_date: pd.Timestamp,
                       week_monday: pd.Timestamp,
                       mode: str) -> pd.DataFrame:
    today = pd.Timestamp(as_of_date).normalize()
    promo_map = build_promo_map(daily_df)

    buckets = buckets_for_mode(mode)

    cutoffs: Dict[str, pd.Timestamp] = {}
    for b in buckets:
        rule = decision_cutoff_for_bucket_by_mode(week_monday, b, mode).normalize()
        cutoffs[b] = min(rule, today, HISTORY_END.normalize())

    max_cutoff = max(cutoffs.values())
    train_all = bucket_df.loc[bucket_df["BUCKET_END"] <= max_cutoff].copy()
    if len(train_all) == 0:
        return pd.DataFrame()

    fam_ladder = collect_observed_price_ladder_by_family(train_all)

    feature_names = [
        "LOG_PRICE_L1",
        "LOG_PRICE",
        "DISCOUNT_DEPTH",
        "PRICE_CHANGE_LOG",
        "PROMO_DEPTH",
        "PROMO_SHARE",
        "BUCKET_IDX",
        "SIN_WOY",
        "COS_WOY",
        "TREND_W",
        "LOW_STOCK_FLAG",
        "STOCK_END_L1",
        "OOS_L1",
        "QTY_L1",
        "PROMO_L1",
    ]

    hier = train_hier_dual_models(train_all, cutoff=max_cutoff, feature_names=feature_names, alpha=POISSON_ALPHA)

    out_rows: List[dict] = []

    for bucket in buckets:
        start, end = bucket_start_end(week_monday, bucket)
        cutoff_rule = decision_cutoff_for_bucket_by_mode(week_monday, bucket, mode).normalize()
        cutoff = cutoffs[bucket]

        ctx = build_context(bucket_df, cutoff=cutoff)
        if len(ctx) == 0:
            continue

        # Cross models
        context_keys = set((str(f), str(s), bucket) for f, s in zip(ctx["FAMILY_CODE"].astype(str), ctx["STORE"].astype(str)))
        topk_map = build_family_topk_for_context(train_all, cutoff=cutoff, context_keys=context_keys)
        cross_models = train_cross_family_models(train_all, cutoff=cutoff, topk_map=topk_map)

        used_in_cross: Set[Tuple[str, str]] = set()

        # ---- Cross-family optimize ----
        for (fam, store), gfam in ctx.groupby(["FAMILY_CODE", "STORE"]):
            key = (str(fam), str(store), str(bucket))
            cfm = cross_models.get(key)
            if cfm is None:
                continue

            base_prices: Dict[str, Optional[float]] = {}
            stock_caps: Dict[str, Optional[float]] = {}
            promo_shares: Dict[str, float] = {}

            # REG_PRICE per sku, grid anchored by REG_PRICE, then promo-cap at REG_PRICE
            reg_prices: Dict[str, Optional[float]] = {}
            reg_sources: Dict[str, str] = {}
            grids: Dict[str, np.ndarray] = {}
            promo_caps: Dict[str, Optional[float]] = {}

            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                store_s = str(store)

                ranges = promo_map.get((sku, store_s), [])
                ps = promo_share_in_range(ranges, start, end) if ranges else 0.0

                base_p, stock_end = get_base_price_and_stock_end(train_all, sku, store_s, bucket, cutoff=cutoff)

                reg_p, reg_src = compute_reg_price(train_all, sku, store_s, bucket, cutoff=cutoff, base_price=base_p,
                                                   median_weeks=REG_PRICE_MEDIAN_WEEKS)
                if not is_valid_price(reg_p):
                    continue

                ladder = fam_ladder.get(str(fam))
                grid = build_price_grid(float(reg_p), observed_prices=ladder)

                grid2, cap_price = apply_promo_cap_to_grid(grid, reg_p, ps)

                base_prices[sku] = base_p
                stock_caps[sku] = stock_end
                promo_shares[sku] = float(ps)

                reg_prices[sku] = float(reg_p)
                reg_sources[sku] = reg_src
                grids[sku] = grid2
                promo_caps[sku] = cap_price

            active = [sku for sku in cfm.family_products if sku in reg_prices and sku in cfm.models_by_product and len(grids.get(sku, [])) > 0]
            if len(active) < 2:
                continue

            prices = {sku: float(reg_prices[sku]) for sku in active}

            for _ in range(CROSS_ITERS):
                for sku in active:
                    grid = grids.get(sku)
                    if grid is None or len(grid) == 0:
                        continue
                    best = None
                    for p in grid:
                        tmp = dict(prices)
                        tmp[sku] = float(p)
                        mu = float(predict_family_demands(cfm, tmp).get(sku, 0.0))

                        cap = stock_caps.get(sku)
                        if cap is not None and np.isfinite(cap):
                            mu = min(mu, float(cap))

                        score = objective(float(p), mu, reg_prices[sku], qty_scale=1.0)
                        if best is None or score > best[0]:
                            best = (score, float(p), mu)
                    if best is not None:
                        prices[sku] = best[1]

            mu_all = predict_family_demands(cfm, prices)

            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                if sku not in prices:
                    continue

                ps = promo_shares.get(sku, 0.0)
                stock_end = stock_caps.get(sku)

                base_p = base_prices.get(sku)
                reg_p = reg_prices.get(sku)
                reg_src = reg_sources.get(sku, "FALLBACK_NONE")
                cap_p = promo_caps.get(sku)

                chosen_price = float(prices[sku])
                mu = float(mu_all.get(sku, 0.0))
                if stock_end is not None and np.isfinite(stock_end):
                    mu = min(mu, float(stock_end))

                p = chosen_price
                p_lo = p * (1.0 - LOCAL_ELASTICITY_DELTA)
                p_hi = p * (1.0 + LOCAL_ELASTICITY_DELTA)
                tmp_lo = dict(prices); tmp_lo[sku] = p_lo
                tmp_hi = dict(prices); tmp_hi[sku] = p_hi
                mu_lo = float(predict_family_demands(cfm, tmp_lo).get(sku, 0.0))
                mu_hi = float(predict_family_demands(cfm, tmp_hi).get(sku, 0.0))
                if stock_end is not None and np.isfinite(stock_end):
                    mu_lo = min(mu_lo, float(stock_end))
                    mu_hi = min(mu_hi, float(stock_end))
                e_loc = local_elasticity(p, mu_lo, mu_hi, LOCAL_ELASTICITY_DELTA)

                out_rows.append({
                    "PRODUCT_CODE": sku,
                    "STORE": str(store),
                    "FAMILY_CODE": str(fam),

                    "AS_OF_DATE": today,
                    "TARGET_WEEK": week_monday,
                    "TARGET_BUCKET": bucket,
                    "BUCKET_START": start,
                    "BUCKET_END": end,

                    "DECISION_CUTOFF_RULE": cutoff_rule,
                    "CUTOFF_EFFECTIVE": cutoff,

                    "PROMO_SHARE": ps,
                    "STOCK_END_BUCKET_CAP": stock_end,

                    "BASE_PRICE": base_p,
                    "REG_PRICE": reg_p,
                    "REG_PRICE_SOURCE": reg_src,

                    "PROMO_CAP_PRICE": cap_p,
                    "PROMO_PRICE_CAPPED": bool(ps > PROMO_THRESHOLD and is_valid_price(cap_p)),

                    "CHOSEN_PRICE": chosen_price,
                    "PRICE_MULT": chosen_price / max(float(reg_p), 1e-6) if reg_p else np.nan,

                    "PRED_QTY": mu,
                    "PRED_REV": chosen_price * mu,

                    "USED_CROSS_PRICE_MODEL": True,

                    "OWN_ELASTICITY_RAW": None,
                    "OWN_ELASTICITY_USED": None,
                    "LOCAL_ELASTICITY": e_loc,
                    "ELASTICITY_QUALITY": "OK",

                    "DISCOUNT_DEPTH": discount_depth(chosen_price, float(reg_p)) if reg_p else 0.0,
                    "PRICE_CHANGE_LOG": price_change_log(chosen_price, float(reg_p)) if reg_p else 0.0,
                    "PROMO_DEPTH": discount_depth(chosen_price, float(reg_p)) * float(ps) if reg_p else 0.0,
                })

                used_in_cross.add((sku, str(store)))

        # ---- Single fallback ----
        for _, r in ctx.iterrows():
            sku = str(r["PRODUCT_CODE"])
            store = str(r["STORE"])
            fam = str(r["FAMILY_CODE"])

            if (sku, store) in used_in_cross:
                continue

            ranges = promo_map.get((sku, store), [])
            promo_share = promo_share_in_range(ranges, start, end) if ranges else 0.0

            base_p, stock_end = get_base_price_and_stock_end(train_all, sku, store, bucket, cutoff=cutoff)

            reg_p, reg_src = compute_reg_price(train_all, sku, store, bucket, cutoff=cutoff, base_price=base_p,
                                               median_weeks=REG_PRICE_MEDIAN_WEEKS)
            if not is_valid_price(reg_p):
                continue

            base_row = r.copy()
            base_row["PROMO_SHARE"] = promo_share

            ladder = fam_ladder.get(fam)
            grid = build_price_grid(float(reg_p), observed_prices=ladder)

            res = optimize_single(
                base_row=base_row,
                bucket=bucket,
                week_monday=week_monday,
                promo_share=promo_share,
                feature_names=feature_names,
                hier=hier,
                grid=grid,
                reg_price=float(reg_p),
                stock_cap_end=stock_end
            )
            if res is None:
                continue

            out_rows.append({
                "PRODUCT_CODE": sku,
                "STORE": store,
                "FAMILY_CODE": fam,

                "AS_OF_DATE": today,
                "TARGET_WEEK": week_monday,
                "TARGET_BUCKET": bucket,
                "BUCKET_START": start,
                "BUCKET_END": end,

                "DECISION_CUTOFF_RULE": cutoff_rule,
                "CUTOFF_EFFECTIVE": cutoff,

                "PROMO_SHARE": promo_share,
                "STOCK_END_BUCKET_CAP": stock_end,

                "BASE_PRICE": base_p,
                "REG_PRICE": float(reg_p),
                "REG_PRICE_SOURCE": reg_src,

                "PROMO_CAP_PRICE": res.get("PROMO_CAP_PRICE"),
                "PROMO_PRICE_CAPPED": bool(promo_share > PROMO_THRESHOLD and is_valid_price(res.get("PROMO_CAP_PRICE"))),

                "CHOSEN_PRICE": float(res["CHOSEN_PRICE"]),
                "PRICE_MULT": float(res["CHOSEN_PRICE"]) / max(float(reg_p), 1e-6),

                "PRED_QTY": float(res["PRED_QTY"]),
                "PRED_REV": float(res["PRED_REV"]),

                "USED_CROSS_PRICE_MODEL": False,

                "OWN_ELASTICITY_RAW": res["OWN_ELASTICITY_RAW"],
                "OWN_ELASTICITY_USED": res["OWN_ELASTICITY_USED"],
                "LOCAL_ELASTICITY": res["LOCAL_ELASTICITY"],
                "ELASTICITY_QUALITY": res["ELASTICITY_QUALITY"],

                "USED_LEVEL": res["USED_LEVEL"],
                "POS_SALES_USED": res["POS_SALES_USED"],

                "DISCOUNT_DEPTH": res["DISCOUNT_DEPTH"],
                "PRICE_CHANGE_LOG": res["PRICE_CHANGE_LOG"],
                "PROMO_DEPTH": res["PROMO_DEPTH"],
            })

    return pd.DataFrame(out_rows)


def run_for_weeks(bucket_df: pd.DataFrame,
                  daily_df: pd.DataFrame,
                  as_of_date: pd.Timestamp,
                  mode: str,
                  weeks_back: int = 0) -> pd.DataFrame:
    today = pd.Timestamp(as_of_date).normalize()
    frames = []
    for w in range(0, int(weeks_back) + 1):
        wk = week_start(today) - pd.Timedelta(weeks=w)
        df_w = calculate_for_week(bucket_df, daily_df, today, wk, mode)
        if len(df_w) > 0:
            df_w["WEEKS_BACK"] = w
            df_w["RUN_MODE"] = mode
            frames.append(df_w)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =========================
# Pipeline
# =========================

def run_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    daily = preprocess_raw(df_raw, history_end=HISTORY_END)
    buckets = aggregate_to_buckets(daily)
    buckets = add_lag_features(buckets, lags=1)
    return buckets, daily


# =========================
# Пример использования
# =========================

"""
1) Собираем bucket-таблицу и daily:
    buckets_df, daily_df = run_pipeline(df_raw)

2) Запуск в понедельник утром (считаем цены на MON_THU) и, например, ещё на 2 предыдущие недели:
    out_mon = run_for_weeks(
        bucket_df=buckets_df,
        daily_df=daily_df,
        as_of_date=pd.Timestamp("2026-02-12"),  # дата запуска (факт. "сегодня")
        mode="monday",
        weeks_back=2
    )

3) Запуск в пятницу утром (считаем цены на FRI и SAT_SUN) с тем же weeks_back:
    out_fri = run_for_weeks(
        bucket_df=buckets_df,
        daily_df=daily_df,
        as_of_date=pd.Timestamp("2026-02-16"),
        mode="friday",
        weeks_back=2
    )

4) Дальше можно объединить:
    out_all = pd.concat([out_mon, out_fri], ignore_index=True)

Ключевые колонки результата:
- TARGET_BUCKET, BUCKET_START/END
- REG_PRICE, REG_PRICE_SOURCE (как нашли регулярную цену)
- PROMO_SHARE, PROMO_CAP_PRICE, PROMO_PRICE_CAPPED
- CHOSEN_PRICE, PRICE_MULT
- USED_CROSS_PRICE_MODEL (учли ли каннибализацию)
- OWN_ELASTICITY_*, LOCAL_ELASTICITY
"""







# === FIX (v5): add missing training-time features DISCOUNT_DEPTH / PRICE_CHANGE_LOG / PROMO_DEPTH ===
# Причина: эти фичи создавались только при scoring кандидатов, но не были рассчитаны в train-таблице,
# поэтому g[feature_names] падал с "not in index".

def add_training_regular_price_features(bucket_df: pd.DataFrame,
                                        median_weeks: int = REG_PRICE_MEDIAN_WEEKS,
                                        min_points: int = REG_PRICE_MIN_POINTS) -> pd.DataFrame:
    """
    Добавляет в bucket_df "обучающие" аналоги regular-price фичей:
      - REG_PRICE_TRAIN: регулярная цена (медиана цены за последние N недель, без текущей точки)
      - DISCOUNT_DEPTH: max(1 - PRICE/REG_PRICE_TRAIN, 0)
      - PRICE_CHANGE_LOG: log(PRICE) - log(REG_PRICE_TRAIN)
      - PROMO_DEPTH: DISCOUNT_DEPTH * PROMO_SHARE

    Это нужно, чтобы те же фичи существовали и в train, и в scoring.
    """
    d = bucket_df.copy().sort_values(["PRODUCT_CODE", "STORE", "BUCKET", "WEEK"])

    # rolling median регулярной цены (по SKU+STORE+BUCKET), исключая текущую неделю
    grp = d.groupby(["PRODUCT_CODE", "STORE", "BUCKET"], sort=False)
    roll = (
        grp["PRICE"]
        .apply(lambda s: s.shift(1).rolling(window=int(median_weeks), min_periods=int(min_points)).median())
        .reset_index(level=[0, 1, 2], drop=True)
    )
    d["REG_PRICE_TRAIN"] = roll

    # fallback: если медианы нет, используем прошлую цену (PRICE_L1), а если и её нет — текущую PRICE
    if "PRICE_L1" in d.columns:
        d["REG_PRICE_TRAIN"] = d["REG_PRICE_TRAIN"].fillna(d["PRICE_L1"])
    d["REG_PRICE_TRAIN"] = d["REG_PRICE_TRAIN"].fillna(d["PRICE"])

    # безопасные вычисления
    rp = pd.to_numeric(d["REG_PRICE_TRAIN"], errors="coerce").astype(float)
    p = pd.to_numeric(d["PRICE"], errors="coerce").astype(float)

    rp = rp.where(np.isfinite(rp) & (rp > 0), np.nan)
    p = p.where(np.isfinite(p) & (p > 0), np.nan)

    d["DISCOUNT_DEPTH"] = (1.0 - (p / rp)).clip(lower=0.0).fillna(0.0)
    d["PRICE_CHANGE_LOG"] = (p.apply(_safe_log) - rp.apply(_safe_log)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["PROMO_DEPTH"] = (d["DISCOUNT_DEPTH"] * pd.to_numeric(d.get("PROMO_SHARE", 0.0), errors="coerce").fillna(0.0)).astype(float)

    return d


# --- 1) В train_hier_dual_models() добавь одну строку сразу после train = ... ---
def train_hier_dual_models(bucket_df: pd.DataFrame,
                           cutoff: pd.Timestamp,
                           feature_names: List[str],
                           alpha: float = POISSON_ALPHA) -> HierDualModels:
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()

    # FIX: добавляем недостающие фичи для обучения (чтобы X = g[feature_names] работал)
    train = add_training_regular_price_features(train)

    # ... дальше код без изменений ...


# --- 2) На всякий случай сделай защиту в fit_poisson() (если кто-то забудет вызвать функцию выше) ---
def fit_poisson(df: pd.DataFrame, feature_names: List[str], alpha: float = POISSON_ALPHA) -> Optional[PoissonModel]:
    g = df.copy()

    # FIX: если фич нет (например, модель обучают на "сыром" bucket_df), добавим их локально
    need = {"DISCOUNT_DEPTH", "PRICE_CHANGE_LOG", "PROMO_DEPTH"}
    if len(need - set(g.columns)) > 0:
        g = add_training_regular_price_features(g)

    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    g["LOG_PRICE"] = g["PRICE"].astype(float).apply(_safe_log)
    y = np.clip(g["QTY"].astype(float).values, 0.0, None)
    if not np.isfinite(y).all():
        return None

    # теперь колонки гарантированно существуют
    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = PoissonRegressor(alpha=max(alpha, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
    reg.fit(Xs, y)

    return PoissonModel(reg, scaler, list(feature_names), int(len(g)), int((y > 0).sum()))




def fit_poisson(df: pd.DataFrame, feature_names: List[str], alpha: float = POISSON_ALPHA) -> Optional[PoissonModel]:
    g = df.copy()

    # FIX: если фич нет (например, модель обучают на "сыром" bucket_df), добавим их локально
    need = {"DISCOUNT_DEPTH", "PRICE_CHANGE_LOG", "PROMO_DEPTH"}
    if len(need - set(g.columns)) > 0:
        g = add_training_regular_price_features(g)

    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    g["LOG_PRICE"] = g["PRICE"].astype(float).apply(_safe_log)
    y = np.clip(g["QTY"].astype(float).values, 0.0, None)
    if not np.isfinite(y).all():
        return None

    # теперь колонки гарантированно существуют
    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = PoissonRegressor(alpha=max(alpha, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
    reg.fit(Xs, y)

    return PoissonModel(reg, scaler, list(feature_names), int(len(g)), int((y > 0).sum()))






