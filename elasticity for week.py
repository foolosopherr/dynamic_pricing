# Основные требования:
# 1) Товары появляются/исчезают, строки с 0 продаж есть (панель).
# 2) Иерархия: PRODUCT -> FAMILY -> CATEGORY -> SEGMENT
# 3) Конец истории: 2025-09-15 (HISTORY_END)
# 4) Цены меняются только между бакетами MON_THU / FRI / SAT_SUN
# 5) Rolling оценка последние 12 недель (есть функция backtest)
# 7) "правильный" cutoff для бакета: MON_THU=вс прошлой недели, FRI=чт, SAT_SUN=пт
#    Но в понедельник мы НЕ видим будущие дни => cutoff_effective = min(rule_cutoff, as_of_date)
# 8) PROMO_PERIOD строка "01-01-2024 - 03-01-2024"
#
# Важно про "явную эластичность":
# - Модель: log(E[Q]) = ... + beta * log(P) + ...
# - При StandardScaler коэффициенты в масштабе z-score.
# - Эластичность в исходных единицах:
#       beta_raw = beta_scaled / std(log(P))
# - Для cross: gamma_raw = coef_scaled[j] / std(log(P_j))

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


# =========================
# Конфигурация
# =========================

HISTORY_END = pd.Timestamp("2025-09-15")

BUCKET_ORDER = ["MON_THU", "FRI", "SAT_SUN"]

# Ограничения цены
MIN_MULT = 0.9
MAX_MULT = 1.3

# Poisson GLM
POISSON_ALPHA = 1e-2          # регуляризация (увеличивайте, если плохо сходится)
POISSON_MAX_ITER = 3000

# Backoff shrinkage (на уровне предсказаний)
DEFAULT_SHRINK_K = 30

# Минимум строк/недель для обучения уровней
MIN_ROWS_GLOBAL = 500
MIN_ROWS_SEGMENT = 600
MIN_ROWS_CATEGORY = 300
MIN_ROWS_FAMILY = 120
MIN_ROWS_PRODUCT_STORE = 40

# Cross внутри family (каннибализация/кросс-эластичность)
CROSS_TOP_K = 5               # было 10 -> сильно быстрее и стабильнее
CROSS_LOOKBACK_WEEKS = 26
CROSS_MIN_ROWS_PER_SKU = 40
CROSS_ITERS = 4

# Фильтры для ускорения cross:
CROSS_MIN_FAMILY_SUM_QTY = 50     # если за lookback в семье мало продаж -> пропускаем cross
CROSS_MIN_FAMILY_WEEKS = 30       # если мало недель -> пропускаем cross


# =========================
# Вспомогательные даты/бакеты
# =========================

def bucket_id_for_date(dt: pd.Timestamp) -> str:
    """
    Бакет по дню недели:
      MON_THU (пн-чт), FRI (пт), SAT_SUN (сб-вс)
    """
    wd = pd.Timestamp(dt).weekday()  # Mon=0..Sun=6
    if wd <= 3:
        return "MON_THU"
    if wd == 4:
        return "FRI"
    return "SAT_SUN"


def week_start(dt: pd.Timestamp) -> pd.Timestamp:
    """Понедельник недели для даты dt."""
    dt = pd.Timestamp(dt).normalize()
    return dt - pd.Timedelta(days=dt.weekday())


def bucket_start_end(week_monday: pd.Timestamp, bucket: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Интервал бакета внутри недели (включительно)."""
    m = pd.Timestamp(week_monday).normalize()
    if bucket == "MON_THU":
        return m, m + pd.Timedelta(days=3)
    if bucket == "FRI":
        d = m + pd.Timedelta(days=4)
        return d, d
    if bucket == "SAT_SUN":
        return m + pd.Timedelta(days=5), m + pd.Timedelta(days=6)
    raise ValueError(f"Unknown bucket={bucket}")


def decision_cutoff_for_bucket(week_monday: pd.Timestamp, bucket: str) -> pd.Timestamp:
    """
    Cutoff по правилу (условие 7):
      - для MON_THU: воскресенье предыдущей недели
      - для FRI: четверг текущей недели
      - для SAT_SUN: пятница текущей недели
    """
    m = pd.Timestamp(week_monday).normalize()
    if bucket == "MON_THU":
        return m - pd.Timedelta(days=1)
    if bucket == "FRI":
        return m + pd.Timedelta(days=3)
    if bucket == "SAT_SUN":
        return m + pd.Timedelta(days=4)
    raise ValueError(f"Unknown bucket={bucket}")


def _safe_log(x: float) -> float:
    """Безопасный лог для устойчивости."""
    return float(np.log(max(float(x), 1e-6)))


# =========================
# PROMO_PERIOD
# =========================

def parse_promo_period(s: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Парсит PROMO_PERIOD вида "01-01-2024 - 03-01-2024" (dd-mm-yyyy).
    """
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
    """
    Доля дней в [start, end], покрытых промо.
    """
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
# Препроцессинг дневных данных
# =========================

def compute_unit_price(df: pd.DataFrame) -> pd.Series:
    """
    Цена за единицу:
      - если есть SALE_PRICE, используем
      - иначе SALE_PRICE_TOTAL / max(SALE_QTY,1)
    """
    if "SALE_PRICE" in df.columns:
        return pd.to_numeric(df["SALE_PRICE"], errors="coerce")
    total = pd.to_numeric(df.get("SALE_PRICE_TOTAL"), errors="coerce")
    qty = pd.to_numeric(df.get("SALE_QTY"), errors="coerce").fillna(0.0)
    return total / qty.clip(lower=1.0)


def preprocess_raw(df: pd.DataFrame, history_end: pd.Timestamp = HISTORY_END) -> pd.DataFrame:
    """
    Чистим дневные данные:
      - даты, фильтр <= history_end
      - QTY_TOTAL
      - UNIT_PRICE
      - BUCKET
      - PROMO_RANGE
      - нормализация ключей
      - важный фикс: QTY_TOTAL >= 0 (Poisson требует y>=0)
    """
    out = df.copy()

    out["TRADE_DT"] = pd.to_datetime(out["TRADE_DT"], errors="coerce").dt.normalize()
    out = out.loc[out["TRADE_DT"].notna()]
    out = out.loc[out["TRADE_DT"] <= pd.Timestamp(history_end).normalize()]

    qty_off = pd.to_numeric(out.get("SALE_QTY"), errors="coerce").fillna(0.0)
    qty_on = pd.to_numeric(out.get("SALE_QTY_ONLINE"), errors="coerce").fillna(0.0)

    qty_total = (qty_off + qty_on).astype(float)
    # ФИКС: клипуем неотрицательно (бывают отрицательные коррекции)
    out["QTY_TOTAL"] = qty_total.clip(lower=0.0)

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
    """Карта промо-интервалов по (PRODUCT_CODE, STORE)."""
    promo_map: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if "PROMO_RANGE" not in daily.columns:
        return promo_map
    for (p, s), g in daily.groupby(["PRODUCT_CODE", "STORE"]):
        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)]
        if ranges:
            promo_map[(str(p), str(s))] = ranges
    return promo_map


# =========================
# Агрегация до бакетов
# =========================

def aggregate_to_buckets(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Дневные -> бакет-панель:
      (PRODUCT, STORE, WEEK, BUCKET) -> QTY, PRICE, PROMO_SHARE, STOCK_*, flows

    ФИКС: QTY >= 0 (Poisson)
    """
    d = daily.copy()
    d["WEEK"] = d["TRADE_DT"].apply(week_start)

    group_cols = [
        "PRODUCT_CODE", "STORE", "WEEK", "BUCKET",
        "FAMILY_CODE", "CATEGORY_CODE", "SEGMENT_CODE",
        "REGION_NAME", "STORE_TYPE", "PLACE_TYPE"
    ]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        qty = float(g["QTY_TOTAL"].sum())
        qty = max(qty, 0.0)  # критично для Poisson

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

        stock_start = float(g["START_STOCK"].iloc[0]) if "START_STOCK" in g else 0.0
        stock_end = float(g["END_STOCK"].iloc[-1]) if "END_STOCK" in g else 0.0
        delivery = float(g["DELIVERY_QTY"].sum()) if "DELIVERY_QTY" in g else 0.0
        loss = float(g["LOSS_QTY"].sum()) if "LOSS_QTY" in g else 0.0
        ret = float(g["RETURN_QTY"].sum()) if "RETURN_QTY" in g else 0.0

        return pd.Series({
            "QTY": qty,
            "PRICE": price,
            "PROMO_SHARE": promo_share,
            "STOCK_START": stock_start,
            "STOCK_END": stock_end,
            "DELIVERY": delivery,
            "LOSS": loss,
            "RETURN": ret,
            "BUCKET_START": start,
            "BUCKET_END": end,
        })

    bucket_df = d.groupby(group_cols, as_index=False).apply(agg_group).reset_index(drop=True)

    bucket_df["WEEK_OF_YEAR"] = bucket_df["WEEK"].dt.isocalendar().week.astype(int)
    bucket_df["YEAR"] = bucket_df["WEEK"].dt.year.astype(int)
    bucket_df["BUCKET_IDX"] = bucket_df["BUCKET"].map({b: i for i, b in enumerate(BUCKET_ORDER)}).astype(int)

    # страховка по типам
    bucket_df["QTY"] = pd.to_numeric(bucket_df["QTY"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket_df["PRICE"] = pd.to_numeric(bucket_df["PRICE"], errors="coerce")

    return bucket_df


# =========================
# Лаги
# =========================

def add_lag_features(bucket_df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    """
    Лаги по (PRODUCT_CODE, STORE, BUCKET).
    """
    d = bucket_df.copy().sort_values(["PRODUCT_CODE", "STORE", "BUCKET", "WEEK"])
    grp = d.groupby(["PRODUCT_CODE", "STORE", "BUCKET"], sort=False)

    for k in range(1, lags + 1):
        d[f"QTY_L{k}"] = grp["QTY"].shift(k).fillna(0.0)
        d[f"PRICE_L{k}"] = grp["PRICE"].shift(k)
        d[f"PROMO_L{k}"] = grp["PROMO_SHARE"].shift(k).fillna(0.0)
        d[f"STOCK_END_L{k}"] = grp["STOCK_END"].shift(k).fillna(0.0)

    d["LOW_STOCK_FLAG"] = (d["STOCK_END_L1"].fillna(0.0) <= 1.0).astype(int)
    return d


# =========================
# Single Poisson модель + scaler
# =========================

@dataclass
class PoissonModel:
    """
    PoissonRegressor + StandardScaler.

    Замечание про эластичность:
      - мы используем признак LOG_PRICE = log(P)
      - модель на масштабированных признаках
      - own-elasticity (в исходных единицах) =
            coef_scaled[idx_LOG_PRICE] / scaler.scale_[idx_LOG_PRICE]
    """
    reg: PoissonRegressor
    scaler: StandardScaler
    feature_names: List[str]
    n: int


def fit_poisson_model(df: pd.DataFrame,
                      feature_names: List[str],
                      alpha: float = POISSON_ALPHA) -> Optional[PoissonModel]:
    """
    Обучает PoissonRegressor (с масштабированием признаков).

    ФИКСЫ:
      - y клипуем >=0
      - alpha >= 1e-2
      - max_iter увеличен
    """
    g = df.copy()
    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    y = g["QTY"].astype(float).values
    y = np.clip(y, 0.0, None)
    if not np.isfinite(y).all():
        return None

    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = PoissonRegressor(
        alpha=max(alpha, 1e-2),
        fit_intercept=True,
        max_iter=POISSON_MAX_ITER
    )
    reg.fit(Xs, y)

    return PoissonModel(reg=reg, scaler=scaler, feature_names=list(feature_names), n=int(len(g)))


def poisson_predict_mean(model: PoissonModel, x_row_raw: np.ndarray) -> float:
    """
    Предсказывает E[Q] (x_row в исходном масштабе признаков).
    """
    Xs = model.scaler.transform(x_row_raw.reshape(1, -1))
    mu = float(model.reg.predict(Xs)[0])
    return max(mu, 0.0)


def model_elasticity_raw(model: PoissonModel, feature_name: str = "LOG_PRICE") -> Optional[float]:
    """
    Возвращает эластичность в исходных единицах для признака (по умолчанию LOG_PRICE).

    beta_raw = beta_scaled / std(feature)
    """
    if feature_name not in model.feature_names:
        return None
    j = model.feature_names.index(feature_name)
    std = float(model.scaler.scale_[j]) if model.scaler.scale_ is not None else 1.0
    if std <= 0:
        return None
    return float(model.reg.coef_[j]) / std


# =========================
# Иерархические single-модели (backoff по предсказаниям)
# =========================

@dataclass
class HierModels:
    """
    Модели по уровням:
      product_store, family, category, segment, global
    """
    global_model: Optional[PoissonModel]
    segment: Dict[str, PoissonModel]
    category: Dict[str, PoissonModel]
    family: Dict[str, PoissonModel]
    product_store: Dict[Tuple[str, str], PoissonModel]


def train_hier_models(bucket_df: pd.DataFrame,
                      cutoff: pd.Timestamp,
                      feature_names: List[str],
                      alpha: float = POISSON_ALPHA) -> HierModels:
    """
    Обучает single Poisson модели на данных BUCKET_END <= cutoff.
    """
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()

    global_model = None
    if len(train) >= MIN_ROWS_GLOBAL:
        global_model = fit_poisson_model(train, feature_names, alpha=alpha)

    seg_models: Dict[str, PoissonModel] = {}
    for seg, g in train.groupby("SEGMENT_CODE"):
        if len(g) >= MIN_ROWS_SEGMENT:
            m = fit_poisson_model(g, feature_names, alpha=alpha)
            if m is not None:
                seg_models[str(seg)] = m

    cat_models: Dict[str, PoissonModel] = {}
    for cat, g in train.groupby("CATEGORY_CODE"):
        if len(g) >= MIN_ROWS_CATEGORY:
            m = fit_poisson_model(g, feature_names, alpha=alpha)
            if m is not None:
                cat_models[str(cat)] = m

    fam_models: Dict[str, PoissonModel] = {}
    for fam, g in train.groupby("FAMILY_CODE"):
        if len(g) >= MIN_ROWS_FAMILY:
            m = fit_poisson_model(g, feature_names, alpha=alpha)
            if m is not None:
                fam_models[str(fam)] = m

    ps_models: Dict[Tuple[str, str], PoissonModel] = {}
    for (p, s), g in train.groupby(["PRODUCT_CODE", "STORE"]):
        if len(g) >= MIN_ROWS_PRODUCT_STORE:
            m = fit_poisson_model(g, feature_names, alpha=alpha)
            if m is not None:
                ps_models[(str(p), str(s))] = m

    return HierModels(
        global_model=global_model,
        segment=seg_models,
        category=cat_models,
        family=fam_models,
        product_store=ps_models,
    )


def shrink_weight(n_child: int, k: int = DEFAULT_SHRINK_K) -> float:
    """Вес детской модели для shrinkage по предсказанию."""
    return float(n_child / (n_child + k))


def predict_mu_backoff(base_row: pd.Series,
                       candidate_price: float,
                       feature_names: List[str],
                       models: HierModels,
                       k: int = DEFAULT_SHRINK_K) -> Optional[float]:
    """
    Предсказывает mu = E[Q] для кандидата цены через backoff по предсказаниям:
      (PRODUCT,STORE) -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL

    Логика:
      - считаем mu на каждом доступном уровне
      - смешиваем последовательно shrink_weight(n_child)
    """
    feat = make_single_features_for_candidate(base_row, candidate_price)
    x = np.array([feat.get(fn, 0.0) for fn in feature_names], dtype=float)

    p = str(base_row["PRODUCT_CODE"])
    s = str(base_row["STORE"])
    fam = str(base_row["FAMILY_CODE"])
    cat = str(base_row["CATEGORY_CODE"])
    seg = str(base_row["SEGMENT_CODE"])

    mu = None

    # product_store
    m_ps = models.product_store.get((p, s))
    if m_ps is not None:
        mu = poisson_predict_mean(m_ps, x)

    # family
    m_fam = models.family.get(fam)
    if m_fam is not None:
        mu_fam = poisson_predict_mean(m_fam, x)
        if mu is None:
            mu = mu_fam
        else:
            w = shrink_weight(m_ps.n if m_ps is not None else 0, k=k)
            mu = w * mu + (1 - w) * mu_fam

    # category
    m_cat = models.category.get(cat)
    if m_cat is not None:
        mu_cat = poisson_predict_mean(m_cat, x)
        if mu is None:
            mu = mu_cat
        else:
            # вес текущей "детской" оценки считаем как n на более детальном уровне
            n_child = m_ps.n if m_ps is not None else (m_fam.n if m_fam is not None else 0)
            w = shrink_weight(n_child, k=k)
            mu = w * mu + (1 - w) * mu_cat

    # segment
    m_seg = models.segment.get(seg)
    if m_seg is not None:
        mu_seg = poisson_predict_mean(m_seg, x)
        if mu is None:
            mu = mu_seg
        else:
            n_child = m_ps.n if m_ps is not None else (m_fam.n if m_fam is not None else (m_cat.n if m_cat is not None else 0))
            w = shrink_weight(n_child, k=k)
            mu = w * mu + (1 - w) * mu_seg

    # global
    m_glb = models.global_model
    if m_glb is not None:
        mu_glb = poisson_predict_mean(m_glb, x)
        if mu is None:
            mu = mu_glb
        else:
            n_child = m_ps.n if m_ps is not None else (m_fam.n if m_fam is not None else (m_cat.n if m_cat is not None else (m_seg.n if m_seg is not None else 0)))
            w = shrink_weight(n_child, k=k)
            mu = w * mu + (1 - w) * mu_glb

    if mu is None or not np.isfinite(mu):
        return None
    return float(max(mu, 0.0))


# =========================
# Single признаки и оптимизация
# =========================

def make_single_features_for_candidate(base_row: pd.Series, candidate_price: float) -> Dict[str, float]:
    """
    Single признаки для кандидата цены.

    LOG_PRICE = log(P) -> базовый канал own-price elasticity.
    """
    return {
        "LOG_PRICE": float(_safe_log(candidate_price)),
        "PROMO_SHARE": float(base_row.get("PROMO_SHARE", 0.0)),
        "BUCKET_IDX": float(base_row.get("BUCKET_IDX", 0.0)),
        "WEEK_OF_YEAR": float(base_row.get("WEEK_OF_YEAR", 0.0)),
        "LOW_STOCK_FLAG": float(base_row.get("LOW_STOCK_FLAG", 0.0)),
        "QTY_L1": float(base_row.get("QTY_L1", 0.0)),
        "PRICE_L1": float(base_row.get("PRICE_L1", 0.0)) if np.isfinite(base_row.get("PRICE_L1", np.nan)) else 0.0,
        "PROMO_L1": float(base_row.get("PROMO_L1", 0.0)),
    }


def build_price_grid(base_price: float,
                     min_mult: float = MIN_MULT,
                     max_mult: float = MAX_MULT,
                     n_mult: int = 15,
                     observed_prices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Дискретная сетка цен:
      base_price * linspace(min_mult, max_mult, n_mult)
    + снап к лестнице observed_prices (если задана).
    """
    base_price = float(base_price)
    if not np.isfinite(base_price) or base_price <= 0:
        return np.array([], dtype=float)

    grid = base_price * np.linspace(min_mult, max_mult, n_mult)

    if observed_prices is not None and len(observed_prices) > 0:
        ladder = np.unique(observed_prices[np.isfinite(observed_prices) & (observed_prices > 0)])
        if len(ladder) > 0:
            grid = np.array([ladder[np.argmin(np.abs(ladder - p))] for p in grid], dtype=float)

    grid = np.unique(np.round(grid, 4))
    return np.sort(grid)


def collect_observed_price_ladder(bucket_df: pd.DataFrame, level: str = "FAMILY_CODE") -> Dict[str, np.ndarray]:
    """Лестница наблюдавшихся цен по уровню (например FAMILY_CODE)."""
    ladders: Dict[str, np.ndarray] = {}
    for key, g in bucket_df.groupby(level):
        prices = pd.to_numeric(g["PRICE"], errors="coerce").astype(float).values
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) > 0:
            ladders[str(key)] = np.unique(np.round(prices, 4))
    return ladders


def optimize_price_single_sku(base_row: pd.Series,
                             feature_names: List[str],
                             models: HierModels,
                             price_grid: np.ndarray,
                             stock_cap: Optional[float] = None,
                             shrink_k: int = DEFAULT_SHRINK_K) -> Optional[Tuple[float, float, float]]:
    """
    Single-SKU оптимизация:
      max_p p * E[Q(p)]  (через backoff предсказание)

    Возвращает:
      (best_price, mu, revenue)
    """
    if price_grid is None or len(price_grid) == 0:
        return None

    best = None
    for p in price_grid:
        mu = predict_mu_backoff(base_row, float(p), feature_names, models, k=shrink_k)
        if mu is None:
            continue

        if stock_cap is not None and np.isfinite(stock_cap) and mu > float(stock_cap):
            continue

        rev = float(p) * float(mu)
        if best is None or rev > best[2]:
            best = (float(p), float(mu), float(rev))

    # если всё отфильтровано по stock_cap, берём минимальный спрос
    if best is None and stock_cap is not None and np.isfinite(stock_cap):
        best2 = None
        for p in price_grid:
            mu = predict_mu_backoff(base_row, float(p), feature_names, models, k=shrink_k)
            if mu is None:
                continue
            rev = float(p) * float(mu)
            if best2 is None or mu < best2[1]:
                best2 = (float(p), float(mu), float(rev))
        return best2

    return best


# =========================
# Cross-family модели (каннибализация) + scaler
# =========================

@dataclass
class CrossFamilyModel:
    """
    Cross-модель внутри (FAMILY, STORE, BUCKET) для Top-K SKU.

    Для каждого SKU i:
      log(E[Q_i]) = a_i + sum_j gamma_{i,j} * log(P_j)
    => gamma_{i,j} — кросс-эластичности.
    """
    family_code: str
    store: str
    bucket: str
    family_products: List[str]                      # порядок признаков
    scaler: StandardScaler                          # общий scaler по X (log prices)
    models_by_product: Dict[str, PoissonRegressor]  # отдельная модель на каждый SKU i
    n_by_product: Dict[str, int]


def build_family_topk(bucket_df: pd.DataFrame,
                      cutoff: pd.Timestamp,
                      top_k: int = CROSS_TOP_K,
                      lookback_weeks: int = CROSS_LOOKBACK_WEEKS) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Top-K SKU по сумме QTY за lookback_weeks до cutoff внутри (FAMILY, STORE, BUCKET).
    + фильтры по объёму/неделям для ускорения.
    """
    df = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()
    min_week = pd.Timestamp(cutoff) - pd.Timedelta(weeks=int(lookback_weeks))
    df = df.loc[df["WEEK"] >= min_week]

    topk_map: Dict[Tuple[str, str, str], List[str]] = {}
    for (fam, store, bucket), g in df.groupby(["FAMILY_CODE", "STORE", "BUCKET"]):
        if g["WEEK"].nunique() < CROSS_MIN_FAMILY_WEEKS:
            continue
        fam_sum = float(g["QTY"].sum())
        if fam_sum < CROSS_MIN_FAMILY_SUM_QTY:
            continue

        sku_sales = g.groupby("PRODUCT_CODE")["QTY"].sum().sort_values(ascending=False)
        skus = [str(x) for x in sku_sales.head(top_k).index.tolist()]
        if len(skus) >= 2:
            topk_map[(str(fam), str(store), str(bucket))] = skus

    return topk_map


def make_family_wide_panel(bucket_df: pd.DataFrame,
                           family_code: str,
                           store: str,
                           bucket: str,
                           family_products: List[str],
                           cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Wide по WEEK:
      Q_<sku>, LOGP_<sku>
    Цены логируем и ffill для контекста конкурентов.
    """
    df = bucket_df.loc[
        (bucket_df["FAMILY_CODE"] == str(family_code)) &
        (bucket_df["STORE"] == str(store)) &
        (bucket_df["BUCKET"] == str(bucket)) &
        (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)) &
        (bucket_df["PRODUCT_CODE"].isin(family_products))
    ].copy()

    if len(df) == 0:
        return pd.DataFrame()

    df = df.groupby(["WEEK", "PRODUCT_CODE"], as_index=False).agg(
        QTY=("QTY", "sum"),
        PRICE=("PRICE", "mean"),
    )

    qty_w = df.pivot(index="WEEK", columns="PRODUCT_CODE", values="QTY").add_prefix("Q_")
    prc_w = df.pivot(index="WEEK", columns="PRODUCT_CODE", values="PRICE").add_prefix("P_")
    wide = qty_w.join(prc_w, how="outer").sort_index()

    for sku in family_products:
        pcol = f"P_{sku}"
        lcol = f"LOGP_{sku}"
        if pcol in wide.columns:
            wide[lcol] = wide[pcol].astype(float).apply(lambda v: _safe_log(v) if np.isfinite(v) else np.nan)
        else:
            wide[lcol] = np.nan

    logp_cols = [f"LOGP_{sku}" for sku in family_products]
    wide[logp_cols] = wide[logp_cols].ffill()

    return wide


def train_cross_family_models(bucket_df: pd.DataFrame,
                              cutoff: pd.Timestamp,
                              topk_map: Dict[Tuple[str, str, str], List[str]],
                              alpha: float = POISSON_ALPHA,
                              min_rows_per_sku: int = CROSS_MIN_ROWS_PER_SKU) -> Dict[Tuple[str, str, str], CrossFamilyModel]:
    """
    Обучает cross-модели:
      - общий scaler на X=[log prices of K skus]
      - отдельный PoissonRegressor на каждый SKU i (y=Q_i)
    """
    out: Dict[Tuple[str, str, str], CrossFamilyModel] = {}

    for (fam, store, bucket), skus in topk_map.items():
        wide = make_family_wide_panel(bucket_df, fam, store, bucket, skus, cutoff=cutoff)
        if wide.empty:
            continue

        X_cols = [f"LOGP_{sku}" for sku in skus]
        X_all = wide[X_cols].astype(float)
        # выкинем строки где вообще нет признаков
        X_all = X_all.replace([np.inf, -np.inf], np.nan)

        # scaler общий (чтобы коэффициенты были сравнимы)
        scaler = StandardScaler()
        X_fit = X_all.fillna(0.0).values
        Xs_all = scaler.fit_transform(X_fit)

        models_by_product: Dict[str, PoissonRegressor] = {}
        n_by_product: Dict[str, int] = {}

        for sku in skus:
            y_col = f"Q_{sku}"
            if y_col not in wide.columns:
                continue

            # строки где y определён + есть собственная LOGP (цена была известна)
            data = pd.concat([wide[y_col], X_all], axis=1).dropna(subset=[y_col, f"LOGP_{sku}"])
            if len(data) < min_rows_per_sku:
                continue

            y = data[y_col].astype(float).values
            # ФИКС: y >= 0 для Poisson
            y = np.clip(y, 0.0, None)
            if not np.isfinite(y).all():
                continue
            if (y > 0).sum() < 5:
                continue

            # соответствующие X строки -> берём из X_all по индексам data
            X_sub = X_all.loc[data.index].fillna(0.0).values
            Xs = scaler.transform(X_sub)

            reg = PoissonRegressor(
                alpha=max(alpha, 1e-2),
                fit_intercept=True,
                max_iter=POISSON_MAX_ITER
            )
            reg.fit(Xs, y)

            models_by_product[sku] = reg
            n_by_product[sku] = int(len(data))

        if len(models_by_product) >= 2:
            out[(fam, store, bucket)] = CrossFamilyModel(
                family_code=fam,
                store=store,
                bucket=bucket,
                family_products=skus,
                scaler=scaler,
                models_by_product=models_by_product,
                n_by_product=n_by_product
            )

    return out


def predict_family_demands(cfm: CrossFamilyModel, prices: Dict[str, float]) -> Dict[str, float]:
    """
    Предсказывает E[Q_i] для всех SKU i из cross-модели при ценах всех Top-K.
    """
    skus = cfm.family_products
    X_raw = np.array([_safe_log(prices[sku]) for sku in skus], dtype=float).reshape(1, -1)
    Xs = cfm.scaler.transform(X_raw)

    out: Dict[str, float] = {}
    for sku, reg in cfm.models_by_product.items():
        mu = float(reg.predict(Xs)[0])
        out[sku] = max(mu, 0.0)
    return out


def cross_elasticity_raw(cfm: CrossFamilyModel, demand_sku: str, price_sku: str) -> Optional[float]:
    """
    Возвращает gamma_{i,j} в исходных единицах:
      gamma_raw = coef_scaled[j] / std(log(P_j))
    """
    if demand_sku not in cfm.models_by_product:
        return None
    if price_sku not in cfm.family_products:
        return None
    j = cfm.family_products.index(price_sku)
    std = float(cfm.scaler.scale_[j]) if cfm.scaler.scale_ is not None else 1.0
    if std <= 0:
        return None
    coef_scaled = float(cfm.models_by_product[demand_sku].coef_[j])
    return coef_scaled / std


def optimize_family_prices_iterative(cfm: CrossFamilyModel,
                                     base_prices: Dict[str, float],
                                     price_grids: Dict[str, np.ndarray],
                                     iters: int = CROSS_ITERS,
                                     stock_caps: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    Итеративная оптимизация family (best-response):
      - фиксируем цены конкурентов
      - перебор цен SKU i по сетке
      - max p_i * E[Q_i | все цены]
    """
    prices = dict(base_prices)

    def best_for_sku(sku: str) -> Tuple[float, float, float]:
        grid = price_grids.get(sku)
        if grid is None or len(grid) == 0:
            p0 = float(prices[sku])
            mu0 = float(predict_family_demands(cfm, prices).get(sku, 0.0))
            return p0, mu0, float(p0 * mu0)

        best = None
        for p in grid:
            tmp = dict(prices)
            tmp[sku] = float(p)
            mu = float(predict_family_demands(cfm, tmp).get(sku, 0.0))
            cap = None if stock_caps is None else stock_caps.get(sku)
            if cap is not None and np.isfinite(cap) and mu > float(cap):
                continue
            rev = float(p) * mu
            if best is None or rev > best[2]:
                best = (float(p), mu, float(rev))

        if best is None:
            # если всё отфильтровано cap-ом
            best2 = None
            for p in grid:
                tmp = dict(prices)
                tmp[sku] = float(p)
                mu = float(predict_family_demands(cfm, tmp).get(sku, 0.0))
                rev = float(p) * mu
                if best2 is None or mu < best2[1]:
                    best2 = (float(p), mu, float(rev))
            return best2

        return best

    for _ in range(int(iters)):
        for sku in cfm.family_products:
            if sku not in prices:
                continue
            p_star, _, _ = best_for_sku(sku)
            prices[sku] = p_star

    mu_all = predict_family_demands(cfm, prices)
    out: Dict[str, Tuple[float, float, float]] = {}
    for sku in cfm.family_products:
        p = float(prices[sku])
        mu = float(mu_all.get(sku, 0.0))
        out[sku] = (p, mu, float(p * mu))
    return out


# =========================
# Прод: цены на всю неделю (3 бакета), запуск в понедельник
# =========================

def price_week_3_buckets(bucket_df: pd.DataFrame,
                         daily_df: pd.DataFrame,
                         as_of_date: pd.Timestamp,
                         shrink_k: int = DEFAULT_SHRINK_K,
                         alpha: float = POISSON_ALPHA,
                         cross_top_k: int = CROSS_TOP_K,
                         cross_lookback_weeks: int = CROSS_LOOKBACK_WEEKS,
                         cross_min_rows_per_sku: int = CROSS_MIN_ROWS_PER_SKU,
                         cross_iters: int = CROSS_ITERS) -> pd.DataFrame:
    """
    Понедельник утром: считаем цены на текущую неделю для MON_THU/FRI/SAT_SUN.

    Важный фикс скорости:
      - single hier модели обучаем ОДИН раз на cutoff=today
      - cross модели обучаем по бакетам (3 раза), но тоже на cutoff=today

    Cutoff для вывода:
      decision_cutoff_rule = decision_cutoff_for_bucket(week, bucket)
      cutoff_effective = min(decision_cutoff_rule, today)
      В понедельник: для FRI/SAT_SUN это будет today (без "подглядывания").
    """
    today = pd.Timestamp(as_of_date).normalize()
    target_week = week_start(today)

    # обучающие данные только до today
    train_all = bucket_df.loc[bucket_df["BUCKET_END"] <= today].copy()
    if len(train_all) == 0:
        return pd.DataFrame()

    promo_map = build_promo_map(daily_df)

    # кандидаты SKU (контекст): последние 12 записей на (product,store)
    recent = train_all.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(12)
    last_ctx = recent.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(1).copy()

    # single признаки
    single_feature_names = [
        "LOG_PRICE",
        "PROMO_SHARE",
        "BUCKET_IDX",
        "WEEK_OF_YEAR",
        "LOW_STOCK_FLAG",
        "QTY_L1",
        "PRICE_L1",
        "PROMO_L1",
    ]

    # (A) single модели: 1 раз
    hier_models = train_hier_models(train_all, cutoff=today, feature_names=single_feature_names, alpha=alpha)

    # (B) cross модели: по бакетам (быстро благодаря фильтрам + Top-K=5)
    cross_by_bucket: Dict[str, Dict[Tuple[str, str, str], CrossFamilyModel]] = {}
    for bucket in BUCKET_ORDER:
        topk_map = build_family_topk(
            train_all, cutoff=today,
            top_k=cross_top_k,
            lookback_weeks=cross_lookback_weeks
        )
        # build_family_topk уже вернул по всем бакетам, но ключ содержит bucket.
        # train_cross_family_models сам фильтрует по bucket внутри wide-panel.
        cross_models = train_cross_family_models(
            train_all, cutoff=today,
            topk_map=topk_map,
            alpha=alpha,
            min_rows_per_sku=cross_min_rows_per_sku
        )
        cross_by_bucket[bucket] = cross_models

    # ladder для снапа (на today)
    fam_ladder = collect_observed_price_ladder(train_all, level="FAMILY_CODE")

    out_all: List[pd.DataFrame] = []

    for bucket in BUCKET_ORDER:
        start, end = bucket_start_end(target_week, bucket)
        decision_cutoff = decision_cutoff_for_bucket(target_week, bucket)
        cutoff_effective = min(pd.Timestamp(decision_cutoff).normalize(), today)

        # контекст под целевой бакет
        ctx = last_ctx.copy()
        ctx["WEEK"] = pd.Timestamp(target_week)
        ctx["BUCKET"] = bucket
        ctx["BUCKET_IDX"] = {"MON_THU": 0, "FRI": 1, "SAT_SUN": 2}[bucket]
        ctx["WEEK_OF_YEAR"] = int(pd.Timestamp(target_week).isocalendar().week)

        # cross модели для этого бакета
        cross_models = cross_by_bucket.get(bucket, {})

        used_in_cross = set()
        out_rows: List[dict] = []

        # ---------- 1) семьи с cross-моделью ----------
        for (fam, store), gfam in ctx.groupby(["FAMILY_CODE", "STORE"]):
            key = (str(fam), str(store), str(bucket))
            cfm = cross_models.get(key)
            if cfm is None:
                continue

            present_skus = set(gfam["PRODUCT_CODE"].astype(str))
            fam_skus = [sku for sku in cfm.family_products if sku in present_skus]
            if len(fam_skus) < 2:
                continue

            base_prices: Dict[str, float] = {}
            grids: Dict[str, np.ndarray] = {}
            stock_caps: Dict[str, float] = {}
            tmp_prices: List[float] = []

            # базовые цены/сетки для SKU, которые реально есть в контексте
            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                if sku not in cfm.family_products:
                    continue

                # base price: последняя цена по этому бакету до today, иначе последняя известная до today
                hist_b = train_all.loc[
                    (train_all["PRODUCT_CODE"] == sku) &
                    (train_all["STORE"] == str(store)) &
                    (train_all["BUCKET"] == bucket)
                ].sort_values("BUCKET_END")

                if len(hist_b) and np.isfinite(hist_b["PRICE"].iloc[-1]):
                    base_price = float(hist_b["PRICE"].iloc[-1])
                else:
                    hist_any = train_all.loc[
                        (train_all["PRODUCT_CODE"] == sku) &
                        (train_all["STORE"] == str(store))
                    ].sort_values("BUCKET_END")
                    base_price = float(hist_any["PRICE"].iloc[-1]) if len(hist_any) and np.isfinite(hist_any["PRICE"].iloc[-1]) else np.nan

                if not np.isfinite(base_price) or base_price <= 0:
                    continue

                base_prices[sku] = base_price
                tmp_prices.append(base_price)

                ladder = fam_ladder.get(str(fam))
                grids[sku] = build_price_grid(base_price, observed_prices=ladder)

                # stock cap: последний END_STOCK
                hist_any2 = train_all.loc[
                    (train_all["PRODUCT_CODE"] == sku) &
                    (train_all["STORE"] == str(store))
                ].sort_values("BUCKET_END")
                stock_caps[sku] = float(hist_any2["STOCK_END"].iloc[-1]) if len(hist_any2) else np.nan

            fam_skus = [x for x in fam_skus if x in base_prices]
            if len(fam_skus) < 2:
                continue

            # Для predict_family_demands нужны цены всех Top-K -> заполняем медианой
            median_price = float(np.median(tmp_prices)) if len(tmp_prices) else 1.0
            for sku in cfm.family_products:
                if sku not in base_prices:
                    base_prices[sku] = median_price
                if sku not in grids:
                    grids[sku] = np.array([base_prices[sku]], dtype=float)
                if sku not in stock_caps:
                    stock_caps[sku] = np.nan

            fam_solution = optimize_family_prices_iterative(
                cfm=cfm,
                base_prices=base_prices,
                price_grids=grids,
                iters=cross_iters,
                stock_caps=stock_caps
            )

            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                if sku not in fam_solution:
                    continue

                # промо доля по плану
                ranges = promo_map.get((sku, str(store)), [])
                promo_share = promo_share_in_range(ranges, start, end) if ranges else 0.0

                chosen_price, pred_qty, pred_rev = fam_solution[sku]

                out_rows.append({
                    "PRODUCT_CODE": sku,
                    "STORE": str(store),
                    "FAMILY_CODE": str(fam),
                    "AS_OF_DATE": today,
                    "TARGET_WEEK": pd.Timestamp(target_week),
                    "TARGET_BUCKET": bucket,
                    "BUCKET_START": start,
                    "BUCKET_END": end,
                    "DECISION_CUTOFF_RULE": pd.Timestamp(decision_cutoff).normalize(),
                    "CUTOFF_EFFECTIVE": cutoff_effective,
                    "PROMO_SHARE": promo_share,
                    "STOCK_CAP": stock_caps.get(sku),
                    "BASE_PRICE": float(base_prices[sku]),
                    "CHOSEN_PRICE": chosen_price,
                    "PRED_QTY": pred_qty,
                    "PRED_REV": pred_rev,
                    "USED_CROSS_PRICE_MODEL": True,
                })
                used_in_cross.add((sku, str(store)))

        # ---------- 2) остальные SKU: single fallback ----------
        for _, r in ctx.iterrows():
            sku = str(r["PRODUCT_CODE"])
            store = str(r["STORE"])
            if (sku, store) in used_in_cross:
                continue

            ranges = promo_map.get((sku, store), [])
            promo_share = promo_share_in_range(ranges, start, end) if ranges else 0.0

            # base price: последняя цена по бакету до today, иначе последняя известная
            hist_b = train_all.loc[
                (train_all["PRODUCT_CODE"] == sku) &
                (train_all["STORE"] == store) &
                (train_all["BUCKET"] == bucket)
            ].sort_values("BUCKET_END")
            if len(hist_b) and np.isfinite(hist_b["PRICE"].iloc[-1]):
                base_price = float(hist_b["PRICE"].iloc[-1])
            else:
                hist_any = train_all.loc[
                    (train_all["PRODUCT_CODE"] == sku) &
                    (train_all["STORE"] == store)
                ].sort_values("BUCKET_END")
                base_price = float(hist_any["PRICE"].iloc[-1]) if len(hist_any) and np.isfinite(hist_any["PRICE"].iloc[-1]) else np.nan

            if not np.isfinite(base_price) or base_price <= 0:
                continue

            # stock cap
            hist_any2 = train_all.loc[
                (train_all["PRODUCT_CODE"] == sku) &
                (train_all["STORE"] == store)
            ].sort_values("BUCKET_END")
            stock_cap = float(hist_any2["STOCK_END"].iloc[-1]) if len(hist_any2) else None

            base_row = r.copy()
            base_row["PROMO_SHARE"] = promo_share

            ladder = fam_ladder.get(str(r["FAMILY_CODE"]))
            grid = build_price_grid(base_price, observed_prices=ladder)

            best = optimize_price_single_sku(
                base_row=base_row,
                feature_names=single_feature_names,
                models=hier_models,
                price_grid=grid,
                stock_cap=stock_cap,
                shrink_k=shrink_k
            )
            if best is None:
                continue

            chosen_price, pred_qty, pred_rev = best

            out_rows.append({
                "PRODUCT_CODE": sku,
                "STORE": store,
                "FAMILY_CODE": str(r["FAMILY_CODE"]),
                "AS_OF_DATE": today,
                "TARGET_WEEK": pd.Timestamp(target_week),
                "TARGET_BUCKET": bucket,
                "BUCKET_START": start,
                "BUCKET_END": end,
                "DECISION_CUTOFF_RULE": pd.Timestamp(decision_cutoff).normalize(),
                "CUTOFF_EFFECTIVE": cutoff_effective,
                "PROMO_SHARE": promo_share,
                "STOCK_CAP": stock_cap,
                "BASE_PRICE": float(base_price),
                "CHOSEN_PRICE": chosen_price,
                "PRED_QTY": pred_qty,
                "PRED_REV": pred_rev,
                "USED_CROSS_PRICE_MODEL": False,
            })

        out_all.append(pd.DataFrame(out_rows))

    return pd.concat(out_all, ignore_index=True) if out_all else pd.DataFrame()


# =========================
# Rolling backtest (опционально)
# =========================

def last_n_weeks(weeks: Iterable[pd.Timestamp], n: int) -> List[pd.Timestamp]:
    """Последние n уникальных недель (по возрастанию)."""
    u = sorted(pd.Series(list(weeks)).dropna().unique())
    return u[-n:]


def rolling_backtest_12w(bucket_df: pd.DataFrame,
                         daily_df: pd.DataFrame,
                         eval_weeks: int = 12,
                         shrink_k: int = DEFAULT_SHRINK_K,
                         alpha: float = POISSON_ALPHA) -> pd.DataFrame:
    """
    Rolling backtest последних eval_weeks недель.
    Тут оставлено просто как базовая проверка (можно расширять).

    Важно: backtest медленнее прода, потому что обучаем модели много раз.
    """
    bdf = bucket_df.loc[bucket_df["BUCKET_END"] <= HISTORY_END].copy()
    promo_map = build_promo_map(daily_df)

    single_feature_names = [
        "LOG_PRICE", "PROMO_SHARE", "BUCKET_IDX", "WEEK_OF_YEAR",
        "LOW_STOCK_FLAG", "QTY_L1", "PRICE_L1", "PROMO_L1"
    ]
    fam_ladder_all = collect_observed_price_ladder(bdf, level="FAMILY_CODE")

    weeks = last_n_weeks(bdf["WEEK"].unique(), eval_weeks)
    out_rows: List[dict] = []

    for wk in weeks:
        wk = pd.Timestamp(wk)
        for bucket in BUCKET_ORDER:
            start, end = bucket_start_end(wk, bucket)
            cutoff = min(decision_cutoff_for_bucket(wk, bucket), HISTORY_END)

            train = bdf.loc[bdf["BUCKET_END"] <= cutoff].copy()
            if len(train) == 0:
                continue

            # single models
            hier = train_hier_models(train, cutoff=cutoff, feature_names=single_feature_names, alpha=alpha)

            # target rows
            target = bdf.loc[(bdf["WEEK"] == wk) & (bdf["BUCKET"] == bucket)].copy()
            if len(target) == 0:
                continue

            for _, r in target.iterrows():
                sku = str(r["PRODUCT_CODE"])
                store = str(r["STORE"])
                fam = str(r["FAMILY_CODE"])

                ranges = promo_map.get((sku, store), [])
                promo_share = promo_share_in_range(ranges, start, end) if ranges else float(r.get("PROMO_SHARE", 0.0))

                # base price
                hist_b = train.loc[
                    (train["PRODUCT_CODE"] == sku) &
                    (train["STORE"] == store) &
                    (train["BUCKET"] == bucket)
                ].sort_values("BUCKET_END")
                base_price = float(hist_b["PRICE"].iloc[-1]) if len(hist_b) and np.isfinite(hist_b["PRICE"].iloc[-1]) else float(r["PRICE"])
                if not np.isfinite(base_price) or base_price <= 0:
                    continue

                # stock cap
                hist_any = train.loc[
                    (train["PRODUCT_CODE"] == sku) &
                    (train["STORE"] == store)
                ].sort_values("BUCKET_END")
                stock_cap = float(hist_any["STOCK_END"].iloc[-1]) if len(hist_any) else None

                base_row = r.copy()
                base_row["PROMO_SHARE"] = promo_share

                ladder = fam_ladder_all.get(fam)
                grid = build_price_grid(base_price, observed_prices=ladder)

                best = optimize_price_single_sku(
                    base_row=base_row,
                    feature_names=single_feature_names,
                    models=hier,
                    price_grid=grid,
                    stock_cap=stock_cap,
                    shrink_k=shrink_k
                )
                if best is None:
                    continue

                chosen_price, pred_qty, pred_rev = best
                out_rows.append({
                    "PRODUCT_CODE": sku,
                    "STORE": store,
                    "FAMILY_CODE": fam,
                    "WEEK": wk,
                    "BUCKET": bucket,
                    "CUTOFF": cutoff,
                    "CHOSEN_PRICE": chosen_price,
                    "PRED_QTY": pred_qty,
                    "PRED_REV": pred_rev,
                    "ACTUAL_QTY": float(r["QTY"]),
                    "ACTUAL_PRICE": float(r["PRICE"]) if np.isfinite(r["PRICE"]) else np.nan,
                })

    return pd.DataFrame(out_rows)


# =========================
# Пайплайн
# =========================

def run_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн:
      1) preprocess daily
      2) aggregate to buckets
      3) add lags

    Returns:
      (bucket_panel, daily_clean)
    """
    daily = preprocess_raw(df_raw, history_end=HISTORY_END)
    buckets = aggregate_to_buckets(daily)
    buckets = add_lag_features(buckets, lags=1)
    return buckets, daily


# ------------------------------------------------------------
# Пример использования:
#
# import pandas as pd
# df = pd.read_parquet("data.parquet")
# bucket_panel, daily = run_pipeline(df)
#
# # Понедельник утром:
# decisions = price_week_3_buckets(bucket_panel, daily, as_of_date=pd.Timestamp("2025-09-15"))
# decisions.to_csv("week_prices.csv", index=False)
#
# # Опционально backtest:
# bt = rolling_backtest_12w(bucket_panel, daily, eval_weeks=12)
# ------------------------------------------------------------
