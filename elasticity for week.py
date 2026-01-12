# Динамическое ценообразование с явной эластичностью:
#
# 1) Own-price elasticity (эластичность собственного спроса по своей цене):
#    Poisson GLM (log-link):
#      log(E[Q]) = ... + b_own * log(P) + ...
#    => b_own — явная (константная) эластичность по log-log.
#
# 2) Cross-price elasticity (кросс-эластичности внутри FAMILY):
#    Для выбранных Top-K SKU в рамках (FAMILY_CODE, STORE, BUCKET) обучаем систему:
#      log(E[Q_i]) = a_i + sum_j gamma_{i,j} * log(P_j) + controls
#    => gamma_{i,j} — явные кросс-эластичности.
#
# 3) Иерархический backoff (условие 2) + shrinkage:
#    Для SKU, которые НЕ попали в cross-модель (длинный хвост), используем
#    single-SKU Poisson с backoff:
#      (PRODUCT,STORE) -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL
#
# 4) Бакеты цен MON_THU / FRI / SAT_SUN, цена меняется только между бакетами.
# 5) Rolling оценка последние 12 недель с правильным cutoff.
# 6) PROMO_PERIOD парсим из строки "01-01-2024 - 03-01-2024".
# 7) Неполная история: товары появляются/исчезают, нули по спросу допустимы.
#
# Важно про оптимизацию cross-модели:
# - Оптимизация семейства связанная (coupled), поэтому используем итеративный best-response
#   (несколько проходов по SKU внутри семьи).
#
# Требования:
#   pip install pandas numpy scikit-learn
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


# =========================
# Конфигурация
# =========================

HISTORY_END = pd.Timestamp("2025-09-15")

BUCKET_ORDER = ["MON_THU", "FRI", "SAT_SUN"]

# Сила "усадки" (shrinkage): чем больше K, тем сильнее откат к родительскому уровню,
# особенно когда наблюдений мало. Вес для "детской" модели: w = n / (n + K).
# При малом n -> w маленький -> больше доверяем родителю (семейство/категория/сегмент/глобально).
DEFAULT_SHRINK_K = 30

# Минимальное число строк для обучения модели на каждом уровне иерархии.
# Нужно для устойчивости коэффициентов (эластичности), иначе модель на малых выборках
# даёт шумные оценки и может рекомендовать экстремальные цены.
MIN_ROWS_PRODUCT_STORE = 40
MIN_ROWS_FAMILY = 120
MIN_ROWS_CATEGORY = 300
MIN_ROWS_SEGMENT = 600

# Ограничения множителей цены
MIN_MULT = 0.9
MAX_MULT = 1.3

# Регуляризация PoissonRegressor (L2)
POISSON_ALPHA = 1e-2

# Cross-price: параметры Top-K
CROSS_TOP_K = 10
CROSS_LOOKBACK_WEEKS = 26
CROSS_MIN_ROWS_PER_SKU = 40
CROSS_ITERS = 4


# =========================
# Даты и бакеты
# =========================

def bucket_id_for_date(dt: pd.Timestamp) -> str:
    """
    Определяет бакет цены по дате.

    Args:
      dt: дата

    Returns:
      "MON_THU" (пн-чт), "FRI" (пт), "SAT_SUN" (сб-вс)
    """
    wd = pd.Timestamp(dt).weekday()
    if wd <= 3:
        return "MON_THU"
    if wd == 4:
        return "FRI"
    return "SAT_SUN"


def week_start(dt: pd.Timestamp) -> pd.Timestamp:
    """
    Возвращает понедельник недели для даты dt.

    Args:
      dt: дата

    Returns:
      дата понедельника (normalize)
    """
    dt = pd.Timestamp(dt).normalize()
    return dt - pd.Timedelta(days=dt.weekday())


def bucket_start_end(week_monday: pd.Timestamp, bucket: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Интервал бакета внутри недели (включительно).

    Args:
      week_monday: понедельник недели
      bucket: MON_THU / FRI / SAT_SUN

    Returns:
      (start_date, end_date)
    """
    week_monday = pd.Timestamp(week_monday).normalize()
    if bucket == "MON_THU":
        return week_monday, week_monday + pd.Timedelta(days=3)
    if bucket == "FRI":
        d = week_monday + pd.Timedelta(days=4)
        return d, d
    if bucket == "SAT_SUN":
        return week_monday + pd.Timedelta(days=5), week_monday + pd.Timedelta(days=6)
    raise ValueError(f"Unknown bucket={bucket}")


def decision_cutoff_for_bucket(week_monday: pd.Timestamp, bucket: str) -> pd.Timestamp:
    """
    Cutoff дата "по правилам" (условие 7), когда принимаем решение для целевого бакета.

    Args:
      week_monday: понедельник целевой недели
      bucket: целевой бакет

    Returns:
      cutoff дата (включительно)
    """
    week_monday = pd.Timestamp(week_monday).normalize()
    if bucket == "MON_THU":
        return week_monday - pd.Timedelta(days=1)  # воскресенье предыдущей недели
    if bucket == "FRI":
        return week_monday + pd.Timedelta(days=3)  # четверг этой недели
    if bucket == "SAT_SUN":
        return week_monday + pd.Timedelta(days=4)  # пятница этой недели
    raise ValueError(f"Unknown bucket={bucket}")


# =========================
# PROMO_PERIOD
# =========================

def parse_promo_period(s: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Парсит PROMO_PERIOD вида "01-01-2024 - 03-01-2024" (dd-mm-yyyy).

    Args:
      s: строка PROMO_PERIOD

    Returns:
      (start, end) включительно или None
    """
    if not isinstance(s, str) or s.strip() == "":
        return None
    if " - " not in s:
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
    Доля дней в [start, end], покрытых хотя бы одним промо-интервалом.

    Args:
      promo_ranges: список (promo_start, promo_end)
      start: начало интервала
      end: конец интервала

    Returns:
      float в [0, 1]
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
# Дневной препроцессинг
# =========================

def compute_unit_price(df: pd.DataFrame) -> pd.Series:
    """
    Цена за единицу:
      - если есть SALE_PRICE, используем её
      - иначе SALE_PRICE_TOTAL / max(SALE_QTY,1)

    Args:
      df: датафрейм

    Returns:
      Series UNIT_PRICE
    """
    if "SALE_PRICE" in df.columns:
        return pd.to_numeric(df["SALE_PRICE"], errors="coerce")
    total = pd.to_numeric(df.get("SALE_PRICE_TOTAL"), errors="coerce")
    qty = pd.to_numeric(df.get("SALE_QTY"), errors="coerce").fillna(0.0)
    return total / qty.clip(lower=1.0)


def preprocess_raw(df: pd.DataFrame, history_end: pd.Timestamp = HISTORY_END) -> pd.DataFrame:
    """
    Очистка дневных данных:
      - даты, фильтр по history_end
      - QTY_TOTAL = SALE_QTY + SALE_QTY_ONLINE
      - UNIT_PRICE
      - BUCKET по дате
      - PROMO_RANGE из PROMO_PERIOD
      - нормализация ключей и числовых колонок

    Args:
      df: сырой датафрейм
      history_end: конец истории

    Returns:
      cleaned daily dataframe
    """
    out = df.copy()

    out["TRADE_DT"] = pd.to_datetime(out["TRADE_DT"], errors="coerce").dt.normalize()
    out = out.loc[out["TRADE_DT"].notna()]
    out = out.loc[out["TRADE_DT"] <= pd.Timestamp(history_end).normalize()]

    qty_off = pd.to_numeric(out.get("SALE_QTY"), errors="coerce").fillna(0.0)
    qty_on = pd.to_numeric(out.get("SALE_QTY_ONLINE"), errors="coerce").fillna(0.0)
    out["QTY_TOTAL"] = (qty_off + qty_on).astype(float)

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
    """
    Карта промо-интервалов по (PRODUCT_CODE, STORE).

    Args:
      daily: дневные данные

    Returns:
      dict[(product, store)] -> list[(start, end)]
    """
    promo_map: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
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
    Агрегирует дневные данные в бакет-панель:
      (PRODUCT_CODE, STORE, WEEK, BUCKET) -> QTY, PRICE, PROMO_SHARE, stock/flows

    Args:
      daily: очищенные дневные данные

    Returns:
      bucket-level dataframe
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

        price = np.nan
        pvals = g["UNIT_PRICE"].astype(float).values
        if qty > 0 and np.isfinite(pvals).any():
            w = g["QTY_TOTAL"].astype(float).values
            if np.nansum(w) > 0:
                price = float(np.nansum(pvals * w) / np.nansum(w))
        if not np.isfinite(price):
            price = float(np.nanmean(pvals)) if np.isfinite(pvals).any() else np.nan

        wk = g["WEEK"].iloc[0]
        b = g["BUCKET"].iloc[0]
        start, end = bucket_start_end(wk, b)

        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)]
        promo_share = promo_share_in_range(ranges, start, end) if ranges else float(g["IS_PROMO"].mean())

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

    return bucket_df


# =========================
# Лаги
# =========================

def add_lag_features(bucket_df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    """
    Добавляет лаги по (PRODUCT_CODE, STORE, BUCKET).

    Args:
      bucket_df: бакет-панель
      lags: число лагов

    Returns:
      dataframe с лагами
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
# Single-SKU Poisson + backoff
# =========================

@dataclass
class PoissonCoeffs:
    """
    Коэффициенты single-SKU PoissonRegressor.

    Важно:
      - coef(LOG_PRICE) = own-price эластичность.
    """
    coef: np.ndarray
    intercept: float
    feature_names: List[str]
    n: int


def _safe_log(x: float) -> float:
    """Безопасный лог(цены) для численной устойчивости."""
    return float(np.log(max(float(x), 1e-6)))


def fit_poisson_group(df: pd.DataFrame,
                      feature_names: List[str],
                      alpha: float = POISSON_ALPHA) -> Optional[PoissonCoeffs]:
    """
    Обучает PoissonRegressor на группе.

    Args:
      df: данные группы
      feature_names: признаки
      alpha: регуляризация

    Returns:
      PoissonCoeffs или None
    """
    g = df.copy()
    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    y = g["QTY"].astype(float).values
    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=500)
    model.fit(X, y)

    return PoissonCoeffs(
        coef=model.coef_.astype(float),
        intercept=float(model.intercept_),
        feature_names=list(feature_names),
        n=int(len(g)),
    )


def shrink_blend(child: Optional[PoissonCoeffs],
                 parent: Optional[PoissonCoeffs],
                 k: int = DEFAULT_SHRINK_K) -> Optional[PoissonCoeffs]:
    """
    Усадка коэффициентов:
      w = n_child / (n_child + k)

    Args:
      child: детская модель
      parent: родительская
      k: сила усадки

    Returns:
      blended модель
    """
    if child is None:
        return parent
    if parent is None:
        return child
    if child.feature_names != parent.feature_names:
        raise ValueError("feature_names differ between child and parent models")

    w = child.n / (child.n + k)
    coef = w * child.coef + (1 - w) * parent.coef
    intercept = float(w * child.intercept + (1 - w) * parent.intercept)
    return PoissonCoeffs(coef=coef, intercept=intercept, feature_names=child.feature_names, n=child.n)


@dataclass
class HierPoissonModels:
    """Набор single-SKU моделей по иерархии."""
    global_model: Optional[PoissonCoeffs]
    segment: Dict[str, PoissonCoeffs]
    category: Dict[str, PoissonCoeffs]
    family: Dict[str, PoissonCoeffs]
    product_store: Dict[Tuple[str, str], PoissonCoeffs]


def train_hierarchical_models(bucket_df: pd.DataFrame,
                              train_until: pd.Timestamp,
                              feature_names: List[str],
                              alpha: float = POISSON_ALPHA) -> HierPoissonModels:
    """
    Обучает single-SKU модели по уровням иерархии.

    Args:
      bucket_df: бакет-панель
      train_until: cutoff
      feature_names: признаки
      alpha: регуляризация

    Returns:
      HierPoissonModels
    """
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(train_until)].copy()

    global_model = fit_poisson_group(train, feature_names, alpha=alpha)

    seg_models: Dict[str, PoissonCoeffs] = {}
    for seg, g in train.groupby("SEGMENT_CODE"):
        if len(g) >= MIN_ROWS_SEGMENT:
            m = fit_poisson_group(g, feature_names, alpha=alpha)
            if m is not None:
                seg_models[str(seg)] = m

    cat_models: Dict[str, PoissonCoeffs] = {}
    for cat, g in train.groupby("CATEGORY_CODE"):
        if len(g) >= MIN_ROWS_CATEGORY:
            m = fit_poisson_group(g, feature_names, alpha=alpha)
            if m is not None:
                cat_models[str(cat)] = m

    fam_models: Dict[str, PoissonCoeffs] = {}
    for fam, g in train.groupby("FAMILY_CODE"):
        if len(g) >= MIN_ROWS_FAMILY:
            m = fit_poisson_group(g, feature_names, alpha=alpha)
            if m is not None:
                fam_models[str(fam)] = m

    ps_models: Dict[Tuple[str, str], PoissonCoeffs] = {}
    for (p, s), g in train.groupby(["PRODUCT_CODE", "STORE"]):
        if len(g) >= MIN_ROWS_PRODUCT_STORE:
            m = fit_poisson_group(g, feature_names, alpha=alpha)
            if m is not None:
                ps_models[(str(p), str(s))] = m

    return HierPoissonModels(
        global_model=global_model,
        segment=seg_models,
        category=cat_models,
        family=fam_models,
        product_store=ps_models,
    )


def get_effective_model(row: pd.Series,
                        models: HierPoissonModels,
                        k: int = DEFAULT_SHRINK_K) -> Optional[PoissonCoeffs]:
    """
    Backoff:
      (PRODUCT,STORE) -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL

    Args:
      row: строка
      models: модели
      k: shrinkage

    Returns:
      PoissonCoeffs или None
    """
    p = str(row["PRODUCT_CODE"])
    s = str(row["STORE"])
    fam = str(row["FAMILY_CODE"])
    cat = str(row["CATEGORY_CODE"])
    seg = str(row["SEGMENT_CODE"])

    m_ps = models.product_store.get((p, s))
    m_fam = models.family.get(fam)
    m_cat = models.category.get(cat)
    m_seg = models.segment.get(seg)
    m_glb = models.global_model

    m = shrink_blend(m_ps, m_fam, k=k)
    m = shrink_blend(m, m_cat, k=k)
    m = shrink_blend(m, m_seg, k=k)
    m = shrink_blend(m, m_glb, k=k)
    return m


def poisson_predict_mean(model: PoissonCoeffs, x_row: np.ndarray) -> float:
    """
    mu = exp(intercept + x @ coef) с клипом для стабильности.

    Args:
      model: PoissonCoeffs
      x_row: 1D массив признаков

    Returns:
      mu
    """
    eta = float(model.intercept + np.dot(x_row, model.coef))
    eta = np.clip(eta, -20, 20)
    return float(np.exp(eta))


# =========================
# Cross-family модели
# =========================

@dataclass
class CrossFamilyModel:
    """
    Кросс-эластичности внутри (FAMILY_CODE, STORE, BUCKET) для Top-K SKU.

    Для каждого SKU i хранится PoissonRegressor:
      y = Q_i
      X = [log(P_sku1), ..., log(P_skuK)]
    """
    family_code: str
    store: str
    bucket: str
    family_products: List[str]
    models_by_product: Dict[str, PoissonRegressor]
    n_by_product: Dict[str, int]


def build_family_topk(bucket_df: pd.DataFrame,
                      cutoff: pd.Timestamp,
                      top_k: int = CROSS_TOP_K,
                      lookback_weeks: int = CROSS_LOOKBACK_WEEKS) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Top-K SKU по продажам за lookback до cutoff внутри (FAMILY, STORE, BUCKET).

    Args:
      bucket_df: бакет-панель
      cutoff: cutoff
      top_k: K
      lookback_weeks: окно

    Returns:
      dict[(fam, store, bucket)] -> list[sku]
    """
    df = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()
    min_week = pd.Timestamp(cutoff) - pd.Timedelta(weeks=int(lookback_weeks))
    df = df.loc[df["WEEK"] >= min_week]

    topk_map: Dict[Tuple[str, str, str], List[str]] = {}
    for (fam, store, bucket), g in df.groupby(["FAMILY_CODE", "STORE", "BUCKET"]):
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
    Wide-панель по WEEK для family Top-K:
      Q_<sku>, P_<sku>, LOGP_<sku>

    Args:
      bucket_df: бакет-панель
      family_code, store, bucket: ключи
      family_products: список SKU
      cutoff: BUCKET_END <= cutoff

    Returns:
      wide dataframe index=WEEK
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
    Обучает cross-модели для каждого (FAMILY, STORE, BUCKET).

    Args:
      bucket_df: бакет-панель
      cutoff: cutoff
      topk_map: top-k карты
      alpha: регуляризация
      min_rows_per_sku: минимум недель для обучения SKU

    Returns:
      dict key -> CrossFamilyModel
    """
    out: Dict[Tuple[str, str, str], CrossFamilyModel] = {}

    for (fam, store, bucket), skus in topk_map.items():
        wide = make_family_wide_panel(bucket_df, fam, store, bucket, skus, cutoff=cutoff)
        if wide.empty:
            continue

        X_cols = [f"LOGP_{sku}" for sku in skus]
        X_all = wide[X_cols].astype(float)

        models_by_product: Dict[str, PoissonRegressor] = {}
        n_by_product: Dict[str, int] = {}

        for sku in skus:
            y_col = f"Q_{sku}"
            if y_col not in wide.columns:
                continue

            data = pd.concat([wide[y_col], X_all], axis=1).dropna(subset=[y_col, f"LOGP_{sku}"])
            if len(data) < min_rows_per_sku:
                continue

            y = data[y_col].astype(float).values
            X = data[X_cols].astype(float).fillna(0.0).values

            m = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=500)
            m.fit(X, y)

            models_by_product[sku] = m
            n_by_product[sku] = int(len(data))

        if len(models_by_product) >= 2:
            out[(fam, store, bucket)] = CrossFamilyModel(
                family_code=fam,
                store=store,
                bucket=bucket,
                family_products=skus,
                models_by_product=models_by_product,
                n_by_product=n_by_product
            )

    return out


def predict_family_demands(cfm: CrossFamilyModel, prices: Dict[str, float]) -> Dict[str, float]:
    """
    Предсказывает E[Q_i] для каждого SKU i при заданных ценах всех Top-K.

    Args:
      cfm: cross модель
      prices: dict[sku] -> price (для всех sku из family_products)

    Returns:
      dict[sku] -> mu
    """
    skus = cfm.family_products
    X = np.array([_safe_log(prices[sku]) for sku in skus], dtype=float).reshape(1, -1)

    out: Dict[str, float] = {}
    for sku, model in cfm.models_by_product.items():
        mu = float(model.predict(X)[0])
        out[sku] = max(mu, 0.0)
    return out


def optimize_family_prices_iterative(cfm: CrossFamilyModel,
                                     base_prices: Dict[str, float],
                                     price_grids: Dict[str, np.ndarray],
                                     iters: int = CROSS_ITERS,
                                     stock_caps: Optional[Dict[str, float]] = None) -> Dict[str, Tuple[float, float, float]]:
    """
    Итеративная оптимизация цен семьи (best-response).

    Args:
      cfm: cross модель
      base_prices: базовые цены для всех Top-K
      price_grids: сетки кандидатов
      iters: число итераций
      stock_caps: ограничения по запасам (опционально)

    Returns:
      dict[sku] -> (chosen_price, pred_qty, pred_rev)
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
            rev = float(p) * mu

            cap = None if stock_caps is None else stock_caps.get(sku)
            if cap is not None and np.isfinite(cap) and mu > float(cap):
                continue

            if best is None or rev > best[2]:
                best = (float(p), mu, float(rev))

        if best is None:
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
# Сетка цен + single fallback оптимизация
# =========================

def build_price_grid(base_price: float,
                     min_mult: float = MIN_MULT,
                     max_mult: float = MAX_MULT,
                     n_mult: int = 21,
                     observed_prices: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Сетка цен: base_price * linspace(min_mult, max_mult, n_mult)
    + снап к observed_prices (если задано).

    Args:
      base_price: базовая цена
      min_mult, max_mult: ограничения
      n_mult: число точек
      observed_prices: лестница цен (опционально)

    Returns:
      np.ndarray цен
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
    """
    Лестницы наблюдавшихся цен по уровню (например FAMILY_CODE).

    Args:
      bucket_df: бакет-панель
      level: колонка уровня

    Returns:
      dict[level_value] -> array(prices)
    """
    ladders: Dict[str, np.ndarray] = {}
    for key, g in bucket_df.groupby(level):
        prices = g["PRICE"].astype(float).values
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) > 0:
            ladders[str(key)] = np.unique(np.round(prices, 4))
    return ladders


def make_single_features_for_candidate(base_row: pd.Series, candidate_price: float) -> Dict[str, float]:
    """
    Single-SKU признаки.

    Args:
      base_row: строка контекста
      candidate_price: цена кандидат

    Returns:
      dict признаков
    """
    lp = _safe_log(candidate_price)
    return {
        "LOG_PRICE": float(lp),
        "PROMO_SHARE": float(base_row.get("PROMO_SHARE", 0.0)),
        "BUCKET_IDX": float(base_row.get("BUCKET_IDX", 0.0)),
        "WEEK_OF_YEAR": float(base_row.get("WEEK_OF_YEAR", 0.0)),
        "LOW_STOCK_FLAG": float(base_row.get("LOW_STOCK_FLAG", 0.0)),
        "QTY_L1": float(base_row.get("QTY_L1", 0.0)),
        "PRICE_L1": float(base_row.get("PRICE_L1", 0.0)) if np.isfinite(base_row.get("PRICE_L1", np.nan)) else 0.0,
        "PROMO_L1": float(base_row.get("PROMO_L1", 0.0)),
    }


def optimize_price_single_sku(base_row: pd.Series,
                             model: PoissonCoeffs,
                             feature_names: List[str],
                             price_grid: np.ndarray,
                             stock_cap: Optional[float] = None) -> Optional[Tuple[float, float, float]]:
    """
    Single-SKU max p * E[Q(p)].

    Args:
      base_row: контекст
      model: модель
      feature_names: порядок
      price_grid: кандидаты цен
      stock_cap: ограничение спроса

    Returns:
      (price, mu, revenue) или None
    """
    if price_grid is None or len(price_grid) == 0:
        return None

    best = None
    for p in price_grid:
        feat = make_single_features_for_candidate(base_row, p)
        x = np.array([feat.get(fn, 0.0) for fn in feature_names], dtype=float)
        mu = poisson_predict_mean(model, x)
        rev = float(p) * float(mu)

        if stock_cap is not None and np.isfinite(stock_cap) and mu > float(stock_cap):
            continue

        if best is None or rev > best[2]:
            best = (float(p), float(mu), float(rev))

    if best is None and stock_cap is not None and np.isfinite(stock_cap):
        best2 = None
        for p in price_grid:
            feat = make_single_features_for_candidate(base_row, p)
            x = np.array([feat.get(fn, 0.0) for fn in feature_names], dtype=float)
            mu = poisson_predict_mean(model, x)
            rev = float(p) * float(mu)
            if best2 is None or mu < best2[1]:
                best2 = (float(p), float(mu), float(rev))
        return best2

    return best


# =========================
# NEW: Продовый расчёт на всю неделю (3 бакета) при запуске в понедельник
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
    Рассчитывает цены сразу на всю целевую неделю (3 бакета), если код запускается в понедельник утром.

    Ключевая идея:
      Для каждого бакета применяем свой decision_cutoff_for_bucket(week, bucket),
      но фактически в проде используем только то, что доступно на момент запуска:
        cutoff_effective = min(decision_cutoff, as_of_date)

    Это:
      - корректно для MON_THU (decision_cutoff уже в прошлом),
      - "best effort" для FRI/SAT_SUN на понедельник (т.к. четверг/пятница ещё не наступили).

    Args:
      bucket_df: бакет-панель
      daily_df: дневные данные
      as_of_date: дата запуска (понедельник утром)
      shrink_k: shrinkage для single fallback
      alpha: регуляризация Poisson
      cross_*: параметры cross-модели

    Returns:
      dataframe с решениями на 3 бакета целевой недели
    """
    today = pd.Timestamp(as_of_date).normalize()
    target_week = week_start(today)  # целевая неделя = текущая (т.к. запуск в понедельник)

    promo_map = build_promo_map(daily_df)

    out_all: List[pd.DataFrame] = []

    # Кандидаты (product,store): все, кто встречался недавно до today
    train_base = bucket_df.loc[bucket_df["BUCKET_END"] <= today].copy()
    recent = train_base.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(12)
    last_ctx = recent.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(1).copy()

    # Для каждого бакета недели считаем решения отдельно (модели могут отличаться из-за cutoff)
    for bucket in BUCKET_ORDER:
        start, end = bucket_start_end(target_week, bucket)

        # "правильный" cutoff по условию 7, но в проде ограничиваем текущей датой
        decision_cutoff = decision_cutoff_for_bucket(target_week, bucket)
        cutoff = min(pd.Timestamp(decision_cutoff).normalize(), today)

        train = bucket_df.loc[bucket_df["BUCKET_END"] <= cutoff].copy()
        if len(train) == 0:
            continue

        fam_ladder = collect_observed_price_ladder(train, level="FAMILY_CODE")

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

        # cross + single модели на данном cutoff
        topk_map = build_family_topk(train, cutoff=cutoff, top_k=cross_top_k, lookback_weeks=cross_lookback_weeks)
        cross_models = train_cross_family_models(train, cutoff=cutoff, topk_map=topk_map, alpha=alpha, min_rows_per_sku=cross_min_rows_per_sku)
        hier_models = train_hierarchical_models(train, train_until=cutoff, feature_names=single_feature_names, alpha=alpha)

        # Контекст на целевой бакет: берем last_ctx и переназначаем bucket/week
        ctx = last_ctx.copy()
        ctx["WEEK"] = pd.Timestamp(target_week)
        ctx["BUCKET"] = bucket
        ctx["BUCKET_IDX"] = {"MON_THU": 0, "FRI": 1, "SAT_SUN": 2}[bucket]
        ctx["WEEK_OF_YEAR"] = int(pd.Timestamp(target_week).isocalendar().week)

        used_in_cross = set()
        out_rows: List[dict] = []

        # A) cross семьи
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
            tmp_prices = []

            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                if sku not in cfm.family_products:
                    continue

                # base price: последняя цена для этого бакета до cutoff, иначе последняя известная до cutoff
                hist_b = train.loc[
                    (train["PRODUCT_CODE"] == sku) &
                    (train["STORE"] == str(store)) &
                    (train["BUCKET"] == bucket)
                ].sort_values("BUCKET_END")
                if len(hist_b) and np.isfinite(hist_b["PRICE"].iloc[-1]):
                    base_price = float(hist_b["PRICE"].iloc[-1])
                else:
                    hist_any = train.loc[
                        (train["PRODUCT_CODE"] == sku) &
                        (train["STORE"] == str(store))
                    ].sort_values("BUCKET_END")
                    base_price = float(hist_any["PRICE"].iloc[-1]) if len(hist_any) and np.isfinite(hist_any["PRICE"].iloc[-1]) else np.nan

                if not np.isfinite(base_price) or base_price <= 0:
                    continue

                base_prices[sku] = base_price
                tmp_prices.append(base_price)

                ladder = fam_ladder.get(str(fam))
                grids[sku] = build_price_grid(base_price, observed_prices=ladder)

                # stock cap: последний END_STOCK до cutoff
                hist_any2 = train.loc[
                    (train["PRODUCT_CODE"] == sku) &
                    (train["STORE"] == str(store))
                ].sort_values("BUCKET_END")
                stock_caps[sku] = float(hist_any2["STOCK_END"].iloc[-1]) if len(hist_any2) else np.nan

            fam_skus = [x for x in fam_skus if x in base_prices]
            if len(fam_skus) < 2:
                continue

            # заполняем отсутствующие Top-K медианой
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
                    "CUTOFF_EFFECTIVE": cutoff,
                    "PROMO_SHARE": promo_share,
                    "STOCK_CAP": stock_caps.get(sku),
                    "BASE_PRICE": float(base_prices[sku]),
                    "CHOSEN_PRICE": chosen_price,
                    "PRED_QTY": pred_qty,
                    "PRED_REV": pred_rev,
                    "USED_CROSS_PRICE_MODEL": True,
                })
                used_in_cross.add((sku, str(store)))

        # B) single fallback
        for _, r in ctx.iterrows():
            sku = str(r["PRODUCT_CODE"])
            store = str(r["STORE"])
            if (sku, store) in used_in_cross:
                continue

            ranges = promo_map.get((sku, store), [])
            promo_share = promo_share_in_range(ranges, start, end) if ranges else 0.0

            # base price: последняя цена для бакета до cutoff, иначе последняя известная
            hist_b = train.loc[
                (train["PRODUCT_CODE"] == sku) &
                (train["STORE"] == store) &
                (train["BUCKET"] == bucket)
            ].sort_values("BUCKET_END")
            if len(hist_b) and np.isfinite(hist_b["PRICE"].iloc[-1]):
                base_price = float(hist_b["PRICE"].iloc[-1])
            else:
                hist_any = train.loc[
                    (train["PRODUCT_CODE"] == sku) &
                    (train["STORE"] == store)
                ].sort_values("BUCKET_END")
                base_price = float(hist_any["PRICE"].iloc[-1]) if len(hist_any) and np.isfinite(hist_any["PRICE"].iloc[-1]) else np.nan

            if not np.isfinite(base_price) or base_price <= 0:
                continue

            hist_any2 = train.loc[
                (train["PRODUCT_CODE"] == sku) &
                (train["STORE"] == store)
            ].sort_values("BUCKET_END")
            stock_cap = float(hist_any2["STOCK_END"].iloc[-1]) if len(hist_any2) else None

            base_row = r.copy()
            base_row["PROMO_SHARE"] = promo_share

            m_eff = get_effective_model(base_row, hier_models, k=shrink_k)
            if m_eff is None:
                continue

            ladder = fam_ladder.get(str(r["FAMILY_CODE"]))
            grid = build_price_grid(base_price, observed_prices=ladder)

            best = optimize_price_single_sku(
                base_row=base_row,
                model=m_eff,
                feature_names=single_feature_names,
                price_grid=grid,
                stock_cap=stock_cap
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
                "CUTOFF_EFFECTIVE": cutoff,
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
# Обёртки
# =========================

def run_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн:
      1) preprocess daily
      2) aggregate to buckets
      3) add lags
      4) rolling backtest (12 weeks)

    Args:
      df_raw: сырой датафрейм

    Returns:
      (bucket_panel, backtest_results)
    """
    daily = preprocess_raw(df_raw, history_end=HISTORY_END)
    buckets = aggregate_to_buckets(daily)
    buckets = add_lag_features(buckets, lags=1)

    backtest = rolling_backtest_pricing(
        bucket_df=buckets,
        daily_df=daily,
        eval_weeks=12,
        train_window_weeks=None,
        shrink_k=DEFAULT_SHRINK_K,
        alpha=POISSON_ALPHA,
        cross_top_k=CROSS_TOP_K,
        cross_lookback_weeks=CROSS_LOOKBACK_WEEKS,
        cross_min_rows_per_sku=CROSS_MIN_ROWS_PER_SKU,
        cross_iters=CROSS_ITERS
    )
    return buckets, backtest


def summarize_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    """
    Сводка backtest (по предсказанной выручке).

    Args:
      backtest: результат rolling_backtest_pricing

    Returns:
      1-row dataframe с метриками
    """
    if backtest is None or len(backtest) == 0:
        return pd.DataFrame([{"rows": 0}])

    pred_rev_sum = float(backtest["PRED_REV"].sum())
    base_rev_sum = float(backtest["BASE_PRED_REV"].sum())
    lift = pred_rev_sum - base_rev_sum
    lift_pct = lift / max(base_rev_sum, 1e-9)

    actual_rev = (backtest["ACTUAL_QTY"] * backtest["ACTUAL_PRICE"]).replace([np.inf, -np.inf], np.nan).sum(skipna=True)

    return pd.DataFrame([{
        "rows": int(len(backtest)),
        "pred_rev_sum": pred_rev_sum,
        "base_pred_rev_sum": base_rev_sum,
        "pred_lift": float(lift),
        "pred_lift_pct": float(lift_pct),
        "actual_rev_sum_reference": float(actual_rev),
        "share_used_cross_model": float((backtest["USED_CROSS_PRICE_MODEL"] == True).mean()),
    }])


# Пример:
# ------------------------------------------------------------
# import pandas as pd
# df = pd.read_parquet("data.parquet")
# bucket_panel, backtest = run_pipeline(df)
# print(summarize_backtest(backtest))
#
# # Прод:
# decisions = price_next_bucket(bucket_panel, preprocess_raw(df), as_of_date=pd.Timestamp("2025-09-14"))
# decisions.to_csv("next_bucket_prices.csv", index=False)
# ------------------------------------------------------------