# dynamic_pricing_elasticity_prod_v6.py
# ------------------------------------------------------------
# Pricing engine (elasticity + cannibalization) for retail buckets:
#   - Buckets: MON_THU, FRI, SAT_SUN (prices change only between buckets)
#   - Runs:
#       * Monday morning: compute prices for MON_THU (using data up to Sunday prev week)
#       * Friday morning: compute prices for FRI and SAT_SUN (using data up to Thu of same week)
#   - Elasticity:
#       * Poisson (log-link) demand model with price features + seasonality + promo + OOS + lags
#       * Hierarchical backoff: PRODUCT_STORE -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL
#       * Local elasticity around chosen price (±delta)
#   - Cannibalization:
#       * Cross-family Poisson models for top-K SKUs inside FAMILY (within STORE+BUCKET)
#       * Coordinate-descent optimization across SKUs
#   - Promo rules:
#       * REGULAR price is BASE_PRICE (input column). We optimize sale price (PRICE) around BASE_PRICE.
#       * If PROMO_SHARE > threshold => hard cap: CHOSEN_PRICE <= BASE_PRICE (regular).
#       * For computing regular anchor (BASE_PRICE fallback) we can exclude promo periods.
#   - Active products:
#       * Predict only SKU+STORE that sold within last N days before cutoff (otherwise skip).
#
# Требования к данным (минимум):
#   TRADE_DT, PRODUCT_CODE, STORE
#   SALE_QTY (optional) / SALE_QTY_ONLINE (optional) -> QTY_TOTAL
#   SALE_PRICE (желательно) или SALE_PRICE_TOTAL (вместе с SALE_QTY) -> фактическая цена продажи
#   BASE_PRICE (регулярная цена) желательно
#   END_STOCK (желательно), START_STOCK (опционально)
#   PROMO_PERIOD (строка "dd-mm-yyyy - dd-mm-yyyy") или IS_PROMO (0/1)
#   Иерархия: FAMILY_CODE, CATEGORY_CODE, SEGMENT_CODE (желательно)
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler


# =========================
# CONFIG
# =========================

# История данных (чтобы не "видеть будущее")
HISTORY_END = pd.Timestamp("2025-09-15")

# Бакеты недели и их порядок (для кодирования)
BUCKET_ORDER = ["MON_THU", "FRI", "SAT_SUN"]

# Бизнес-ограничения: разрешённые мультипликаторы цены относительно BASE_PRICE
MIN_MULT = 0.9
MAX_MULT = 1.3

# Poisson demand model
POISSON_ALPHA = 1e-2          # регуляризация, повышай если коэффициенты "взрываются"
POISSON_MAX_ITER = 3000       # чтобы меньше было ConvergenceWarning

# Иерархическое смешивание уровней (backoff)
DEFAULT_SHRINK_K = 30         # больше => сильнее тянем к более общим уровням

# Промо-порог (если доля промо-дней в бакете больше порога => считаем бакет промо)
PROMO_THRESHOLD = 0.15
MIN_ROWS_SPLIT_MODEL = 60     # минимум строк для обучения отдельной promo/nonpromo модели

# Guardrails для эластичности (чтобы оптимизация не уезжала в MAX_MULT от "плоского" спроса)
BETA_FLOOR = 0.7              # если beta_raw > -BETA_FLOOR или beta_raw плохой => используем -BETA_FLOOR

# Штраф удерживает цену ближе к BASE_PRICE (регулярной)
LAMBDA_PRICE_PENALTY = 0.25

# Локальная эластичность вокруг выбранной цены (±delta)
LOCAL_ELASTICITY_DELTA = 0.05

# Минимум данных для уровней и качества эластичности
MIN_ROWS_GLOBAL = 600
MIN_ROWS_LEVEL = 120
MIN_ROWS_PS = 40
MIN_POS_SALES_FOR_ELAST = 6

# Активные товары: считаем цены только тем, кто продавался недавно до cutoff
ACTIVE_SOLD_LOOKBACK_DAYS = 90
ACTIVE_MIN_QTY = 1.0

# Праздники РФ: флаг "праздник или неделя до него" на бакет
HOLIDAY_LEAD_DAYS = 7

# Регулярная цена: если BASE_PRICE отсутствует, берём медиану за последние N недель
REG_PRICE_MEDIAN_WEEKS = 4
REG_PRICE_MIN_POINTS = 2

# Исключать ли промо-периоды при оценке регулярной цены (BASE_PRICE fallback)
EXCLUDE_PROMO_PRICES_FROM_REG_PRICE = True

# Cross-family (каннибализация)
CROSS_TOP_K = 7
CROSS_LOOKBACK_WEEKS = 26
CROSS_MIN_ROWS_PER_SKU = 30
CROSS_ITERS = 4
CROSS_MIN_FAMILY_WEEKS = 20
CROSS_MIN_FAMILY_SUM_QTY = 20

# Численная стабильность Poisson: mu = exp(eta), clip eta чтобы не было overflow
MAX_LINEAR_PRED = 20.0  # exp(20) ~ 4.8e8


# =========================
# BUCKETS / DATES
# =========================

def bucket_id_for_date(dt: pd.Timestamp) -> str:
    """Определяем бакет по дню недели: Mon-Thu / Fri / Sat-Sun."""
    wd = pd.Timestamp(dt).weekday()
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
    """Границы бакета внутри недели (по датам)."""
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
    Cutoff (до какой даты можно использовать факты) в зависимости от режима запуска.

    - mode="monday": считаем MON_THU в понедельник утром => используем факты до воскресенья прошлой недели.
    - mode="friday": считаем FRI и SAT_SUN в пятницу утром => используем факты до четверга текущей недели.

    Важно: цены внутри бакета не меняем, но рассчитываем заранее по доступным фактам.
    """
    mode = mode.lower().strip()
    m = pd.Timestamp(week_monday).normalize()

    if mode == "monday":
        return m - pd.Timedelta(days=1)  # Sunday prev week
    if mode == "friday":
        return m + pd.Timedelta(days=3)  # Thu current week

    # fallback (если кто-то зовёт иначе)
    if bucket == "MON_THU":
        return m - pd.Timedelta(days=1)
    if bucket == "FRI":
        return m + pd.Timedelta(days=3)
    if bucket == "SAT_SUN":
        return m + pd.Timedelta(days=4)
    raise ValueError(bucket)


def _safe_log(x: float) -> float:
    """Безопасный лог (чтобы не было log(0))."""
    return float(np.log(max(float(x), 1e-6)))


# =========================
# PROMO_PERIOD parsing
# =========================

def parse_promo_period(s: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Парсим PROMO_PERIOD вида '01-01-2024 - 03-01-2024' в (start, end)."""
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
    """Доля дней промо на интервале [start, end]."""
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
# HOLIDAYS (Russia)
# =========================

try:
    import holidays  # pip install holidays
    _RU_HOL = holidays.RU()
except Exception:
    _RU_HOL = None

_FIXED_RU_MMDD = {
    "01-01", "01-02", "01-03", "01-04", "01-05", "01-06", "01-07", "01-08",
    "02-23",
    "03-08",
    "05-01",
    "05-09",
    "06-12",
    "11-04",
}

def is_ru_holiday(dt: pd.Timestamp) -> bool:
    dt = pd.Timestamp(dt).normalize()
    if _RU_HOL is not None:
        return dt in _RU_HOL
    return dt.strftime("%m-%d") in _FIXED_RU_MMDD

def is_ru_holiday_or_preweek(dt: pd.Timestamp, lead_days: int = HOLIDAY_LEAD_DAYS) -> bool:
    """
    True если дата:
      - праздник РФ
      - или попадает в "неделю до праздника" (lead_days)
    """
    dt = pd.Timestamp(dt).normalize()
    if is_ru_holiday(dt):
        return True
    for k in range(1, int(lead_days) + 1):
        if is_ru_holiday(dt + pd.Timedelta(days=k)):
            return True
    return False

def holiday_flag_for_bucket(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Флаг бакета: 1 если есть день праздника или неделя до него внутри бакета."""
    days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")
    return int(any(is_ru_holiday_or_preweek(d) for d in days))


# =========================
# PREPROCESS DAILY
# =========================

def compute_sale_price_per_unit(df: pd.DataFrame) -> pd.Series:
    """
    Фактическая цена продажи (за единицу).
    - Если есть SALE_PRICE (уже unit price) — используем.
    - Иначе пытаемся вычислить из SALE_PRICE_TOTAL / QTY.
    """
    if "SALE_PRICE" in df.columns:
        return pd.to_numeric(df["SALE_PRICE"], errors="coerce")

    total = pd.to_numeric(df.get("SALE_PRICE_TOTAL"), errors="coerce")
    qty = pd.to_numeric(df.get("SALE_QTY"), errors="coerce").fillna(0.0)
    return total / qty.clip(lower=1.0)


def preprocess_raw(df: pd.DataFrame, history_end: pd.Timestamp = HISTORY_END) -> pd.DataFrame:
    """
    Подготовка дневных данных:
      - TRADE_DT -> datetime
      - QTY_TOTAL = SALE_QTY + SALE_QTY_ONLINE
      - SALE_PRICE_UNIT = фактическая цена продажи (за единицу)
      - BASE_PRICE (если есть) оставляем
      - BUCKET по дню недели
      - PROMO_RANGE из PROMO_PERIOD (если есть)
    """
    out = df.copy()

    out["TRADE_DT"] = pd.to_datetime(out["TRADE_DT"], errors="coerce").dt.normalize()
    out = out.loc[out["TRADE_DT"].notna()]
    out = out.loc[out["TRADE_DT"] <= pd.Timestamp(history_end).normalize()]

    qty_off = pd.to_numeric(out.get("SALE_QTY"), errors="coerce").fillna(0.0)
    qty_on = pd.to_numeric(out.get("SALE_QTY_ONLINE"), errors="coerce").fillna(0.0)
    out["QTY_TOTAL"] = (qty_off + qty_on).astype(float).clip(lower=0.0)

    # Фактическая цена продажи (unit)
    out["SALE_PRICE_UNIT"] = compute_sale_price_per_unit(out).astype(float)

    # Регулярная цена (если в данных есть BASE_PRICE)
    if "BASE_PRICE" in out.columns:
        out["BASE_PRICE"] = pd.to_numeric(out["BASE_PRICE"], errors="coerce").astype(float)
    else:
        out["BASE_PRICE"] = np.nan

    out["BUCKET"] = out["TRADE_DT"].apply(bucket_id_for_date)

    if "PROMO_PERIOD" in out.columns:
        out["PROMO_RANGE"] = out["PROMO_PERIOD"].apply(parse_promo_period)
    else:
        out["PROMO_RANGE"] = None

    if "IS_PROMO" in out.columns:
        out["IS_PROMO"] = pd.to_numeric(out["IS_PROMO"], errors="coerce").fillna(0.0).astype(int)
    else:
        out["IS_PROMO"] = 0

    # Приводим ключи к строкам (чтобы join/groupby не ломался)
    keys = ["PRODUCT_CODE", "FAMILY_CODE", "CATEGORY_CODE", "SEGMENT_CODE",
            "STORE", "REGION_NAME", "STORE_TYPE", "PLACE_TYPE"]
    for k in keys:
        if k in out.columns:
            out[k] = out[k].astype(str).fillna("NA")
        else:
            # если кода нет — создаём NA, чтобы иерархия работала хотя бы как "NA"
            out[k] = "NA"

    for c in ["START_STOCK", "END_STOCK", "DELIVERY_QTY", "LOSS_QTY", "RETURN_QTY"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
        else:
            out[c] = 0.0

    return out


def build_promo_map(daily: pd.DataFrame) -> Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """Сохраняем промо-диапазоны по (SKU, STORE)."""
    promo_map: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if "PROMO_RANGE" not in daily.columns:
        return promo_map
    for (p, s), g in daily.groupby(["PRODUCT_CODE", "STORE"]):
        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)]
        if ranges:
            promo_map[(str(p), str(s))] = ranges
    return promo_map


def build_active_sku_store_set(daily_df: pd.DataFrame,
                               cutoff: pd.Timestamp,
                               lookback_days: int = ACTIVE_SOLD_LOOKBACK_DAYS,
                               min_qty: float = ACTIVE_MIN_QTY) -> Set[Tuple[str, str]]:
    """
    Множество (SKU, STORE) которые продавались в окне [cutoff - lookback_days, cutoff].
    Это убирает товары, которые давно не продаются, но по ним всё равно есть строки.
    """
    cutoff = pd.Timestamp(cutoff).normalize()
    start = cutoff - pd.Timedelta(days=int(lookback_days))

    d = daily_df.loc[(daily_df["TRADE_DT"] >= start) & (daily_df["TRADE_DT"] <= cutoff)].copy()
    if len(d) == 0:
        return set()

    qty = pd.to_numeric(d["QTY_TOTAL"], errors="coerce").fillna(0.0)
    d["_Q"] = qty

    g = d.groupby(["PRODUCT_CODE", "STORE"], as_index=False)["_Q"].sum()
    g = g.loc[g["_Q"] >= float(min_qty)]

    return set((str(p), str(s)) for p, s in zip(g["PRODUCT_CODE"].astype(str), g["STORE"].astype(str)))


def get_last_daily_prices(daily_df: pd.DataFrame,
                          sku: str,
                          store: str,
                          cutoff: pd.Timestamp,
                          exclude_is_promo: bool = False) -> Tuple[Optional[float], Optional[float]]:
    """
    Возвращает (last_sale_price, last_base_price) по дневным данным до cutoff.
    - last_sale_price берём из SALE_PRICE_UNIT (или SALE_PRICE если в данных он уже unit).
    - last_base_price берём из BASE_PRICE (если есть).
    - exclude_is_promo=True: игнорируем дни где IS_PROMO=1.
    """
    d = daily_df.loc[
        (daily_df["PRODUCT_CODE"].astype(str) == str(sku)) &
        (daily_df["STORE"].astype(str) == str(store)) &
        (daily_df["TRADE_DT"] <= pd.Timestamp(cutoff))
    ].copy()

    if len(d) == 0:
        return None, None

    if exclude_is_promo and "IS_PROMO" in d.columns:
        d = d.loc[d["IS_PROMO"].astype(int) == 0]
        if len(d) == 0:
            return None, None

    d = d.sort_values("TRADE_DT")

    sp = pd.to_numeric(d["SALE_PRICE_UNIT"].iloc[-1], errors="coerce")
    sp = float(sp) if np.isfinite(sp) and float(sp) > 0 else None

    bp = pd.to_numeric(d["BASE_PRICE"].iloc[-1], errors="coerce")
    bp = float(bp) if np.isfinite(bp) and float(bp) > 0 else None

    return sp, bp


# =========================
# AGGREGATE DAILY -> BUCKET
# =========================

def aggregate_to_buckets(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегируем дневные данные до уровня:
      SKU, STORE, WEEK, BUCKET, + иерархия

    В bucket_df:
      - QTY: сумма QTY_TOTAL
      - PRICE: фактическая цена продажи на бакете (weighted avg по QTY)
      - BASE_PRICE: регулярная цена на бакете (последнее наблюдение BASE_PRICE внутри бакета)
      - PROMO_SHARE: доля промо-дней в бакете (по PROMO_PERIOD или IS_PROMO)
      - STOCK_END: остаток на конец бакета, OOS_FLAG
      - HOLIDAY_FLAG: праздник или неделя до него попадает внутрь бакета
      - Сезонность/тренд: SIN_WOY/COS_WOY, SIN_MONTH/COS_MONTH, TREND_W
    """
    d = daily.copy()
    d["WEEK"] = d["TRADE_DT"].apply(week_start)

    group_cols = [
        "PRODUCT_CODE", "STORE", "WEEK", "BUCKET",
        "FAMILY_CODE", "CATEGORY_CODE", "SEGMENT_CODE",
        "REGION_NAME", "STORE_TYPE", "PLACE_TYPE"
    ]

    def agg_group(g: pd.DataFrame) -> pd.Series:
        qty = float(pd.to_numeric(g["QTY_TOTAL"], errors="coerce").fillna(0.0).sum())
        qty = max(qty, 0.0)

        # Фактическая цена продажи (SALE_PRICE_UNIT)
        pvals = pd.to_numeric(g["SALE_PRICE_UNIT"], errors="coerce").astype(float).values
        price = np.nan
        if qty > 0 and np.isfinite(pvals).any():
            w = pd.to_numeric(g["QTY_TOTAL"], errors="coerce").fillna(0.0).astype(float).values
            if np.nansum(w) > 0:
                price = float(np.nansum(pvals * w) / np.nansum(w))
        if not np.isfinite(price):
            price = float(np.nanmean(pvals)) if np.isfinite(pvals).any() else np.nan

        # Регулярная цена (BASE_PRICE): берём последнее наблюдение внутри бакета
        bpvals = pd.to_numeric(g["BASE_PRICE"], errors="coerce").astype(float).values
        base_price = np.nan
        if np.isfinite(bpvals).any():
            base_price = float(pd.to_numeric(g["BASE_PRICE"], errors="coerce").dropna().iloc[-1]) \
                if len(pd.to_numeric(g["BASE_PRICE"], errors="coerce").dropna()) > 0 else np.nan

        wk = g["WEEK"].iloc[0]
        b = g["BUCKET"].iloc[0]
        start, end = bucket_start_end(wk, b)

        # PROMO_SHARE: по PROMO_PERIOD если есть, иначе по IS_PROMO
        ranges = [x for x in g["PROMO_RANGE"].tolist() if isinstance(x, tuple)] if "PROMO_RANGE" in g else []
        if ranges:
            promo_share = promo_share_in_range(ranges, start, end)
        else:
            promo_share = float(pd.to_numeric(g.get("IS_PROMO", 0), errors="coerce").fillna(0.0).mean())

        stock_end = float(pd.to_numeric(g["END_STOCK"], errors="coerce").fillna(0.0).iloc[-1])
        stock_start = float(pd.to_numeric(g["START_STOCK"], errors="coerce").fillna(0.0).iloc[0])

        holiday_flag = holiday_flag_for_bucket(start, end)

        return pd.Series({
            "QTY": qty,
            "PRICE": price,                 # фактическая цена продажи на бакете
            "BASE_PRICE": base_price,       # регулярная цена (если есть)
            "PROMO_SHARE": promo_share,
            "STOCK_START": stock_start,
            "STOCK_END": stock_end,
            "BUCKET_START": start,
            "BUCKET_END": end,
            "HOLIDAY_FLAG": holiday_flag,
        })

    bucket_df = d.groupby(group_cols, as_index=False).apply(agg_group).reset_index(drop=True)

    # Сезонность и тренд
    bucket_df["WEEK_OF_YEAR"] = bucket_df["WEEK"].dt.isocalendar().week.astype(int)
    bucket_df["BUCKET_IDX"] = bucket_df["BUCKET"].map({b: i for i, b in enumerate(BUCKET_ORDER)}).astype(int)

    woy = bucket_df["WEEK_OF_YEAR"].astype(float)
    bucket_df["SIN_WOY"] = np.sin(2.0 * np.pi * woy / 52.0)
    bucket_df["COS_WOY"] = np.cos(2.0 * np.pi * woy / 52.0)

    bucket_df["MONTH"] = bucket_df["WEEK"].dt.month.astype(int)
    bucket_df["SIN_MONTH"] = np.sin(2.0 * np.pi * bucket_df["MONTH"].astype(float) / 12.0)
    bucket_df["COS_MONTH"] = np.cos(2.0 * np.pi * bucket_df["MONTH"].astype(float) / 12.0)

    min_week = bucket_df["WEEK"].min()
    bucket_df["TREND_W"] = ((bucket_df["WEEK"] - min_week) / np.timedelta64(1, "W")).astype(float)

    # Cleaning
    bucket_df["QTY"] = pd.to_numeric(bucket_df["QTY"], errors="coerce").fillna(0.0).clip(lower=0.0)
    bucket_df["PRICE"] = pd.to_numeric(bucket_df["PRICE"], errors="coerce").astype(float)
    bucket_df["BASE_PRICE"] = pd.to_numeric(bucket_df["BASE_PRICE"], errors="coerce").astype(float)
    bucket_df["STOCK_END"] = pd.to_numeric(bucket_df["STOCK_END"], errors="coerce").fillna(0.0).clip(lower=0.0)

    bucket_df["OOS_FLAG"] = (bucket_df["STOCK_END"] <= 0.0).astype(int)

    return bucket_df


# =========================
# LAGS (price today -> sales tomorrow)
# =========================

def add_lag_features(bucket_df: pd.DataFrame, lags: int = 1) -> pd.DataFrame:
    """
    Лаги внутри (SKU, STORE, BUCKET).
    Важно: помогает не смешивать "цена сейчас" и "продажи сейчас" (в ритейле часто есть задержка).
    """
    d = bucket_df.copy().sort_values(["PRODUCT_CODE", "STORE", "BUCKET", "WEEK"])
    grp = d.groupby(["PRODUCT_CODE", "STORE", "BUCKET"], sort=False)

    for k in range(1, lags + 1):
        d[f"QTY_L{k}"] = grp["QTY"].shift(k).fillna(0.0)
        d[f"PRICE_L{k}"] = grp["PRICE"].shift(k)
        d[f"BASE_PRICE_L{k}"] = grp["BASE_PRICE"].shift(k)
        d[f"PROMO_L{k}"] = grp["PROMO_SHARE"].shift(k).fillna(0.0)
        d[f"STOCK_END_L{k}"] = grp["STOCK_END"].shift(k).fillna(0.0)
        d[f"OOS_L{k}"] = grp["OOS_FLAG"].shift(k).fillna(0).astype(int)

    d["LOG_PRICE_L1"] = d["PRICE_L1"].apply(lambda v: _safe_log(v) if np.isfinite(v) and v > 0 else 0.0)
    d["LOW_STOCK_FLAG"] = (d["STOCK_END_L1"].fillna(0.0) <= 1.0).astype(int)

    return d


# =========================
# TRAIN-TIME regular-price features (fix: columns not in index)
# =========================

def add_training_regular_price_features(bucket_df: pd.DataFrame,
                                        median_weeks: int = REG_PRICE_MEDIAN_WEEKS,
                                        min_points: int = REG_PRICE_MIN_POINTS) -> pd.DataFrame:
    """
    Фичи должны существовать и в обучении, и при scoring кандидатов.
    Эти фичи завязаны на регулярную цену (BASE_PRICE):
      - BASE_PRICE_TRAIN: rolling median базовой цены (без текущей точки)
      - DISCOUNT_DEPTH: max(1 - PRICE / BASE_PRICE_TRAIN, 0)
      - PRICE_CHANGE_LOG: log(PRICE) - log(BASE_PRICE_TRAIN)
      - PROMO_DEPTH: DISCOUNT_DEPTH * PROMO_SHARE

    Почему:
      - PRICE = фактическая цена продажи
      - BASE_PRICE_TRAIN = регулярная цена, относительно которой считаем "скидку"
    """
    d = bucket_df.copy().sort_values(["PRODUCT_CODE", "STORE", "BUCKET", "WEEK"])
    grp = d.groupby(["PRODUCT_CODE", "STORE", "BUCKET"], sort=False)

    # rolling median по BASE_PRICE, исключая текущую точку (shift(1))
    roll = grp["BASE_PRICE"].apply(
        lambda s: s.shift(1).rolling(window=int(median_weeks), min_periods=int(min_points)).median()
    ).reset_index(level=[0, 1, 2], drop=True)

    d["BASE_PRICE_TRAIN"] = roll

    # fallback: если медианы нет, используем прошлую базовую цену, затем текущую
    if "BASE_PRICE_L1" in d.columns:
        d["BASE_PRICE_TRAIN"] = d["BASE_PRICE_TRAIN"].fillna(d["BASE_PRICE_L1"])
    d["BASE_PRICE_TRAIN"] = d["BASE_PRICE_TRAIN"].fillna(d["BASE_PRICE"])

    bp = pd.to_numeric(d["BASE_PRICE_TRAIN"], errors="coerce").astype(float)
    sp = pd.to_numeric(d["PRICE"], errors="coerce").astype(float)

    bp = bp.where(np.isfinite(bp) & (bp > 0), np.nan)
    sp = sp.where(np.isfinite(sp) & (sp > 0), np.nan)

    d["DISCOUNT_DEPTH"] = (1.0 - (sp / bp)).clip(lower=0.0).fillna(0.0)
    d["PRICE_CHANGE_LOG"] = (sp.apply(_safe_log) - bp.apply(_safe_log)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d["PROMO_DEPTH"] = (d["DISCOUNT_DEPTH"] * pd.to_numeric(d.get("PROMO_SHARE", 0.0), errors="coerce").fillna(0.0)).astype(float)

    return d


# =========================
# MODELS: Poisson + dual promo/nonpromo + hierarchical backoff
# =========================

@dataclass
class PoissonModel:
    reg: PoissonRegressor
    scaler: StandardScaler
    feature_names: List[str]
    n: int
    pos_sales: int


def fit_poisson(df: pd.DataFrame, feature_names: List[str], alpha: float = POISSON_ALPHA) -> Optional[PoissonModel]:
    """
    Обучаем PoissonRegressor на бакетных данных.
    Важно:
      - y=QTY должно быть >= 0 (иначе HalfPoissonLoss падает)
      - PRICE > 0
      - Все фичи должны существовать (поэтому подстраховываемся add_training_regular_price_features)
    """
    g = df.copy()

    # FIX: если фичи ещё не рассчитаны — добавим
    need = {"DISCOUNT_DEPTH", "PRICE_CHANGE_LOG", "PROMO_DEPTH"}
    if len(need - set(g.columns)) > 0:
        g = add_training_regular_price_features(g)

    g = g.loc[np.isfinite(g["PRICE"].values) & (g["PRICE"].values > 0)]
    if len(g) < 20:
        return None

    y = np.clip(pd.to_numeric(g["QTY"], errors="coerce").fillna(0.0).astype(float).values, 0.0, None)
    if not np.isfinite(y).all():
        return None

    X = g[feature_names].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    reg = PoissonRegressor(alpha=max(alpha, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
    reg.fit(Xs, y)

    return PoissonModel(reg, scaler, list(feature_names), int(len(g)), int((y > 0).sum()))


def predict_mu(model: PoissonModel, x_raw: np.ndarray) -> float:
    """
    Безопасный прогноз mu для PoissonRegressor:
      mu = exp(eta), eta = линейный предиктор.
    Чтобы избежать overflow в exp(), ограничиваем eta.
    """
    Xs = model.scaler.transform(x_raw.reshape(1, -1))
    eta = float(model.reg._linear_predictor(Xs)[0])  # type: ignore[attr-defined]
    eta = float(np.clip(eta, -MAX_LINEAR_PRED, MAX_LINEAR_PRED))
    return float(np.exp(eta))


def coef_raw(model: PoissonModel, feature: str) -> Optional[float]:
    """
    Восстанавливаем "сырой" коэффициент по фиче после StandardScaler:
      beta_raw = beta_scaled / scale
    Для LOG_PRICE_L1 это и есть d log(mu) / d log(price) (эластичность).
    """
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
    """
    ВАЖНО ПРО ПРОМО:
    Мы не выбрасываем промо из обучения полностью.
    Вместо этого обучаем две модели:
      - nonpromo: PROMO_SHARE <= threshold
      - promo:    PROMO_SHARE > threshold
    Это снижает смещение, когда промо-дни имеют другую динамику спроса.
    """
    nonpromo = df.loc[df["PROMO_SHARE"] <= PROMO_THRESHOLD].copy()
    promo = df.loc[df["PROMO_SHARE"] > PROMO_THRESHOLD].copy()
    return nonpromo, promo


def train_dual(df: pd.DataFrame, feature_names: List[str], alpha: float) -> DualModels:
    nonpromo_df, promo_df = split_promo(df)
    m_np = fit_poisson(nonpromo_df, feature_names, alpha=alpha) if len(nonpromo_df) >= MIN_ROWS_SPLIT_MODEL else None
    m_p = fit_poisson(promo_df, feature_names, alpha=alpha) if len(promo_df) >= MIN_ROWS_SPLIT_MODEL else None
    return DualModels(nonpromo=m_np, promo=m_p)


def choose_model_with_backoff(dm: DualModels, promo_share: float) -> Optional[PoissonModel]:
    """Выбираем promo/nonpromo модель под текущий бакет (с fallback)."""
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

    # FIX: чтобы DISCOUNT_DEPTH/PRICE_CHANGE_LOG/PROMO_DEPTH были в train
    train = add_training_regular_price_features(train)

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
    """Вес смешивания: чем больше данных на нижнем уровне, тем больше ему доверяем."""
    return float(n_child / (n_child + k)) if n_child > 0 else 0.0


# =========================
# REGULAR PRICE (BASE_PRICE) + promo cap
# =========================

def is_valid_price(p: Optional[float]) -> bool:
    return p is not None and np.isfinite(p) and float(p) > 0


def compute_regular_base_price(bucket_df: pd.DataFrame,
                               sku: str,
                               store: str,
                               bucket: str,
                               cutoff: pd.Timestamp,
                               base_price_last: Optional[float],
                               median_weeks: int = REG_PRICE_MEDIAN_WEEKS,
                               exclude_promo_prices: bool = EXCLUDE_PROMO_PRICES_FROM_REG_PRICE) -> Tuple[Optional[float], str]:
    """
    Определяем регулярную цену (BASE_PRICE) для якоря и промо-cap.

    Источник (по приоритету):
      1) BASE_BUCKET_LAST: base_price_last валиден (последняя BASE_PRICE в этом бакете до cutoff)
      2) MEDIAN_LAST_NW_BUCKET: медиана BASE_PRICE за последние N недель в этом бакете
      3) MEDIAN_LAST_NW_ANY_BUCKET: медиана BASE_PRICE за последние N недель по любому бакету
      4) FALLBACK_NONE: ничего нет

    ВАЖНО ПРО ПРОМО:
      Если exclude_promo_prices=True, то при вычислении медианы используем только строки,
      где PROMO_SHARE <= PROMO_THRESHOLD (то есть "не промо" режим).
      Это не значит "мы не используем промо цены вообще" — промо используется в demand модели,
      но регулярную цену мы хотим получать из обычного режима.
    """
    if is_valid_price(base_price_last):
        return float(base_price_last), "BASE_BUCKET_LAST"

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

    if exclude_promo_prices and "PROMO_SHARE" in df_b.columns:
        df_b = df_b.loc[df_b["PROMO_SHARE"] <= PROMO_THRESHOLD]

    pr = pd.to_numeric(df_b["BASE_PRICE"], errors="coerce").astype(float)
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

    if exclude_promo_prices and "PROMO_SHARE" in df_a.columns:
        df_a = df_a.loc[df_a["PROMO_SHARE"] <= PROMO_THRESHOLD]

    pr = pd.to_numeric(df_a["BASE_PRICE"], errors="coerce").astype(float)
    pr = pr[np.isfinite(pr) & (pr > 0)]
    if len(pr) >= REG_PRICE_MIN_POINTS:
        return float(np.median(pr)), "MEDIAN_LAST_NW_ANY_BUCKET"

    return None, "FALLBACK_NONE"


def discount_depth(candidate_price: float, base_price: float) -> float:
    """Глубина скидки относительно BASE_PRICE."""
    if not is_valid_price(base_price):
        return 0.0
    return float(max(1.0 - float(candidate_price) / float(base_price), 0.0))


def price_change_log(candidate_price: float, base_price: float) -> float:
    """Лог-изменение цены относительно BASE_PRICE."""
    if not is_valid_price(base_price):
        return 0.0
    return float(_safe_log(candidate_price) - _safe_log(base_price))


def apply_promo_cap_to_grid(grid: np.ndarray,
                            base_price: Optional[float],
                            promo_share: float) -> Tuple[np.ndarray, Optional[float]]:
    """
    Жёсткое правило:
      если PROMO_SHARE > PROMO_THRESHOLD => CHOSEN_PRICE <= BASE_PRICE (регулярная цена).

    Реализация:
      - обрезаем сетку цен сверху по base_price.
      - возвращаем cap_price, чтобы сохранить в output.
    """
    if grid is None or len(grid) == 0:
        return grid, None
    if promo_share > PROMO_THRESHOLD and is_valid_price(base_price):
        cap = float(base_price)
        grid2 = grid[grid <= cap + 1e-9]
        if len(grid2) == 0:
            return np.array([cap], dtype=float), cap
        return grid2, cap
    return grid, None


def objective(price: float, mu: float, base_price: float, qty_scale: float) -> float:
    """
    Целевая функция:
      revenue - penalty(price deviation from base)
    penalty удерживает цену ближе к BASE_PRICE.
    """
    revenue = float(price) * float(mu)
    rel = float(price) / max(float(base_price), 1e-6) - 1.0
    penalty_price = LAMBDA_PRICE_PENALTY * (rel * rel) * (float(base_price) * float(qty_scale))
    return revenue - penalty_price


def make_features_for_candidate(base_row: pd.Series,
                                candidate_price: float,
                                base_price: float,
                                bucket: str,
                                week_monday: pd.Timestamp) -> Dict[str, float]:
    """
    Формируем фичи для scoring кандидата.
    BASE_PRICE используется как регулярный якорь (скидки, промо-глубина).
    """
    woy = int(pd.Timestamp(week_monday).isocalendar().week)
    sin_woy = float(np.sin(2.0 * np.pi * woy / 52.0))
    cos_woy = float(np.cos(2.0 * np.pi * woy / 52.0))

    month = int(pd.Timestamp(week_monday).month)
    sin_month = float(np.sin(2.0 * np.pi * month / 12.0))
    cos_month = float(np.cos(2.0 * np.pi * month / 12.0))

    trend_w = float(base_row.get("TREND_W", 0.0))
    promo_share = float(base_row.get("PROMO_SHARE", 0.0))
    holiday_flag = float(base_row.get("HOLIDAY_FLAG", 0.0))

    dd = discount_depth(candidate_price, base_price)
    pcl = price_change_log(candidate_price, base_price)
    promo_depth = float(dd * promo_share)

    return {
        # Цена кандидата
        "LOG_PRICE_L1": float(_safe_log(candidate_price)),
        "LOG_PRICE": float(_safe_log(candidate_price)),

        # "скидка" относительно BASE_PRICE
        "DISCOUNT_DEPTH": float(dd),
        "PRICE_CHANGE_LOG": float(pcl),
        "PROMO_DEPTH": float(promo_depth),

        # контекст бакета
        "PROMO_SHARE": promo_share,
        "BUCKET_IDX": float({"MON_THU": 0, "FRI": 1, "SAT_SUN": 2}[bucket]),
        "SIN_WOY": sin_woy,
        "COS_WOY": cos_woy,
        "SIN_MONTH": sin_month,
        "COS_MONTH": cos_month,
        "TREND_W": trend_w,
        "HOLIDAY_FLAG": holiday_flag,

        # запасы и OOS
        "LOW_STOCK_FLAG": float(base_row.get("LOW_STOCK_FLAG", 0.0)),
        "STOCK_END_L1": float(base_row.get("STOCK_END_L1", 0.0)),
        "OOS_L1": float(base_row.get("OOS_L1", 0.0)),

        # лаги
        "QTY_L1": float(base_row.get("QTY_L1", 0.0)),
        "PROMO_L1": float(base_row.get("PROMO_L1", 0.0)),
    }


def predict_backoff_dual(base_row: pd.Series,
                         candidate_price: float,
                         base_price: float,
                         bucket: str,
                         week_monday: pd.Timestamp,
                         promo_share: float,
                         feature_names: List[str],
                         hier: HierDualModels) -> Tuple[Optional[float], Optional[float], str, int]:
    """
    Прогноз спроса mu для кандидата + оценка эластичности beta_raw
    с иерархическим backoff (PRODUCT_STORE -> FAMILY -> CATEGORY -> SEGMENT -> GLOBAL).
    """
    feats = make_features_for_candidate(base_row, candidate_price, base_price, bucket, week_monday)
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
    """Локальная эластичность вокруг p: d log(mu) / d log(p)."""
    p_lo = p * (1.0 - delta)
    p_hi = p * (1.0 + delta)
    num = np.log(max(mu_hi, 1e-9)) - np.log(max(mu_lo, 1e-9))
    den = np.log(max(p_hi, 1e-9)) - np.log(max(p_lo, 1e-9))
    return float(num / den) if den != 0 else np.nan


def elasticity_used(beta_raw: Optional[float], beta_floor: float = BETA_FLOOR) -> float:
    """
    Guardrail:
      - если beta_raw None/NaN или beta_raw > -beta_floor => используем -beta_floor
    """
    if beta_raw is None or (not np.isfinite(beta_raw)):
        return -float(beta_floor)
    if beta_raw > -float(beta_floor):
        return -float(beta_floor)
    return float(beta_raw)


# =========================
# PRICE GRID (price ladder)
# =========================

def collect_observed_price_ladder_by_family(bucket_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Наблюдавшиеся фактические цены продажи (PRICE) по FAMILY (чтобы snap к ladder)."""
    ladders: Dict[str, np.ndarray] = {}
    for fam, g in bucket_df.groupby("FAMILY_CODE"):
        prices = pd.to_numeric(g["PRICE"], errors="coerce").astype(float).values
        prices = prices[np.isfinite(prices) & (prices > 0)]
        if len(prices) > 0:
            ladders[str(fam)] = np.unique(np.round(prices, 4))
    return ladders


def build_price_grid(base_price: float, observed_prices: Optional[np.ndarray], n_mult: int = 15) -> np.ndarray:
    """
    Сетка цен вокруг BASE_PRICE с ограничениями MIN_MULT..MAX_MULT.
    observed_prices (price ladder) помогает не предлагать "неестественные" цены.
    """
    base_price = float(base_price)
    if not np.isfinite(base_price) or base_price <= 0:
        return np.array([], dtype=float)

    lo = base_price * MIN_MULT
    hi = base_price * MAX_MULT
    grid = base_price * np.linspace(MIN_MULT, MAX_MULT, n_mult)

    if observed_prices is not None and len(observed_prices) > 0:
        ladder = np.unique(observed_prices[np.isfinite(observed_prices) & (observed_prices > 0)])
        ladder = ladder[(ladder >= lo) & (ladder <= hi)]
        if len(ladder) > 0:
            grid = np.array([ladder[np.argmin(np.abs(ladder - p))] for p in grid], dtype=float)

    grid = grid[(grid >= lo) & (grid <= hi)]
    return np.sort(np.unique(np.round(grid, 4)))


# =========================
# SINGLE optimization
# =========================

def optimize_single(base_row: pd.Series,
                    bucket: str,
                    week_monday: pd.Timestamp,
                    promo_share: float,
                    feature_names: List[str],
                    hier: HierDualModels,
                    grid: np.ndarray,
                    base_price: float,
                    stock_cap_end: Optional[float]) -> Optional[dict]:
    """
    Подбор цены по grid для одного SKU (без учёта каннибализации).
    """
    grid, promo_cap_price = apply_promo_cap_to_grid(grid, base_price, promo_share)
    if grid is None or len(grid) == 0:
        return None

    qty_scale = float(max(base_row.get("QTY_L1", 0.0), 1.0))

    best = None
    meta = None

    for p in grid:
        mu, beta_raw, used_level, pos_sales_used = predict_backoff_dual(
            base_row, float(p), float(base_price), bucket, week_monday, promo_share, feature_names, hier
        )
        if mu is None:
            continue

        # Cap спроса остатками на конец бакета (если доступно)
        if stock_cap_end is not None and np.isfinite(stock_cap_end):
            mu = min(mu, float(stock_cap_end))

        beta_used = elasticity_used(beta_raw, beta_floor=BETA_FLOOR)

        # Лёгкая калибровка наклона около BASE_PRICE (чтобы "плоская" оценка не уводила в максимум)
        mu_adj = mu
        if beta_raw is not None and np.isfinite(beta_raw):
            rel_log = _safe_log(float(p)) - _safe_log(float(base_price))
            mu_adj = float(mu) * float(np.exp((beta_used - beta_raw) * rel_log))
            mu_adj = max(mu_adj, 0.0)
            if stock_cap_end is not None and np.isfinite(stock_cap_end):
                mu_adj = min(mu_adj, float(stock_cap_end))

        score = objective(float(p), float(mu_adj), float(base_price), qty_scale)
        rev = float(p) * float(mu_adj)

        if best is None or score > best[0]:
            best = (score, float(p), float(mu_adj), float(rev))
            meta = (beta_raw, beta_used, used_level, pos_sales_used, promo_cap_price)

    if best is None:
        return None

    _, p_star, mu_star, rev_star = best
    beta_raw, beta_used, used_level, pos_sales_used, promo_cap_price = meta

    # local elasticity around chosen price
    p_lo = p_star * (1.0 - LOCAL_ELASTICITY_DELTA)
    p_hi = p_star * (1.0 + LOCAL_ELASTICITY_DELTA)

    mu_lo, _, _, _ = predict_backoff_dual(base_row, p_lo, base_price, bucket, week_monday, promo_share, feature_names, hier)
    mu_hi, _, _, _ = predict_backoff_dual(base_row, p_hi, base_price, bucket, week_monday, promo_share, feature_names, hier)
    mu_lo = float(mu_lo) if mu_lo is not None else 0.0
    mu_hi = float(mu_hi) if mu_hi is not None else 0.0

    if stock_cap_end is not None and np.isfinite(stock_cap_end):
        mu_lo = min(mu_lo, float(stock_cap_end))
        mu_hi = min(mu_hi, float(stock_cap_end))

    e_loc = local_elasticity(p_star, mu_lo, mu_hi, LOCAL_ELASTICITY_DELTA)
    quality = "OK" if (pos_sales_used >= MIN_POS_SALES_FOR_ELAST and base_price > 0) else "LOW_DATA"

    dd = discount_depth(p_star, base_price)
    pcl = price_change_log(p_star, base_price)
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
# CONTEXT helpers
# =========================

def get_last_bucket_prices_and_stock(bucket_df: pd.DataFrame,
                                     sku: str,
                                     store: str,
                                     bucket: str,
                                     cutoff: pd.Timestamp) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Возвращает (last_sale_bucket_price, last_base_bucket_price, last_stock_end) до cutoff в нужном bucket.
    - sale bucket price = PRICE (агрегированная фактическая цена продажи)
    - base bucket price = BASE_PRICE (регулярная)
    """
    df_b = bucket_df.loc[
        (bucket_df["PRODUCT_CODE"] == sku) &
        (bucket_df["STORE"] == store) &
        (bucket_df["BUCKET"] == bucket) &
        (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff))
    ].sort_values("BUCKET_END")

    if len(df_b) == 0:
        # fallback: любой bucket (чтобы не терять полностью)
        df_any = bucket_df.loc[
            (bucket_df["PRODUCT_CODE"] == sku) &
            (bucket_df["STORE"] == store) &
            (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff))
        ].sort_values("BUCKET_END")
        if len(df_any) == 0:
            return None, None, None
        sp = float(df_any["PRICE"].iloc[-1]) if np.isfinite(df_any["PRICE"].iloc[-1]) else None
        bp = float(df_any["BASE_PRICE"].iloc[-1]) if np.isfinite(df_any["BASE_PRICE"].iloc[-1]) else None
        st = float(df_any["STOCK_END"].iloc[-1]) if np.isfinite(df_any["STOCK_END"].iloc[-1]) else None
        return sp, bp, st

    sp = float(df_b["PRICE"].iloc[-1]) if np.isfinite(df_b["PRICE"].iloc[-1]) else None
    bp = float(df_b["BASE_PRICE"].iloc[-1]) if np.isfinite(df_b["BASE_PRICE"].iloc[-1]) else None
    st = float(df_b["STOCK_END"].iloc[-1]) if np.isfinite(df_b["STOCK_END"].iloc[-1]) else None
    return sp, bp, st


def build_context(bucket_df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Берём последнюю доступную строку по SKU+STORE до cutoff (как "текущий контекст").
    """
    train = bucket_df.loc[bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff)].copy()
    recent = train.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(12)
    return recent.sort_values("BUCKET_END").groupby(["PRODUCT_CODE", "STORE"]).tail(1).copy()


# =========================
# CROSS-FAMILY (cannibalization)
# =========================

@dataclass
class CrossFamilyModel:
    family_code: str
    store: str
    bucket: str
    family_products: List[str]               # порядок признаков (SKU list)
    scaler: StandardScaler
    models_by_product: Dict[str, PoissonRegressor]  # регрессия для каждого target SKU


def build_family_topk_for_context(bucket_df: pd.DataFrame,
                                  cutoff: pd.Timestamp,
                                  context_keys: Set[Tuple[str, str, str]]) -> Dict[Tuple[str, str, str], List[str]]:
    """
    Для каждого (FAMILY, STORE, BUCKET) берём топ-K SKU по продажам в lookback.
    """
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
    """
    Wide panel на уровне недель:
      Q_sku, LOGP_sku для каждого sku в top-K.
    """
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
        if pcol in wide.columns:
            wide[lcol] = wide[pcol].apply(lambda v: _safe_log(v) if np.isfinite(v) and v > 0 else np.nan)
        else:
            wide[lcol] = np.nan

    log_cols = [f"LOGP_{s}" for s in skus]
    wide[log_cols] = wide[log_cols].ffill()

    return wide


def train_cross_family_models(bucket_df: pd.DataFrame,
                              cutoff: pd.Timestamp,
                              topk_map: Dict[Tuple[str, str, str], List[str]]) -> Dict[Tuple[str, str, str], CrossFamilyModel]:
    """
    Обучаем cross-family Poisson:
      mu_target = exp( a + sum_j beta_j * LOGP_j )
    """
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

            y = np.clip(pd.to_numeric(data[y_col], errors="coerce").fillna(0.0).astype(float).values, 0.0, None)
            if (y > 0).sum() < 5:
                continue

            Xs = scaler.transform(X_all.loc[data.index].fillna(0.0).values)

            reg = PoissonRegressor(alpha=max(POISSON_ALPHA, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
            reg.fit(Xs, y)

            models_by_product[str(sku)] = reg

        if len(models_by_product) >= 2:
            out[(str(fam), str(store), str(bucket))] = CrossFamilyModel(str(fam), str(store), str(bucket), skus, scaler, models_by_product)

    return out


def predict_family_demands(cfm: CrossFamilyModel,
                           prices: Dict[str, float],
                           default_price: Optional[float] = None) -> Dict[str, float]:
    """
    FIX:
      - если какого-то sku нет в prices dict, подставляем default_price (иначе KeyError)
      - ограничиваем eta перед exp(), чтобы не было overflow

    Это нужно, потому что оптимизируем только active subset, но модель хочет вектор цен по всем family_products.
    """
    if default_price is None:
        default_price = float(np.median(list(prices.values()))) if prices else 1.0

    X_raw = np.array([_safe_log(prices.get(sku, default_price)) for sku in cfm.family_products], dtype=float).reshape(1, -1)
    Xs = cfm.scaler.transform(X_raw)

    out: Dict[str, float] = {}
    for sku, reg in cfm.models_by_product.items():
        eta = float(reg._linear_predictor(Xs)[0])  # type: ignore[attr-defined]
        eta = float(np.clip(eta, -MAX_LINEAR_PRED, MAX_LINEAR_PRED))
        out[sku] = float(np.exp(eta))

    return out


def cross_coef_raw(cfm: CrossFamilyModel, target_sku: str, price_sku: str) -> Optional[float]:
    """
    Cross-эластичность:
      d log(mu_target) / d log(price_price_sku)

    Учитываем StandardScaler:
      beta_raw = beta_scaled / scale_j
    """
    target_sku = str(target_sku)
    price_sku = str(price_sku)

    reg = cfm.models_by_product.get(target_sku)
    if reg is None:
        return None

    if price_sku not in cfm.family_products:
        return None
    j = cfm.family_products.index(price_sku)

    scale = getattr(cfm.scaler, "scale_", None)
    if scale is None or j >= len(scale) or float(scale[j]) <= 0:
        return None

    coef = getattr(reg, "coef_", None)
    if coef is None or j >= len(coef):
        return None

    return float(coef[j]) / float(scale[j])


def cross_own_elasticity_raw(cfm: CrossFamilyModel, sku: str) -> Optional[float]:
    """Own-эластичность в cross-family: d log(mu_sku)/d log(price_sku)."""
    return cross_coef_raw(cfm, target_sku=sku, price_sku=sku)


# =========================
# MODES
# =========================

def buckets_for_mode(mode: str) -> List[str]:
    mode = mode.lower().strip()
    if mode == "monday":
        return ["MON_THU"]
    if mode == "friday":
        return ["FRI", "SAT_SUN"]
    raise ValueError("mode must be 'monday' or 'friday'")


# =========================
# MAIN: calculate prices for given week start
# =========================

def calculate_for_week(bucket_df: pd.DataFrame,
                       daily_df: pd.DataFrame,
                       as_of_date: pd.Timestamp,
                       week_monday: pd.Timestamp,
                       mode: str) -> pd.DataFrame:
    """
    Считает цены для указанной недели week_monday и режима запуска.
    В output возвращает также последние фактические SALE_PRICE и BASE_PRICE (где возможно).

    Важно:
      - применяет фильтр "active sold" по daily_df до cutoff
      - в промо применяет hard cap: CHOSEN_PRICE <= BASE_PRICE (регулярная)
      - учитывает сезонность и HOLIDAY_FLAG как признаки
      - считает OWN_ELASTICITY для single и cross
    """
    today = pd.Timestamp(as_of_date).normalize()
    promo_map = build_promo_map(daily_df)

    buckets = buckets_for_mode(mode)

    # Cutoffs per bucket
    cutoffs: Dict[str, pd.Timestamp] = {}
    for b in buckets:
        rule = decision_cutoff_for_bucket_by_mode(week_monday, b, mode).normalize()
        cutoffs[b] = min(rule, today, HISTORY_END.normalize())

    max_cutoff = max(cutoffs.values())

    # Данные для обучения (до max_cutoff)
    train_all = bucket_df.loc[bucket_df["BUCKET_END"] <= max_cutoff].copy()
    if len(train_all) == 0:
        return pd.DataFrame()

    # Price ladder по семье (по фактическим PRICE)
    fam_ladder = collect_observed_price_ladder_by_family(train_all)

    # Feature set (должны существовать в train и scoring)
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
        "SIN_MONTH",
        "COS_MONTH",
        "TREND_W",
        "HOLIDAY_FLAG",
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

        # Активные (SKU, STORE) по продажам в окне до cutoff
        active_set = build_active_sku_store_set(daily_df, cutoff=cutoff,
                                                lookback_days=ACTIVE_SOLD_LOOKBACK_DAYS,
                                                min_qty=ACTIVE_MIN_QTY)

        ctx = build_context(bucket_df, cutoff=cutoff)
        if len(ctx) == 0:
            continue

        # Фильтр активных товаров
        ctx = ctx.loc[[(str(p), str(s)) in active_set for p, s in zip(ctx["PRODUCT_CODE"], ctx["STORE"])]].copy()
        if len(ctx) == 0:
            continue

        # ---- Cross-family models ----
        context_keys = set((str(f), str(s), bucket) for f, s in zip(ctx["FAMILY_CODE"].astype(str), ctx["STORE"].astype(str)))
        topk_map = build_family_topk_for_context(train_all, cutoff=cutoff, context_keys=context_keys)
        cross_models = train_cross_family_models(train_all, cutoff=cutoff, topk_map=topk_map)

        used_in_cross: Set[Tuple[str, str]] = set()

        # ---- CROSS optimization per family+store ----
        for (fam, store), gfam in ctx.groupby(["FAMILY_CODE", "STORE"]):
            key = (str(fam), str(store), str(bucket))
            cfm = cross_models.get(key)
            if cfm is None:
                continue

            # Готовим данные по SKU семьи: promo_share, base_price (regular), grid, stock cap
            base_prices: Dict[str, float] = {}           # BASE_PRICE (regular)
            base_sources: Dict[str, str] = {}
            last_sale_bucket: Dict[str, Optional[float]] = {}
            stock_caps: Dict[str, Optional[float]] = {}
            promo_shares: Dict[str, float] = {}
            grids: Dict[str, np.ndarray] = {}
            promo_caps: Dict[str, Optional[float]] = {}

            # Также — последние дневные цены (для output)
            last_sale_daily: Dict[str, Optional[float]] = {}
            last_base_daily: Dict[str, Optional[float]] = {}
            last_sale_daily_np: Dict[str, Optional[float]] = {}
            last_base_daily_np: Dict[str, Optional[float]] = {}

            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                store_s = str(store)

                # PROMO_SHARE по PROMO_PERIOD (если есть) на интервале бакета
                ranges = promo_map.get((sku, store_s), [])
                ps = promo_share_in_range(ranges, start, end) if ranges else 0.0

                # Последние bucket цены/остатки в этом bucket
                last_sp_bucket, last_bp_bucket, stock_end = get_last_bucket_prices_and_stock(train_all, sku, store_s, bucket, cutoff=cutoff)

                # Регулярная цена для якоря и cap
                reg_bp, reg_src = compute_regular_base_price(train_all, sku, store_s, bucket, cutoff=cutoff,
                                                             base_price_last=last_bp_bucket,
                                                             median_weeks=REG_PRICE_MEDIAN_WEEKS,
                                                             exclude_promo_prices=EXCLUDE_PROMO_PRICES_FROM_REG_PRICE)
                if not is_valid_price(reg_bp):
                    continue

                ladder = fam_ladder.get(str(fam))
                grid = build_price_grid(float(reg_bp), observed_prices=ladder)

                # HARD CAP в промо: chosen_price <= BASE_PRICE (regular)
                grid2, cap_price = apply_promo_cap_to_grid(grid, reg_bp, ps)

                if grid2 is None or len(grid2) == 0:
                    continue

                base_prices[sku] = float(reg_bp)
                base_sources[sku] = reg_src
                last_sale_bucket[sku] = last_sp_bucket
                stock_caps[sku] = stock_end
                promo_shares[sku] = float(ps)
                grids[sku] = grid2
                promo_caps[sku] = cap_price

                # Последние дневные цены до cutoff (для output)
                sp_d, bp_d = get_last_daily_prices(daily_df, sku, store_s, cutoff=cutoff, exclude_is_promo=False)
                sp_np, bp_np = get_last_daily_prices(daily_df, sku, store_s, cutoff=cutoff, exclude_is_promo=True)
                last_sale_daily[sku] = sp_d
                last_base_daily[sku] = bp_d
                last_sale_daily_np[sku] = sp_np
                last_base_daily_np[sku] = bp_np

            # Активные для cross (нужно: есть model + есть base_price + grid)
            active = [sku for sku in cfm.family_products
                      if sku in base_prices and sku in cfm.models_by_product and (sku in grids and len(grids[sku]) > 0)]
            if len(active) < 2:
                continue

            # Для отсутствующих SKU в prices — fallback (чтобы predict_family_demands не падал)
            prices = {sku: float(base_prices[sku]) for sku in active}
            default_price = float(np.median(list(prices.values()))) if prices else 1.0

            # Итеративная оптимизация (coordinate descent) с учётом каннибализации
            for _ in range(CROSS_ITERS):
                for sku in active:
                    grid = grids.get(sku)
                    if grid is None or len(grid) == 0:
                        continue
                    best = None
                    for p in grid:
                        tmp = dict(prices)
                        tmp[sku] = float(p)

                        mu = float(predict_family_demands(cfm, tmp, default_price=default_price).get(sku, 0.0))

                        cap = stock_caps.get(sku)
                        if cap is not None and np.isfinite(cap):
                            mu = min(mu, float(cap))

                        score = objective(float(p), mu, base_prices[sku], qty_scale=1.0)
                        if best is None or score > best[0]:
                            best = (score, float(p), mu)
                    if best is not None:
                        prices[sku] = best[1]

            mu_all = predict_family_demands(cfm, prices, default_price=default_price)

            # Записываем output по SKU, которые реально оптимизировались
            for _, rr in gfam.iterrows():
                sku = str(rr["PRODUCT_CODE"])
                if sku not in prices:
                    continue

                bp = base_prices.get(sku)                # regular
                ps = promo_shares.get(sku, 0.0)
                cap_p = promo_caps.get(sku)
                stock_end = stock_caps.get(sku)
                sp_bucket = last_sale_bucket.get(sku)

                chosen_price = float(prices[sku])
                mu = float(mu_all.get(sku, 0.0))
                if stock_end is not None and np.isfinite(stock_end):
                    mu = min(mu, float(stock_end))

                # Эластичности cross
                beta_raw = cross_own_elasticity_raw(cfm, sku)
                beta_used = elasticity_used(beta_raw, beta_floor=BETA_FLOOR)

                # Local elasticity (±delta) при прочих ценах семьи фиксированных
                p = chosen_price
                p_lo = p * (1.0 - LOCAL_ELASTICITY_DELTA)
                p_hi = p * (1.0 + LOCAL_ELASTICITY_DELTA)
                tmp_lo = dict(prices); tmp_lo[sku] = p_lo
                tmp_hi = dict(prices); tmp_hi[sku] = p_hi

                mu_lo = float(predict_family_demands(cfm, tmp_lo, default_price=default_price).get(sku, 0.0))
                mu_hi = float(predict_family_demands(cfm, tmp_hi, default_price=default_price).get(sku, 0.0))
                if stock_end is not None and np.isfinite(stock_end):
                    mu_lo = min(mu_lo, float(stock_end))
                    mu_hi = min(mu_hi, float(stock_end))

                e_loc = local_elasticity(p, mu_lo, mu_hi, LOCAL_ELASTICITY_DELTA)

                out_rows.append({
                    # keys
                    "PRODUCT_CODE": sku,
                    "STORE": str(store),
                    "FAMILY_CODE": str(fam),

                    # timing
                    "AS_OF_DATE": today,
                    "TARGET_WEEK": pd.Timestamp(week_monday).normalize(),
                    "TARGET_BUCKET": bucket,
                    "BUCKET_START": start,
                    "BUCKET_END": end,

                    # cutoff
                    "DECISION_CUTOFF_RULE": cutoff_rule,
                    "CUTOFF_EFFECTIVE": cutoff,

                    # promo/stock
                    "PROMO_SHARE": ps,
                    "STOCK_END_BUCKET_CAP": stock_end,

                    # last observed prices (bucket + daily)
                    "LAST_SALE_PRICE": last_sale_daily.get(sku),
                    "LAST_BASE_PRICE": last_base_daily.get(sku),
                    "LAST_SALE_PRICE_NONPROMO": last_sale_daily_np.get(sku),
                    "LAST_BASE_PRICE_NONPROMO": last_base_daily_np.get(sku),

                    "LAST_BUCKET_SALE_PRICE": sp_bucket,     # PRICE bucket
                    "BASE_PRICE": bp,                         # regular anchor used
                    "BASE_PRICE_SOURCE": base_sources.get(sku, "FALLBACK_NONE"),

                    # promo cap
                    "PROMO_CAP_PRICE": cap_p,
                    "PROMO_PRICE_CAPPED": bool(ps > PROMO_THRESHOLD and is_valid_price(cap_p)),

                    # decision
                    "CHOSEN_PRICE": chosen_price,
                    "PRICE_MULT": chosen_price / max(float(bp), 1e-6) if bp else np.nan,

                    # forecast
                    "PRED_QTY": mu,
                    "PRED_REV": chosen_price * mu,

                    # model type
                    "USED_CROSS_PRICE_MODEL": True,

                    # elasticities
                    "OWN_ELASTICITY_RAW": beta_raw,
                    "OWN_ELASTICITY_USED": beta_used,
                    "LOCAL_ELASTICITY": e_loc,
                    "ELASTICITY_QUALITY": "OK" if np.isfinite(e_loc) else "LOW_DATA",

                    # diagnostics
                    "USED_LEVEL": "CROSS_FAMILY",
                    "POS_SALES_USED": None,

                    # price vs base
                    "DISCOUNT_DEPTH": discount_depth(chosen_price, float(bp)) if bp else 0.0,
                    "PRICE_CHANGE_LOG": price_change_log(chosen_price, float(bp)) if bp else 0.0,
                    "PROMO_DEPTH": discount_depth(chosen_price, float(bp)) * float(ps) if bp else 0.0,
                })

                used_in_cross.add((sku, str(store)))

        # ---- SINGLE fallback for SKUs not optimized by cross ----
        for _, r in ctx.iterrows():
            sku = str(r["PRODUCT_CODE"])
            store = str(r["STORE"])
            fam = str(r["FAMILY_CODE"])

            if (sku, store) in used_in_cross:
                continue

            ranges = promo_map.get((sku, store), [])
            promo_share = promo_share_in_range(ranges, start, end) if ranges else 0.0

            last_sp_bucket, last_bp_bucket, stock_end = get_last_bucket_prices_and_stock(train_all, sku, store, bucket, cutoff=cutoff)

            base_price, base_src = compute_regular_base_price(train_all, sku, store, bucket, cutoff=cutoff,
                                                              base_price_last=last_bp_bucket,
                                                              median_weeks=REG_PRICE_MEDIAN_WEEKS,
                                                              exclude_promo_prices=EXCLUDE_PROMO_PRICES_FROM_REG_PRICE)
            if not is_valid_price(base_price):
                continue

            # Last daily prices for output
            sp_d, bp_d = get_last_daily_prices(daily_df, sku, store, cutoff=cutoff, exclude_is_promo=False)
            sp_np, bp_np = get_last_daily_prices(daily_df, sku, store, cutoff=cutoff, exclude_is_promo=True)

            # grid around BASE_PRICE
            ladder = fam_ladder.get(fam)
            grid = build_price_grid(float(base_price), observed_prices=ladder)

            base_row = r.copy()
            base_row["PROMO_SHARE"] = promo_share

            res = optimize_single(
                base_row=base_row,
                bucket=bucket,
                week_monday=week_monday,
                promo_share=promo_share,
                feature_names=feature_names,
                hier=hier,
                grid=grid,
                base_price=float(base_price),
                stock_cap_end=stock_end
            )
            if res is None:
                continue

            out_rows.append({
                "PRODUCT_CODE": sku,
                "STORE": store,
                "FAMILY_CODE": fam,

                "AS_OF_DATE": today,
                "TARGET_WEEK": pd.Timestamp(week_monday).normalize(),
                "TARGET_BUCKET": bucket,
                "BUCKET_START": start,
                "BUCKET_END": end,

                "DECISION_CUTOFF_RULE": cutoff_rule,
                "CUTOFF_EFFECTIVE": cutoff,

                "PROMO_SHARE": promo_share,
                "STOCK_END_BUCKET_CAP": stock_end,

                # last observed prices (daily + bucket)
                "LAST_SALE_PRICE": sp_d,
                "LAST_BASE_PRICE": bp_d,
                "LAST_SALE_PRICE_NONPROMO": sp_np,
                "LAST_BASE_PRICE_NONPROMO": bp_np,

                "LAST_BUCKET_SALE_PRICE": last_sp_bucket,
                "BASE_PRICE": float(base_price),
                "BASE_PRICE_SOURCE": base_src,

                "PROMO_CAP_PRICE": res.get("PROMO_CAP_PRICE"),
                "PROMO_PRICE_CAPPED": bool(promo_share > PROMO_THRESHOLD and is_valid_price(res.get("PROMO_CAP_PRICE"))),

                "CHOSEN_PRICE": float(res["CHOSEN_PRICE"]),
                "PRICE_MULT": float(res["CHOSEN_PRICE"]) / max(float(base_price), 1e-6),

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


# =========================
# NEW runner: specify week start, no weeks_back
# =========================

def run_for_week_start(bucket_df: pd.DataFrame,
                       daily_df: pd.DataFrame,
                       as_of_date: pd.Timestamp,
                       mode: str,
                       week_monday: pd.Timestamp,
                       active_sold_lookback_days: int = ACTIVE_SOLD_LOOKBACK_DAYS,
                       active_min_qty: float = ACTIVE_MIN_QTY) -> pd.DataFrame:
    """
    Запуск "как будто" в as_of_date для недели week_monday.
    Режимы:
      - monday: MON_THU
      - friday: FRI + SAT_SUN

    Фильтр активных товаров настраивается аргументами.
    """
    global ACTIVE_SOLD_LOOKBACK_DAYS, ACTIVE_MIN_QTY
    ACTIVE_SOLD_LOOKBACK_DAYS = int(active_sold_lookback_days)
    ACTIVE_MIN_QTY = float(active_min_qty)

    df = calculate_for_week(bucket_df=bucket_df,
                            daily_df=daily_df,
                            as_of_date=pd.Timestamp(as_of_date),
                            week_monday=pd.Timestamp(week_monday),
                            mode=mode)
    df["RUN_MODE"] = mode
    return df


# =========================
# PIPELINE
# =========================

def run_pipeline(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Полный пайплайн подготовки:
      raw daily -> preprocess -> aggregate to buckets -> add lags
    Возвращает:
      - bucket_df: данные по бакетам (для обучения/контекста)
      - daily_df: дневные данные (для active filter, last prices, promo map)
    """
    daily = preprocess_raw(df_raw, history_end=HISTORY_END)
    buckets = aggregate_to_buckets(daily)
    buckets = add_lag_features(buckets, lags=1)
    return buckets, daily