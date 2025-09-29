# ============================================
# Dynamic Pricing — Robust Start (full script)
# ============================================
# Requirements: pandas, numpy, scikit-learn
import pandas as pd
import numpy as np
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# 0) Constants & helpers
# ----------------------------
BUCKET_MAP = {  # 0=Mon ... 6=Sun
    0: "MonThu", 1: "MonThu", 2: "MonThu", 3: "MonThu",
    4: "Fri",
    5: "SatSun", 6: "SatSun"
}

def to_week_start(d: pd.Timestamp) -> pd.Timestamp:
    return d - pd.Timedelta(days=d.weekday())  # Monday

def parse_promo_period(s: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    PROMO_PERIOD like '01-01-2024 - 03-01-2024' (day-first) or None/NaN.
    Returns (start, end) where either can be NaT.
    """
    if not isinstance(s, str) or " - " not in s:
        return (pd.NaT, pd.NaT)
    a, b = [x.strip() for x in s.split(" - ", 1)]
    start = pd.to_datetime(a, dayfirst=True, errors="coerce")
    end   = pd.to_datetime(b, dayfirst=True, errors="coerce")
    return (start, end)

def safe_div(num, den):
    den = np.where(pd.isna(den) | (den == 0), np.nan, den)
    return np.divide(num, den)

def _is_promo_for_week(row: pd.Series, week_start: pd.Timestamp) -> int:
    ps, pe = row.get("promo_start", pd.NaT), row.get("promo_end", pd.NaT)
    if pd.isna(ps) or pd.isna(pe):
        return 0
    return int(ps <= week_start <= pe)

# ----------------------------
# 1) Aggregate to week × bucket
# ----------------------------
def make_week_bucket_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates & buckets
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"], errors="coerce")
    df = df[df["TRADE_DT"].notna()]
    df["week_start"] = df["TRADE_DT"].apply(to_week_start)
    df["bucket"] = df["TRADE_DT"].dt.weekday.map(BUCKET_MAP)

    # Robust numeric casting
    num_cols = ["SALE_QTY","SALE_QTY_ONLINE","SALE_PRICE","BASE_PRICE",
                "START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Targets
    df["SALE_QTY"] = df["SALE_QTY"].fillna(0.0)
    df["SALE_QTY_ONLINE"] = df["SALE_QTY_ONLINE"].fillna(0.0)
    df["q_total"] = df["SALE_QTY"] + df["SALE_QTY_ONLINE"]

    # PROMO_PERIOD parsing (NaN-safe)
    if "PROMO_PERIOD" in df:
        starts, ends = zip(*df["PROMO_PERIOD"].apply(parse_promo_period))
    else:
        starts, ends = [], []
    df["promo_start"] = list(starts) if len(starts) else pd.NaT
    df["promo_end"]   = list(ends)   if len(ends)   else pd.NaT

    # Stock/flows (keep NaNs → later impute in pipeline)
    for col in ["START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY"]:
        if col in df:
            df[col] = df[col].fillna(0.0)

    grp_cols = ["STORE","PRODUCT_CODE","bucket","week_start",
                "FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE",
                "STORE_TYPE","REGION_NAME","PLACE_TYPE"]

    agg = (df.groupby(grp_cols, dropna=False)
             .agg(
                 q_total=("q_total","sum"),
                 price=("SALE_PRICE","mean"),          # realized average price in that bucket-week
                 base_price=("BASE_PRICE","mean"),
                 is_promo=("IS_PROMO","max"),
                 start_stock=("START_STOCK","sum"),
                 end_stock=("END_STOCK","sum"),
                 delivery_qty=("DELIVERY_QTY","sum"),
                 loss_qty=("LOSS_QTY","sum"),
                 return_qty=("RETURN_QTY","sum"),
                 promo_start=("promo_start","min"),
                 promo_end=("promo_end","max")
             )
             .reset_index())

    # Derived
    agg["promo_depth"] = safe_div(agg["base_price"] - agg["price"], agg["base_price"])
    agg["promo_depth"] = agg["promo_depth"].fillna(0.0)

    agg["oos_flag"] = (agg["start_stock"] <= 0) | \
        ((agg["start_stock"] + agg["delivery_qty"] - agg["end_stock"] - agg["loss_qty"] - agg["return_qty"]) <= 0)

    agg["promo_active_now"] = (agg["promo_start"].notna()) & (agg["promo_start"] <= agg["week_start"]) & \
                              (agg["week_start"] <= agg["promo_end"].fillna(pd.Timestamp.max))

    # If 'is_promo' was non-numeric, coerce now
    if "is_promo" in agg:
        agg["is_promo"] = pd.to_numeric(agg["is_promo"], errors="coerce").fillna(0.0)

    return agg

# ----------------------------
# 2) Lag/rolling features (no leakage)
# ----------------------------
def add_time_features(panel: pd.DataFrame, max_lag_weeks: int = 4) -> pd.DataFrame:
    panel = panel.sort_values(["STORE","PRODUCT_CODE","bucket","week_start"]).copy()
    keys = ["STORE","PRODUCT_CODE","bucket"]

    def add_grp_features(g):
        for L in range(1, max_lag_weeks + 1):
            g[f"q_lag_{L}"] = g["q_total"].shift(L)
            g[f"price_lag_{L}"] = g["price"].shift(L)
            g[f"promo_lag_{L}"] = g["is_promo"].shift(L)
        g["q_ma_4"] = g["q_total"].rolling(4, min_periods=1).mean().shift(1)
        g["price_ma_4"] = g["price"].rolling(4, min_periods=1).mean().shift(1)
        return g

    panel = panel.groupby(keys, group_keys=False).apply(add_grp_features)

    # Fill only the numeric new columns (avoid touching categoricals)
    num_new = [f"q_lag_{i}" for i in range(1,5)] + [f"price_lag_{i}" for i in range(1,5)] + \
              [f"promo_lag_{i}" for i in range(1,5)] + ["q_ma_4","price_ma_4"]
    for c in num_new:
        if c in panel:
            panel[c] = panel[c].fillna(0.0)

    return panel

# ----------------------------
# 3) Model (hierarchical pooled ridge)
# ----------------------------
CAT_FEATS = ["STORE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE",
             "STORE_TYPE","REGION_NAME","PLACE_TYPE","PRODUCT_CODE"]

NUM_FEATS_BASE = ["price","base_price","promo_depth","is_promo",
                  "q_ma_4","price_ma_4","start_stock","end_stock",
                  "delivery_qty","loss_qty","return_qty","oos_flag"] + \
                 [f"q_lag_{i}" for i in range(1,5)] + \
                 [f"price_lag_{i}" for i in range(1,5)] + \
                 [f"promo_lag_{i}" for i in range(1,5)]

def make_bucket_model(alpha: float = 5.0) -> Pipeline:
    cat_pipe = Pipeline(steps=[
        ("to_str", FunctionTransformer(lambda X: X.astype(str))),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
    ])
    num_pipe = Pipeline(steps=[
        ("to_num", FunctionTransformer(lambda X: X.apply(pd.to_numeric, errors="coerce"))),
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0))
    ])
    ct = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_FEATS),
            ("num", num_pipe, NUM_FEATS_BASE),
        ],
        sparse_threshold=0.3
    )
    return Pipeline(steps=[("prep", ct), ("ridge", Ridge(alpha=alpha, fit_intercept=True, random_state=0))])

@dataclass
class FittedBucket:
    model: Pipeline
    feature_cols: List[str]

def fit_per_bucket_models(train_panel: pd.DataFrame, alpha: float = 5.0) -> Dict[str, FittedBucket]:
    models = {}
    for b in ["MonThu","Fri","SatSun"]:
        tr = train_panel[train_panel["bucket"] == b].copy()
        if tr.empty:
            continue
        X_cols = CAT_FEATS + NUM_FEATS_BASE
        y = np.log1p(tr["q_total"].clip(lower=0))
        m = make_bucket_model(alpha)
        m.fit(tr[X_cols], y)
        models[b] = FittedBucket(model=m, feature_cols=X_cols)
    return models

def predict_qty(models: Dict[str, FittedBucket], df_rows: pd.DataFrame) -> np.ndarray:
    b = df_rows["bucket"].iloc[0]
    fb = models.get(b)
    if fb is None:
        return np.zeros(len(df_rows))
    yhat = fb.model.predict(df_rows[fb.feature_cols])
    return np.expm1(yhat).clip(min=0)

# ----------------------------
# 4) Price ladder & optimizer
# ----------------------------
def make_price_ladder(base_price: float, last_price: float) -> List[float]:
    candidates = []
    anchors = [x for x in [base_price, last_price] if np.isfinite(x) and x > 0]
    if not anchors:
        anchors = [max(0.01, last_price if np.isfinite(last_price) else 0.01)]
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3]
    for a in anchors:
        for m in multipliers:
            candidates.append(max(0.01, round(a * m, 2)))
    # .99 endings
    candidates += [np.floor(x) - 0.01 for x in candidates if x > 1.0]
    c = sorted(set([round(x, 2) for x in candidates if np.isfinite(x) and x > 0]))
    return c

@dataclass
class OptimizerConfig:
    min_margin_pct: float = 0.05
    oos_penalty: float = 0.0
    use_profit: bool = False
    unit_cost_col: Optional[str] = None  # e.g., "UNIT_COST"

def choose_price_for_row(row: pd.Series,
                         models: Dict[str, FittedBucket],
                         cfg: OptimizerConfig) -> Tuple[float, float, float]:
    base = float(row.get("base_price", np.nan))
    last = float(row.get("price", base if np.isfinite(base) else np.nan))
    ladder = make_price_ladder(base, last if np.isfinite(last) else 0.01)
    if not ladder:
        return (float(last if np.isfinite(last) else 0.01), 0.0, 0.0)

    rows = []
    for p in ladder:
        r = row.copy()
        r["price"] = p
        r["promo_depth"] = 0.0 if not np.isfinite(r.get("base_price", np.nan)) or r["base_price"] <= 0 \
                           else (r["base_price"] - p) / r["base_price"]
        rows.append(r)
    cand_df = pd.DataFrame(rows)

    q_hat = predict_qty(models, cand_df)
    cand_df["q_hat"] = q_hat

    # Objective
    if cfg.use_profit and cfg.unit_cost_col and cfg.unit_cost_col in row.index and np.isfinite(row[cfg.unit_cost_col]):
        cost = float(row[cfg.unit_cost_col])
        cand_df = cand_df[cand_df["price"] >= cost * (1 + cfg.min_margin_pct)]
        cand_df["objective"] = (cand_df["price"] - cost) * cand_df["q_hat"]
    else:
        cand_df["objective"] = cand_df["price"] * cand_df["q_hat"]

    if cfg.oos_penalty > 0:
        supply = float(row.get("start_stock", 0) + row.get("delivery_qty", 0))
        shortage = np.maximum(0.0, cand_df["q_hat"].values - supply)
        cand_df["objective"] = cand_df["objective"].values - cfg.oos_penalty * shortage

    if cand_df.empty:
        return (float(last if np.isfinite(last) else 0.01), float(row.get("q_ma_4", 0.0)), 0.0)

    idx = int(np.argmax(cand_df["objective"].values))
    best = cand_df.iloc[idx]
    return (float(best["price"]), float(best["q_hat"]), float(best["objective"]))

# ----------------------------
# 5) Future-row builder (used by next-week & evaluation)
# ----------------------------
def _build_future_rows(panel_feats: pd.DataFrame,
                       week_past: pd.Timestamp,
                       week_target: pd.Timestamp) -> pd.DataFrame:
    """
    Build one target-week row per (STORE, PRODUCT, bucket) using information available
    up to week_past (typically week_target - 7 days).
    """
    keys = ["STORE","PRODUCT_CODE","bucket"]
    last_rows = (panel_feats[panel_feats["week_start"] == week_past]
                 .sort_values(keys)
                 .copy())

    # Shift lags forward
    for i in range(4, 1, -1):
        last_rows[f"q_lag_{i}"] = last_rows[f"q_lag_{i-1}"]
        last_rows[f"price_lag_{i}"] = last_rows[f"price_lag_{i-1}"]
        last_rows[f"promo_lag_{i}"] = last_rows[f"promo_lag_{i-1}"]
    last_rows["q_lag_1"] = last_rows["q_total"]
    last_rows["price_lag_1"] = last_rows["price"]
    last_rows["promo_lag_1"] = last_rows["is_promo"]

    # Recompute MAs from history ≤ week_past
    def recompute_ma(group):
        g_hist = (panel_feats[(panel_feats["STORE"]==group.name[0]) &
                              (panel_feats["PRODUCT_CODE"]==group.name[1]) &
                              (panel_feats["bucket"]==group.name[2]) &
                              (panel_feats["week_start"]<=week_past)]
                  .sort_values("week_start").tail(4))
        q_ma = g_hist["q_total"].mean() if len(g_hist) else 0.0
        p_ma = g_hist["price"].mean() if len(g_hist) else group["price"].iloc[0]
        out = group.copy()
        out["q_ma_4"] = q_ma
        out["price_ma_4"] = p_ma
        return out

    last_rows = last_rows.groupby(keys, group_keys=False).apply(recompute_ma).reset_index(drop=True)

    # Set target-week values derived from known calendars
    last_rows["week_start"] = week_target
    last_rows["last_price"] = last_rows["price"]  # realized last price
    last_rows["is_promo"] = last_rows.apply(lambda r: _is_promo_for_week(r, week_target), axis=1)

    # We forecast q_total; not known at target
    last_rows["q_total"] = 0.0
    return last_rows

# ----------------------------
# 6) Next-week recommender (single shot)
# ----------------------------
@dataclass
class NextWeekRow:
    STORE: str
    PRODUCT_CODE: str
    FAMILY_CODE: object
    CATEGORY_CODE: object
    SEGMENT_CODE: object
    REGION_NAME: object
    STORE_TYPE: object
    PLACE_TYPE: object
    bucket: str
    week_start_next: pd.Timestamp
    base_price: float
    last_price: float
    is_promo_next: int
    price_star: float
    q_hat_star: float
    revenue_star: float
    promo_depth_at_star: float

def recommend_next_week_prices(raw_df: pd.DataFrame,
                               end_date: str = "2025-09-15",
                               cfg: OptimizerConfig = OptimizerConfig()) -> pd.DataFrame:
    panel = make_week_bucket_panel(raw_df)
    panel = add_time_features(panel)

    week_curr = min(to_week_start(pd.Timestamp(end_date)), panel["week_start"].max())
    week_next = week_curr + pd.Timedelta(days=7)

    # Train on ≤ week_curr
    train = panel[panel["week_start"] <= week_curr].copy()
    models = fit_per_bucket_models(train, alpha=5.0)

    # Build target rows for next week from last available week (week_curr)
    next_rows = _build_future_rows(train, week_curr, week_next)

    out_records = []
    for _, row in next_rows.iterrows():
        p_star, q_hat, obj = choose_price_for_row(row, models, cfg)
        base = float(row.get("base_price", np.nan))
        promo_depth_star = 0.0 if not np.isfinite(base) or base <= 0 else (base - p_star) / base
        out_records.append(NextWeekRow(
            STORE=str(row["STORE"]),
            PRODUCT_CODE=str(row["PRODUCT_CODE"]),
            FAMILY_CODE=row.get("FAMILY_CODE"),
            CATEGORY_CODE=row.get("CATEGORY_CODE"),
            SEGMENT_CODE=row.get("SEGMENT_CODE"),
            REGION_NAME=row.get("REGION_NAME"),
            STORE_TYPE=row.get("STORE_TYPE"),
            PLACE_TYPE=row.get("PLACE_TYPE"),
            bucket=row["bucket"],
            week_start_next=week_next,
            base_price=base,
            last_price=float(row.get("last_price", np.nan)),
            is_promo_next=int(row.get("is_promo", 0)),
            price_star=p_star,
            q_hat_star=q_hat,
            revenue_star=p_star * q_hat,
            promo_depth_at_star=promo_depth_star
        ).__dict__)

    return pd.DataFrame(out_records).sort_values(["STORE","PRODUCT_CODE","bucket"]).reset_index(drop=True)

# ----------------------------
# 7) Rolling evaluation (same future-row logic)
# ----------------------------
@dataclass
class EvalRow:
    week_start: pd.Timestamp
    bucket: str
    STORE: str
    PRODUCT_CODE: str
    base_price_hist: float
    last_price_hist: float
    is_promo_week: int
    price_star: float
    q_hat_star: float
    revenue_star: float
    realized_sale_price: float  # actual SALE_PRICE mean that week
    realized_qty: float         # actual q_total that week
    realized_revenue: float

def rolling_evaluation(raw_df: pd.DataFrame,
                       end_date: str = "2025-09-15",
                       horizon_weeks: int = 12,
                       cfg: OptimizerConfig = OptimizerConfig()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - eval_rows: per-row detailed comparison (policy vs realized) for each of the last `horizon_weeks`
      - kpis: aggregated KPIs by week and bucket
    """
    panel = make_week_bucket_panel(raw_df)
    panel = add_time_features(panel)

    end_week = to_week_start(pd.Timestamp(end_date))
    weeks = sorted([w for w in panel["week_start"].unique() if w <= end_week])
    if len(weeks) < (horizon_weeks + 1):
        raise ValueError(f"Need at least {horizon_weeks+1} weeks of data for rolling evaluation.")
    eval_weeks = weeks[-horizon_weeks:]  # last N weeks
    eval_rows = []

    for w in eval_weeks:
        week_past = weeks[weeks.index(w) - 1]  # previous week exists due to check above

        # Train ≤ week_past
        train = panel[panel["week_start"] <= week_past].copy()
        models = fit_per_bucket_models(train, alpha=5.0)

        # Build target rows for week w from lags at week_past
        target_rows = _build_future_rows(train, week_past, w)

        # Choose prices per target row
        chosen = []
        for _, row in target_rows.iterrows():
            p_star, q_hat, obj = choose_price_for_row(row, models, cfg)
            base = float(row.get("base_price", np.nan))
            chosen.append((row, p_star, q_hat))

        # Join realized week w to compare (sale price & qty)
        realized_w = panel[panel["week_start"] == w][
            ["STORE","PRODUCT_CODE","bucket","price","q_total"]
        ].rename(columns={"price":"realized_sale_price","q_total":"realized_qty"})

        # Build eval detail rows
        for (row, p_star, q_hat) in chosen:
            key = (str(row["STORE"]), str(row["PRODUCT_CODE"]), row["bucket"])
            rw = realized_w[(realized_w["STORE"]==key[0]) &
                            (realized_w["PRODUCT_CODE"]==key[1]) &
                            (realized_w["bucket"]==key[2])]

            realized_price = float(rw["realized_sale_price"].iloc[0]) if not rw.empty else np.nan
            realized_qty   = float(rw["realized_qty"].iloc[0]) if not rw.empty else 0.0

            eval_rows.append(EvalRow(
                week_start=w,
                bucket=row["bucket"],
                STORE=key[0],
                PRODUCT_CODE=key[1],
                base_price_hist=float(row.get("base_price", np.nan)),
                last_price_hist=float(row.get("last_price", np.nan)),
                is_promo_week=int(row.get("is_promo", 0)),
                price_star=float(p_star),
                q_hat_star=float(q_hat),
                revenue_star=float(p_star*q_hat),
                realized_sale_price=realized_price,
                realized_qty=realized_qty,
                realized_revenue=float((realized_price if np.isfinite(realized_price) else 0.0) * realized_qty)
            ).__dict__)

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        return eval_df, eval_df

    # Aggregated KPIs
    kpis = (eval_df
            .groupby(["week_start","bucket"], as_index=False)
            .agg(
                n_rows=("PRODUCT_CODE","count"),
                revenue_policy=("revenue_star","sum"),
                revenue_realized=("realized_revenue","sum")
            ))
    kpis["uplift_vs_realized_%"] = 100.0 * (kpis["revenue_policy"] - kpis["revenue_realized"]) / \
                                   kpis["revenue_realized"].replace(0, np.nan)
    kpis = kpis.sort_values(["week_start","bucket"]).reset_index(drop=True)

    return eval_df.sort_values(["week_start","STORE","PRODUCT_CODE","bucket"]).reset_index(drop=True), kpis

# ----------------------------
# 8) How to run
# ----------------------------
# df = pd.read_csv("your_data.csv")
# cfg = OptimizerConfig(min_margin_pct=0.05, oos_penalty=0.0, use_profit=False)  # set unit_cost_col if you have costs

# A) Next week recommendations (training on all history ≤ 2025-09-15)
# next_week = recommend_next_week_prices(df, end_date="2025-09-15", cfg=cfg)
# print(next_week.head())

# B) Rolling evaluation for last 12 weeks (same future-row logic)
# eval_rows, kpis = rolling_evaluation(df, end_date="2025-09-15", horizon_weeks=12, cfg=cfg)
# print(kpis)
# print(eval_rows.head())
