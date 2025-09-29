# ============================
# Dynamic Pricing — Robust Start (Hierarchical log-demand + optimizer)
# ============================
# Requirements: pandas, numpy, scikit-learn
import pandas as pd
import numpy as np
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------
# 1) Utilities
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
    Returns (start, end); NaT if unknown.
    """
    if not isinstance(s, str) or "-" not in s:
        return (pd.NaT, pd.NaT)
    try:
        a, b = [x.strip() for x in s.split(" - ")]
        start = pd.to_datetime(a, dayfirst=True, errors="coerce")
        end   = pd.to_datetime(b, dayfirst=True, errors="coerce")
        return (start, end)
    except Exception:
        return (pd.NaT, pd.NaT)

def safe_div(a, b):
    b = np.where(b == 0, np.nan, b)
    return np.divide(a, b)

# ----------------------------
# 2) Aggregation to week × bucket
# ----------------------------
ddef make_week_bucket_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"])
    df["week_start"] = df["TRADE_DT"].apply(to_week_start)
    df["bucket"] = df["TRADE_DT"].dt.weekday.map(BUCKET_MAP)

    # Cast numeric cols robustly
    num_cols = ["SALE_QTY","SALE_QTY_ONLINE","SALE_PRICE","BASE_PRICE",
                "START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY"]
    for c in num_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["SALE_QTY"] = df["SALE_QTY"].fillna(0)
    df["SALE_QTY_ONLINE"] = df["SALE_QTY_ONLINE"].fillna(0)
    df["q_total"] = df["SALE_QTY"] + df["SALE_QTY_ONLINE"]

    starts, ends = zip(*df["PROMO_PERIOD"].apply(parse_promo_period))
    df["promo_start"] = list(starts)
    df["promo_end"] = list(ends)

    for col in ["START_STOCK", "END_STOCK", "DELIVERY_QTY", "LOSS_QTY", "RETURN_QTY"]:
        if col in df:
            df[col] = df[col].fillna(0)

    grp_cols = ["STORE","PRODUCT_CODE","bucket","week_start",
                "FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE",
                "STORE_TYPE","REGION_NAME","PLACE_TYPE"]

    agg = (df.groupby(grp_cols, dropna=False)
             .agg(q_total=("q_total","sum"),
                  price=("SALE_PRICE","mean"),
                  base_price=("BASE_PRICE","mean"),
                  is_promo=("IS_PROMO","max"),
                  start_stock=("START_STOCK","sum"),
                  end_stock=("END_STOCK","sum"),
                  delivery_qty=("DELIVERY_QTY","sum"),
                  loss_qty=("LOSS_QTY","sum"),
                  return_qty=("RETURN_QTY","sum"),
                  promo_start=("promo_start","min"),
                  promo_end=("promo_end","max"))
             .reset_index())

    agg["promo_depth"] = safe_div(agg["base_price"] - agg["price"], agg["base_price"]).fillna(0.0)
    agg["oos_flag"] = (agg["start_stock"] <= 0) | \
        ((agg["start_stock"] + agg["delivery_qty"] - agg["end_stock"] - agg["loss_qty"] - agg["return_qty"]) <= 0)
    agg["promo_active_now"] = (agg["promo_start"].notna()) & (agg["promo_start"] <= agg["week_start"]) & \
                              (agg["week_start"] <= agg["promo_end"].fillna(pd.Timestamp.max))
    return agg


# ----------------------------
# 3) Feature builder (lags & rollings)
# ----------------------------
def add_time_features(panel: pd.DataFrame, max_lag_weeks: int = 4) -> pd.DataFrame:
    panel = panel.sort_values(["STORE", "PRODUCT_CODE", "bucket", "week_start"]).copy()
    keys = ["STORE", "PRODUCT_CODE", "bucket"]

    def add_grp_features(g):
        for L in range(1, max_lag_weeks + 1):
            g[f"q_lag_{L}"] = g["q_total"].shift(L)
            g[f"price_lag_{L}"] = g["price"].shift(L)
            g[f"promo_lag_{L}"] = g["is_promo"].shift(L)
        g["q_ma_4"] = g["q_total"].rolling(4, min_periods=1).mean().shift(1)
        g["price_ma_4"] = g["price"].rolling(4, min_periods=1).mean().shift(1)
        return g

    panel = panel.groupby(keys, group_keys=False).apply(add_grp_features)
    panel.fillna(0, inplace=True)
    return panel

# ----------------------------
# 4) Hierarchical pooled model per bucket (ridge)
# ----------------------------
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

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

def fit_per_bucket_models(train_panel: pd.DataFrame, alpha: float = 5.0):
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

def predict_qty(models, df_rows):
    b = df_rows["bucket"].iloc[0]
    fb = models.get(b)
    if fb is None:
        return np.zeros(len(df_rows))
    yhat = fb.model.predict(df_rows[fb.feature_cols])
    return np.expm1(yhat).clip(min=0)


# ----------------------------
# 5) Price ladder & optimizer
# ----------------------------
def make_price_ladder(base_price: float, last_price: float) -> List[float]:
    """
    Business-friendly ladder around BASE_PRICE & last price; clamp >0.
    """
    candidates = []
    anchors = [base_price, last_price]
    multipliers = [0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.3]
    for a in anchors:
        for m in multipliers:
            candidates.append(max(0.01, round(a * m, 2)))
    # .99 endings
    candidates += [np.floor(x) - 0.01 for x in candidates if x > 1.0]
    # unique & sorted
    c = sorted(set([round(x, 2) for x in candidates if np.isfinite(x) and x > 0]))
    return c

@dataclass
class OptimizerConfig:
    min_margin_pct: float = 0.05   # floor
    oos_penalty: float = 0.0       # set >0 to penalize OOS risk
    use_profit: bool = False       # if False -> maximize revenue
    unit_cost_col: Optional[str] = None  # if provided and use_profit=True

def choose_price_for_row(row: pd.Series,
                         models: Dict[str, FittedBucket],
                         cfg: OptimizerConfig) -> Tuple[float, float, float]:
    """
    Returns (best_price, q_hat_best, obj_best) for a single (store, product, bucket, week).
    Uses only info in row (lagged features).
    """
    base = float(row.get("base_price", np.nan)) if np.isfinite(row.get("base_price", np.nan)) else float(row.get("price", 0))
    last = float(row.get("price", base if np.isfinite(base) else 0))
    ladder = make_price_ladder(base, last)
    if len(ladder) == 0:
        return (last, 0.0, 0.0)

    # Clone row across candidate prices
    rows = []
    for p in ladder:
        r = row.copy()
        r["price"] = p
        r["promo_depth"] = 0.0 if not np.isfinite(r.get("base_price", np.nan)) or r["base_price"] <= 0 else (r["base_price"] - p) / r["base_price"]
        rows.append(r)
    cand_df = pd.DataFrame(rows)

    q_hat = predict_qty(models, cand_df)
    cand_df["q_hat"] = q_hat

    # Constraints: margin floor if profit used and cost available
    if cfg.use_profit and cfg.unit_cost_col is not None and cfg.unit_cost_col in row.index and np.isfinite(row[cfg.unit_cost_col]):
        cost = float(row[cfg.unit_cost_col])
        cand_df = cand_df[cand_df["price"] >= cost * (1 + cfg.min_margin_pct)]
        cand_df["profit"] = (cand_df["price"] - cost) * cand_df["q_hat"]
        objective = cand_df["profit"].values
    else:
        # Revenue objective by default
        objective = (cand_df["price"] * cand_df["q_hat"]).values

    # Optional OOS risk penalty (requires stock fields already in features)
    if cfg.oos_penalty > 0:
        supply = float(row.get("start_stock", 0) + row.get("delivery_qty", 0))
        risk = np.maximum(0.0, cand_df["q_hat"].values - supply)  # expected shortage
        objective = objective - cfg.oos_penalty * risk

    if len(objective) == 0:
        # Fallback: keep last price
        return (last, float(row.get("q_ma_4", 0.0)), 0.0)

    idx = int(np.argmax(objective))
    return (float(cand_df.iloc[idx]["price"]), float(cand_df.iloc[idx]["q_hat"]), float(objective[idx]))

# ----------------------------
# 6) Rolling evaluation (last 12 weeks until 2025-09-15)
# ----------------------------
@dataclass
class EvalResult:
    week_start: pd.Timestamp
    bucket: str
    revenue_realized: float
    revenue_policy: float
    count_rows: int

def rolling_evaluation(raw_df: pd.DataFrame,
                       end_date: str = "2025-09-15",
                       cfg: OptimizerConfig = OptimizerConfig()) -> pd.DataFrame:
    panel = make_week_bucket_panel(raw_df)
    panel = add_time_features(panel)

    # Determine eval horizon
    end_week = to_week_start(pd.Timestamp(end_date))
    weeks = sorted(panel["week_start"].unique())
    weeks = [w for w in weeks if w <= end_week]
    if len(weeks) < 13:
        raise ValueError("Need at least 13 weeks of data to run 12-week rolling evaluation.")
    eval_weeks = weeks[-12:]  # last 12

    results = []
    for w in eval_weeks:
        # Train with data strictly before current eval week
        train = panel[panel["week_start"] < w]
        test  = panel[panel["week_start"] == w].copy()
        if test.empty or train.empty:
            continue

        models = fit_per_bucket_models(train, alpha=5.0)

        # Choose prices per row (STORE×PRODUCT×bucket)
        best_prices = []
        for i, row in test.iterrows():
            p_star, q_hat, obj = choose_price_for_row(row, models, cfg)
            best_prices.append((i, p_star, q_hat, obj))
        choose_df = pd.DataFrame(best_prices, columns=["idx", "price_star", "q_hat_star", "obj_star"]).set_index("idx")
        test = test.join(choose_df, how="left")

        # Evaluate vs realized (counterfactual unknown; we log policy revenue using predicted q_hat at chosen p)
        test["revenue_realized"] = test["price"] * test["q_total"]
        test["revenue_policy"] = test["price_star"] * test["q_hat_star"]

        # Aggregate by bucket (optional)
        for b, g in test.groupby("bucket"):
            results.append(EvalResult(
                week_start=w, bucket=b,
                revenue_realized=float((g["revenue_realized"]).sum()),
                revenue_policy=float((g["revenue_policy"]).sum()),
                count_rows=len(g)
            ))

    out = pd.DataFrame([r.__dict__ for r in results])
    if out.empty:
        return out
    # Overall KPIs
    out["uplift_vs_realized_%"] = 100 * (out["revenue_policy"] - out["revenue_realized"]) / (out["revenue_realized"].replace(0, np.nan))
    return out.sort_values(["week_start", "bucket"]).reset_index(drop=True)

# ----------------------------
# 7) How to run
# ----------------------------
# df = pd.read_csv("your_data.csv")  # must contain the columns from your screenshot/list
# eval_report = rolling_evaluation(df, end_date="2025-09-15",
#                                  cfg=OptimizerConfig(min_margin_pct=0.05,
#                                                      oos_penalty=0.0,      # set >0 to penalize shortages
#                                                      use_profit=False))    # set True if you have unit costs
# print(eval_report.head())
# print("Mean uplift %:", eval_report["uplift_vs_realized_%"].mean())
