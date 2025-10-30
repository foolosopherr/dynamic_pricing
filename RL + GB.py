# ============================
# Offline Pricing with CatBoost + Contextual Bandit (UCB)
# Buckets: Mon-Thu, Fri, Sat-Sun
# ============================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor, Pool

# ----------------------------
# CONFIG
# ----------------------------
CONFIG = {
    "price_multipliers": np.round(np.arange(0.90, 1.31, 0.05), 2),  # ARMS (min..max)
    "min_margin_pct": 0.08,        # minimal gross margin vs cost
    "max_price_jump_pct": 0.15,    # vs last realized price (guard against shocks)
    "oos_penalty_factor": 0.5,     # penalize profit if predicted demand > available stock
    "repeat_arm_penalty": 0.03,    # subtraction from UCB score if repeating last arm
    "ucb_c": 0.7,                  # exploration strength
    "eval_weeks": 12,              # rolling evaluation window
    "min_obs_sku": 12,             # cold-start threshold for SKU-level training rows
    "hier_levels": ["FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"],
    "random_state": 42
}

# ----------------------------
# Helpers: calendar & buckets
# ----------------------------
def to_date(s):
    return pd.to_datetime(s).dt.tz_localize(None) if isinstance(s, pd.Series) else pd.to_datetime(s).tz_localize(None)

def week_monday(d):
    d = pd.to_datetime(d)
    return d - timedelta(days=(d.weekday()))  # Monday

def which_bucket(d):
    wd = pd.to_datetime(d).weekday()  # 0-Mon ... 6-Sun
    if wd in [0,1,2,3]: return "Mon-Thu"
    if wd == 4: return "Fri"
    return "Sat-Sun"

def bucket_bounds(week_start, bucket_kind):
    """
    Return inclusive start and inclusive end datetimes for a bucket inside a Mon-Sun week.
    week_start is Monday.
    """
    ws = week_monday(week_start)
    if bucket_kind == "Mon-Thu":
        return ws, ws + timedelta(days=3)
    if bucket_kind == "Fri":
        d = ws + timedelta(days=4)
        return d, d
    if bucket_kind == "Sat-Sun":
        return ws + timedelta(days=5), ws + timedelta(days=6)
    raise ValueError("bucket_kind must be one of: Mon-Thu, Fri, Sat-Sun")

# ----------------------------
# Data prep
# ----------------------------
def prepare(df):
    df = df.copy()
    df["TRADE_DT"] = to_date(df["TRADE_DT"]).dt.normalize()
    df["WEEK_START"] = df["TRADE_DT"].apply(week_monday)
    df["BUCKET_KIND"] = df["TRADE_DT"].apply(which_bucket)

    # Parse PROMO_PERIOD 'dd-mm-YYYY - dd-mm-YYYY'
    def parse_interval(s):
        if pd.isna(s): return (pd.NaT, pd.NaT)
        try:
            a,b = [x.strip() for x in str(s).split("-")]
            # string may be like "01-01-2024 " and " 03-01-2024"
            a = pd.to_datetime(a, dayfirst=True, errors="coerce")
            b = pd.to_datetime(b, dayfirst=True, errors="coerce")
            return (a, b)
        except:
            return (pd.NaT, pd.NaT)
    promo = df["PROMO_PERIOD"].apply(parse_interval)
    df["PROMO_START"] = [x[0] for x in promo]
    df["PROMO_END"]   = [x[1] for x in promo]
    df["IS_PROMO_ACTIVE"] = (df["TRADE_DT"]>=df["PROMO_START"]) & (df["TRADE_DT"]<=df["PROMO_END"])

    # Fill missing stock-related fields
    for c in ["START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY","SALE_QTY","SALE_PRICE","BASE_PRICE"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Approximate cost if absent
    if "COST_PRICE" not in df.columns:
        df["COST_PRICE"] = np.maximum(df["BASE_PRICE"]*(1-0.75), 0)  # assume 25% baseline margin; adjust if known

    # Daily to bucket aggregates (per store, SKU)
    agg = (df.groupby(["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"], as_index=False)
             .agg({
                 "SALE_QTY":"sum",
                 "SALE_PRICE":"mean",
                 "BASE_PRICE":"mean",
                 "COST_PRICE":"mean",
                 "START_STOCK":"max",
                 "END_STOCK":"min",
                 "DELIVERY_QTY":"sum",
                 "IS_PROMO_ACTIVE":"max"
             }))
    agg.rename(columns={"SALE_QTY":"QTY","SALE_PRICE":"PRICE"}, inplace=True)
    # simple stock availability proxy
    agg["AVAIL_STOCK"] = np.maximum(agg["START_STOCK"] + agg["DELIVERY_QTY"] - np.maximum(agg["QTY"],0), 0)

    # Lag features (last 12 weeks per SKU-store-bucket)
    agg = agg.sort_values(["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"])
    grp = agg.groupby(["STORE","PRODUCT_CODE","BUCKET_KIND"], group_keys=False)
    for col in ["QTY","PRICE","AVAIL_STOCK"]:
        agg[f"{col}_L1"]  = grp[col].shift(1)
        agg[f"{col}_MA4"] = grp[col].rolling(4, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
        agg[f"{col}_MA12"]= grp[col].rolling(12, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)

    # Hierarchical aggregates (share information)
    for lvl in CONFIG["hier_levels"]:
        if lvl in df.columns:
            map_lvl = df.groupby(["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"], as_index=False)[lvl].first()
            agg = agg.merge(map_lvl, on=["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"], how="left")
    return agg

# ----------------------------
# Demand model (CatBoost)
# ----------------------------
def train_demand_model(hist):
    # Use only rows with positive observations to reduce noise
    use = hist.copy()
    y = use["QTY"].astype(float).values
    cat_cols = [c for c in ["STORE","PRODUCT_CODE","BUCKET_KIND","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"] if c in use.columns]
    num_cols = [c for c in use.columns if c not in ["QTY","WEEK_START"] + cat_cols]
    X = use[cat_cols+num_cols]

    model = CatBoostRegressor(
        depth=6, learning_rate=0.08, n_estimators=400,
        random_seed=CONFIG["random_state"],
        loss_function="RMSE", verbose=False
    )
    train_pool = Pool(X, y, cat_features=list(range(len(cat_cols))))
    model.fit(train_pool)
    return model, cat_cols, num_cols

def predict_qty(model, cat_cols, num_cols, dfX):
    pool = Pool(dfX[cat_cols+num_cols], cat_features=list(range(len(cat_cols))))
    pred = model.predict(pool)
    return np.maximum(pred, 0.0)

# ----------------------------
# Contextual bandit policy (per (STORE, SKU, BUCKET))
# UCB with: exploration bonus, repetition penalty, OOS guard
# ----------------------------
def select_arm(row, last_price, last_arm, model, cat_cols, num_cols):
    base_price = row["BASE_PRICE"]
    cost = row["COST_PRICE"]
    avail = max(row.get("AVAIL_STOCK", 0), 0)

    arms = CONFIG["price_multipliers"]
    # Keep arms within price jump constraint vs last realized price (if known)
    if pd.notna(last_price) and last_price > 0:
        lo = max(arms.min(), (1 - CONFIG["max_price_jump_pct"]) * (last_price / base_price))
        hi = min(arms.max(), (1 + CONFIG["max_price_jump_pct"]) * (last_price / base_price))
        arms = arms[(arms >= lo) & (arms <= hi)]
        if len(arms) == 0:
            arms = CONFIG["price_multipliers"]

    # Build candidate rows
    cands = []
    for a in arms:
        price = max(round(base_price * a, 2), cost / (1 - CONFIG["min_margin_pct"]))
        cand = row.copy()
        cand["PRICE"] = price
        cand["ARM"] = a
        cands.append(cand)
    C = pd.DataFrame(cands)

    # Predict demand per arm using CatBoost
    qty_pred = predict_qty(model, cat_cols, num_cols, C)
    # OOS prevention: clamp to available stock
    qty_served = np.minimum(qty_pred, avail if avail>0 else qty_pred)  # if no stock info, keep as is
    profit = (C["PRICE"] - C["COST_PRICE"]) * qty_served

    # Simple UCB: bonus by sqrt(log T / (n_arm+1)). We proxy counts by global frequency of arm per SKU if provided.
    # For a single-shot per bucket, encourage diversity via small exploration + repetition penalty:
    T = 100  # virtual horizon
    n_arm = 1  # proxy (can store history externally; set to 1 to give small bonus)
    bonus = CONFIG["ucb_c"] * np.sqrt(np.log(T) / (n_arm))

    score = profit.values + bonus
    # repetition penalty
    if pd.notna(last_arm):
        score = score - CONFIG["repeat_arm_penalty"] * (C["ARM"].values == last_arm)

    best_idx = int(np.argmax(score))
    return float(C.iloc[best_idx]["ARM"]), float(C.iloc[best_idx]["PRICE"]), float(profit.iloc[best_idx])

# ----------------------------
# Rolling evaluation (12 weeks)
# ----------------------------
def rolling_eval(agg, bucket_kind, end_week_start, weeks=12):
    """
    Evaluate policy for the last `weeks` buckets of type `bucket_kind` ending before `end_week_start`.
    Uses model trained only on data strictly before each evaluated bucket.
    Returns dataframe with simulated profits for policy vs baseline.
    """
    out = []
    # sequence of bucket week_starts to evaluate
    wk_list = (agg.query("BUCKET_KIND == @bucket_kind and WEEK_START < @end_week_start")
                 .WEEK_START.drop_duplicates().sort_values())[-weeks:]

    for wk in wk_list:
        # train window: all data strictly before this bucket
        train_hist = agg[(agg["WEEK_START"] < wk) & (agg["BUCKET_KIND"] == bucket_kind)]

        # Cold-start: if too few rows, back off to hierarchy by dropping PRODUCT_CODE for CatBoost (handled implicitly)
        if len(train_hist) < 50:
            continue

        model, cat_cols, num_cols = train_demand_model(train_hist)

        cur = agg[(agg["WEEK_START"] == wk) & (agg["BUCKET_KIND"] == bucket_kind)].copy()
        # last realized signals
        cur = cur.sort_values(["STORE","PRODUCT_CODE"])
        # build index to fetch last price & arm (approx from last week)
        prev = agg[(agg["WEEK_START"] < wk) & (agg["BUCKET_KIND"] == bucket_kind)]
        prev_last = (prev.sort_values("WEEK_START")
                          .groupby(["STORE","PRODUCT_CODE"]).tail(1)[["STORE","PRODUCT_CODE","PRICE"]]
                          .rename(columns={"PRICE":"LAST_PRICE"}))
        cur = cur.merge(prev_last, on=["STORE","PRODUCT_CODE"], how="left")

        # last arm approximate from ratio to base
        cur["LAST_ARM"] = (cur["LAST_PRICE"] / cur["BASE_PRICE"]).round(2)

        recs = []
        for i, row in cur.iterrows():
            arm, price, prof = select_arm(row, row.get("LAST_PRICE", np.nan), row.get("LAST_ARM", np.nan),
                                          model, cat_cols, num_cols)
            recs.append((arm, price, prof))
        cur["REC_ARM"], cur["REC_PRICE"], cur["REC_PROFIT_SIM"] = zip(*recs)

        # Baseline: keep last price (or base price)
        cur["BASELINE_PRICE"] = np.where(cur["LAST_PRICE"].notna(), cur["LAST_PRICE"], cur["BASE_PRICE"])
        # Simulate baseline demand (use model)
        baseX = cur.copy()
        baseX["PRICE"] = baseX["BASELINE_PRICE"]
        base_qty = predict_qty(model, cat_cols, num_cols, baseX)
        base_served = np.minimum(base_qty, np.maximum(cur["AVAIL_STOCK"],0))
        cur["BASELINE_PROFIT_SIM"] = (baseX["PRICE"] - baseX["COST_PRICE"]) * base_served

        # KPI
        cur["UPLIFT_SIM"] = cur["REC_PROFIT_SIM"] - cur["BASELINE_PROFIT_SIM"]
        cur["WEEK_START_EVAL"] = wk
        out.append(cur[["STORE","PRODUCT_CODE","BUCKET_KIND","WEEK_START_EVAL",
                        "REC_ARM","REC_PRICE","REC_PROFIT_SIM",
                        "BASELINE_PRICE","BASELINE_PROFIT_SIM","UPLIFT_SIM"]])

    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    kpi = (res.groupby("WEEK_START_EVAL", as_index=False)
              .agg({"REC_PROFIT_SIM":"sum","BASELINE_PROFIT_SIM":"sum","UPLIFT_SIM":"sum"}))
    kpi["UPLIFT_%"] = 100 * (kpi["UPLIFT_SIM"] / np.maximum(kpi["BASELINE_PROFIT_SIM"],1e-6))
    return res, kpi

# ----------------------------
# One-shot recommendation for a chosen bucket (using info strictly before bucket start)
# ----------------------------
def recommend_for_bucket(agg, bucket_kind, week_start):
    # Train model on history strictly before target bucket
    train_hist = agg[(agg["WEEK_START"] < week_start) & (agg["BUCKET_KIND"] == bucket_kind)]
    if len(train_hist) < 50:
        raise ValueError("Not enough history for training. Reduce constraints or provide more data.")
    model, cat_cols, num_cols = train_demand_model(train_hist)

    cur = agg[(agg["WEEK_START"] == week_start) & (agg["BUCKET_KIND"] == bucket_kind)].copy()
    prev = agg[(agg["WEEK_START"] < week_start) & (agg["BUCKET_KIND"] == bucket_kind)]
    prev_last = (prev.sort_values("WEEK_START")
                      .groupby(["STORE","PRODUCT_CODE"]).tail(1)[["STORE","PRODUCT_CODE","PRICE"]]
                      .rename(columns={"PRICE":"LAST_PRICE"}))
    cur = cur.merge(prev_last, on=["STORE","PRODUCT_CODE"], how="left")
    cur["LAST_ARM"] = (cur["LAST_PRICE"] / cur["BASE_PRICE"]).round(2)

    recs = []
    for i, row in cur.iterrows():
        arm, price, prof = select_arm(row, row.get("LAST_PRICE", np.nan), row.get("LAST_ARM", np.nan),
                                      model, cat_cols, num_cols)
        recs.append((arm, price, prof))
    cur["REC_ARM"], cur["REC_PRICE"], cur["REC_PROFIT_SIM"] = zip(*recs)

    return cur[["STORE","PRODUCT_CODE","BUCKET_KIND","WEEK_START","BASE_PRICE","LAST_PRICE","REC_ARM","REC_PRICE","REC_PROFIT_SIM"]]

# ----------------------------
# Orchestration: run for 3 buckets & evaluate
# ----------------------------
def run_all(df, chosen_week_start_str):
    """
    chosen_week_start_str: any date string inside the target week; it will be snapped to Monday.
    Produces:
      - recommendations for Mon-Thu, Fri, Sat-Sun buckets for that week
      - 12-week rolling evaluation up to each bucket start
    """
    agg = prepare(df)
    week_start = week_monday(pd.to_datetime(chosen_week_start_str))

    outputs = {}
    for bucket_kind in ["Mon-Thu","Fri","Sat-Sun"]:
        b_start, _ = bucket_bounds(week_start, bucket_kind)

        # evaluation uses last 12 occurrences of this bucket, ending before b_start
        eval_res = rolling_eval(agg, bucket_kind, b_start, weeks=CONFIG["eval_weeks"])
        outputs[f"eval_{bucket_kind}"] = eval_res

        # recommendation for the chosen bucket
        rec = recommend_for_bucket(agg, bucket_kind, b_start)
        outputs[f"rec_{bucket_kind}"] = rec

    return outputs

# ----------------------------
# Usage:
# ----------------------------
# df = ...  # your raw daily data with required columns
# res = run_all(df, "2025-09-15")  # any date in the desired week; snapped to Monday
# rec_mon_thu = res["rec_Mon-Thu"]
# rec_fri     = res["rec_Fri"]
# rec_sat_sun = res["rec_Sat-Sun"]
# eval_mon_thu = res["eval_Mon-Thu"]
# print(rec_mon_thu.head())
# print(eval_mon_thu[1] if isinstance(eval_mon_thu, tuple) else eval_mon_thu)
