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
            a = pd.to_datetime(a, dayfirst=True, errors="coerce")
            b = pd.to_datetime(b, dayfirst=True, errors="coerce")
            return (a, b)
        except:
            return (pd.NaT, pd.NaT)
    promo = df["PROMO_PERIOD"].apply(parse_interval)
    df["PROMO_START"] = [x[0] for x in promo]
    df["PROMO_END"]   = [x[1] for x in promo]
    df["IS_PROMO_ACTIVE"] = (df["TRADE_DT"]>=df["PROMO_START"]) & (df["TRADE_DT"]<=df["PROMO_END"])

    for c in ["START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY",
              "SALE_QTY","SALE_PRICE","BASE_PRICE"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    if "COST_PRICE" not in df.columns:
        df["COST_PRICE"] = np.maximum(df["BASE_PRICE"]*(1-0.75), 0)  # fallback

    # Daily → bucket aggregates
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

    # BUCKET_START actual date (Mon..Sun depending on kind)
    bs = []
    for _, r in agg.iterrows():
        s, _ = bucket_bounds(r["WEEK_START"], r["BUCKET_KIND"])
        bs.append(pd.to_datetime(s).normalize())
    agg["BUCKET_START"] = pd.to_datetime(bs)

    # availability proxy
    agg["AVAIL_STOCK"] = np.maximum(agg["START_STOCK"] + agg["DELIVERY_QTY"] - np.maximum(agg["QTY"],0), 0)

    # Sort & lags (within SKU×STORE×BUCKET_KIND for behavior; keep also cross-bucket info via MA12 over all buckets if desired)
    agg = agg.sort_values(["STORE","PRODUCT_CODE","BUCKET_START","BUCKET_KIND"])
    grp = agg.groupby(["STORE","PRODUCT_CODE","BUCKET_KIND"], group_keys=False)
    for col in ["QTY","PRICE","AVAIL_STOCK"]:
        agg[f"{col}_L1"]  = grp[col].shift(1)
        agg[f"{col}_MA4"] = grp[col].rolling(4, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
        agg[f"{col}_MA12"]= grp[col].rolling(12, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)

    # Attach hierarchy from raw df (first available per group)
    for lvl in CONFIG["hier_levels"]:
        if lvl in df.columns:
            map_lvl = df.groupby(["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"], as_index=False)[lvl].first()
            agg = agg.merge(map_lvl, on=["STORE","PRODUCT_CODE","WEEK_START","BUCKET_KIND"], how="left")

    return agg

# ----------------------------
# Demand model (CatBoost)
# ----------------------------
# Replace train_demand_model()
def train_demand_model(hist):
    use = hist.copy()

    # Fixed categorical list (use only those that exist)
    cat_cols_all = ["STORE","PRODUCT_CODE","BUCKET_KIND","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"]
    cat_cols = [c for c in cat_cols_all if c in use.columns]

    # Candidate numeric columns we expect to use
    num_candidates = [
        "PRICE","BASE_PRICE","COST_PRICE","AVAIL_STOCK","IS_PROMO_ACTIVE",
        # lags / MAs (may or may not exist yet)
        "QTY_L1","QTY_MA4","QTY_MA12",
        "PRICE_L1","PRICE_MA4","PRICE_MA12",
        "AVAIL_STOCK_L1","AVAIL_STOCK_MA4","AVAIL_STOCK_MA12"
    ]
    # plus any _L1/_MA4/_MA12 that were created beyond the above
    extra_lag_ma = [c for c in use.columns if any(c.endswith(suf) for suf in ["_L1","_MA4","_MA12"])]
    # keep unique order and only those present in data
    seen = set()
    num_cols = []
    for c in num_candidates + extra_lag_ma:
        if c in use.columns and c not in seen and c not in ["QTY","WEEK_START","BUCKET_START"]:
            num_cols.append(c); seen.add(c)

    y = use["QTY"].astype(float).values
    X = use[cat_cols + num_cols].copy()

    model = CatBoostRegressor(
        depth=6, learning_rate=0.08, n_estimators=400,
        random_seed=CONFIG["random_state"], loss_function="RMSE", verbose=False
    )
    train_pool = Pool(X, y, cat_features=list(range(len(cat_cols))))
    model.fit(train_pool)
    return model, cat_cols, num_cols

# Replace predict_qty()
def predict_qty(model, cat_cols, num_cols, dfX):
    X = dfX.copy()

    # Add any missing columns with safe defaults
    for c in cat_cols:
        if c not in X.columns:
            X[c] = "NA"

    for c in num_cols:
        if c not in X.columns:
            # sensible defaults
            if c == "IS_PROMO_ACTIVE":
                X[c] = 0
            elif c in ("PRICE","BASE_PRICE","COST_PRICE"):
                X[c] = np.nan  # will be filled per-candidate or left NaN (CatBoost can handle)
            else:
                X[c] = np.nan

    # Keep column order consistent
    X = X[cat_cols + num_cols]
    pool = Pool(X, cat_features=list(range(len(cat_cols))))
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
def rolling_eval(agg, bucket_kind, end_bucket_start, weeks=12):
    out = []
    to_eval = (agg.query("BUCKET_KIND == @bucket_kind and BUCKET_START < @end_bucket_start")
                 .drop_duplicates(subset=["WEEK_START","BUCKET_KIND"])
                 .sort_values("BUCKET_START"))[-weeks:]

    for _, r in to_eval.iterrows():
        target_start = r["BUCKET_START"]

        train_hist = agg[agg["BUCKET_START"] < target_start]
        if len(train_hist) < 50:
            continue

        model, cat_cols, num_cols = train_demand_model(train_hist)

        cur = build_bucket_snapshot(agg, bucket_kind, target_start).copy()
        if cur.empty:
            continue

        cur["LAST_ARM"] = (cur.get("LAST_PRICE", np.nan) / cur["BASE_PRICE"]).round(2)

        recs = []
        for _, row in cur.iterrows():
            arm, price, prof = select_arm(row, row.get("LAST_PRICE", np.nan), row.get("LAST_ARM", np.nan),
                                          model, cat_cols, num_cols)
            recs.append((arm, price, prof))
        if len(recs) == 0:
            continue

        cur["REC_ARM"], cur["REC_PRICE"], cur["REC_PROFIT_SIM"] = zip(*recs)

        # baseline
        cur["BASELINE_PRICE"] = np.where(cur.get("LAST_PRICE").notna(), cur["LAST_PRICE"], cur["BASE_PRICE"])
        baseX = cur.copy(); baseX["PRICE"] = baseX["BASELINE_PRICE"]
        base_qty = predict_qty(model, cat_cols, num_cols, baseX)
        base_served = np.minimum(base_qty, np.maximum(cur["AVAIL_STOCK"],0))
        cur["BASELINE_PROFIT_SIM"] = (baseX["PRICE"] - baseX["COST_PRICE"]) * base_served

        cur["UPLIFT_SIM"] = cur["REC_PROFIT_SIM"] - cur["BASELINE_PROFIT_SIM"]
        cur["WEEK_START_EVAL"] = cur["WEEK_START"]
        cur["BUCKET_START_EVAL"] = target_start

        out.append(cur[["STORE","PRODUCT_CODE","BUCKET_KIND","WEEK_START_EVAL","BUCKET_START_EVAL",
                        "REC_ARM","REC_PRICE","REC_PROFIT_SIM",
                        "BASELINE_PRICE","BASELINE_PROFIT_SIM","UPLIFT_SIM"]])

    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    kpi = (res.groupby("BUCKET_START_EVAL", as_index=False)
              .agg({"REC_PROFIT_SIM":"sum","BASELINE_PROFIT_SIM":"sum","UPLIFT_SIM":"sum"}))
    kpi["UPLIFT_%"] = 100 * (kpi["UPLIFT_SIM"] / np.maximum(kpi["BASELINE_PROFIT_SIM"],1e-6))
    return res, kpi


# ----------------------------
# One-shot recommendation for a chosen bucket (using info strictly before bucket start)
# ----------------------------
# Replace build_bucket_snapshot()
def build_bucket_snapshot(agg, bucket_kind, bucket_start):
    week_start = week_monday(bucket_start)

    # Actual rows for this bucket (if any)
    cur = agg[(agg["BUCKET_KIND"] == bucket_kind) & (agg["BUCKET_START"] == bucket_start)].copy()

    hist = agg[agg["BUCKET_START"] < bucket_start]
    if hist.empty:
        return cur  # nothing to synthesize from

    last = (hist.sort_values("BUCKET_START")
                 .groupby(["STORE","PRODUCT_CODE"]).tail(1).copy())

    skel = last[["STORE","PRODUCT_CODE"]].drop_duplicates().copy()
    skel["WEEK_START"]   = week_start
    skel["BUCKET_START"] = pd.to_datetime(bucket_start).normalize()
    skel["BUCKET_KIND"]  = bucket_kind

    # Bring last-known basics
    keep_cols = [
        "BASE_PRICE","COST_PRICE","PRICE","AVAIL_STOCK","START_STOCK","END_STOCK",
        "DELIVERY_QTY","QTY","IS_PROMO_ACTIVE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"
    ]
    # Also bring over any lag/MA columns present in history
    lag_ma_cols = [c for c in hist.columns if any(c.endswith(suf) for suf in ["_L1","_MA4","_MA12"])]
    keep_cols += lag_ma_cols

    for c in keep_cols:
        if c in last.columns:
            skel = skel.merge(last[["STORE","PRODUCT_CODE",c]], on=["STORE","PRODUCT_CODE"], how="left")

    # LAST_PRICE and defaults
    if "PRICE" in skel.columns:
        skel.rename(columns={"PRICE":"LAST_PRICE"}, inplace=True)
    if "BASE_PRICE" not in skel.columns: skel["BASE_PRICE"] = np.nan
    if "COST_PRICE" not in skel.columns: skel["COST_PRICE"] = np.nan
    if "IS_PROMO_ACTIVE" not in skel.columns: skel["IS_PROMO_ACTIVE"] = 0

    if not cur.empty:
        cur = cur.merge(skel[["STORE","PRODUCT_CODE","LAST_PRICE"]], on=["STORE","PRODUCT_CODE"], how="left")
        snap = cur
    else:
        snap = skel.copy()
        snap["QTY"] = snap.get("QTY", 0).fillna(0.0)
        # AVAIL_STOCK fallback if missing
        if "AVAIL_STOCK" not in snap.columns or snap["AVAIL_STOCK"].isna().all():
            snap["AVAIL_STOCK"] = np.maximum(
                snap.get("START_STOCK", 0).fillna(0)
                + snap.get("DELIVERY_QTY", 0).fillna(0)
                - snap.get("QTY", 0).fillna(0), 0
            )
    return snap



def recommend_for_bucket(agg, bucket_kind, bucket_start):
    train_hist = agg[agg["BUCKET_START"] < bucket_start]
    if len(train_hist) < 50:
        raise ValueError("Not enough history for training before this bucket_start.")

    model, cat_cols, num_cols = train_demand_model(train_hist)

    cur = build_bucket_snapshot(agg, bucket_kind, bucket_start).copy()
    if cur.empty:
        return cur  # nothing to recommend

    # LAST_ARM from LAST_PRICE if present
    cur["LAST_ARM"] = (cur.get("LAST_PRICE", np.nan) / cur["BASE_PRICE"]).round(2)

    recs = []
    for _, row in cur.iterrows():
        arm, price, prof = select_arm(row, row.get("LAST_PRICE", np.nan), row.get("LAST_ARM", np.nan),
                                      model, cat_cols, num_cols)
        recs.append((arm, price, prof))

    if len(recs) == 0:
        # return structure with NaNs instead of throwing
        cur["REC_ARM"] = np.nan
        cur["REC_PRICE"] = np.nan
        cur["REC_PROFIT_SIM"] = np.nan
        return cur[["STORE","PRODUCT_CODE","BUCKET_KIND","WEEK_START","BUCKET_START",
                    "BASE_PRICE","LAST_PRICE","REC_ARM","REC_PRICE","REC_PROFIT_SIM"]]

    cur["REC_ARM"], cur["REC_PRICE"], cur["REC_PROFIT_SIM"] = zip(*recs)
    return cur[["STORE","PRODUCT_CODE","BUCKET_KIND","WEEK_START","BUCKET_START",
                "BASE_PRICE","LAST_PRICE","REC_ARM","REC_PRICE","REC_PROFIT_SIM"]]


# ----------------------------
# Orchestration: run for 3 buckets & evaluate
# ----------------------------
def run_for_run_date(df, run_date_str):
    """
    Run the pipeline for a specific calendar date that is guaranteed to be Mon, Fri or Sat.
    Uses ALL history strictly before the start of that bucket (so Fri sees Mon–Thu; Sat–Sun sees Fri).
    Returns: {"rec": recommendation_df, "eval": (eval_items_df, eval_kpi_df)}
    """
    agg = prepare(df)
    run_date = pd.to_datetime(run_date_str).normalize()

    # determine bucket by actual date
    bucket_kind = which_bucket(run_date)
    # bucket start is the same calendar date for Fri and Sat, or Monday for Mon-Thu
    week_start = week_monday(run_date)
    bucket_start, _ = bucket_bounds(week_start, bucket_kind)

    eval_res = rolling_eval(agg, bucket_kind, bucket_start, weeks=CONFIG["eval_weeks"])
    rec = recommend_for_bucket(agg, bucket_kind, bucket_start)
    return {"bucket_kind": bucket_kind, "bucket_start": bucket_start, "rec": rec, "eval": eval_res}

# ----------------------------
# Usage:
# ----------------------------