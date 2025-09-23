import numpy as np
import pandas as pd
from contextualbandits.online import LinUCB
from tqdm.auto import tqdm

# ---------------- CONFIG ----------------
ARMS = np.array([0.90, 1.00, 1.10])
HISTORY_END = "2025-09-14"
BASELINE_ARM_IDX = int(np.argmin(np.abs(ARMS - 1.00)))


# ---------------- HELPERS ----------------
def day_bucket(ts: pd.Timestamp) -> int:
    wd = ts.weekday()
    if wd <= 3: return 0   # Mon–Thu
    if wd == 4: return 1   # Fri
    return 2               # Sat–Sun


def prepare(df):
    df = df.copy()
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"])
    df = df[df["TRADE_DT"] <= pd.Timestamp(HISTORY_END)]
    df["DAY_BUCKET"] = df["TRADE_DT"].apply(day_bucket)
    df["WEEK_START"] = (df["TRADE_DT"] -
                        pd.to_timedelta(df["TRADE_DT"].dt.weekday, unit="D")).dt.normalize()
    return df


# ---------------- FEATURE ENGINEERING ----------------
def add_features(df, holiday_calendar=None):
    df = df.copy()

    # base qtys
    df["SALE_QTY_TOTAL"] = df["SALE_QTY"].fillna(0) + df["SALE_QTY_ONLINE"].fillna(0)
    df["NET_QTY"] = df["SALE_QTY_TOTAL"] - df["LOSS_QTY"].fillna(0) + df["RETURN_QTY"].fillna(0)

    # sorting & groups
    df = df.sort_values(["STORE","PRODUCT_CODE","TRADE_DT"])
    gp_sp = df.groupby(["STORE","PRODUCT_CODE"])

    # rolling demand
    df["SALE_QTY_7"]  = gp_sp["SALE_QTY_TOTAL"].transform(lambda s: s.rolling(7,  min_periods=1).sum().shift(1))
    df["SALE_QTY_28"] = gp_sp["SALE_QTY_TOTAL"].transform(lambda s: s.rolling(28, min_periods=1).sum().shift(1))

    # returns
    df["RETURN_RATIO"] = (df["RETURN_QTY"].fillna(0) /
                          df["SALE_QTY_TOTAL"].replace(0, np.nan)).fillna(0)

    # stock
    avg_7 = (df["SALE_QTY_7"]/7).replace(0, np.nan)
    df["stock_cover"]   = (df["END_STOCK"].replace(0, np.nan) / avg_7).fillna(1.0)
    df["stock_change"]  = df["END_STOCK"] - df["START_STOCK"]
    df["delivery_ratio"] = (df["DELIVERY_QTY"].fillna(0) /
                            df["SALE_QTY_TOTAL"].replace(0, np.nan)).fillna(0)

    # price
    df["discount"] = ((df["BASE_PRICE"] - df["SALE_PRICE"]) /
                      df["BASE_PRICE"].replace(0, np.nan)).fillna(0)
    df["online_price_ratio"] = (
        df["SALE_PRICE_ONLINE"].fillna(df["SALE_PRICE"]) /
        df["SALE_PRICE"].replace(0, np.nan)
    ).fillna(1.0)

    # promo period parse
    def _parse_promo_period(s):
        if not s or pd.isna(s): return None
        try:
            a, b = s.split("-")
            start = pd.to_datetime(a.strip(), dayfirst=True, errors="coerce")
            end   = pd.to_datetime(b.strip(), dayfirst=True, errors="coerce")
            if pd.isna(start) or pd.isna(end): return None
            return (start, end)
        except Exception:
            return None

    def _in_promo_period(row):
        p = _parse_promo_period(row.get("PROMO_PERIOD"))
        if not p: return 0
        start, end = p
        return int(start <= row["TRADE_DT"] <= end)

    df["is_in_promo_period"] = df.apply(_in_promo_period, axis=1)

    # cannibalization proxy
    gp_sfd = df.groupby(["STORE","FAMILY_CODE","TRADE_DT"])
    df["competitor_on_promo"] = gp_sfd["is_in_promo_period"].transform("mean")
    df["sibling_promo_overlap"] = gp_sfd["is_in_promo_period"].transform("sum") - df["is_in_promo_period"]

    # seasonality
    df["week_of_year"] = df["TRADE_DT"].dt.isocalendar().week.astype(int)
    df["month"] = df["TRADE_DT"].dt.month
    df["is_holiday"] = 0 if holiday_calendar is None else df["TRADE_DT"].isin(holiday_calendar).astype(int)

    # lifecycle
    first_seen = gp_sp["TRADE_DT"].transform("min")
    df["days_since_first_seen"]  = (df["TRADE_DT"] - first_seen).dt.days.clip(lower=0)
    df["weeks_since_first_seen"] = df["days_since_first_seen"] / 7.0

    # assortment width / active siblings
    df["family_assortment_width"] = gp_sfd["PRODUCT_CODE"].transform("nunique")
    df["active_siblings"] = (gp_sfd["PRODUCT_CODE"].transform("count") - 1).clip(lower=0)

    # sibling price gaps
    df["fam_price_median"] = gp_sfd["SALE_PRICE"].transform("median")
    df["fam_price_min"]    = gp_sfd["SALE_PRICE"].transform("min")
    df["price_gap_vs_family_med"] = (
        (df["SALE_PRICE"] - df["fam_price_median"]) /
        df["fam_price_median"].replace(0, np.nan)
    ).fillna(0)
    df["price_gap_vs_family_min"] = (
        (df["SALE_PRICE"] - df["fam_price_min"]) /
        df["fam_price_min"].replace(0, np.nan)
    ).fillna(0)

    # momentum (family/category)
    fam_daily = (df.groupby(["STORE","FAMILY_CODE","TRADE_DT"], as_index=False)
                   .agg(fam_qty=("SALE_QTY_TOTAL","sum")))
    fam_daily = fam_daily.sort_values(["STORE","FAMILY_CODE","TRADE_DT"])
    fam_daily["fam_qty_7"]  = fam_daily.groupby(["STORE","FAMILY_CODE"])["fam_qty"]\
                                     .transform(lambda s: s.rolling(7,  min_periods=1).sum().shift(1))
    fam_daily["fam_qty_28"] = fam_daily.groupby(["STORE","FAMILY_CODE"])["fam_qty"]\
                                     .transform(lambda s: s.rolling(28, min_periods=1).sum().shift(1))

    cat_daily = (df.groupby(["STORE","CATEGORY_CODE","TRADE_DT"], as_index=False)
                   .agg(cat_qty=("SALE_QTY_TOTAL","sum")))
    cat_daily = cat_daily.sort_values(["STORE","CATEGORY_CODE","TRADE_DT"])
    cat_daily["cat_qty_7"]  = cat_daily.groupby(["STORE","CATEGORY_CODE"])["cat_qty"]\
                                     .transform(lambda s: s.rolling(7,  min_periods=1).sum().shift(1))
    cat_daily["cat_qty_28"] = cat_daily.groupby(["STORE","CATEGORY_CODE"])["cat_qty"]\
                                     .transform(lambda s: s.rolling(28, min_periods=1).sum().shift(1))

    df = df.merge(
        fam_daily[["STORE","FAMILY_CODE","TRADE_DT","fam_qty_7","fam_qty_28"]],
        on=["STORE","FAMILY_CODE","TRADE_DT"], how="left", validate="many_to_one"
    ).merge(
        cat_daily[["STORE","CATEGORY_CODE","TRADE_DT","cat_qty_7","cat_qty_28"]],
        on=["STORE","CATEGORY_CODE","TRADE_DT"], how="left", validate="many_to_one"
    )

    df["fam_momentum"] = (df["fam_qty_7"] / df["fam_qty_28"].replace(0, np.nan)).fillna(0)
    df["cat_momentum"] = (df["cat_qty_7"] / df["cat_qty_28"].replace(0, np.nan)).fillna(0)

    # charm pricing
    df["price_ends_99"] = ((df["SALE_PRICE"]*100 % 100).round(0) == 99).astype(int)
    df["price_ends_95"] = ((df["SALE_PRICE"]*100 % 100).round(0) == 95).astype(int)
    df["price_ends_49"] = ((df["SALE_PRICE"]*100 % 100).round(0) == 49).astype(int)

    # z-scores
    df["zprice_family"] = (
        (df["SALE_PRICE"] - gp_sfd["SALE_PRICE"].transform("mean")) /
         gp_sfd["SALE_PRICE"].transform("std").replace(0, np.nan)
    ).fillna(0)
    gp_scd = df.groupby(["STORE","CATEGORY_CODE","TRADE_DT"])
    df["zprice_category"] = (
        (df["SALE_PRICE"] - gp_scd["SALE_PRICE"].transform("mean")) /
         gp_scd["SALE_PRICE"].transform("std").replace(0, np.nan)
    ).fillna(0)

    # promo timing
    def _promo_timing(r):
        p = _parse_promo_period(r.get("PROMO_PERIOD"))
        if not p: return (0, 0, 0)
        start, end = p
        if r["TRADE_DT"] < start: return (0, 0, 0)
        if r["TRADE_DT"] > end:   return ((end-start).days+1, 0, 0)
        return ((r["TRADE_DT"]-start).days, (end-r["TRADE_DT"]).days, 1)

    tt = df.apply(_promo_timing, axis=1, result_type="expand")
    tt.columns = ["promo_days_since_start","promo_days_until_end","promo_is_mid"]
    df = pd.concat([df, tt], axis=1)
    df["promo_is_mid"] = df["promo_is_mid"].fillna(0).astype(int)

    # stockout risk
    exp7 = (df["SALE_QTY_7"]/7.0).fillna(0)
    x = (df["END_STOCK"].fillna(0) - 1.5*exp7)
    df["stockout_risk"] = 1.0 / (1.0 + np.exp(0.2 * x))

    return df


# ---------------- CONTEXT VECTOR ----------------
def build_features(row):
    return np.array([
        1.0,
        # demand & returns
        np.log1p(row.get("SALE_QTY_7",0.0)),
        np.log1p(row.get("SALE_QTY_28",0.0)),
        row.get("RETURN_RATIO",0.0),
        row.get("NET_QTY",0.0),
        # stock & delivery
        row.get("stock_cover",1.0),
        row.get("stock_change",0.0),
        row.get("delivery_ratio",0.0),
        # price & discount
        row.get("discount",0.0),
        row.get("online_price_ratio",1.0),
        np.log1p(row.get("SALE_PRICE", row.get("BASE_PRICE",1.0))),
        row.get("SALE_PRICE",0)/max(row.get("BASE_PRICE",1.0),1e-6),
        # promo & cannibalization
        float(row.get("is_in_promo",0)),
        float(row.get("is_in_promo_period",0)),
        row.get("competitor_on_promo",0.0),
        row.get("sibling_promo_overlap",0.0),
        # hashed cats
        (hash(str(row.get("REGION_NAME",""))) % 100) / 100.0,
        (hash(str(row.get("STORE_TYPE",""))) % 50) / 50.0,
        (hash(str(row.get("PLACE_TYPE",""))) % 20) / 20.0,
        # seasonality
        np.sin(2*np.pi*row.get("week_of_year",0)/52.0),
        np.cos(2*np.pi*row.get("week_of_year",0)/52.0),
        float(row.get("is_holiday",0)),
        float(row.get("month",0))/12.0,
        # buckets
        1.0 if row["DAY_BUCKET"]==1 else 0.0,
        1.0 if row["DAY_BUCKET"]==2 else 0.0,
        # lifecycle & assortment
        row.get("weeks_since_first_seen",0.0),
        row.get("family_assortment_width",0.0),
        row.get("active_siblings",0.0),
        # sibling price gaps
        row.get("price_gap_vs_family_med",0.0),
        row.get("price_gap_vs_family_min",0.0),
        # momentum
        row.get("fam_momentum",0.0),
        row.get("cat_momentum",0.0),
        # charm pricing
        float(row.get("price_ends_99",0)),
        float(row.get("price_ends_95",0)),
        float(row.get("price_ends_49",0)),
        # z-scores
        row.get("zprice_family",0.0),
        row.get("zprice_category",0.0),
        # promo timing
        row.get("promo_days_since_start",0.0),
        row.get("promo_days_until_end",0.0),
        float(row.get("promo_is_mid",0)),
        # OOS risk
        row.get("stockout_risk",0.0),
    ], dtype=float)


# ---------------- TRAIN & EVAL ----------------
def train_and_eval(df_daily, horizon_weeks=12, use_progress=False):
    """
    Train contextual bandit on history, evaluate on last N weeks,
    and produce next-week price recommendations.
    """
    # --- Prepare and feature-engineer ---
    df = prepare(df_daily)
    df = add_features(df)
    agg = df.groupby(
        ["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE","WEEK_START","DAY_BUCKET"],
        as_index=False
    ).agg(
        BASE_PRICE=("BASE_PRICE","last"),
        SALE_PRICE=("SALE_PRICE","last"),
        SALE_QTY_TOTAL=("SALE_QTY_TOTAL","sum"),
        IS_PROMO=("IS_PROMO","max"),
        # keep engineered fields
        discount=("discount","last"),
        online_price_ratio=("online_price_ratio","last"),
        SALE_QTY_7=("SALE_QTY_7","last"),
        SALE_QTY_28=("SALE_QTY_28","last"),
        RETURN_RATIO=("RETURN_RATIO","last"),
        stock_cover=("stock_cover","last"),
        stock_change=("stock_change","last"),
        delivery_ratio=("delivery_ratio","last"),
        is_in_promo=("is_in_promo","max"),
        is_in_promo_period=("is_in_promo_period","max"),
        competitor_on_promo=("competitor_on_promo","mean"),
        sibling_promo_overlap=("sibling_promo_overlap","mean"),
        week_of_year=("week_of_year","last"),
        month=("month","last"),
        is_holiday=("is_holiday","max"),
        weeks_since_first_seen=("weeks_since_first_seen","last"),
        family_assortment_width=("family_assortment_width","last"),
        active_siblings=("active_siblings","last"),
        price_gap_vs_family_med=("price_gap_vs_family_med","last"),
        price_gap_vs_family_min=("price_gap_vs_family_min","last"),
        zprice_family=("zprice_family","last"),
        zprice_category=("zprice_category","last"),
        fam_momentum=("fam_momentum","last"),
        cat_momentum=("cat_momentum","last"),
        price_ends_99=("price_ends_99","max"),
        price_ends_95=("price_ends_95","max"),
        price_ends_49=("price_ends_49","max"),
        promo_days_since_start=("promo_days_since_start","last"),
        promo_days_until_end=("promo_days_until_end","last"),
        promo_is_mid=("promo_is_mid","max"),
        stockout_risk=("stockout_risk","last"),
    )

    # --- Split history ---
    max_week = agg["WEEK_START"].max()
    eval_cutoff = max_week - pd.Timedelta(weeks=horizon_weeks)

    train_df = agg[agg["WEEK_START"] < eval_cutoff]
    eval_df  = agg[(agg["WEEK_START"] >= eval_cutoff) & (agg["WEEK_START"] <= max_week)]

    # --- Train model ---
    model = LinUCB(nchoices=len(ARMS), alpha=1.0, fit_intercept=True, random_state=42)

        # --- Train model ---
    model = LinUCB(nchoices=len(ARMS), alpha=1.0, fit_intercept=True, random_state=42)

    for _, row in train_df.iterrows():
        X = build_features(row).reshape(1, -1)

        # observed sales at baseline price
        base_price = row["SALE_PRICE"]
        base_sales = row["SALE_QTY_TOTAL"]

        # for each arm, simulate reward using simple price–demand elasticity assumption
        for arm_idx, mult in enumerate(ARMS):
            price = row["BASE_PRICE"] * mult

            # elasticity proxy: assume demand changes inversely with price ratio
            demand = base_sales * (base_price / price) if price > 0 else base_sales

            # reward: revenue = price × demand
            reward = price * demand

            model.fit(X, np.array([arm_idx]), np.array([reward]))


    # --- Evaluate last horizon_weeks ---
    eval_results = []
    for _, row in eval_df.iterrows():
        X = build_features(row).reshape(1, -1)
        arm = model.predict(X)[0]
        price = row["BASE_PRICE"] * ARMS[arm]
        eval_results.append({
            "STORE": row["STORE"],
            "PRODUCT_CODE": row["PRODUCT_CODE"],
            "WEEK_START": row["WEEK_START"],
            "DAY_BUCKET": row["DAY_BUCKET"],
            "BASE_PRICE": row["BASE_PRICE"],
            "CHOSEN_ARM": arm,
            "CHOSEN_PRICE": price,
            "ACTUAL_SALES": row["SALE_QTY_TOTAL"],
        })
    eval_results = pd.DataFrame(eval_results)

    # --- Next-week prediction ---
    next_week = max_week + pd.Timedelta(weeks=1)
    future = agg[agg["WEEK_START"] == max_week].copy()
    future["WEEK_START"] = next_week

    preds = []
    for _, row in future.iterrows():
        X = build_features(row).reshape(1, -1)
        arm = model.predict(X)[0]
        price = row["BASE_PRICE"] * ARMS[arm]
        preds.append({
            "STORE": row["STORE"],
            "PRODUCT_CODE": row["PRODUCT_CODE"],
            "WEEK_START": row["WEEK_START"],
            "DAY_BUCKET": row["DAY_BUCKET"],
            "RECOMM_PRICE": price,
            "CHOSEN_ARM": arm,
        })
    next_week_results = pd.DataFrame(preds)

    return eval_results, next_week_results
