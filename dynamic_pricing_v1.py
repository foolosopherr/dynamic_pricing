import numpy as np
import pandas as pd
from contextualbandits.online import LinUCB
from datetime import datetime

# ---------------- config ----------------
ARMS = np.array([0.90, 1.00, 1.10])
HISTORY_END = "2025-09-14"
EVAL_WEEKS = 12

# ---------------- helpers ----------------
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

# ---------------- feature engineering ----------------
def add_features(df, holiday_calendar=None):
    df = df.copy()

    df['SALE_QTY'] = df['SALE_QTY'].fillna(0) - df["SALE_QTY_ONLINE"].fillna(0)

    # base qtys
    df["SALE_QTY_TOTAL"] = df["SALE_QTY"] + df["SALE_QTY_ONLINE"].fillna(0)
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

# ---------------- model wrapper ----------------
class HierLinUCB:
    def __init__(self, arms=ARMS, alpha=1.0, min_n=10):
        self.arms = arms
        self.alpha = alpha
        self.min_n = min_n
        self.models = {}   # (level,value,bucket) -> (model, n_obs)
        self.cache = {}

    def _key_order(self,row):
        return [("PRODUCT_CODE",row["PRODUCT_CODE"]),
                ("FAMILY_CODE",row["FAMILY_CODE"]),
                ("CATEGORY_CODE",row["CATEGORY_CODE"]),
                ("SEGMENT_CODE",row["SEGMENT_CODE"]),
                ("__GLOBAL__","__GLOBAL__")]

    def _get_model(self, level, value, bucket):
        k=(level,value,bucket)
        if k not in self.models:
            self.models[k] = [LinUCB(nchoices=len(self.arms),
                                     alpha=self.alpha,
                                     fit_intercept=True,
                                     random_state=42),
                              0]
        return self.models[k]

    def _select_level(self,row,bucket):
        for level,value in self._key_order(row):
            model,nobs=self._get_model(level,value,bucket)
            if level=="PRODUCT_CODE" or nobs>=self.min_n or level=="__GLOBAL__":
                return level,value
        return "__GLOBAL__","__GLOBAL__"

    def choose(self,row,X):
        ckey=(row["STORE"],row["PRODUCT_CODE"],row["WEEK_START"],row["DAY_BUCKET"])
        if ckey in self.cache:
            return self.cache[ckey]
        level,value=self._select_level(row,row["DAY_BUCKET"])
        model,_=self._get_model(level,value,row["DAY_BUCKET"])
        a=int(np.asarray(model.predict(X.reshape(1,-1))).item())
        self.cache[ckey]=a
        return a

    def update(self,row,X,realized_price,realized_qty):
        level,value=self._select_level(row,row["DAY_BUCKET"])
        model,nobs=self._get_model(level,value,row["DAY_BUCKET"])
        a=np.argmin(np.abs(self.arms-realized_price/max(row["BASE_PRICE"],1e-6)))
        r=float(realized_price*realized_qty)
        model.fit(X.reshape(1,-1),
                  a=np.array([a],dtype=int),
                  r=np.array([r],dtype=float))
        self.models[(level,value,row["DAY_BUCKET"])][1]=nobs+1

    def safe_choose(self, row, X, fallback_arm=BASELINE_ARM_IDX):
        ckey = (row["STORE"], row["PRODUCT_CODE"], row["WEEK_START"], row["DAY_BUCKET"])
        if ckey in self.cache:
            return self.cache[ckey]
        level, value = self._select_level(row, row["DAY_BUCKET"])
        model, nobs = self._get_model(level, value, row["DAY_BUCKET"])
        if nobs == 0:
            a = fallback_arm
        else:
            a = int(np.asarray(model.predict(X.reshape(1,-1))).item())
        self.cache[ckey] = a
        return a

# ---------------- training & prediction ----------------
def train_and_eval(df_daily, arms=ARMS, alpha=1.0, min_n=10, report_last_weeks=12):
    """
    Rolling walk-forward:
      - sort weeks chronologically
      - for each week W: choose prices using model trained on weeks < W,
        log the recommendation & realized outcome, then update the model with week W
      - after the last history week (ending 2025-09-14), predict next week (2025-09-15..21)

    Returns:
      rolling_eval_df: per-bucket rolling eval across all historical weeks
      eval_lastN_df : last `report_last_weeks` weeks from rolling eval
      nextweek_df   : recommendations for 2025-09-15..2025-09-21 (Mon..Sun)
    """
    # ---------- prep + features ----------
    df = prepare(df_daily)        # must set TRADE_DT, WEEK_START (Mon), DAY_BUCKET
    df = add_features(df)

    # ---------- aggregate to decision buckets (KEEPING all fields) ----------
    keys = ["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE","WEEK_START","DAY_BUCKET"]
    agg = (df.groupby(keys, as_index=False)
             .agg(
                # core prices and qtys
                BASE_PRICE=("BASE_PRICE","last"),
                SALE_PRICE=("SALE_PRICE","last"),
                SALE_PRICE_ONLINE=("SALE_PRICE_ONLINE","last"),
                SALE_QTY=("SALE_QTY","sum"),
                SALE_QTY_ONLINE=("SALE_QTY_ONLINE","sum"),
                SALE_QTY_TOTAL=("SALE_QTY_TOTAL","sum"),
                NET_QTY=("NET_QTY","sum"),
                LOSS_QTY=("LOSS_QTY","sum"),
                RETURN_QTY=("RETURN_QTY","sum"),
                DELIVERY_QTY=("DELIVERY_QTY","sum"),
                START_STOCK=("START_STOCK","last"),
                END_STOCK=("END_STOCK","last"),

                # store-level meta
                REGION_NAME=("REGION_NAME","last"),
                STORE_TYPE=("STORE_TYPE","last"),
                PLACE_TYPE=("PLACE_TYPE","last"),

                # engineered (existing + extended)
                discount=("discount","last"),
                online_price_ratio=("online_price_ratio","last"),
                SALE_QTY_7=("SALE_QTY_7","last"),
                SALE_QTY_28=("SALE_QTY_28","last"),
                RETURN_RATIO=("RETURN_RATIO","last"),
                stock_cover=("stock_cover","last"),
                stock_change=("stock_change","last"),
                delivery_ratio=("delivery_ratio","last"),
                is_in_promo=("IS_PROMO","max"),
                is_in_promo_period=("is_in_promo_period","max"),
                competitor_on_promo=("competitor_on_promo","mean"),
                week_of_year=("week_of_year","last"),
                month=("month","last"),
                is_holiday=("is_holiday","max"),

                # NEW lifecycle & assortment
                weeks_since_first_seen=("weeks_since_first_seen","last"),
                family_assortment_width=("family_assortment_width","last"),
                active_siblings=("active_siblings","last"),

                # NEW price gaps & z-scores
                price_gap_vs_family_med=("price_gap_vs_family_med","last"),
                price_gap_vs_family_min=("price_gap_vs_family_min","last"),
                zprice_family=("zprice_family","last"),
                zprice_category=("zprice_category","last"),

                # NEW momentum
                fam_momentum=("fam_momentum","last"),
                cat_momentum=("cat_momentum","last"),

                # NEW charm pricing
                price_ends_99=("price_ends_99","max"),
                price_ends_95=("price_ends_95","max"),
                price_ends_49=("price_ends_49","max"),

                # NEW promo timing within period
                promo_days_since_start=("promo_days_since_start","last"),
                promo_days_until_end=("promo_days_until_end","last"),
                promo_is_mid=("promo_is_mid","max"),

                # NEW stockout risk
                stockout_risk=("stockout_risk","last"),
             ))

    # ---------- rolling walk-forward ----------
    policy = HierLinUCB(arms=arms, alpha=alpha, min_n=min_n)

    week_list = np.sort(agg["WEEK_START"].unique())
    rolling_rows = []

    # walk-forward over all available weeks
    for wk in week_list:
        wk_rows = agg[agg["WEEK_START"] == wk].sort_values(["STORE","PRODUCT_CODE","DAY_BUCKET"])
        for _, row in wk_rows.iterrows():
            rd = row.to_dict()
            X  = build_features(rd)

            # choose price for this week using knowledge from previous weeks
            arm = policy.safe_choose(rd, X, fallback_arm=BASELINE_ARM_IDX)
            rec_price = float(arms[arm] * max(rd["BASE_PRICE"], 1e-6))

            # "realized" for offline eval: use historical sale price if present, else our rec
            realized_price = float(rd["SALE_PRICE"]) if pd.notnull(rd["SALE_PRICE"]) and rd["SALE_PRICE"] > 0 else rec_price
            realized_qty   = float(rd["SALE_QTY_TOTAL"]) if pd.notnull(rd["SALE_QTY_TOTAL"]) else 0.0

            # log evaluation row
            rolling_rows.append({
                **{k: rd[k] for k in keys},
                "RECOMMENDED_PRICE": rec_price,
                "ARM": int(arm),
                "REALIZED_PRICE": realized_price,
                "REALIZED_QTY": realized_qty,
                "BASE_PRICE": rd["BASE_PRICE"],
                "DISCOUNT": rd.get("discount", 0.0),
            })

            # update model with what actually happened this week
            policy.update(rd, X, realized_price=realized_price, realized_qty=realized_qty)

    rolling_eval_df = pd.DataFrame(rolling_rows)

    # ---------- slice last N weeks for reporting convenience ----------
    if not rolling_eval_df.empty:
        weeks_sorted = np.sort(rolling_eval_df["WEEK_START"].unique())
        last_weeks = set(weeks_sorted[-min(report_last_weeks, len(weeks_sorted)):])
        eval_lastN_df = rolling_eval_df[rolling_eval_df["WEEK_START"].isin(last_weeks)].copy()
    else:
        eval_lastN_df = rolling_eval_df.copy()

    # ---------- predict NEXT WEEK (Mon after last history week) ----------
    # last history week start:
    if len(week_list) > 0:
        last_week_start = pd.Timestamp(week_list.max())
        next_week_start = last_week_start + pd.Timedelta(days=7)
    else:
        # if no data, default to Monday after HISTORY_END
        last_week_start = (pd.Timestamp(HISTORY_END) - pd.to_timedelta(pd.Timestamp(HISTORY_END).weekday(), unit="D"))
        next_week_start = last_week_start + pd.Timedelta(days=7)

    base_next = agg[agg["WEEK_START"] == last_week_start].copy()
    next_rows = []
    if not base_next.empty:
        for _, row in base_next.sort_values(["STORE","PRODUCT_CODE","DAY_BUCKET"]).iterrows():
            rd = row.to_dict()
            rd["WEEK_START"] = next_week_start
            # update seasonal week number to next week
            rd["week_of_year"] = int(next_week_start.isocalendar().week)
            X = build_features(rd)
            arm = policy.safe_choose(rd, X, fallback_arm=BASELINE_ARM_IDX)
            rec_price = float(arms[arm] * max(rd["BASE_PRICE"], 1e-6))
            next_rows.append({
                "STORE": rd["STORE"],
                "PRODUCT_CODE": rd["PRODUCT_CODE"],
                "FAMILY_CODE": rd["FAMILY_CODE"],
                "CATEGORY_CODE": rd["CATEGORY_CODE"],
                "SEGMENT_CODE": rd["SEGMENT_CODE"],
                "WEEK_START": next_week_start,
                "DAY_BUCKET": rd["DAY_BUCKET"],
                "RECOMMENDED_PRICE": rec_price,
                "ARM": int(arm),
                "BASE_PRICE": rd["BASE_PRICE"]
            })
    nextweek_df = pd.DataFrame(next_rows)

    return rolling_eval_df, eval_lastN_df, nextweek_df




eval_df, nextweek_df = train_and_eval(df_daily)