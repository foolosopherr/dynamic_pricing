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
def add_features(df):
    keys = ["STORE","PRODUCT_CODE"]
    df = df.sort_values(keys+["TRADE_DT"])
    g = df.groupby(keys)

    # rolling demand
    df["qty7"]   = g["SALE_QTY"].transform(lambda s: s.rolling(7,min_periods=1).sum().shift(1))
    df["qty28"]  = g["SALE_QTY"].transform(lambda s: s.rolling(28,min_periods=1).sum().shift(1))
    df["qty84"]  = g["SALE_QTY"].transform(lambda s: s.rolling(84,min_periods=1).sum().shift(1))
    df["qty28_mean"] = g["SALE_QTY"].transform(lambda s: s.rolling(28,min_periods=1).mean().shift(1))
    df["qty28_std"]  = g["SALE_QTY"].transform(lambda s: s.rolling(28,min_periods=2).std().shift(1))

    # demand recency
    df["days_since_sale"] = g["SALE_QTY"].apply(lambda s: (s==0).astype(int).cumsum())

    # activity
    df["weeks_with_sales_12"] = (
        g["SALE_QTY"].transform(lambda s: s.rolling(84,min_periods=1).apply(lambda x: (x>0).sum(),raw=True).shift(1))
    )

    # promo
    df["discount"] = ((df["BASE_PRICE"] - df["SALE_PRICE"]) / 
                      df["BASE_PRICE"].replace(0,np.nan)).fillna(0.0)
    df["weeks_since_promo"] = g["IS_PROMO"].cumsum().where(df["IS_PROMO"]==0).fillna(0)
    df["promo_streak"] = g["IS_PROMO"].cumsum().where(df["IS_PROMO"]==1).fillna(0)

    # stock
    if "END_STOCK" in df:
        df["stock_cover"] = df["END_STOCK"] / df["qty7"].replace(0,1)
    else:
        df["stock_cover"] = 1.0

    # seasonality
    df["week_of_year"] = df["TRADE_DT"].dt.isocalendar().week.astype(int)

    return df

def build_features(row):
    return np.array([
        1.0,
        np.log1p(row.get("qty7",0.0)),
        np.log1p(row.get("qty28",0.0)),
        np.log1p(row.get("qty84",0.0)),
        (row.get("qty28_std",0.0)/(row.get("qty28_mean",1.0)+1e-6)),
        row.get("days_since_sale",0.0),
        row.get("weeks_with_sales_12",0.0)/12.0,
        float(row.get("IS_PROMO",0)),
        row.get("discount",0.0),
        row.get("weeks_since_promo",0.0),
        row.get("promo_streak",0.0),
        row.get("stock_cover",1.0),
        np.sin(2*np.pi*row.get("week_of_year",0)/52.0),
        np.cos(2*np.pi*row.get("week_of_year",0)/52.0),
        1.0 if row["DAY_BUCKET"]==1 else 0.0,
        1.0 if row["DAY_BUCKET"]==2 else 0.0,
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
def train_and_eval(df_daily):
    # ---------- prep + features (same as before) ----------
    df = prepare(df_daily)             # cuts at 2025-09-14, adds WEEK_START & DAY_BUCKET
    df = add_features(df)

    keys = ["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE","WEEK_START","DAY_BUCKET"]
    agg = (df.groupby(keys, as_index=False)
             .agg(BASE_PRICE=("BASE_PRICE","last"),
                  SALE_PRICE=("SALE_PRICE","last"),
                  PRODAJA_QTY=("PRODAJA_QTY","sum"),
                  IS_PROMO=("IS_PROMO","max"),
                  qty7=("qty7","last"),
                  qty28=("qty28","last"),
                  qty84=("qty84","last"),
                  qty28_mean=("qty28_mean","last"),
                  qty28_std=("qty28_std","last"),
                  days_since_sale=("days_since_sale","last"),
                  weeks_with_sales_12=("weeks_with_sales_12","last"),
                  discount=("discount","last"),
                  weeks_since_promo=("weeks_since_promo","last"),
                  promo_streak=("promo_streak","last"),
                  stock_cover=("stock_cover","last"),
                  week_of_year=("week_of_year","last")))

    # ---------- align weeks properly ----------
    history_end_ts = pd.Timestamp(HISTORY_END)              # Sun 2025-09-14
    last_week_start = _monday(history_end_ts)               # Mon 2025-09-08
    next_week_start = last_week_start + pd.Timedelta(days=7)  # Mon 2025-09-15

    # evaluation window: last 12 Mondays up to last_week_start
    eval_cutoff = last_week_start - pd.Timedelta(weeks=EVAL_WEEKS)

    train = agg[agg["WEEK_START"] < eval_cutoff].copy()
    eval_ = agg[(agg["WEEK_START"] >= eval_cutoff) & (agg["WEEK_START"] <= last_week_start)].copy()

    # if train is empty, use older rows as train; if eval is empty, shrink window
    if train.empty and not eval_.empty:
        # use earliest half of eval_ as train to warm-start
        mid_date = eval_["WEEK_START"].sort_values().unique()
        mid_date = mid_date[len(mid_date)//2] if len(mid_date)>0 else last_week_start
        train = agg[agg["WEEK_START"] < mid_date].copy()
        eval_  = agg[(agg["WEEK_START"] >= mid_date) & (agg["WEEK_START"] <= last_week_start)].copy()

    policy = HierLinUCB(arms=ARMS, alpha=1.0, min_n=10)

    # ---------- TRAIN (older weeks) ----------
    # use itertuples for speed and consistent types
    for row in train.itertuples(index=False):
        rd = row._asdict()
        X  = build_features(rd)
        # update using realized (historical) price & qty
        realized_price = rd["SALE_PRICE"]
        # guard against missing base price
        realized_price = float(realized_price) if pd.notnull(realized_price) else float(rd["BASE_PRICE"])
        policy.update(rd, X, realized_price, rd["PRODAJA_QTY"])

    # ---------- EVAL (last 12 weeks) ----------
    eval_rows = []
    if not eval_.empty:
        for row in eval_.sort_values(["WEEK_START","STORE","PRODUCT_CODE","DAY_BUCKET"]).itertuples(index=False):
            rd = row._asdict()
            X  = build_features(rd)
            # safe choose: fallback to baseline if node has no history
            a = policy.safe_choose(rd, X, fallback_arm=BASELINE_ARM_IDX)
            rec_price = ARMS[a] * max(rd["BASE_PRICE"], 1e-6)
            realized_price = rd["SALE_PRICE"] if rd["SALE_PRICE"]>0 else rec_price
            policy.update(rd, X, realized_price, rd["PRODAJA_QTY"])
            eval_rows.append({
                **{k: rd[k] for k in keys},
                "RECOMMENDED_PRICE": rec_price,
                "REALIZED_PRICE": realized_price,
                "REALIZED_QTY": rd["PRODAJA_QTY"],
                "ARM": int(a)
            })
    eval_df = pd.DataFrame(eval_rows)

    # ---------- PREDICT NEXT WEEK (Mon 2025-09-15 → Sun 2025-09-21) ----------
    # Base the next-week contexts on the latest observed week (last_week_start)
    base = agg[agg["WEEK_START"] == last_week_start].copy()
    next_rows = []
    if not base.empty:
        for row in base.sort_values(["STORE","PRODUCT_CODE","DAY_BUCKET"]).itertuples(index=False):
            rd = row._asdict()
            rd["WEEK_START"] = next_week_start   # roll forward a week
            # seasonal features: update week_of_year for next week
            rd["week_of_year"] = int((pd.Timestamp(rd["WEEK_START"]).isocalendar().week))
            X = build_features(rd)
            a = policy.safe_choose(rd, X, fallback_arm=BASELINE_ARM_IDX)
            rec_price = ARMS[a] * max(rd["BASE_PRICE"], 1e-6)
            next_rows.append({
                "STORE": rd["STORE"],
                "PRODUCT_CODE": rd["PRODUCT_CODE"],
                "FAMILY_CODE": rd["FAMILY_CODE"],
                "CATEGORY_CODE": rd["CATEGORY_CODE"],
                "SEGMENT_CODE": rd["SEGMENT_CODE"],
                "WEEK_START": next_week_start,
                "DAY_BUCKET": rd["DAY_BUCKET"],
                "RECOMMENDED_PRICE": rec_price,
                "ARM": int(a)
            })
    nextweek_df = pd.DataFrame(next_rows)

    return eval_df, nextweek_df

eval_df, nextweek_df = train_and_eval(df_daily)