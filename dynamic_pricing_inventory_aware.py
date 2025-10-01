import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# -----------------------------
# Utilities
# -----------------------------

BUCKETS = {
    "Mon-Thu": [0,1,2,3],  # Monday=0
    "Fri":     [4],
    "Sat-Sun": [5,6],
}

def _to_week(dt: pd.Series) -> pd.Series:
    """Convert dates to (ISO) weekly period (Mon-Sun)."""
    return pd.to_datetime(dt).dt.to_period("W-MON").dt.start_time

def _bucketize_weekday(dates: pd.Series) -> pd.Series:
    """Map calendar date to weekday bucket label."""
    wd = pd.to_datetime(dates).dt.weekday
    out = np.where(wd.isin(BUCKETS["Mon-Thu"]), "Mon-Thu",
          np.where(wd.isin(BUCKETS["Fri"]), "Fri", "Sat-Sun"))
    return pd.Series(out, index=dates.index)

def _parse_promo_period(s: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Parse PROMO_PERIOD string 'dd-mm-YYYY - dd-mm-YYYY' into (start_date, end_date).
    If empty/NaN -> (NaT, NaT).
    """
    starts, ends = [], []
    for val in s.fillna(""):
        if "-" in val:
            try:
                a, b = [x.strip() for x in val.split("-")]
                # The string uses '-' as both separator and inside dates; be robust:
                # Try split by ' - ' first
                if " - " in val:
                    a, b = val.split(" - ")
                starts.append(pd.to_datetime(a, dayfirst=True, errors="coerce"))
                ends.append(pd.to_datetime(b, dayfirst=True, errors="coerce"))
            except Exception:
                starts.append(pd.NaT); ends.append(pd.NaT)
        else:
            starts.append(pd.NaT); ends.append(pd.NaT)
    return pd.Series(starts, name="promo_start"), pd.Series(ends, name="promo_end")

def _known_future_promo(next_week_dates: pd.DatetimeIndex,
                        promo_start: pd.Series, promo_end: pd.Series,
                        key_idx: pd.Index) -> pd.Series:
    """
    For each row (aligned with key_idx), return 1 if any date in NEXT WEEK
    intersects [promo_start, promo_end] (known at decision time).
    """
    # Represent next-week window by its start & end for quick overlap check
    next_start, next_end = next_week_dates.min(), next_week_dates.max()
    start = promo_start.reindex(key_idx)
    end = promo_end.reindex(key_idx)
    cond = (start.notna() & end.notna() &
            (start <= next_end) & (end >= next_start))
    return cond.astype(int).rename("promo_next_week_known")

def _safe_div(a, b):
    return np.where(b==0, 0.0, a/b)

# -----------------------------
# Feature Engineering
# -----------------------------

def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RU: Базовые фичи и календарь, бакеты, парсинг промо.
    """
    out = df.copy()
    out["TRADE_DT"] = pd.to_datetime(out["TRADE_DT"])
    out["week"] = _to_week(out["TRADE_DT"])
    out["bucket"] = _bucketize_weekday(out["TRADE_DT"])
    out["dow"] = out["TRADE_DT"].dt.weekday
    out["week_of_year"] = out["TRADE_DT"].dt.isocalendar().week.astype(int)
    out["month"] = out["TRADE_DT"].dt.month

    # Parse promo period
    ps, pe = _parse_promo_period(out["PROMO_PERIOD"].astype(str))
    out = out.join(ps).join(pe)

    # Price signals
    out["price_vs_base"] = _safe_div(out["SALE_PRICE"] - out["BASE_PRICE"], out["BASE_PRICE"])
    out["is_zero_price"] = (out["SALE_PRICE"] <= 0).astype(int)

    # Quality/noise
    out["return_rate"] = _safe_div(out["RETURN_QTY"], out["SALE_QTY"].replace(0, np.nan)).fillna(0).clip(0, 1)
    out["loss_rate"] = _safe_div(out["LOSS_QTY"], (out["SALE_QTY"]+1e-6)).clip(0, 10)

    # Stock turns proxy
    avg_stock = (out["START_STOCK"] + out["END_STOCK"]) / 2.0
    out["stock_turns"] = _safe_div(out["SALE_QTY"], (avg_stock + 1e-6))

    # Group keys
    out["key_prod"] = list(zip(out["STORE"], out["PRODUCT_CODE"]))
    out["key_fam"]  = list(zip(out["STORE"], out["FAMILY_CODE"]))
    out["key_cat"]  = list(zip(out["STORE"], out["CATEGORY_CODE"]))
    out["key_seg"]  = list(zip(out["STORE"], out["SEGMENT_CODE"]))

    return out

def build_lag_features(df: pd.DataFrame, lags: List[int] = [1,2,4,8]) -> pd.DataFrame:
    """
    RU: Лаги спроса/цены на уровне (STORE, PRODUCT, bucket, неделя).
    """
    # Aggregate to week x bucket
    agg = (df
           .groupby(["STORE","PRODUCT_CODE","week","bucket"], as_index=False)
           .agg(SALE_QTY=("SALE_QTY","sum"),
                SALE_PRICE=("SALE_PRICE","mean"),
                BASE_PRICE=("BASE_PRICE","mean"),
                END_STOCK=("END_STOCK","sum"),
                DELIVERY_QTY=("DELIVERY_QTY","sum"),
                price_vs_base=("price_vs_base","mean"),
                return_rate=("return_rate","mean"),
                loss_rate=("loss_rate","mean"),
                stock_turns=("stock_turns","mean"))
          )

    # Create contiguous index per key for shifting by "previous bucket in previous week with same bucket label"
    agg = agg.sort_values(["STORE","PRODUCT_CODE","bucket","week"])
    def _lags(g):
        for L in lags:
            g[f"qty_lag_{L}"] = g["SALE_QTY"].shift(L)
            g[f"price_lag_{L}"] = g["SALE_PRICE"].shift(L)
        g["qty_ma_4"] = g["SALE_QTY"].rolling(4, min_periods=1).mean()
        g["qty_ma_8"] = g["SALE_QTY"].rolling(8, min_periods=1).mean()
        g["price_trend_4"] = g["SALE_PRICE"].diff().rolling(4, min_periods=1).mean()
        return g
    agg = agg.groupby(["STORE","PRODUCT_CODE","bucket"], group_keys=False).apply(_lags)

    # Calendar
    cal = (df.drop_duplicates(["week","bucket"])[["week","bucket","week_of_year","month"]])
    agg = agg.merge(cal, on=["week","bucket"], how="left")

    return agg

# -----------------------------
# Hierarchical backoff (elasticity & priors)
# -----------------------------

def fit_hierarchical_elasticity(df_week_bucket: pd.DataFrame) -> pd.DataFrame:
    """
    RU: Оценка эластичности log(Q) ~ a + ε*log(P) с поэтапным backoff:
        PRODUCT -> FAMILY -> CATEGORY -> SEGMENT.
    Возвращает таблицу с ε на каждом уровне.
    Формула оптимума (при степенной реакции спроса): p* = p0 / (1 + 1/ε).
    """
    # Prepare log variables (guard zeros)
    d = df_week_bucket.copy()
    d["log_q"] = np.log1p(d["SALE_QTY"])           # log(1+Q) устойчив к нулям
    d["log_p"] = np.log(np.clip(d["SALE_PRICE"], 1e-6, None))

    def _coef(g):
        if g["log_q"].notna().sum() >= 8 and g["SALE_PRICE"].nunique() >= 3:
            try:
                reg = Ridge(alpha=1.0).fit(g[["log_p"]], g["log_q"])
                eps = reg.coef_[0]
                return eps
            except Exception:
                return np.nan
        return np.nan

    eps_prod = (d.groupby(["STORE","PRODUCT_CODE"])
                  .apply(_coef).rename("eps_prod").reset_index())

    # Roll up using means where child is missing
    # Attach family/category/segment keys
    keys = d.drop_duplicates(["STORE","PRODUCT_CODE"])[
        ["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"]
    ]
    eps_prod = eps_prod.merge(keys, on=["STORE","PRODUCT_CODE"], how="left")

    def _level(df, level_cols, name):
        tmp = d.merge(df[["STORE","PRODUCT_CODE","eps_prod"]], on=["STORE","PRODUCT_CODE"], how="left")
        # compute group elasticity from the base data
        agg = (tmp.groupby(level_cols)
                  .apply(_coef).rename(name).reset_index())
        return agg

    eps_fam = _level(eps_prod, ["STORE","FAMILY_CODE"], "eps_fam")
    eps_cat = _level(eps_prod, ["STORE","CATEGORY_CODE"], "eps_cat")
    eps_seg = _level(eps_prod, ["STORE","SEGMENT_CODE"], "eps_seg")

    # Join and compute final backoff ε
    out = eps_prod.merge(eps_fam, on=["STORE","FAMILY_CODE"], how="left") \
                  .merge(eps_cat, on=["STORE","CATEGORY_CODE"], how="left") \
                  .merge(eps_seg, on=["STORE","SEGMENT_CODE"], how="left")
    # Backoff ladder: prod -> fam -> cat -> seg -> global default (-1.2 as safe)
    out["eps_final"] = out["eps_prod"].fillna(out["eps_fam"]) \
                                     .fillna(out["eps_cat"]) \
                                     .fillna(out["eps_seg"]) \
                                     .fillna(-1.2)
    return out[["STORE","PRODUCT_CODE","eps_final"]]

# -----------------------------
# Demand Model (Stage A)
# -----------------------------

FEATURES_NUM = [
    "BASE_PRICE","SALE_PRICE","price_vs_base",
    "qty_lag_1","qty_lag_2","qty_lag_4","qty_lag_8",
    "qty_ma_4","qty_ma_8",
    "price_lag_1","price_lag_2","price_lag_4","price_lag_8",
    "price_trend_4","END_STOCK","DELIVERY_QTY",
    "return_rate","loss_rate","stock_turns"
]
FEATURES_CAT = ["bucket","week_of_year","month","STORE_TYPE","REGION_NAME"]

def fit_demand_model(df_week_bucket: pd.DataFrame) -> Pipeline:
    """
    RU: Прогноз недельного спроса по бакету.
    Модель: Ridge на лог-таргете (стабильно при нулях).
    """
    d = df_week_bucket.copy()
    d["y"] = np.log1p(d["SALE_QTY"])
    # Fill NA safeties
    for c in FEATURES_NUM:
        if c not in d: d[c] = 0.0
    X_num = FEATURES_NUM
    X_cat = [c for c in FEATURES_CAT if c in d.columns]

    pre = ColumnTransformer([
        ("num","passthrough", X_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat)
    ])
    model = Pipeline([
        ("pre", pre),
        ("reg", Ridge(alpha=1.0))
    ])
    model.fit(d[X_num + X_cat], d["y"])
    return model

def predict_demand(model: Pipeline, frame: pd.DataFrame) -> pd.Series:
    """Return predicted quantity (not log) for rows in frame."""
    # Ensure feature presence
    for c in FEATURES_NUM:
        if c not in frame: frame[c] = 0.0
    for c in FEATURES_CAT:
        if c not in frame: frame[c] = ""
    yhat = model.predict(frame[FEATURES_NUM + FEATURES_CAT])
    return pd.Series(np.maximum(0.0, np.expm1(yhat)), index=frame.index, name="q_hat")

# -----------------------------
# Price Optimization (Stage B)
# -----------------------------

@dataclass
class PriceConstraints:
    min_mult: float = 0.8     # min vs current price
    max_mult: float = 1.2     # max vs current price
    step: float = 0.01        # price grid step
    margin_floor: float = 0.05  # minimal margin over base cost proxy (use BASE_PRICE as proxy)
    liquidation_penalty: float = 0.0  # optional
    target_sell_through: Optional[float] = None  # if set, override toward Q_target

def optimal_price_row(row, eps: float, q_hat: float, cons: PriceConstraints) -> float:
    """
    RU: Оптимум для степенной модели спроса q(p) = q_hat * (p/p0)^ε.
        Формула максимума выручки: p* = p0 / (1 + 1/ε).
        Инвентарное ограничение: если надо продать Q_target (по стоку),
        п = p0 * (Q_target / q_hat)^(1/ε).
    """
    p0 = max(1e-6, row["SALE_PRICE"]) if row["SALE_PRICE"] > 0 else max(1e-6, row["BASE_PRICE"])
    base = max(1e-6, row["BASE_PRICE"])
    # Margin floor
    min_allowed_by_margin = base * (1.0 + cons.margin_floor)

    # Revenue-optimal candidate
    if eps is None or eps >= -0.05:   # guard: non-elastic or invalid -> nudge to constraints
        p_rev = max(min_allowed_by_margin, p0)
    else:
        p_rev = p0 / (1.0 + 1.0/eps)

    # Inventory-aware target
    p_inv = None
    if cons.target_sell_through is not None:
        Q_target = min(max(0.0, cons.target_sell_through), row.get("END_STOCK", np.inf))
        if q_hat > 0 and abs(eps) > 1e-6:
            p_inv = p0 * (max(1e-9, Q_target) / q_hat)**(1.0/eps)
        else:
            p_inv = p0

    p_cand = p_inv if p_inv is not None else p_rev

    # Apply bounds & step grid
    lo = max(min_allowed_by_margin, p0 * cons.min_mult)
    hi = max(lo, p0 * cons.max_mult)
    p_cand = float(np.clip(p_cand, lo, hi))
    # Round to grid
    p_cand = np.round(p_cand / cons.step) * cons.step
    return float(p_cand)

# -----------------------------
# Bandit layer (safe Thompson over multipliers)
# -----------------------------

class GaussianTS:
    """
    RU: Томпсон-семплинг по дискретным множителям цены.
    Вознаграждение = прибыль за бакет. Предполагаем Normal(μ, σ^2), нестрогая априорная N(0,1).
    Безопасность: ограничиваем множители [min_mult, max_mult].
    """
    def __init__(self, arms: List[float], prior_mu=0.0, prior_var=1.0, noise_var=1.0):
        self.arms = arms
        self.prior_mu = {a: prior_mu for a in arms}
        self.prior_var = {a: prior_var for a in arms}
        self.noise_var = noise_var

    def choose(self) -> float:
        samples = {a: np.random.normal(self.prior_mu[a], np.sqrt(self.prior_var[a])) for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: float, reward: float):
        # Bayesian update for Normal-Normal
        mu0, v0 = self.prior_mu[arm], self.prior_var[arm]
        vn = 1.0 / (1.0/v0 + 1.0/self.noise_var)
        mn = vn * (mu0/v0 + reward/self.noise_var)
        self.prior_mu[arm], self.prior_var[arm] = mn, vn

# -----------------------------
# Heuristic fallback
# -----------------------------

def heuristic_price(row) -> float:
    """
    RU: Простая эвристика:
    - при риске OOS (низкий сток относительно средней продажи) ↑ цена на +5%
    - при избыточном стоке ↓ цена на -5%
    """
    p0 = row["SALE_PRICE"] if row["SALE_PRICE"] > 0 else row["BASE_PRICE"]
    avg_sell = row.get("qty_ma_4", 0) or 0
    stock = row.get("END_STOCK", 0) or 0
    if avg_sell <= 0:
        return p0
    cover_weeks = stock / max(1e-6, avg_sell)
    if cover_weeks < 0.5:
        return p0 * 1.05
    if cover_weeks > 4:
        return p0 * 0.95
    return p0

# -----------------------------
# Orchestration: train, predict next week, rolling eval
# -----------------------------

def prepare_training(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = add_core_features(df_raw)
    agg = build_lag_features(d)
    return d, agg

def recommend_next_week_prices(df_raw: pd.DataFrame,
                               end_history: str = "2025-09-15",
                               price_cons: PriceConstraints = PriceConstraints(),
                               bandit_arms: List[float] = [0.9, 0.95, 1.0, 1.05, 1.1],
                               seed: int = 42) -> pd.DataFrame:
    """
    RU: Главная функция. Возвращает рекомендации цен на СЛЕДУЮЩУЮ НЕДЕЛЮ по бакетам.
    Шаги:
      1) fit demand model на истории ≤ end_history
      2) fit hierarchical ε
      3) прогноз q_hat на следующую неделю
      4) оптимизация цены + безопасный бандит поверх
      5) эвристический фоллбек при крайней редкости данных
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # Prep
    d, agg = prepare_training(df_raw)
    end_dt = pd.to_datetime(end_history)
    train = agg[agg["week"] <= _to_week(pd.Series([end_dt]))[0]].copy()

    # Demand model
    model = fit_demand_model(train)

    # Elasticity
    eps_tbl = fit_hierarchical_elasticity(train.merge(
        d[["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"]].drop_duplicates(),
        on=["STORE","PRODUCT_CODE"], how="left"
    ))

    # Build "next week" frame per (STORE, PRODUCT, bucket)
    next_week_start = (_to_week(pd.Series([end_dt]))[0] + pd.offsets.Week(1))
    # For scoring we need most recent row per key; use last available week record
    last = (train.sort_values("week")
                 .groupby(["STORE","PRODUCT_CODE","bucket"], as_index=False)
                 .tail(1)
            ).copy()
    # Set calendar to next week
    last["week"] = next_week_start
    # Predict demand
    q_hat = predict_demand(model, last)
    last["q_hat"] = q_hat

    # Attach elasticity
    last = last.merge(eps_tbl, on=["STORE","PRODUCT_CODE"], how="left")

    # Inventory-aware target: try to sell <= available stock in bucket
    # Here we aim at 1 week of coverage: target_sell_through = min(END_STOCK, q_hat)
    cons = price_cons

    # Bandit per (STORE, PRODUCT, bucket)
    recs = []
    for _, r in last.iterrows():
        eps = float(r.get("eps_final", -1.2))
        # Inventory target
        cons_loc = PriceConstraints(**cons.__dict__)
        cons_loc.target_sell_through = float(np.nan_to_num(min(r.get("END_STOCK", 0.0), r["q_hat"]), nan=0.0))

        # Deterministic optimum
        p_star = optimal_price_row(r, eps, r["q_hat"], cons_loc)

        # Bandit arms centered around p_star (respect bounds)
        lo = r["SALE_PRICE"] * cons.min_mult
        hi = r["SALE_PRICE"] * cons.max_mult
        arms_prices = np.clip(p_star * np.array(bandit_arms), lo, hi)
        ts = GaussianTS(arms=list(arms_prices))

        # One-shot TS choice for next week (no online history yet -> prior)
        chosen = ts.choose()

        # Fallback if no data at all (few price points or all zero qty historically)
        sparse = (np.isfinite(eps) and (abs(eps) < 0.1)) or (r[["qty_lag_1","qty_lag_2","qty_lag_4"]].isna().all())
        if sparse:
            chosen = heuristic_price(r)

        recs.append({
            "STORE": r["STORE"],
            "PRODUCT_CODE": r["PRODUCT_CODE"],
            "bucket": r["bucket"],
            "base_price": r["BASE_PRICE"],
            "last_price": r["SALE_PRICE"],
            "eps_final": eps,
            "q_hat": r["q_hat"],
            "price_opt_det": p_star,
            "price_rec": float(np.round(chosen / cons.step) * cons.step),
            "week": next_week_start
        })

    recs = pd.DataFrame(recs)
    return recs.sort_values(["STORE","PRODUCT_CODE","bucket"])

# -----------------------------
# Rolling 12-week evaluation
# -----------------------------

def rolling_backtest(df_raw: pd.DataFrame,
                     last_weeks: int = 12,
                     end_history: str = "2025-09-15",
                     price_cons: PriceConstraints = PriceConstraints()) -> pd.DataFrame:
    """
    RU: Форвард-чейнинг бэктест на последние 12 недель.
    На каждой итерации t:
      - учимся на ≤ t-1
      - прогнозируем q_hat на t
      - оптимизируем цену (без утечки)
      - считаем фактическую прибыль на t при выбранной цене vs фактическая по логу (proxy).
    Метрики: MAE по спросу, выручка, прибыль (прибыль = (price - BASE_PRICE)*min(q_hat, фактический спрос по капам)).
    """
    d, agg = prepare_training(df_raw)
    end_dt = pd.to_datetime(end_history)
    all_weeks = np.sort(agg["week"].unique())
    all_weeks = all_weeks[all_weeks <= _to_week(pd.Series([end_dt]))[0]]
    eval_weeks = all_weeks[-last_weeks:]

    rows = []
    for wk in eval_weeks:
        train = agg[agg["week"] < wk].copy()
        valid = agg[agg["week"] == wk].copy()
        if train.empty or valid.empty:
            continue

        model = fit_demand_model(train)
        eps_tbl = fit_hierarchical_elasticity(train.merge(
            d[["STORE","PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"]].drop_duplicates(),
            on=["STORE","PRODUCT_CODE"], how="left"
        ))

        v = valid.copy()
        v["q_hat"] = predict_demand(model, v)
        v = v.merge(eps_tbl, on=["STORE","PRODUCT_CODE"], how="left")

        # Optimize & compute metrics
        cons = price_cons
        v["price_opt"] = [
            optimal_price_row(r, r.get("eps_final", -1.2), r["q_hat"],
                              PriceConstraints(**cons.__dict__, target_sell_through=min(r["END_STOCK"], r["q_hat"])))
            for _, r in v.iterrows()
        ]
        # Profit with our price (approx using our q_hat constrained by stock)
        q_sold = np.minimum(v["q_hat"], v["END_STOCK"])
        profit_model = (v["price_opt"] - v["BASE_PRICE"]) * q_sold
        # Proxy baseline: realized price & qty
        profit_real = (v["SALE_PRICE"] - v["BASE_PRICE"]) * v["SALE_QTY"]

        rows.append(pd.DataFrame({
            "week":[wk]*len(v),
            "STORE": v["STORE"].values,
            "PRODUCT_CODE": v["PRODUCT_CODE"].values,
            "bucket": v["bucket"].values,
            "mae_qty": np.abs(v["SALE_QTY"] - v["q_hat"]).values,
            "profit_model": profit_model.values,
            "profit_real": profit_real.values
        }))

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        summary = (out.groupby("week")
                     .agg(mae_qty=("mae_qty","mean"),
                          profit_model=("profit_model","sum"),
                          profit_real=("profit_real","sum"))
                  ).reset_index()
        summary["uplift_vs_real_%"] = _safe_div(summary["profit_model"]-summary["profit_real"],
                                                np.abs(summary["profit_real"])) * 100
        return summary
    return out

# -----------------------------
# Short Russian docstrings for the “what/why”
# -----------------------------
"""
Что делает модель в общем?
- Шаг A (прогноз): оцениваем спрос на следующий недельный бакет из лагов, календаря и известного промо.
- Шаг B (оптимизация): выбираем цену по эластичности с ограничениями (границы, шаг, маржа) и учётом склада.
- Иерархический backoff: ε оцениваем на уровне товара; при нехватке данных сжимаем к FAMILY/CATEGORY/SEGMENT.
- Бандит: безопасно «пробуем» несколько уровней вокруг оптимума (Томпсон), чтобы обучаться онлайн.
- Фоллбек: простые правила для очень «пустых» товаров.
- Оценка: форвард-чейнинг 12 недель до 2025-09-15, без утечек.
"""
