# -*- coding: utf-8 -*-
# Полный, самодостаточный каркас с:
# 1) weekly_df (агрегаты по неделе и 3 бакетам Mon–Thu, Fri, Sat–Sun; неделя = Mon..Sun)
# 2) устойчивым к «shape» бандитом LinUCB (фикс. размер признаков, паддинги)
# 3) безопасным выбором цены (без 0-кандидатов, greedy fallback)
# 4) predict_next_week_prices (исправлено)
# 5) recommend_for_run_date: запуск по дням Mon/Fri/Sat только нужного бакета
#
# Зависимости: numpy, pandas, scikit-learn

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# =========================
# Константы и настройки
# =========================

# Бакеты: Mon–Thu, Fri, Sat–Sun (понедельник=0)
BUCKETS = {
    "Mon-Thu": [0, 1, 2, 3],
    "Fri":     [4],
    "Sat-Sun": [5, 6]
}

# Глобальная сетка мультипликаторов (фиксированная размерность действий)
ARM_GRID = np.round(np.arange(0.90, 1.30 + 0.001, 0.02), 2)

MAX_RELATIVE_STEP = 0.10     # ограничение изменения цены vs прошлый бакет
MIN_GROSS_MARGIN = 0.05      # минимальная валовая маржа
OOS_PENALTY = 0.30           # штраф за риск OOS
EXPLORATION_EPS = 0.05       # эпсилон-исследование

# Исторический срез (не читаем будущее)
END_OF_HISTORY = pd.Timestamp("2025-09-15")


# ==================================
# Утилиты дат/недель/промо
# ==================================

def week_start_monday(d):
    # возвращает monday (00:00) недели, где d — Timestamp/датa
    d = pd.to_datetime(d)
    return (d - pd.Timedelta(days=int(d.weekday()))).normalize()

def week_end_sunday(d):
    # конец недели (вс) 23:59:59
    start = week_start_monday(d)
    return start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)

def week_id(dt):
    iso = pd.Timestamp(dt).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"

def date_bucket(ts):
    wd = pd.Timestamp(ts).weekday()
    for name, wds in BUCKETS.items():
        if wd in wds:
            return name
    return "Mon-Thu"

def parse_promo_period(s):
    if pd.isna(s) or not isinstance(s, str) or "-" not in s:
        return None, None
    try:
        parts = s.split("-")
        left = "-".join(parts[:3]).strip()
        right = "-".join(parts[3:]).strip()
        start = datetime.strptime(left, "%d-%m-%Y").date()
        end = datetime.strptime(right, "%d-%m-%Y").date()
        return start, end
    except:
        return None, None


# ==================================
# Подготовка сырых данных
# ==================================

def prepare_raw(df):
    df = df.copy()
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"]).dt.tz_localize(None)
    df = df[df["TRADE_DT"] <= END_OF_HISTORY].copy()

    df["WEEK_START"] = df["TRADE_DT"].apply(week_start_monday)
    df["WEEK_END"] = df["TRADE_DT"].apply(week_end_sunday)
    df["WEEK_ID"] = df["TRADE_DT"].apply(week_id)
    df["BUCKET"] = df["TRADE_DT"].apply(date_bucket)
    df["DOW"] = df["TRADE_DT"].dt.weekday
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)

    se = df["PROMO_PERIOD"].apply(parse_promo_period)
    df["PROMO_START"] = se.apply(lambda x: x[0])
    df["PROMO_END"] = se.apply(lambda x: x[1])
    dts = df["TRADE_DT"].dt.date
    df["IS_PROMO_NOW"] = ((~df["PROMO_START"].isna()) &
                          (dts >= df["PROMO_START"]) &
                          (dts <= df["PROMO_END"])).astype(int)

    for c in ["SALE_QTY", "SALE_QTY_ONLINE", "LOSS_QTY", "RETURN_QTY",
              "DELIVERY_QTY", "START_STOCK", "END_STOCK"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["SALE_PRICE", "SALE_PRICE_ONLINE", "BASE_PRICE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["BASE_PRICE"] = df["BASE_PRICE"].fillna(df["SALE_PRICE"])
    return df


# ============================================
# weekly_df: недельно-бакетные аггрегаты (+лаги)
# ============================================

def weekly_aggregates(df):
    key = ["STORE", "PRODUCT_CODE", "WEEK_ID", "BUCKET"]
    agg = df.groupby(key).agg({
        "SALE_QTY": "sum",
        "SALE_QTY_ONLINE": "sum",
        "SALE_PRICE": "mean",
        "SALE_PRICE_ONLINE": "mean",
        "IS_PROMO_NOW": "max",
        "START_STOCK": "mean",
        "END_STOCK": "mean",
        "DELIVERY_QTY": "sum",
        "LOSS_QTY": "sum",
        "RETURN_QTY": "sum",
        "BASE_PRICE": "mean",
        "FAMILY_CODE": "first",
        "CATEGORY_CODE": "first",
        "SEGMENT_CODE": "first",
        "STORE_TYPE": "first",
        "REGION_NAME": "first"
    }).reset_index()

    agg = agg.sort_values(["STORE", "PRODUCT_CODE", "BUCKET", "WEEK_ID"])

    def add_lags(g):
        g = g.copy()
        for w in [1, 2, 4, 8, 12]:
            g[f"LAG_SALE_QTY_{w}"] = g["SALE_QTY"].shift(w)
            g[f"LAG_PRICE_{w}"]    = g["SALE_PRICE"].shift(w)
            g[f"LAG_PROMO_{w}"]    = g["IS_PROMO_NOW"].shift(w)
        g["ELAST1"] = (g["SALE_QTY"] - g["LAG_SALE_QTY_1"]) / (g["SALE_PRICE"] - g["LAG_PRICE_1"] + 1e-6)
        g["ELAST4"] = (g["SALE_QTY"] - g["LAG_SALE_QTY_4"]) / (g["SALE_PRICE"] - g["LAG_PRICE_4"] + 1e-6)
        g["RISK_OOS"] = ((g["END_STOCK"] < np.maximum(1.0, 0.3 * (g["SALE_QTY"] + 1))) &
                         (g["SALE_QTY"] > 0)).astype(int)
        total_qty = g["SALE_QTY"] + g["SALE_QTY_ONLINE"]
        g["ONLINE_SHARE"] = np.where(total_qty > 0, g["SALE_QTY_ONLINE"] / total_qty, 0.0)
        g["VOLATILITY_QTY_4W"] = g["SALE_QTY"].rolling(4, min_periods=1).std().fillna(0.0)
        return g

    agg = agg.groupby(["STORE", "PRODUCT_CODE", "BUCKET"], group_keys=False).apply(add_lags).reset_index(drop=True)

    def make_hier(dfwk, key_cols, prefix):
        tmp = dfwk.groupby(key_cols).agg({
            "SALE_QTY": "sum",
            "SALE_PRICE": "mean",
            "IS_PROMO_NOW": "mean"
        }).rename(columns={
            "SALE_QTY": f"{prefix}_SALE_QTY",
            "SALE_PRICE": f"{prefix}_PRICE",
            "IS_PROMO_NOW": f"{prefix}_PROMO"
        }).reset_index()
        return tmp

    fam = make_hier(agg, ["STORE", "FAMILY_CODE", "WEEK_ID", "BUCKET"], "FAM")
    cat = make_hier(agg, ["STORE", "CATEGORY_CODE", "WEEK_ID", "BUCKET"], "CAT")
    seg = make_hier(agg, ["STORE", "SEGMENT_CODE", "WEEK_ID", "BUCKET"], "SEG")
    reg = make_hier(agg, ["REGION_NAME", "WEEK_ID", "BUCKET"], "REG")
    stt = make_hier(agg, ["STORE_TYPE", "WEEK_ID", "BUCKET"], "STT")

    m = agg.merge(fam, on=["STORE", "FAMILY_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(cat, on=["STORE", "CATEGORY_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(seg, on=["STORE", "SEGMENT_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(reg, on=["REGION_NAME", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(stt, on=["STORE_TYPE", "WEEK_ID", "BUCKET"], how="left")

    for c in m.columns:
        if m[c].dtype.kind in "fc" and m[c].isna().any():
            m[c] = m[c].fillna(m[c].median())
    m["AGE_WEEKS"] = m.groupby(["STORE", "PRODUCT_CODE", "BUCKET"]).cumcount() + 1
    return m


def create_future_week_template(raw_df, target_week_monday):
    # Создаёт пустые строки на неделю target_week_monday (Mon..Sun) по всем STORE×PRODUCT_CODE×3 бакета
    # BASE_PRICE берём последнюю известную (SALE_PRICE/BASE_PRICE) до target_week
    df = prepare_raw(raw_df)
    cutoff = pd.Timestamp(target_week_monday) - pd.Timedelta(seconds=1)

    # Последние известные цены/иерархия
    df_hist = df[df["TRADE_DT"] <= cutoff].copy().sort_values("TRADE_DT")
    last_rows = df_hist.groupby(["STORE", "PRODUCT_CODE"]).tail(1)

    if last_rows.empty:
        return pd.DataFrame(columns=[
            "STORE","PRODUCT_CODE","WEEK_ID","BUCKET","BASE_PRICE",
            "FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE","STORE_TYPE","REGION_NAME",
            "SALE_QTY","SALE_QTY_ONLINE","SALE_PRICE","SALE_PRICE_ONLINE","IS_PROMO_NOW",
            "START_STOCK","END_STOCK","DELIVERY_QTY","LOSS_QTY","RETURN_QTY",
        ])

    # Конструируем 3 бакета на целевую неделю
    wk_id = week_id(pd.Timestamp(target_week_monday))
    records = []
    for _, r in last_rows.iterrows():
        base_price = r["BASE_PRICE"] if pd.notna(r["BASE_PRICE"]) and r["BASE_PRICE"]>0 else r["SALE_PRICE"]
        for b in BUCKETS.keys():
            records.append({
                "STORE": r["STORE"],
                "PRODUCT_CODE": r["PRODUCT_CODE"],
                "WEEK_ID": wk_id,
                "BUCKET": b,
                "BASE_PRICE": base_price,
                "FAMILY_CODE": r["FAMILY_CODE"],
                "CATEGORY_CODE": r["CATEGORY_CODE"],
                "SEGMENT_CODE": r["SEGMENT_CODE"],
                "STORE_TYPE": r["STORE_TYPE"],
                "REGION_NAME": r["REGION_NAME"],
                "SALE_QTY": 0.0,
                "SALE_QTY_ONLINE": 0.0,
                "SALE_PRICE": np.nan,
                "SALE_PRICE_ONLINE": np.nan,
                "IS_PROMO_NOW": 0,
                "START_STOCK": 0.0,
                "END_STOCK": 0.0,
                "DELIVERY_QTY": 0.0,
                "LOSS_QTY": 0.0,
                "RETURN_QTY": 0.0
            })
    return pd.DataFrame(records)


# =================================
# Сетка цен и действия (фикс. размер)
# =================================

def make_price_ladder(base_price):
    if pd.isna(base_price) or base_price <= 0:
        return np.array([])
    return np.unique(np.round(base_price * ARM_GRID, 4))

def snap_price_to_grid(candidate, ladder):
    if len(ladder) == 0:
        return np.nan
    idx = np.argmin(np.abs(ladder - candidate))
    return float(ladder[idx])

def generate_candidates(base_price, last_price):
    ladder = make_price_ladder(base_price)
    if len(ladder) == 0:
        return np.array([])
    if pd.isna(last_price) or last_price <= 0:
        return ladder
    lo, hi = last_price*(1-MAX_RELATIVE_STEP), last_price*(1+MAX_RELATIVE_STEP)
    rng = ladder[(ladder >= lo) & (ladder <= hi)]
    if len(rng) == 0:
        return np.array([snap_price_to_grid(last_price, ladder)])
    return rng

def price_to_multiplier(price, base_price):
    if pd.isna(price) or pd.isna(base_price) or base_price <= 0:
        return np.nan
    m = np.round(price / base_price, 2)
    idx = np.argmin(np.abs(ARM_GRID - m))
    return ARM_GRID[idx]

def action_vector_from_price(price, base_price):
    vec = np.zeros(len(ARM_GRID), dtype=float)
    m = price_to_multiplier(price, base_price)
    if pd.isna(m):
        return vec
    idx = int(np.where(np.isclose(ARM_GRID, m))[0][0])
    vec[idx] = 1.0
    return vec


# ============================================
# Модель спроса (ML-приближение отклика)
# ============================================

def build_demand_model():
    cat_cols = ["BUCKET", "STORE_TYPE", "REGION_NAME"]
    num_cols = [
        "SALE_PRICE", "BASE_PRICE",
        "LAG_SALE_QTY_1","LAG_SALE_QTY_2","LAG_SALE_QTY_4","LAG_SALE_QTY_8","LAG_SALE_QTY_12",
        "LAG_PRICE_1","LAG_PRICE_2","LAG_PRICE_4","LAG_PRICE_8","LAG_PRICE_12",
        "LAG_PROMO_1","LAG_PROMO_2","LAG_PROMO_4","LAG_PROMO_8","LAG_PROMO_12",
        "ELAST1","ELAST4","ONLINE_SHARE","VOLATILITY_QTY_4W","IS_PROMO_NOW",
        "FAM_SALE_QTY","CAT_SALE_QTY","SEG_SALE_QTY","REG_SALE_QTY","STT_SALE_QTY",
        "FAM_PRICE","CAT_PRICE","SEG_PRICE","REG_PRICE","STT_PRICE",
        "AGE_WEEKS","RISK_OOS"
    ]
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ],
        remainder="drop"
    )
    model = GradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("gbm", model)])
    return pipe, cat_cols + num_cols

def fit_demand_model(pipe, df_train):
    use = df_train.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = use
    y = use["SALE_QTY"].values
    pipe.fit(X, y)
    return pipe

def predict_demand(pipe, Xcand):
    Xc = Xcand.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pipe.predict(Xc)


# ============================================
# Устойчивый LinUCB (фикс. размер, паддинги)
# ============================================

def _pad_or_trim(vec, target_dim):
    d = vec.shape[0]
    if d == target_dim:
        return vec
    if d < target_dim:
        out = np.zeros(target_dim, dtype=float)
        out[:d] = vec
        return out
    return vec[:target_dim]

class LinUCB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.groups = {}  # gid -> {"A":, "b":, "dim": int}

    def _init_group(self, dim):
        return {"A": np.eye(dim, dtype=float), "b": np.zeros(dim, dtype=float), "dim": dim}

    def _resize_group(self, g, new_dim):
        A_old, b_old, old_dim = g["A"], g["b"], g["dim"]
        A_new = np.eye(new_dim, dtype=float)
        b_new = np.zeros(new_dim, dtype=float)
        m = min(old_dim, new_dim)
        A_new[:m, :m] = A_old[:m, :m]
        b_new[:m] = b_old[:m]
        g["A"], g["b"], g["dim"] = A_new, b_new, new_dim
        return g

    def _ensure_group(self, gid, dim):
        if gid not in self.groups:
            self.groups[gid] = self._init_group(dim)
        elif self.groups[gid]["dim"] != dim:
            self.groups[gid] = self._resize_group(self.groups[gid], dim)

    def select(self, gid, ctx_matrix, actions_matrix):
        n = int(ctx_matrix.shape[0])
        if n == 0:
            return -1
        d_ctx = int(ctx_matrix.shape[1])
        d_act = int(actions_matrix.shape[1])  # == len(ARM_GRID)
        d = d_ctx + d_act
        self._ensure_group(gid, d)
        g = self.groups[gid]
        A, b = g["A"], g["b"]
        A_inv = np.linalg.inv(A)

        scores = []
        for i in range(n):
            x = np.concatenate([ctx_matrix[i], actions_matrix[i]])
            x = _pad_or_trim(x, g["dim"])
            theta = _pad_or_trim(A_inv.dot(b), g["dim"])
            mu = float(theta.dot(x))
            sigma = float(np.sqrt(x.dot(A_inv).dot(x)))
            scores.append(mu + self.alpha * sigma)

        return int(np.argmax(scores))

    def update(self, gid, x, reward):
        d = x.shape[0]
        self._ensure_group(gid, d)
        g = self.groups[gid]
        x = _pad_or_trim(x, g["dim"])
        g["A"] += np.outer(x, x)
        g["b"] += float(reward) * x


# ============================================
# Политика выбора цены (без 0-кандидатов)
# ============================================

def expected_profit(demand, price, cost_price):
    margin = 0.25
    cost = price * (1 - margin)
    min_cost = price * (1 - MIN_GROSS_MARGIN)
    cost = min(cost, min_cost)
    return np.maximum(0.0, price - cost) * np.maximum(0.0, demand)

def choose_price_for_row(row, pipe, bandit, last_price, group_id):
    base_price = row["BASE_PRICE"]

    # Кандидаты и чистка
    candidates = generate_candidates(base_price, last_price)
    candidates = np.array([c for c in np.unique(candidates) if (pd.notna(c) and c > 0)], dtype=float)

    if candidates.size == 0:
        ladder = make_price_ladder(base_price)
        if ladder.size == 0:
            return np.nan, {"reason": "no_candidates"}
        if pd.notna(last_price) and last_price > 0:
            lo, hi = last_price*(1-MAX_RELATIVE_STEP), last_price*(1+MAX_RELATIVE_STEP)
            sl = ladder[(ladder >= lo) & (ladder <= hi)]
            if sl.size > 0:
                candidates = sl
            else:
                candidates = np.array([snap_price_to_grid(last_price, ladder)], dtype=float)
        else:
            idx = int(np.argmin(np.abs(ladder - base_price)))
            candidates = np.array([ladder[idx]], dtype=float)

    Xcand = pd.DataFrame([row] * int(candidates.size)).reset_index(drop=True)
    Xcand["SALE_PRICE"] = candidates
    demand_pred = predict_demand(pipe, Xcand)

    risk = float(row.get("RISK_OOS", 0.0))
    profits = []
    for q, p in zip(demand_pred, candidates):
        prof = expected_profit(q, p, row.get("BASE_PRICE", p)) * (1.0 - OOS_PENALTY * risk)
        profits.append(prof)
    profits = np.array(profits, dtype=float)

    if candidates.size > 1 and np.random.rand() < EXPLORATION_EPS:
        idx = np.random.randint(int(candidates.size))
        return float(candidates[idx]), {"reason": "epsilon_explore", "profit_est": float(profits[idx])}

    ctx_cols = [
        "IS_PROMO_NOW", "ONLINE_SHARE", "VOLATILITY_QTY_4W",
        "FAM_SALE_QTY", "CAT_SALE_QTY", "SEG_SALE_QTY", "REG_SALE_QTY", "STT_SALE_QTY",
        "ELAST1", "ELAST4", "AGE_WEEKS", "RISK_OOS"
    ]
    ctx = np.array([row.get(c, 0.0) for c in ctx_cols], dtype=float)

    ctx_matrix = np.vstack([ctx] * int(candidates.size))
    action_matrix = np.vstack([action_vector_from_price(p, base_price) for p in candidates])

    if ctx_matrix.shape[0] == 0 or action_matrix.shape[0] == 0:
        idx = int(np.argmax(profits))
        return float(candidates[idx]), {"reason": "greedy_fallback_n0", "profit_est": float(profits[idx])}

    idx = bandit.select(group_id, ctx_matrix, action_matrix)
    if idx is None or idx < 0 or idx >= candidates.size:
        idx = int(np.argmax(profits))
        return float(candidates[idx]), {"reason": "greedy_fallback_idx", "profit_est": float(profits[idx])}

    chosen = float(candidates[idx])
    return chosen, {
        "reason": "linucb",
        "profit_est": float(profits[idx]),
        "all_profit_est": profits.tolist()
    }


# ============================================
# Оценка и предсказание
# ============================================

def walk_forward_eval(weekly_df, weeks_sorted, horizon_eval=12):
    pipe, _ = build_demand_model()
    bandit = LinUCB(alpha=1.0)
    results = []
    last_prices = {}

    for i in range(horizon_eval, len(weeks_sorted)):
        train_weeks = weeks_sorted[i - horizon_eval: i]
        pred_week = weeks_sorted[i]

        train_data = weekly_df[weekly_df["WEEK_ID"].isin(train_weeks)].copy()
        pipe = fit_demand_model(pipe, train_data)

        pred_rows = weekly_df[weekly_df["WEEK_ID"] == pred_week].copy()
        pred_rows = pred_rows.sort_values(["STORE", "PRODUCT_CODE", "BUCKET"])

        week_chosen = []
        for _, row in pred_rows.iterrows():
            key_lp = (row["STORE"], row["PRODUCT_CODE"], row["BUCKET"])
            last_price = last_prices.get(key_lp, row["BASE_PRICE"])
            group_id = f"{row['STORE']}|{row['SEGMENT_CODE']}|{row['BUCKET']}"

            chosen_price, info = choose_price_for_row(row, pipe, bandit, last_price, group_id)
            if not pd.isna(chosen_price):
                last_prices[key_lp] = chosen_price

            profit_hat = info.get("profit_est", 0.0)
            reward = profit_hat / (1.0 + abs(profit_hat))

            ctx_cols = [
                "IS_PROMO_NOW", "ONLINE_SHARE", "VOLATILITY_QTY_4W",
                "FAM_SALE_QTY", "CAT_SALE_QTY", "SEG_SALE_QTY", "REG_SALE_QTY", "STT_SALE_QTY",
                "ELAST1", "ELAST4", "AGE_WEEKS", "RISK_OOS"
            ]
            ctx = np.array([row.get(c, 0.0) for c in ctx_cols], dtype=float)
            act = action_vector_from_price(chosen_price, row["BASE_PRICE"])
            x = np.concatenate([ctx, act])
            bandit.update(group_id, x, reward)

            week_chosen.append({
                "STORE": row["STORE"],
                "PRODUCT_CODE": row["PRODUCT_CODE"],
                "BUCKET": row["BUCKET"],
                "WEEK_ID": pred_week,
                "CHOSEN_PRICE": chosen_price,
                "REASON": info.get("reason", "na"),
                "PROFIT_EST": profit_hat
            })

        if week_chosen:
            results.append(pd.DataFrame(week_chosen))

    if len(results) == 0:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def predict_next_week_prices(weekly_df, next_week_id):
    all_weeks = sorted(weekly_df["WEEK_ID"].unique())
    if next_week_id not in all_weeks:
        raise ValueError("next_week_id отсутствует в weekly_df (нужен шаблон строк на эту неделю).")
    idx = all_weeks.index(next_week_id)
    if idx < 12:
        raise ValueError("Недостаточно истории (<12 недель) для обучения.")

    train_weeks = all_weeks[idx - 12: idx]
    pipe, _ = build_demand_model()
    bandit = LinUCB(alpha=1.0)

    train_data = weekly_df[weekly_df["WEEK_ID"].isin(train_weeks)].copy()
    pipe = fit_demand_model(pipe, train_data)

    prev_week_id = all_weeks[idx - 1]
    prev_df = weekly_df[weekly_df["WEEK_ID"] == prev_week_id][
        ["STORE", "PRODUCT_CODE", "BUCKET", "SALE_PRICE", "BASE_PRICE"]
    ].copy()
    prev_df = prev_df.rename(columns={"SALE_PRICE": "LAST_SALE_PRICE"})
    prev_key = list(zip(prev_df["STORE"], prev_df["PRODUCT_CODE"], prev_df["BUCKET"]))
    prev_map = {k: p for k, p in zip(prev_key, prev_df["LAST_SALE_PRICE"])}

    pred_rows = weekly_df[weekly_df["WEEK_ID"] == next_week_id].copy()
    pred_rows = pred_rows.sort_values(["STORE", "PRODUCT_CODE", "BUCKET"])

    recommendations = []
    for _, row in pred_rows.iterrows():
        key_lp = (row["STORE"], row["PRODUCT_CODE"], row["BUCKET"])
        last_price = prev_map.get(key_lp, row["BASE_PRICE"])
        group_id = f"{row['STORE']}|{row['SEGMENT_CODE']}|{row['BUCKET']}"
        price, info = choose_price_for_row(row, pipe, bandit, last_price, group_id)
        recommendations.append({
            "STORE": row["STORE"],
            "PRODUCT_CODE": row["PRODUCT_CODE"],
            "BUCKET": row["BUCKET"],
            "WEEK_ID": next_week_id,
            "RECOMMENDED_PRICE": price,
            "BASE_PRICE": row["BASE_PRICE"],
            "LAST_PRICE_PREV_WEEK": last_price,
            "REASON": info.get("reason", "na"),
            "PROFIT_EST": info.get("profit_est", np.nan)
        })

    return pd.DataFrame(recommendations)


# ============================================
# Рантайм-логика «запуск по дням Mon/Fri/Sat»
# ============================================

def build_weekly_df_with_future(raw_df, run_date):
    # Собираем weekly_df из истории + добавляем пустые строки на неделю run_date (Mon..Sun)
    df_hist = prepare_raw(raw_df)
    wk_hist = weekly_aggregates(df_hist)

    monday = week_start_monday(pd.Timestamp(run_date))
    fut = create_future_week_template(raw_df, monday)

    if fut.empty:
        return wk_hist  # нечего добавлять

    # Склеиваем, пересчитываем фичи уже на объединении, чтобы у будущей недели появились лаги/иерархии
    combined = pd.concat([df_hist, fut.assign(TRADE_DT=monday)], ignore_index=True, sort=False)
    combined = prepare_raw(combined)  # восстановим WEEK_ID/BUCKET и т.д. для новых строк
    weekly = weekly_aggregates(combined)
    return weekly


def bucket_for_run_date(run_date):
    wd = pd.Timestamp(run_date).weekday()
    if wd == 0:
        return "Mon-Thu"
    if wd == 4:
        return "Fri"
    if wd == 5:
        return "Sat-Sun"
    raise ValueError("Код должен запускаться только в понедельник, пятницу или субботу.")

def recommend_for_run_date(raw_df, run_date):
    # 1) weekly_df (история + будущая неделя)
    weekly_df = build_weekly_df_with_future(raw_df, run_date)
    target_week = week_id(week_start_monday(pd.Timestamp(run_date)))
    # 2) Получаем рекомендации на всю неделю
    full_week_reco = predict_next_week_prices(weekly_df, target_week)
    # 3) Оставляем только нужный бакет ( Mon->Mon-Thu, Fri->Fri, Sat->Sat-Sun )
    target_bucket = bucket_for_run_date(run_date)
    return weekly_df, full_week_reco[full_week_reco["BUCKET"] == target_bucket].reset_index(drop=True)


# ============================================
# Пример использования
# ============================================

if __name__ == "__main__":
    # Заготовка структуры входа: Замените на вашу загрузку DWH/CSV
    cols = [
        "TRADE_DT", "IS_PROMO", "SALE_PRICE", "SALE_PRICE_ONLINE",
        "SALE_QTY", "SALE_QTY_ONLINE", "LOSS_QTY", "RETURN_QTY", "DELIVERY_QTY",
        "START_STOCK", "END_STOCK", "PRODUCT_CODE", "FAMILY_CODE", "CATEGORY_CODE",
        "SEGMENT_CODE", "STORE", "STORE_TYPE", "REGION_NAME", "BASE_PRICE",
        "PROMO_PERIOD", "PLACE_TYPE"
    ]
    raw_df = pd.DataFrame(columns=cols)

    # Пример: запускать только в Mon/Fri/Sat
    # run_date = "2025-09-08"  # Понедельник → Mon-Thu
    # run_date = "2025-09-12"  # Пятница → Fri
    # run_date = "2025-09-13"  # Суббота → Sat-Sun
    # weekly_df, reco = recommend_for_run_date(raw_df, run_date)
    # print(weekly_df.head())
    # print(reco.head())
    pass
