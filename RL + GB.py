# -*- coding: utf-8 -*-
# Практический, самодостаточный каркас динамического ценообразования
# с контекстуальными бандитами под 3 бакета: Mon–Thu, Fri, Sat–Sun.
# Без аннотаций типов. Комментарии на русском.
#
# Использует только то, что рекомендовано в плане:
# - Фичи: календарь/история/иерархии/промо/запасы/каналы/бакет-специфика
# - Сетка цен (price ladder) + снаппинг и бизнес-ограничения
# - Контекстуальный бандит (LinUCB / Thompson) со шарингом по иерархии
# - Walk-forward оценка на 12 недель, предсказание следующей недели
#
# Зависимости: numpy, pandas, scikit-learn
# (для простоты используем sklearn GradientBoostingRegressor как модель отклика)


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge


# =========================
# 0) Константы и настройки
# =========================

BUCKETS = {
    "Mon-Thu": [0, 1, 2, 3],   # понедельник=0 ... четверг=3
    "Fri":     [4],
    "Sat-Sun": [5, 6]
}

# Дискретные мультипликаторы к базовой цене (arms) с шагом 0.02 в диапазоне 0.90..1.30
ARM_GRID = np.round(np.arange(0.90, 1.30 + 0.001, 0.02), 2)

# Ограничения на резкость изменений цены между неделями, на маржу и т.п.
MAX_RELATIVE_STEP = 0.10   # не более 10% относительно предыдущей цены
MIN_GROSS_MARGIN = 0.05    # минимальная валовая маржа как доля цены (пример)
OOS_PENALTY = 0.30         # штраф за риск OOS в полезности
STABILITY_PENALTY = 0.02   # штраф за слишком частые изменения
EXPLORATION_EPS = 0.05     # базовая эпсилон-исследование

# Срез истории: конец истории (не обучаемся/смотрим дальше)
END_OF_HISTORY = pd.Timestamp("2025-09-15")

# --- Add these helpers near ARM_GRID ---

ef price_to_multiplier(price, base_price):
    # аккуратно обрабатываем нули/NaN
    if pd.isna(price) or pd.isna(base_price) or base_price <= 0:
        return np.nan
    m = np.round(price / base_price, 2)
    # снап к ARM_GRID, чтобы индекс совпадал с глобальной сеткой
    idx = np.argmin(np.abs(ARM_GRID - m))
    return ARM_GRID[idx]

def action_vector_from_price(price, base_price):
    # one-hot длиной len(ARM_GRID) по глобальному индексу мультипликатора
    vec = np.zeros(len(ARM_GRID), dtype=float)
    m = price_to_multiplier(price, base_price)
    if pd.isna(m):
        return vec
    idx = int(np.where(np.isclose(ARM_GRID, m))[0][0])
    vec[idx] = 1.0
    return vec



# ==================================
# 1) Утилиты для дат и PROMO_PERIOD
# ==================================

def parse_promo_period(s):
    """
    Парсит строку 'dd-mm-YYYY - dd-mm-YYYY' -> (start_date, end_date).
    Может вернуть (None, None), если нет значения.
    """
    if pd.isna(s) or not isinstance(s, str) or "-" not in s:
        return None, None
    try:
        parts = s.split("-")
        # строка может быть '01-01-2024 - 03-01-2024'
        left = "-".join(parts[:3]).strip()
        right = "-".join(parts[3:]).strip()
        start = datetime.strptime(left, "%d-%m-%Y").date()
        end = datetime.strptime(right, "%d-%m-%Y").date()
        return start, end
    except:
        return None, None


def date_bucket(dt):
    """
    Возвращает название бакета для даты.
    """
    wd = dt.weekday()
    for bname, dlist in BUCKETS.items():
        if wd in dlist:
            return bname
    return "Mon-Thu"


def week_id(dt):
    """
    Возвращает идентификатор недели (год-неделя ISO).
    """
    iso = dt.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def monday_of_week(dt):
    """
    Находит понедельник той же недели.
    """
    return dt - timedelta(days=dt.weekday())


# ===========================
# 2) Подготовка исходных данных
# ===========================

def prepare_raw(df):
    """
    Ожидаемые колонки (минимум): 
    TRADE_DT (datetime), IS_PROMO, SALE_PRICE, SALE_PRICE_ONLINE, SALE_QTY, SALE_QTY_ONLINE,
    LOSS_QTY, RETURN_QTY, DELIVERY_QTY, START_STOCK, END_STOCK,
    PRODUCT_CODE, FAMILY_CODE, CATEGORY_CODE, SEGMENT_CODE,
    STORE, STORE_TYPE, REGION_NAME, BASE_PRICE, PROMO_PERIOD, PLACE_TYPE
    Некоторые могут быть None.
    Функция приводит даты, обрезает историю, добивает обязательные поля.
    """
    df = df.copy()
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"]).dt.tz_localize(None)
    df = df[df["TRADE_DT"] <= END_OF_HISTORY].copy()
    # Вычисляем базовые календарные поля
    df["WEEK_ID"] = df["TRADE_DT"].apply(week_id)
    df["BUCKET"] = df["TRADE_DT"].apply(date_bucket)
    df["DOW"] = df["TRADE_DT"].dt.weekday
    df["IS_WEEKEND"] = df["DOW"].isin([5, 6]).astype(int)
    # Промо периоды
    start_end = df["PROMO_PERIOD"].apply(parse_promo_period)
    df["PROMO_START"] = start_end.apply(lambda x: x[0])
    df["PROMO_END"]   = start_end.apply(lambda x: x[1])
    # Флаги «в промо сегодня»
    dts = df["TRADE_DT"].dt.date
    df["IS_PROMO_NOW"] = ((~df["PROMO_START"].isna()) &
                          (dts >= df["PROMO_START"]) &
                          (dts <= df["PROMO_END"])).astype(int)
    # Заполнение пропусков по ценам/количествам/запасам
    for c in ["SALE_QTY", "SALE_QTY_ONLINE", "LOSS_QTY", "RETURN_QTY",
              "DELIVERY_QTY", "START_STOCK", "END_STOCK"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["SALE_PRICE", "SALE_PRICE_ONLINE", "BASE_PRICE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["BASE_PRICE"] = df["BASE_PRICE"].fillna(df["SALE_PRICE"])
    return df


# ===========================
# 3) Фича-инжиниринг
# ===========================

def weekly_aggregates(df):
    """
    Считает недельные аггрегаты и бакет-аггрегаты.
    Это нужно, чтобы строить лаговые фичи на 1/2/4/8/12 недель.
    """
    key = ["STORE", "PRODUCT_CODE", "WEEK_ID", "BUCKET"]
    # Суммы продаж и средние цены по бакету за неделю
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
    # Лаговые окна в неделях (по бакетам отдельно)
    def add_lags(g):
        g = g.copy()
        for w in [1, 2, 4, 8, 12]:
            g[f"LAG_SALE_QTY_{w}"] = g["SALE_QTY"].shift(w)
            g[f"LAG_PRICE_{w}"]    = g["SALE_PRICE"].shift(w)
            g[f"LAG_PROMO_{w}"]    = g["IS_PROMO_NOW"].shift(w)
        # Простая эластичность-набросок: dQ/dP на горизонтах
        g["ELAST1"] = (g["SALE_QTY"] - g["LAG_SALE_QTY_1"]) / (g["SALE_PRICE"] - g["LAG_PRICE_1"] + 1e-6)
        g["ELAST4"] = (g["SALE_QTY"] - g["LAG_SALE_QTY_4"]) / (g["SALE_PRICE"] - g["LAG_PRICE_4"] + 1e-6)
        # Риск OOS (склад низкий + есть продажи)
        g["RISK_OOS"] = ((g["END_STOCK"] < np.maximum(1.0, 0.3 * (g["SALE_QTY"] + 1))) &
                         (g["SALE_QTY"] > 0)).astype(int)
        # Доли онлайн
        total_qty = g["SALE_QTY"] + g["SALE_QTY_ONLINE"]
        g["ONLINE_SHARE"] = np.where(total_qty > 0, g["SALE_QTY_ONLINE"] / total_qty, 0.0)
        # Вспомогательные композитные фичи
        g["VOLATILITY_QTY_4W"] = g["SALE_QTY"].rolling(4, min_periods=1).std().fillna(0.0)
        return g

    agg = agg.groupby(["STORE", "PRODUCT_CODE", "BUCKET"], group_keys=False).apply(add_lags).reset_index(drop=True)

    # Иерархические аггрегаты (FAMILY/CATEGORY/SEGMENT/STORE/REGION)
    hier_keys = {
        "FAMILY": ["STORE", "FAMILY_CODE", "WEEK_ID", "BUCKET"],
        "CATEGORY": ["STORE", "CATEGORY_CODE", "WEEK_ID", "BUCKET"],
        "SEGMENT": ["STORE", "SEGMENT_CODE", "WEEK_ID", "BUCKET"],
        "REGION": ["REGION_NAME", "WEEK_ID", "BUCKET"],
        "STORE_TYPE": ["STORE_TYPE", "WEEK_ID", "BUCKET"]
    }
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

    fam = make_hier(agg, hier_keys["FAMILY"], "FAM")
    cat = make_hier(agg, hier_keys["CATEGORY"], "CAT")
    seg = make_hier(agg, hier_keys["SEGMENT"], "SEG")
    reg = make_hier(agg, hier_keys["REGION"], "REG")
    stt = make_hier(agg, hier_keys["STORE_TYPE"], "STT")

    # Джойним обратно по соответствующим ключам
    m = agg.merge(fam, on=["STORE", "FAMILY_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(cat, on=["STORE", "CATEGORY_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(seg, on=["STORE", "SEGMENT_CODE", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(reg, on=["REGION_NAME", "WEEK_ID", "BUCKET"], how="left")
    m = m.merge(stt, on=["STORE_TYPE", "WEEK_ID", "BUCKET"], how="left")

    # Заполняем пропуски и финальные доп-фичи
    for c in m.columns:
        if m[c].dtype.kind in "fc" and m[c].isna().any():
            m[c] = m[c].fillna(m[c].median())
    m["AGE_WEEKS"] = m.groupby(["STORE", "PRODUCT_CODE", "BUCKET"]).cumcount() + 1
    return m


# =================================
# 4) Прайс-лестница и кандидаты цен
# =================================

def make_price_ladder(base_price):
    """
    Строит сетку допустимых цен (snap к дискретным мультипликаторам).
    """
    if pd.isna(base_price) or base_price <= 0:
        return np.array([])
    grid = np.unique(np.round(base_price * ARM_GRID, 4))
    return grid


def snap_price_to_grid(candidate, ladder):
    """
    Привязывает цену к ближайшей допустимой точке прайс-лестницы.
    """
    if len(ladder) == 0:
        return np.nan
    idx = np.argmin(np.abs(ladder - candidate))
    return float(ladder[idx])


def generate_candidates(base_price, last_price):
    """
    Строит кандидатные цены с учётом:
    - прайс-лестницы
    - ограничения на резкость изменений (MAX_RELATIVE_STEP)
    """
    ladder = make_price_ladder(base_price)
    if len(ladder) == 0:
        return np.array([])
    if pd.isna(last_price) or last_price <= 0:
        # Если нет прошлой цены, берём всю лестницу
        return ladder
    # Ограничиваем диапазон вокруг прошлой цены
    lo = last_price * (1 - MAX_RELATIVE_STEP)
    hi = last_price * (1 + MAX_RELATIVE_STEP)
    rng = ladder[(ladder >= lo) & (ladder <= hi)]
    if len(rng) == 0:
        # Если срез пуст — берём ближайшую точку к last_price
        return np.array([snap_price_to_grid(last_price, ladder)])
    return rng


# ============================================
# 5) Модель отклика (спроса) для оценки profit
# ============================================

def build_demand_model():
    """
    Небольшой градиентный бустинг для прогноза спроса (шт).
    Работаем в пайплайне с OneHot на категориальные поля.
    """
    cat_cols = ["BUCKET", "STORE_TYPE", "REGION_NAME"]
    num_cols = [
        "SALE_PRICE", "BASE_PRICE",
        "LAG_SALE_QTY_1", "LAG_SALE_QTY_2", "LAG_SALE_QTY_4", "LAG_SALE_QTY_8", "LAG_SALE_QTY_12",
        "LAG_PRICE_1", "LAG_PRICE_2", "LAG_PRICE_4", "LAG_PRICE_8", "LAG_PRICE_12",
        "LAG_PROMO_1", "LAG_PROMO_2", "LAG_PROMO_4", "LAG_PROMO_8", "LAG_PROMO_12",
        "ELAST1", "ELAST4", "ONLINE_SHARE", "VOLATILITY_QTY_4W", "IS_PROMO_NOW",
        "FAM_SALE_QTY", "CAT_SALE_QTY", "SEG_SALE_QTY", "REG_SALE_QTY", "STT_SALE_QTY",
        "FAM_PRICE", "CAT_PRICE", "SEG_PRICE", "REG_PRICE", "STT_PRICE",
        "AGE_WEEKS", "RISK_OOS"
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
    """
    Обучает модель спроса на исторических фактах за предыдущие недели.
    Целевая переменная — SALE_QTY.
    """
    use = df_train.copy()
    use = use.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = use
    y = use["SALE_QTY"].values
    pipe.fit(X, y)
    return pipe


def predict_demand(pipe, Xcand):
    """
    Прогноз спроса для кандидатных цен.
    """
    Xc = Xcand.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pipe.predict(Xc)


# =========================================
# 6) Контекстуальный бандит (LinUCB / TS)
# =========================================

def _pad_or_trim(vec, target_dim):
    d = vec.shape[0]
    if d == target_dim: return vec
    if d < target_dim:
        out = np.zeros(target_dim, dtype=float); out[:d] = vec; return out
    return vec[:target_dim]

class LinUCB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.groups = {}  # gid -> {"A":, "b":, "dim": int}

    def _init_group(self, dim):
        return {"A": np.eye(dim, dtype=float), "b": np.zeros(dim, dtype=float), "dim": dim}

    def _resize_group(self, g, new_dim):
        A_old, b_old, old_dim = g["A"], g["b"], g["dim"]
        A_new = np.eye(new_dim, dtype=float); b_new = np.zeros(new_dim, dtype=float)
        m = min(old_dim, new_dim)
        A_new[:m, :m] = A_old[:m, :m]; b_new[:m] = b_old[:m]
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
            return -1  # сигнализируем вызывающему: нечего выбирать

        d_ctx = int(ctx_matrix.shape[1])
        d_act = int(actions_matrix.shape[1])
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

# ====================================================
# 7) Политика выбора цены (бандит + бизнес-ограничения)
# ====================================================

def expected_profit(demand, price, cost_price):
    """
    Ожидаемая прибыль на SKU = (price - cost) * demand.
    cost_price можно аппроксимировать как BASE_PRICE * (1 - средняя маржа).
    Здесь для примера считаем cost = price * (1 - 0.25) => маржа 25%.
    """
    margin = 0.25
    cost = price * (1 - margin)
    # Защита минимальной маржи
    min_cost = price * (1 - MIN_GROSS_MARGIN)
    cost = min(cost, min_cost)
    return np.maximum(0.0, price - cost) * np.maximum(0.0, demand)


 # 3) Полный код функции выбора цены с фиксированным action-вектором (глобальный one-hot по ARM_GRID)

ddef choose_price_for_row(row, pipe, bandit, last_price, group_id):
    base_price = row["BASE_PRICE"]

    # 1) кандидаты
    candidates = generate_candidates(base_price, last_price)
    # убираем NaN/неположительные/дубликаты
    candidates = np.array([c for c in np.unique(candidates) if (pd.notna(c) and c > 0)], dtype=float)

    # если сетка пустая — пробуем fallback на снап last_price → base ladder
    if candidates.size == 0:
        ladder = make_price_ladder(base_price)
        if ladder.size == 0:
            return np.nan, {"reason": "no_candidates"}
        # ограничим шагом относительно last_price, если он валиден
        if pd.notna(last_price) and last_price > 0:
            lo, hi = last_price*(1-MAX_RELATIVE_STEP), last_price*(1+MAX_RELATIVE_STEP)
            sl = ladder[(ladder >= lo) & (ladder <= hi)]
            if sl.size > 0:
                candidates = sl
            else:
                candidates = np.array([snap_price_to_grid(last_price, ladder)], dtype=float)
        else:
            # нет прошлой цены — возьмём ближайшее к BASE_PRICE
            idx = int(np.argmin(np.abs(ladder - base_price)))
            candidates = np.array([ladder[idx]], dtype=float)

    # на этом этапе кандидатов >= 1
    Xcand = pd.DataFrame([row] * int(candidates.size)).reset_index(drop=True)
    Xcand["SALE_PRICE"] = candidates
    demand_pred = predict_demand(pipe, Xcand)

    risk = float(row.get("RISK_OOS", 0.0))
    profits = []
    for q, p in zip(demand_pred, candidates):
        prof = expected_profit(q, p, row.get("BASE_PRICE", p)) * (1.0 - OOS_PENALTY * risk)
        profits.append(prof)
    profits = np.array(profits, dtype=float)

    # ε-explore (только если есть >1 канд.)
    if candidates.size > 1 and np.random.rand() < EXPLORATION_EPS:
        idx = np.random.randint(int(candidates.size))
        return float(candidates[idx]), {"reason": "epsilon_explore", "profit_est": float(profits[idx])}

    # Контекст
    ctx_cols = [
        "IS_PROMO_NOW", "ONLINE_SHARE", "VOLATILITY_QTY_4W",
        "FAM_SALE_QTY", "CAT_SALE_QTY", "SEG_SALE_QTY", "REG_SALE_QTY", "STT_SALE_QTY",
        "ELAST1", "ELAST4", "AGE_WEEKS", "RISK_OOS"
    ]
    ctx = np.array([row.get(c, 0.0) for c in ctx_cols], dtype=float)

    # Матрицы для бандита
    ctx_matrix = np.vstack([ctx] * int(candidates.size))
    action_matrix = np.vstack([action_vector_from_price(p, base_price) for p in candidates])

    # Если по какой-то причине n == 0 (не должно, но на всякий случай) — greedy fallback
    if ctx_matrix.shape[0] == 0 or action_matrix.shape[0] == 0:
        idx = int(np.argmax(profits))
        return float(candidates[idx]), {"reason": "greedy_fallback_n0", "profit_est": float(profits[idx])}

    # Вызов бандита
    idx = bandit.select(group_id, ctx_matrix, action_matrix)

    # Если бандит вернул -1 (наша защита в select), тоже greedy
    if idx is None or idx < 0 or idx >= candidates.size:
        idx = int(np.argmax(profits))
        return float(candidates[idx]), {"reason": "greedy_fallback_idx", "profit_est": float(profits[idx])}

    chosen = float(candidates[idx])
    return chosen, {
        "reason": "linucb",
        "profit_est": float(profits[idx]),
        "all_profit_est": profits.tolist()
    }



# ==================================
# 8) Тренировка, оценка, предсказание
# ==================================

# 4) Полный код walk_forward_eval с обновлением бандита через фиксированный action-вектор

def walk_forward_eval(weekly_df, weeks_sorted, horizon_eval=12):
    pipe, _ = build_demand_model()
    bandit = LinUCB(alpha=1.0)

    results = []
    last_prices = {}

    for i in range(horizon_eval, len(weeks_sorted)):
        train_weeks = weeks_sorted[i - horizon_eval: i]
        pred_week = weeks_sorted[i]

        # Обучение модели спроса
        train_data = weekly_df[weekly_df["WEEK_ID"].isin(train_weeks)].copy()
        pipe = fit_demand_model(pipe, train_data)

        # Предсказание/выбор цен на неделю pred_week (3 бакета)
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

            # Псевдо-награда по оценке прибыли (в проде сюда идут факты)
            profit_hat = info.get("profit_est", 0.0)
            reward = profit_hat / (1.0 + abs(profit_hat))

            # Контекстная часть для обновления
            ctx_cols = [
                "IS_PROMO_NOW", "ONLINE_SHARE", "VOLATILITY_QTY_4W",
                "FAM_SALE_QTY", "CAT_SALE_QTY", "SEG_SALE_QTY", "REG_SALE_QTY", "STT_SALE_QTY",
                "ELAST1", "ELAST4", "AGE_WEEKS", "RISK_OOS"
            ]
            ctx = np.array([row.get(c, 0.0) for c in ctx_cols], dtype=float)

            # Фиксированный action-вектор на основе выбранной цены
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

        results.append(pd.DataFrame(week_chosen))

    if len(results) == 0:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)



# 5) Исправленный predict_next_week_prices с фиксированным action-вектором (глобальный one-hot по ARM_GRID)

def predict_next_week_prices(weekly_df, next_week_id):
    # Проверяем наличие недели и достаточной истории (≥12 недель до неё)
    all_weeks = sorted(weekly_df["WEEK_ID"].unique())
    if next_week_id not in all_weeks:
        raise ValueError("next_week_id отсутствует в weekly_df (нужен шаблон строк на эту неделю).")
    idx = all_weeks.index(next_week_id)
    if idx < 12:
        raise ValueError("Недостаточно истории (<12 недель) для обучения.")

    # Тренируем модель спроса на последних 12 неделях до целевой
    train_weeks = all_weeks[idx - 12: idx]
    pipe, _ = build_demand_model()
    bandit = LinUCB(alpha=1.0)

    train_data = weekly_df[weekly_df["WEEK_ID"].isin(train_weeks)].copy()
    pipe = fit_demand_model(pipe, train_data)

    # Для ограничения шага цены используем цену предыдущей недели по тому же бакету
    prev_week_id = all_weeks[idx - 1]
    prev_df = weekly_df[weekly_df["WEEK_ID"] == prev_week_id][
        ["STORE", "PRODUCT_CODE", "BUCKET", "SALE_PRICE", "BASE_PRICE"]
    ].copy()
    prev_df = prev_df.rename(columns={"SALE_PRICE": "LAST_SALE_PRICE"})
    prev_key = list(zip(prev_df["STORE"], prev_df["PRODUCT_CODE"], prev_df["BUCKET"]))
    prev_map = {k: p for k, p in zip(prev_key, prev_df["LAST_SALE_PRICE"])}

    # Ряды для предсказания на целевую неделю
    pred_rows = weekly_df[weekly_df["WEEK_ID"] == next_week_id].copy()
    pred_rows = pred_rows.sort_values(["STORE", "PRODUCT_CODE", "BUCKET"])

    recommendations = []

    for _, row in pred_rows.iterrows():
        key_lp = (row["STORE"], row["PRODUCT_CODE"], row["BUCKET"])
        last_price = prev_map.get(key_lp, row["BASE_PRICE"])
        group_id = f"{row['STORE']}|{row['SEGMENT_CODE']}|{row['BUCKET']}"

        # Выбор цены с учётом фиксированного action-вектора в choose_price_for_row
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



# ==========================================
# 9) Главная функция: от сырых данных к выходу
# ==========================================

def run_pipeline(raw_df, eval_weeks=12, next_week_id=None):
    """
    Полный цикл:
    1) Преобразовать сырые данные -> prepare_raw
    2) Построить недельные/бакетные фичи -> weekly_aggregates
    3) Walk-forward оценка на 12 недель
    4) (опционально) Предсказать цены на следующую неделю next_week_id
    Возвращает: eval_df, next_df (может быть None)
    """
    df = prepare_raw(raw_df)
    wk = weekly_aggregates(df)

    # Порядок недель
    weeks_sorted = sorted(wk["WEEK_ID"].unique())

    # Оценка
    eval_df = walk_forward_eval(wk, weeks_sorted, horizon_eval=eval_weeks)

    # Предсказание на следующую неделю (если указан идентификатор и в данных есть шаблон строк)
    next_df = None
    if next_week_id is not None:
        next_df = predict_next_week_prices(wk, next_week_id)

    return eval_df, next_df


# ==========================================
# 10) Пример использования (скелет)
# ==========================================

if __name__ == "__main__":
    # Пример: ожидается, что raw_df уже загружен из вашей DWH/CSV и имеет нужные колонки.
    # raw_df = pd.read_csv("transactions.csv", parse_dates=["TRADE_DT"])
    # Здесь просто создадим пустую заглушку структуры:
    cols = [
        "TRADE_DT", "IS_PROMO", "SALE_PRICE", "SALE_PRICE_ONLINE",
        "SALE_QTY", "SALE_QTY_ONLINE", "LOSS_QTY", "RETURN_QTY", "DELIVERY_QTY",
        "START_STOCK", "END_STOCK", "PRODUCT_CODE", "FAMILY_CODE", "CATEGORY_CODE",
        "SEGMENT_CODE", "STORE", "STORE_TYPE", "REGION_NAME", "BASE_PRICE",
        "PROMO_PERIOD", "PLACE_TYPE"
    ]
    raw_df = pd.DataFrame(columns=cols)

    # В бою: перед запуском недели формируете в weekly_df строки для next_week_id (STORE×SKU×3 бакета) с BASE_PRICE и фичами,
    # где фактические продажи на эту неделю пустые. Это позволит predict_next_week_prices выдать рекомендации.

    # Пример скелета запуска:
    # eval_df, next_df = run_pipeline(raw_df, eval_weeks=12, next_week_id="2025-W39")
    # print(eval_df.head())
    # print(next_df.head())
    pass
