# ==========================================================
# ОФЛАЙН ЦЕНООБРАЗОВАНИЕ С КОНТЕКСТНЫМИ БАНДИТАМИ (Jupyter)
# ==========================================================

import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta

# -------------------------
# 0) Глобальные настройки
# -------------------------

# Корзины: цена фиксируется внутри корзины, меняется между ними
BUCKETS = {"Mon-Thu":[0,1,2,3], "Fri":[4], "Sat-Sun":[5,6]}
BUCKET_TO_ORD = {"Mon-Thu":0, "Fri":1, "Sat-Sun":2}
ORD_TO_BUCKET = {v:k for k,v in BUCKET_TO_ORD.items()}
BUCKET_ORDER = ["Mon-Thu","Fri","Sat-Sun"]

# Мультипликаторы цен (действия)
PRICE_MULTS = np.round(np.arange(0.90, 1.301, 0.05), 2)   # 0.90..1.30

# Константы
END_OF_HISTORY = pd.to_datetime("2025-09-15")
ROLL_BUCKETS = 12                 # окно истории в корзинах
COST_RATE = 0.70                  # грубая себестоимость = 70% BASE_PRICE
MIN_MARGIN_RATE = 0.05            # минимум маржи (5%) как гард
USE_SYNTHETIC = True              # включите False и подставьте свою загрузку

# ==========================================================
# 1) Утилиты календаря и парсинга промо
# ==========================================================

def iso_monday(year, week):
    # Возвращает понедельник ISO-недели
    return pd.Timestamp.fromisocalendar(int(year), int(week), 1)

def dow_to_bucket(dow):
    # Преобразует номер дня недели к корзине
    for b, dows in BUCKETS.items():
        if dow in dows:
            return b
    return "Mon-Thu"

def bucket_bounds(week_start, bucket_id):
    # Старт/конец корзины (для оценки активной промо и т.п.)
    if bucket_id == "Mon-Thu":
        start = week_start + pd.Timedelta(days=0)
        end   = week_start + pd.Timedelta(days=3, hours=23, minutes=59, seconds=59)
    elif bucket_id == "Fri":
        start = week_start + pd.Timedelta(days=4)
        end   = week_start + pd.Timedelta(days=4, hours=23, minutes=59, seconds=59)
    else:
        start = week_start + pd.Timedelta(days=5)
        end   = week_start + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
    return start, end

def parse_promo_period_to_intervals(x):
    # Парс строки 'dd-mm-YYYY - dd-mm-YYYY' -> (start,end)
    if not isinstance(x, str) or "-" not in x:
        return pd.NaT, pd.NaT
    try:
        if " - " in x:
            l, r = x.split(" - ")
            start = pd.to_datetime(l.strip(), format="%d-%m-%Y", errors="coerce")
            end   = pd.to_datetime(r.strip(), format="%d-%m-%Y", errors="coerce")
        else:
            parts = [t.strip() for t in x.split("-")]
            if len(parts) >= 6:
                start = f"{parts[0]}-{parts[1]}-{parts[2]}"
                end   = f"{parts[3]}-{parts[4]}-{parts[5]}"
                start = pd.to_datetime(start, format="%d-%m-%Y", errors="coerce")
                end   = pd.to_datetime(end,   format="%d-%m-%Y", errors="coerce")
            else:
                return pd.NaT, pd.NaT
    except:
        return pd.NaT, pd.NaT
    return start, end

def promo_active_in_interval(promo_table, key_tuple, start, end):
    # Проверяет, есть ли промо для (STORE, PRODUCT_CODE) пересекающее [start,end]
    # promo_table: DataFrame с колонками: STORE, PRODUCT_CODE, PROMO_START, PROMO_END
    s, p = key_tuple
    subset = promo_table[(promo_table["STORE"]==s) & (promo_table["PRODUCT_CODE"]==p)]
    if subset.empty:
        return 0
    # Пересечение интервалов
    return int(((subset["PROMO_START"] <= end) & (subset["PROMO_END"] >= start)).any())

# ==========================================================
# 2) Загрузка/генерация данных (day-level)
#    Требуются колонки из условия; ниже — синтетика для примера.
# ==========================================================

if USE_SYNTHETIC:
    np.random.seed(42)
    start_date = END_OF_HISTORY - pd.Timedelta(days=7*22-1)  # ~22 недели
    dates = pd.date_range(start_date, END_OF_HISTORY, freq="D")
    stores = ["S1","S2"]
    skus   = [f"SKU{i}" for i in range(8)]
    families = {f"SKU{i}": f"F{1+i%3}" for i in range(8)}
    cats     = {"F1":"C1","F2":"C2","F3":"C3"}
    segs     = {"C1":"SEGA","C2":"SEGB","C3":"SEGC"}

    rows = []
    for d in dates:
        for s in stores:
            for p in skus:
                base_price = 8 + 2*(int(p[-1])%4)
                # случайные промо-интервалы
                is_promo_today = np.random.rand() < 0.12
                price = base_price * np.random.choice([0.95, 1.00, 1.05])
                if is_promo_today:
                    price = base_price * 0.9
                beta = 0.12 + 0.03*(int(p[-1])%3)  # коэффициент чувствительности
                mu = np.exp(2.2 - beta*(price/base_price))
                qty = np.random.poisson(mu)
                start_stock = 100
                end_stock = max(0, start_stock - qty + np.random.randint(0,3))
                # создадим редкие строки с будущими промо-интервалами
                if np.random.rand()<0.02:
                    # промо на будущую неделю
                    ps = d + pd.Timedelta(days=np.random.randint(7, 21))
                    pe = ps + pd.Timedelta(days=np.random.randint(1, 3))
                    promo_str = f"{ps.strftime('%d-%m-%Y')} - {pe.strftime('%d-%m-%Y')}"
                else:
                    promo_str = f"{(d-pd.Timedelta(days=1)).strftime('%d-%m-%Y')} - {d.strftime('%d-%m-%Y')}" if is_promo_today else None
                rows.append({
                    "TRADE_DT": d,
                    "IS_PROMO": int(is_promo_today),
                    "SALE_QTY": qty,
                    "SALE_PRICE": price,
                    "START_STOCK": start_stock,
                    "END_STOCK": end_stock,
                    "PRODUCT_CODE": p,
                    "FAMILY_CODE": families[p],
                    "CATEGORY_CODE": cats[families[p]],
                    "SEGMENT_CODE": segs[cats[families[p]]],
                    "STORE": s,
                    "STORE_TYPE": "A" if s=="S1" else "B",
                    "REGION_NAME": "North" if s=="S1" else "South",
                    "BASE_PRICE": base_price,
                    "PROMO_PERIOD": promo_str,
                    "PLACE_TYPE": None,
                })
    df = pd.DataFrame(rows)
else:
    # Замените на вашу загрузку:
    # df = pd.read_parquet("your_data.parquet")
    # df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"])
    raise ValueError("Подставьте загрузку данных и выключите USE_SYNTHETIC.")

# ==========================================================
# 3) Предобработка day-level и агрегация до weekly×bucket
# ==========================================================

# Парсим промо интервалы
df["PROMO_START"], df["PROMO_END"] = zip(*df["PROMO_PERIOD"].map(parse_promo_period_to_intervals))
df["DOW"] = df["TRADE_DT"].dt.weekday
df["BUCKET_ID"] = df["DOW"].map(dow_to_bucket)
df["WEEK_ID"] = df["TRADE_DT"].dt.isocalendar().week.astype(int)
df["YEAR"]    = df["TRADE_DT"].dt.isocalendar().year.astype(int)
df["YEARWEEK"]= df["YEAR"]*100 + df["WEEK_ID"]

# Таблица известных промо-интервалов (для будущего)
promo_table = (
    df[["STORE","PRODUCT_CODE","PROMO_START","PROMO_END"]]
    .dropna()
    .drop_duplicates()
    .reset_index(drop=True)
)

# Скелет всех комбинаций (STORE×SKU×YEARWEEK×BUCKET)
keys = ["STORE","PRODUCT_CODE"]
calendar = df[["YEAR","WEEK_ID","YEARWEEK","BUCKET_ID"]].drop_duplicates()
universe = df[keys + ["STORE_TYPE","REGION_NAME","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE"]].drop_duplicates()
skeleton = universe.merge(calendar, how="cross")

# Агрегация
agg = df.groupby(keys + ["YEAR","WEEK_ID","YEARWEEK","BUCKET_ID"], as_index=False).agg({
    "SALE_QTY":"sum",
    "SALE_PRICE":"mean",
    "BASE_PRICE":"mean",
    "START_STOCK":"mean",
    "END_STOCK":"mean",
})
weekly = skeleton.merge(agg, on=keys+["YEAR","WEEK_ID","YEARWEEK","BUCKET_ID"], how="left")

# Заполнения: нули/ffill (аккуратно в бою)
weekly["SALE_QTY"] = weekly["SALE_QTY"].fillna(0)
for c in ["SALE_PRICE","BASE_PRICE","START_STOCK","END_STOCK"]:
    weekly[c] = weekly[c].groupby(keys).ffill()

# Якорная цена без Series.replace ловушки
weekly["ANCHOR_PRICE"] = np.where(
    weekly["SALE_PRICE"].isna() | weekly["SALE_PRICE"].eq(0),
    weekly["BASE_PRICE"],
    weekly["SALE_PRICE"]
)

# Календарь недель и корзин
weekly["WEEK_START"] = [iso_monday(y,w) for y,w in zip(weekly["YEAR"], weekly["WEEK_ID"])]
weekly["BUCKET_ORD"] = weekly["BUCKET_ID"].map(BUCKET_TO_ORD).astype(int)
weekly["BUCKET_START"], weekly["BUCKET_END"] = zip(*[
    bucket_bounds(ws, b) for ws,b in zip(weekly["WEEK_START"], weekly["BUCKET_ID"])
])

# Промо активно в корзине (по исходным дням через max)
# Если day-level промо не доступно после агрегации — используем интервалы:
weekly["PROMO_ACTIVE_BUCKET"] = [
    promo_active_in_interval(promo_table, (r.STORE, r.PRODUCT_CODE), r.BUCKET_START, r.BUCKET_END)
    for r in weekly.itertuples(index=False)
]

# ==========================================================
# 4) Роллинговые фичи по корзинам
# ==========================================================

def add_bucket_roll_feats(g):
    # Добавляет rolling-фичи на окне последних ROLL_BUCKETS корзин
    g = g.sort_values(["WEEK_START","BUCKET_ORD"])
    g["QTY_12B"]  = g["SALE_QTY"].rolling(ROLL_BUCKETS, min_periods=1).sum()
    g["REV_12B"]  = (g["SALE_QTY"]*g["ANCHOR_PRICE"]).rolling(ROLL_BUCKETS, min_periods=1).sum()
    g["AVG_P_12B"]= g["ANCHOR_PRICE"].rolling(ROLL_BUCKETS, min_periods=1).mean()
    g["PRM_12B"]  = g["PROMO_ACTIVE_BUCKET"].rolling(ROLL_BUCKETS, min_periods=1).mean()
    # эластичность-прокси: ковариация qty и лог-цены / var лог-цены
    lp = np.log(np.clip(g["ANCHOR_PRICE"].fillna(method="ffill").replace(0, np.nan), 0.01, None))
    q  = g["SALE_QTY"].fillna(0)
    lp_ma = lp.rolling(ROLL_BUCKETS).mean()
    q_ma  = q.rolling(ROLL_BUCKETS).mean()
    cov = ((lp - lp_ma)*(q - q_ma)).rolling(ROLL_BUCKETS).sum()
    var = ((lp - lp_ma)**2).rolling(ROLL_BUCKETS).sum()
    g["ELAST_PROXY"] = (cov/(var+1e-6)).fillna(0.0)
    # очень грубый риск OOS
    g["OOS_RISK"] = (g["END_STOCK"] < q.rolling(3, min_periods=1).mean()*2).astype(int)
    return g

weekly = (weekly
          .sort_values(keys+["WEEK_START","BUCKET_ORD"])
          .groupby(keys, as_index=False, group_keys=False)
          .apply(add_bucket_roll_feats))

# ==========================================================
# 5) Леддер цен и бизнес-гарды
# ==========================================================

def snap_ending_99(x):
    # Округляем к .99 (безопасно для >0)
    if x <= 0:
        return x
    return np.floor(x) + 0.99

def build_price_candidates(anchor_price, base_price, is_promo):
    # Разные пределы для промо/непромо
    raw = anchor_price * PRICE_MULTS
    if is_promo:
        low, high = 0.70*base_price, 1.10*base_price
    else:
        low, high = 0.85*base_price, 1.50*base_price
    clipped = np.clip(raw, low, high)
    snapped = np.array([snap_ending_99(v) for v in clipped])
    # Убираем отрицательные/NaN и дубликаты
    snapped = snapped[~np.isnan(snapped)]
    snapped = snapped[snapped>0]
    return np.unique(np.round(snapped, 2))

def guard_min_margin(candidate_price, base_price):
    # Гард по минимальной марже: (p - cost)/p >= MIN_MARGIN_RATE
    cost = COST_RATE * max(base_price, 0.01)
    if candidate_price <= cost:
        return False
    margin_rate = (candidate_price - cost)/candidate_price
    return margin_rate >= MIN_MARGIN_RATE

# ==========================================================
# 6) Простой LinUCB и менеджер иерархии
# ==========================================================

class LinUCB:
    # Базовый контекстный бандит с UCB
    def __init__(self, d, alpha):
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d)
        self.b = np.zeros(d)

    def theta(self):
        return np.linalg.solve(self.A, self.b)

    def predict_ucb(self, X):
        A_inv = np.linalg.inv(self.A)
        th = self.theta()
        mu = X.dot(th)
        s = np.sqrt(np.sum(X.dot(A_inv)*X, axis=1))
        return mu + self.alpha * s

    def update(self, x, reward):
        self.A += np.outer(x, x)
        self.b += x * reward

class BanditManager:
    # Держит бандиты на 2 уровнях: локальный (SEGMENT×STORE_TYPE×REGION×BUCKET_ORD) и глобальный (только BUCKET_ORD)
    def __init__(self, d, alpha):
        self.d = d
        self.alpha = alpha
        self.local = {}   # ключ: (seg, stype, region, bucket_ord)
        self.global_ = {} # ключ: bucket_ord

    def get_local(self, seg, stype, region, bucket_ord):
        k = (seg, stype, region, bucket_ord)
        if k not in self.local:
            self.local[k] = LinUCB(self.d, self.alpha)
        return self.local[k]

    def get_global(self, bucket_ord):
        if bucket_ord not in self.global_:
            self.global_[bucket_ord] = LinUCB(self.d, self.alpha)
        return self.global_[bucket_ord]

    def choose_ucb(self, X, seg, stype, region, bucket_ord):
        # Сначала пробуем локальную модель; если «сырая», можно усреднить с глобальной
        loc = self.get_local(seg, stype, region, bucket_ord)
        glob = self.get_global(bucket_ord)
        # Простой трюк: берём максимум UCB между локальной и глобальной
        u_loc = loc.predict_ucb(X)
        u_glb = glob.predict_ucb(X)
        u = np.maximum(u_loc, u_glb)
        return int(np.argmax(u))

    def update_both(self, x, reward, seg, stype, region, bucket_ord):
        self.get_local(seg, stype, region, bucket_ord).update(x, reward)
        self.get_global(bucket_ord).update(x, reward)

# Функция формирования вектора признаков
def make_features_row(r, candidate_price, promo_flag_for_target):
    base_price = r["BASE_PRICE"] if r["BASE_PRICE"]>0 else max(r["ANCHOR_PRICE"], 0.01)
    log_ratio = np.log(max(candidate_price, 0.01)/max(base_price, 0.01))
    return np.array([
        1.0,
        log_ratio,
        float(promo_flag_for_target),
        float(r["OOS_RISK"]),
        float(r["ELAST_PROXY"]),
        float(r["PRM_12B"]),
        float(r["QTY_12B"]>0),
    ])

# Инициализация менеджера бандитов для 3 корзин (размерность d=7)
BANDIT_D = 7
ALPHA = 1.0
bandits = BanditManager(d=BANDIT_D, alpha=ALPHA)

# ==========================================================
# 7) Offline replay для тёплого старта (по истории)
# ==========================================================

def nearest_candidate(cands, observed_price):
    if len(cands)==0:
        return None
    idx = np.argmin(np.abs(cands - observed_price))
    return cands[idx]

def run_offline_replay(weekly_df):
    # Обновляет бандиты, проходя историю по порядку корзин
    ordered = weekly_df.sort_values(["WEEK_START","BUCKET_ORD"])
    for r in ordered.itertuples(index=False):
        anchor = r.ANCHOR_PRICE if pd.notna(r.ANCHOR_PRICE) and r.ANCHOR_PRICE>0 else r.BASE_PRICE
        cands = build_price_candidates(anchor, r.BASE_PRICE, r.PROMO_ACTIVE_BUCKET==1)
        if len(cands)==0:
            continue
        observed = r.SALE_PRICE if pd.notna(r.SALE_PRICE) and r.SALE_PRICE>0 else anchor
        chosen = nearest_candidate(cands, observed)
        if chosen is None:
            continue
        x = make_features_row(weekly_df.loc[r.Index] if hasattr(r, "Index") else weekly_df.iloc[0],
                              chosen, r.PROMO_ACTIVE_BUCKET)
        # Более простой и быстрый способ — собрать вручную из r:
        base_price = r.BASE_PRICE if r.BASE_PRICE>0 else anchor
        promo_flag = r.PROMO_ACTIVE_BUCKET
        row = {
            "BASE_PRICE": base_price,
            "ANCHOR_PRICE": anchor,
            "OOS_RISK": r.OOS_RISK,
            "ELAST_PROXY": r.ELAST_PROXY,
            "PRM_12B": r.PRM_12B,
            "QTY_12B": r.QTY_12B,
        }
        x = make_features_row(row, chosen, promo_flag)
        cost = COST_RATE * max(base_price, 0.01)
        reward = (chosen - cost) * r.SALE_QTY
        bandits.update_both(x, reward, r.SEGMENT_CODE, r.STORE_TYPE, r.REGION_NAME, r.BUCKET_ORD)

run_offline_replay(weekly)

# ==========================================================
# 8) Определяем «следующую корзину» и строим рекомендации
# ==========================================================

def next_bucket_from_history(weekly_df):
    last_ws  = weekly_df["WEEK_START"].max()
    last_ord = weekly_df.loc[weekly_df["WEEK_START"].eq(last_ws), "BUCKET_ORD"].max()
    if last_ord < 2:
        target_ws, target_ord = last_ws, last_ord+1
    else:
        target_ws, target_ord = last_ws + pd.Timedelta(weeks=1), 0
    return target_ws, target_ord, ORD_TO_BUCKET[target_ord]

target_ws, target_ord, target_bucket = next_bucket_from_history(weekly)

def latest_state_per_item(weekly_df):
    # Последняя доступная строка на каждого (STORE, SKU)
    latest = (weekly_df.sort_values(["WEEK_START","BUCKET_ORD"])
                        .groupby(["STORE","PRODUCT_CODE"], as_index=False)
                        .tail(1)
                        .copy())
    return latest

latest = latest_state_per_item(weekly)

def recommend_for_next_bucket(latest_df, target_ws, target_ord, promo_table):
    recs = []
    for r in latest_df.itertuples(index=False):
        # Определяем промо-флаг в целевой корзине по интервалам
        start, end = bucket_bounds(target_ws, ORD_TO_BUCKET[target_ord])
        promo_next = promo_active_in_interval(promo_table, (r.STORE, r.PRODUCT_CODE), start, end)

        anchor = r.ANCHOR_PRICE if pd.notna(r.ANCHOR_PRICE) and r.ANCHOR_PRICE>0 else r.BASE_PRICE
        cands = build_price_candidates(anchor, r.BASE_PRICE, promo_next==1)
        # Бизнес-гарды: мин. маржа
        cands = np.array([c for c in cands if guard_min_margin(c, r.BASE_PRICE)])
        if len(cands)==0:
            continue

        # Формируем X для всех кандидатов
        row = {
            "BASE_PRICE": r.BASE_PRICE,
            "ANCHOR_PRICE": anchor,
            "OOS_RISK": r.OOS_RISK,
            "ELAST_PROXY": r.ELAST_PROXY,
            "PRM_12B": r.PRM_12B,
            "QTY_12B": r.QTY_12B,
        }
        X = np.vstack([make_features_row(row, cp, promo_next) for cp in cands])

        # Выбор действия из иерархического менеджера
        best_idx = bandits.choose_ucb(X, r.SEGMENT_CODE, r.STORE_TYPE, r.REGION_NAME, target_ord)
        best_price = float(cands[best_idx])

        recs.append({
            "STORE": r.STORE,
            "STORE_TYPE": r.STORE_TYPE,
            "REGION_NAME": r.REGION_NAME,
            "PRODUCT_CODE": r.PRODUCT_CODE,
            "SEGMENT_CODE": r.SEGMENT_CODE,
            "FAMILY_CODE": r.FAMILY_CODE,
            "CATEGORY_CODE": r.CATEGORY_CODE,
            "TARGET_WEEK_START": target_ws.date(),
            "TARGET_BUCKET": ORD_TO_BUCKET[target_ord],
            "BASE_PRICE": float(r.BASE_PRICE),
            "ANCHOR_PRICE": float(anchor),
            "PROMO_IN_TARGET": int(promo_next),
            "RECOMMENDED_PRICE": best_price
        })
    return pd.DataFrame(recs).sort_values(["STORE","PRODUCT_CODE"]).reset_index(drop=True)

recs_df = recommend_for_next_bucket(latest, target_ws, target_ord, promo_table)

print("Рекомендации на следующую корзину:",
      target_bucket, "— неделя с", target_ws.date())
recs_df.head(20)
