# === Библиотеки ===
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ==========================================================
# 0) Настройки и вспомогательные таблицы
# ==========================================================

# Корзины: цена фиксируется внутри корзины, меняется только между ними
BUCKETS = {
    "Mon-Thu": [0,1,2,3],   # Пн..Чт (0=Monday)
    "Fri":     [4],         # Пт
    "Sat-Sun": [5,6],       # Сб..Вс
}
BUCKET_ORDER = ["Mon-Thu", "Fri", "Sat-Sun"]

# Дискретные действия: мультипликаторы к "якорной" цене
PRICE_MULTS = np.round(np.arange(0.90, 1.301, 0.05), 2)  # 0.90..1.30 с шагом 0.05

END_OF_HISTORY = pd.to_datetime("2025-09-15")  # как в условиях
ROLL_WEEKS = 12  # окно истории для фич

# Если данных нет — можно включить игрушечную генерацию
USE_SYNTHETIC = True

# ==========================================================
# 1) Загрузка/создание данных
# Столбцы из задачи (минимальный поднабор для примера):
# TRADE_DT, IS_PROMO, SALE_QTY, SALE_PRICE, START_STOCK, END_STOCK,
# PRODUCT_CODE, FAMILY_CODE, CATEGORY_CODE, SEGMENT_CODE,
# STORE, STORE_TYPE, REGION_NAME, BASE_PRICE, PROMO_PERIOD, PLACE_TYPE
# + допускаем SALE_PRICE_TOTAL и пр., если есть.
# ==========================================================

if USE_SYNTHETIC:
    np.random.seed(7)
    # Сгенерируем ~20 недель истории для 2 магазинов × 6 SKU
    start_date = END_OF_HISTORY - pd.Timedelta(days=7*20-1)
    dates = pd.date_range(start_date, END_OF_HISTORY, freq="D")
    stores = ["S1", "S2"]
    skus = [f"SKU{i}" for i in range(6)]
    families = {"SKU0":"F1","SKU1":"F1","SKU2":"F2","SKU3":"F2","SKU4":"F3","SKU5":"F3"}
    cats = {"F1":"C1","F2":"C2","F3":"C3"}
    segs = {"C1":"SEGA","C2":"SEGB","C3":"SEGC"}

    rows = []
    for d in dates:
        for s in stores:
            for p in skus:
                base_price = 10 + 2*(int(p[-1])%3)
                is_promo = np.random.rand() < 0.15
                # спрос ~ exp(α - β*price) + шум
                true_beta = 0.15 + 0.05*(int(p[-1])%3)
                price = base_price * np.random.choice([0.95, 1.0, 1.05])
                if is_promo:
                    price = base_price * 0.9
                mu = np.exp(2.2 - true_beta*(price/base_price))
                qty = np.random.poisson(mu)
                start_stock = 100
                end_stock = max(0, start_stock - qty + np.random.randint(0,3))
                promo_start = d - pd.Timedelta(days=np.random.randint(0,10))
                promo_end = promo_start + pd.Timedelta(days=np.random.randint(0,3))
                promo_str = f"{promo_start.strftime('%d-%m-%Y')} - {promo_end.strftime('%d-%m-%Y')}" if is_promo else None
                rows.append({
                    "TRADE_DT": d,
                    "IS_PROMO": int(is_promo),
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
    # Пример: df = pd.read_parquet("sales.parquet")
    # df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    # Для воспроизводимости оставим как заглушку:
    raise ValueError("Замените на загрузку ваших данных и выключите USE_SYNTHETIC.")

# ==========================================================
# 2) Парсинг PROMO_PERIOD в интервалы и признаки корзины
# ==========================================================

def parse_promo_period_to_intervals(x):
    # Парсим строку вида '01-01-2024 - 03-01-2024' -> (start,end)
    if not isinstance(x, str) or "-" not in x:
        return pd.NaT, pd.NaT
    # Строка может содержать пробелы; разделим по ' - '
    parts = [t.strip() for t in x.split("-")]
    # На всякий случай склеим 2 части обратно на формат 'dd-mm-yyyy - dd-mm-yyyy'
    if len(parts) >= 4:
        start = f"{parts[0]}-{parts[1]}-{parts[2]}"
        end = f"{parts[3]}-{parts[4]}-{parts[5]}" if len(parts)>=6 else None
    else:
        # более простой разбор
        try:
            l, r = x.split(" - ")
            start = l.strip()
            end = r.strip()
        except:
            return pd.NaT, pd.NaT
    try:
        start = pd.to_datetime(start, format="%d-%m-%Y", errors="coerce")
        end = pd.to_datetime(end, format="%d-%m-%Y", errors="coerce")
    except:
        return pd.NaT, pd.NaT
    return start, end

df["PROMO_START"], df["PROMO_END"] = zip(*df["PROMO_PERIOD"].map(parse_promo_period_to_intervals))

# День недели, неделя ISO и корзина
df["DOW"] = df["TRADE_DT"].dt.weekday
def dow_to_bucket(dow):
    for b, dows in BUCKETS.items():
        if dow in dows:
            return b
    return "Mon-Thu"

df["BUCKET_ID"] = df["DOW"].map(dow_to_bucket)
df["WEEK_ID"] = df["TRADE_DT"].dt.isocalendar().week.astype(int)
df["YEAR"] = df["TRADE_DT"].dt.isocalendar().year.astype(int)
df["YEARWEEK"] = df["YEAR"]*100 + df["WEEK_ID"]

# Флаг "промо активно в этот день"
df["IS_PROMO_ACTIVE"] = (
    (df["TRADE_DT"] >= df["PROMO_START"]) &
    (df["TRADE_DT"] <= df["PROMO_END"])
).fillna(False).astype(int)

# ==========================================================
# 3) Агрегация до уровня (STORE, PRODUCT, YEARWEEK, BUCKET_ID)
#    и подготовка «скелета» строк (чтобы присутствовали нули)
# ==========================================================

# Список всех комбинаций для скелета
keys = ["STORE","PRODUCT_CODE"]
calendar = df[["YEARWEEK","BUCKET_ID","YEAR"]].drop_duplicates()
universe_left = df[keys].drop_duplicates()
skeleton = universe_left.merge(calendar, how="cross")

# Реальная агрегация
agg = df.groupby(keys + ["YEARWEEK","BUCKET_ID","YEAR"], as_index=False).agg({
    "SALE_QTY":"sum",
    "SALE_PRICE":"mean",
    "BASE_PRICE":"mean",
    "IS_PROMO_ACTIVE":"max",
    "START_STOCK":"mean",
    "END_STOCK":"mean",
})

# Соединяем: пропуски → нули
weekly = skeleton.merge(agg, on=keys+["YEARWEEK","BUCKET_ID","YEAR"], how="left")
for col in ["SALE_QTY","IS_PROMO_ACTIVE"]:
    weekly[col] = weekly[col].fillna(0)
for col in ["SALE_PRICE","BASE_PRICE","START_STOCK","END_STOCK"]:
    weekly[col] = weekly[col].fillna(method="ffill")  # мягкое заполнение по времени в реальном проекте делайте аккуратнее

# «Якорная» цена: последняя известная или BASE_PRICE
weekly["ANCHOR_PRICE"] = weekly["SALE_PRICE"].fillna(weekly["BASE_PRICE"])
weekly["ANCHOR_PRICE"] = weekly["ANCHOR_PRICE"].replace(0, weekly["BASE_PRICE"])

# ==========================================================
# 4) Признаки на 12-недельном окне (просто и понятно)
# ==========================================================

weekly = weekly.sort_values(keys+["YEARWEEK","BUCKET_ID"])

def add_roll_feats(g):
    g = g.sort_values(["YEARWEEK","BUCKET_ID"])
    # окно последних 12 недель: агрегации по корзинам внутри недели уже сделаны
    g["QTY_12W"] = g["SALE_QTY"].rolling(ROLL_WEEKS, min_periods=1).sum()
    g["REV_12W"] = (g["SALE_QTY"]*g["SALE_PRICE"]).rolling(ROLL_WEEKS, min_periods=1).sum()
    g["AVG_PRICE_12W"] = g["SALE_PRICE"].rolling(ROLL_WEEKS, min_periods=1).mean()
    g["PROMO_RATE_12W"] = g["IS_PROMO_ACTIVE"].rolling(ROLL_WEEKS, min_periods=1).mean()
    # простая «прокси-эластичность»: ковариация qty и price на окне
    qty = g["SALE_QTY"].fillna(0)
    prc = g["SALE_PRICE"].fillna(g["BASE_PRICE"])
    cov = (qty - qty.rolling(ROLL_WEEKS).mean())*(prc - prc.rolling(ROLL_WEEKS).mean())
    var = (prc - prc.rolling(ROLL_WEEKS).mean())**2
    g["ELAST_PROXY"] = (cov.rolling(ROLL_WEEKS).sum() / (var.rolling(ROLL_WEEKS).sum()+1e-6)).fillna(0.0)
    return g

weekly = weekly.groupby(keys, group_keys=False).apply(add_roll_feats)

# Запасы и риск OOS (очень грубо)
weekly["OOS_RISK"] = (weekly["END_STOCK"] < weekly["SALE_QTY"].rolling(3, min_periods=1).mean()*2).astype(int)

# ==========================================================
# 5) Price ladder + округление .99
# ==========================================================

def snap_ending_99(x):
    # Округляем так, чтобы получить ... .99 (примерно)
    r = np.floor(x) + 0.99
    if r <= 0:
        r = x
    return r

def build_price_candidates(anchor_price, base_price, is_promo):
    # Отдельные ладдеры можно сделать для промо/непромо; здесь — один общий
    raw = anchor_price * PRICE_MULTS
    # ограничение относительно BASE_PRICE (пример)
    low = 0.7 * base_price
    high = 1.5 * base_price
    clipped = np.clip(raw, low, high)
    # Округляем к .99
    snapped = np.array([snap_ending_99(v) for v in clipped])
    # Убираем дубликаты
    return np.unique(np.round(snapped, 2))

# ==========================================================
# 6) Простой контекстный бандит LinUCB (глобальный, для простоты)
#    Φ = [1, log_price_candidate/base_price, promo_flag, oos_risk, elast_proxy, ...]
# ==========================================================

class LinUCB:
    def __init__(self, d, alpha):
        # d — размерность признаков; alpha — коэффициент уверенности (exploration)
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d)
        self.b = np.zeros(d)

    def _theta(self):
        return np.linalg.solve(self.A, self.b)

    def predict_ucb(self, X):
        # X: матрица кандидатов (n_actions × d)
        A_inv = np.linalg.inv(self.A)
        theta = self._theta()
        mu = X.dot(theta)
        s = np.sqrt(np.sum(X.dot(A_inv)*X, axis=1))
        return mu + self.alpha * s

    def update(self, x, reward):
        # x — вектор признаков выбранного действия
        self.A += np.outer(x, x)
        self.b += x * reward

def make_features_row(r, candidate_price):
    # Минимальный набор контекста; расширяйте со временем
    base_price = r["BASE_PRICE"] if r["BASE_PRICE"]>0 else max(r["ANCHOR_PRICE"], 1.0)
    log_ratio = np.log(max(candidate_price, 0.01)/max(base_price, 0.01))
    return np.array([
        1.0,
        log_ratio,
        float(r["IS_PROMO_ACTIVE"]),
        float(r["OOS_RISK"]),
        float(r["ELAST_PROXY"]),
        float(r["QTY_12W"]>0),
    ])

# Инициализация бандита
d = 6
alpha = 1.0
bandit = LinUCB(d=d, alpha=alpha)

# ==========================================================
# 7) Offline "replay" для тёплого старта
#    Идея: считаем, что исторически выбранная цена ~ одно из действий (ближайший кандидат),
#    reward = прибыль за корзину = (price - "себестоимость")*qty (здесь себестоимость неизвестна -> используем долю от BASE_PRICE).
# ==========================================================

def nearest_candidate(cands, observed_price):
    idx = np.argmin(np.abs(cands - observed_price))
    return cands[idx]

# Сделаем грубую себестоимость как 0.7*BASE_PRICE (замените на вашу)
COST_RATE = 0.7

weekly = weekly.sort_values(keys+["YEARWEEK","BUCKET_ID"])
history_mask = weekly["TRADE_DT"].isna()  # нет TRADE_DT после агрегации; используем год/неделю как порядок
# Для простоты — используем все строки до END_OF_HISTORY (они и так до него)
history = weekly.copy()

# Реплей: проходим по истории в хронологическом порядке и обновляем бандит
for _, r in history.iterrows():
    anchor = r["ANCHOR_PRICE"] if r["ANCHOR_PRICE"]>0 else r["BASE_PRICE"]
    cands = build_price_candidates(anchor, r["BASE_PRICE"], r["IS_PROMO_ACTIVE"])
    if len(cands)==0:
        continue
    # Какая цена фактически была?
    observed_price = r["SALE_PRICE"] if r["SALE_PRICE"]>0 else anchor
    chosen = nearest_candidate(cands, observed_price)
    x = make_features_row(r, chosen)
    cost = COST_RATE * max(r["BASE_PRICE"], 0.01)
    reward = (chosen - cost) * r["SALE_QTY"]
    bandit.update(x, reward)

# ==========================================================
# 8) Рекомендации на «следующую корзину»
#    Выберем дату-таргет: следующий день после END_OF_HISTORY,
#    определим его корзину и сгенерируем рекомендации.
# ==========================================================

next_day = END_OF_HISTORY + pd.Timedelta(days=1)
target_dow = next_day.weekday()
# Какая корзина у целевого дня?
for b, dows in BUCKETS.items():
    if target_dow in dows:
        target_bucket = b
        break

# Берём последнее наблюдение по каждому (STORE, PRODUCT) перед целевой неделей/корзиной
weekly["DATE_PROXY"] = pd.to_datetime(weekly["YEAR"].astype(str) + "-1", format="%Y-%j") + \
                       (weekly["WEEK_ID"]-1).astype("timedelta64[W]")

latest = weekly.sort_values(keys+["YEARWEEK","BUCKET_ID"]).groupby(keys, as_index=False).tail(1).copy()
latest["TARGET_BUCKET"] = target_bucket

recs = []
for _, r in latest.iterrows():
    anchor = r["ANCHOR_PRICE"] if r["ANCHOR_PRICE"]>0 else r["BASE_PRICE"]
    cands = build_price_candidates(anchor, r["BASE_PRICE"], 0)
    if len(cands)==0:
        continue
    # Собираем X для всех кандидатов и считаем UCB
    X = np.vstack([make_features_row(r, cp) for cp in cands])
    ucb = bandit.predict_ucb(X)
    best_idx = int(np.argmax(ucb))
    best_price = float(cands[best_idx])
    recs.append({
        "STORE": r["STORE"],
        "PRODUCT_CODE": r["PRODUCT_CODE"],
        "TARGET_BUCKET": target_bucket,
        "RECOMMENDED_PRICE": best_price,
        "ANCHOR_PRICE": float(anchor),
        "BASE_PRICE": float(r["BASE_PRICE"]),
    })

recs_df = pd.DataFrame(recs).sort_values(["STORE","PRODUCT_CODE"]).reset_index(drop=True)
print("Рекомендации на следующую корзину:", target_bucket)
recs_df.head(20)
