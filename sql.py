SELECT
    product_code,
    toDate(trade_dt) AS trade_dt,
    sale_price,
    base_price
FROM sales
WHERE store_id IN (101, 205, 330)
  AND segment_code IN (1, 3)
  AND trade_dt >= toDateTime('2024-12-20 00:00:00')   -- min_monday - 14d
  AND trade_dt <  toDateTime('2025-02-01 00:00:00')   -- max_monday + 7d (опционально)


import pandas as pd
import numpy as np

# --- 0) привести названия (если нужно)
df_sales = df_sales.rename(columns={
    "product_code": "PRODUCT_CODE",
    "trade_dt": "TRADE_DT",
    "sale_price": "SALE_PRICE",
    "base_price": "BASE_PRICE",
})

# --- 1) типы
df_sales["TRADE_DT"] = pd.to_datetime(df_sales["TRADE_DT"]).dt.date
df_now["MONDAY_DATE"] = pd.to_datetime(df_now["MONDAY_DATE"]).dt.date

# --- 2) для каждого BUCKET_NOW считаем "последний день предыдущего бакета"
# MON_THU -> prev Sunday = monday - 1
# FRI     -> Thursday    = monday + 3
# SAT_SUN -> Friday      = monday + 4
offset_map = {"MON_THU": -1, "FRI": 3, "SAT_SUN": 4}

df_now["PREV_BUCKET_LAST_DAY"] = df_now["MONDAY_DATE"].map(
    lambda d: d  # placeholder
)

df_now["PREV_BUCKET_LAST_DAY"] = df_now.apply(
    lambda r: r["MONDAY_DATE"] + pd.Timedelta(days=offset_map[r["BUCKET_NOW"]]),
    axis=1
).dt.date

# --- 3) ищем последнюю цену <= PREV_BUCKET_LAST_DAY для каждого продукта
# Для скорости: отсортируем и возьмем merge_asof (он очень быстрый на больших данных)
df_sales_sorted = df_sales.sort_values(["PRODUCT_CODE", "TRADE_DT"])
df_keys = df_now[["PRODUCT_CODE", "MONDAY_DATE", "BUCKET_NOW", "PREV_BUCKET_LAST_DAY"]].copy()
df_keys = df_keys.sort_values(["PRODUCT_CODE", "PREV_BUCKET_LAST_DAY"])

# merge_asof работает только по одному ключу + by, поэтому делаем по PRODUCT_CODE
prev = pd.merge_asof(
    df_keys,
    df_sales_sorted,
    left_on="PREV_BUCKET_LAST_DAY",
    right_on="TRADE_DT",
    by="PRODUCT_CODE",
    direction="backward",   # <=
    allow_exact_matches=True
)

prev_prices = prev[["PRODUCT_CODE", "MONDAY_DATE", "BUCKET_NOW", "SALE_PRICE", "BASE_PRICE"]].rename(columns={
    "SALE_PRICE": "SALE_PRICE_PREV_BUCKET",
    "BASE_PRICE": "BASE_PRICE_PREV_BUCKET"
})

# --- 4) merge обратно в df_now
df_out = df_now.merge(
    prev_prices,
    on=["PRODUCT_CODE", "MONDAY_DATE", "BUCKET_NOW"],
    how="left"
)
