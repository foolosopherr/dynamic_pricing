
import pandas as pd
import numpy as np

def fill_missing_dates_per_product(
    df: pd.DataFrame,
    product_col: str = "PRODUCT_CODE",
    date_col: str = "TRADE_DT",
    qty_col: str = "QTY",
    freq: str = "D",
) -> pd.DataFrame:
    df = df.copy()

    # 1) types
    df[product_col] = df[product_col].astype(str)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # уберём битые даты
    df = df[df[date_col].notna()].copy()

    # 2) уберём возможные дубликаты одной даты на товар:
    #    если их может быть несколько — выбери агрегирование.
    #    здесь: суммируем QTY, а остальное берём "последнее"
    other_cols = [c for c in df.columns if c not in [product_col, date_col, qty_col]]
    if df.duplicated([product_col, date_col]).any():
        df = (df.sort_values([product_col, date_col])
                .groupby([product_col, date_col], as_index=False)
                .agg({**{qty_col: "sum"}, **{c: "last" for c in other_cols}}))

    # 3) основной шаг: для каждого продукта — полный daily диапазон и ffill значений
    df = df.sort_values([product_col, date_col])

    def _expand_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index(date_col)

        full_idx = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g2 = g.reindex(full_idx)

        # QTY: для сгенерированных дат -> 0
        g2[qty_col] = g2[qty_col].fillna(0)

        # остальные колонки: ffill (протянуть последнее известное значение)
        for c in other_cols:
            g2[c] = g2[c].ffill()

        g2[product_col] = g[product_col].iloc[0]
        g2.index.name = date_col
        return g2.reset_index()

    out = (df.groupby(product_col, group_keys=False, sort=False)
             .apply(_expand_one))

    # (опционально) привести TRADE_DT к date, если нужно
    out[date_col] = out[date_col].dt.date

    return out












N_JOBS = -1          # сколько потоков/процессов (-1 = все)
PAR_BACKEND = "loky" # процессы (лучше для sklearn)


from joblib import Parallel, delayed

def _train_one_cross_family(bucket_df: pd.DataFrame,
                            cutoff: pd.Timestamp,
                            key: Tuple[str, str, str],
                            skus: List[str]) -> Optional[Tuple[Tuple[str, str, str], CrossFamilyModel]]:
    fam, store, bucket = key
    wide = make_family_wide_panel(bucket_df, fam, store, bucket, skus, cutoff=cutoff)
    if wide.empty:
        return None

    X_cols = [f"LOGP_{sku}" for sku in skus]
    X_all = wide[X_cols].astype(float).replace([np.inf, -np.inf], np.nan)

    scaler = StandardScaler()
    scaler.fit(X_all.fillna(0.0).values)

    models_by_product: Dict[str, PoissonRegressor] = {}
    for sku in skus:
        y_col = f"Q_{sku}"
        if y_col not in wide.columns:
            continue
        data = pd.concat([wide[y_col], X_all], axis=1).dropna(subset=[y_col, f"LOGP_{sku}"])
        if len(data) < CROSS_MIN_ROWS_PER_SKU:
            continue

        y = np.clip(pd.to_numeric(data[y_col], errors="coerce").fillna(0.0).astype(float).values, 0.0, None)
        if (y > 0).sum() < 5:
            continue

        Xs = scaler.transform(X_all.loc[data.index].fillna(0.0).values)

        reg = PoissonRegressor(alpha=max(POISSON_ALPHA, 1e-2), fit_intercept=True, max_iter=POISSON_MAX_ITER)
        reg.fit(Xs, y)
        models_by_product[str(sku)] = reg

    if len(models_by_product) < 2:
        return None

    cfm = CrossFamilyModel(str(fam), str(store), str(bucket), list(skus), scaler, models_by_product)
    return key, cfm


def train_cross_family_models(bucket_df: pd.DataFrame,
                              cutoff: pd.Timestamp,
                              topk_map: Dict[Tuple[str, str, str], List[str]]) -> Dict[Tuple[str, str, str], CrossFamilyModel]:
    items = list(topk_map.items())
    res = Parallel(n_jobs=N_JOBS, backend=PAR_BACKEND, prefer="processes")(
        delayed(_train_one_cross_family)(bucket_df, cutoff, key, skus) for key, skus in items
    )
    out: Dict[Tuple[str, str, str], CrossFamilyModel] = {}
    for x in res:
        if x is None:
            continue
        k, cfm = x
        out[k] = cfm
    return out



def _solve_one_family_store(gfam: pd.DataFrame,
                            fam: str,
                            store: str,
                            bucket: str,
                            cfm: CrossFamilyModel,
                            train_all: pd.DataFrame,
                            daily_df: pd.DataFrame,
                            promo_map: dict,
                            start: pd.Timestamp,
                            end: pd.Timestamp,
                            cutoff: pd.Timestamp,
                            fam_ladder: dict) -> Tuple[List[dict], Set[Tuple[str, str]]]:

    # <-- сюда просто переносишь твой existing код внутри одного (fam,store)
    # возвращаешь out_rows_local и used_in_cross_local

    return out_rows_local, used_in_cross_local


# В calculate_for_week(), вместо последовательного цикла:
grouped = list(ctx.groupby(["FAMILY_CODE", "STORE"]))

tasks = []
for (fam, store), gfam in grouped:
    key = (str(fam), str(store), str(bucket))
    cfm = cross_models.get(key)
    if cfm is None:
        continue
    tasks.append((str(fam), str(store), gfam, cfm))

res = Parallel(n_jobs=N_JOBS, backend=PAR_BACKEND, prefer="processes")(
    delayed(_solve_one_family_store)(
        gfam=gfam, fam=fam, store=store, bucket=bucket, cfm=cfm,
        train_all=train_all, daily_df=daily_df, promo_map=promo_map,
        start=start, end=end, cutoff=cutoff, fam_ladder=fam_ladder
    )
    for fam, store, gfam, cfm in tasks
)

for out_local, used_local in res:
    out_rows.extend(out_local)
    used_in_cross |= used_local


def make_last_daily_maps(daily_df: pd.DataFrame, cutoff: pd.Timestamp) -> Tuple[dict, dict]:
    d = daily_df.loc[daily_df["TRADE_DT"] <= pd.Timestamp(cutoff)].copy()
    d = d.sort_values("TRADE_DT")

    last_all = d.groupby(["PRODUCT_CODE","STORE"]).tail(1)
    map_all = {
        (str(p), str(s)): (float(sp) if np.isfinite(sp) else None,
                          float(bp) if np.isfinite(bp) else None)
        for p, s, sp, bp in zip(last_all["PRODUCT_CODE"], last_all["STORE"],
                                last_all["SALE_PRICE_UNIT"], last_all["BASE_PRICE"])
    }

    dnp = d.loc[d["IS_PROMO"].astype(int) == 0].copy()
    dnp = dnp.sort_values("TRADE_DT")
    last_np = dnp.groupby(["PRODUCT_CODE","STORE"]).tail(1)
    map_np = {
        (str(p), str(s)): (float(sp) if np.isfinite(sp) else None,
                          float(bp) if np.isfinite(bp) else None)
        for p, s, sp, bp in zip(last_np["PRODUCT_CODE"], last_np["STORE"],
                                last_np["SALE_PRICE_UNIT"], last_np["BASE_PRICE"])
    }
    return map_all, map_np

last_daily_all, last_daily_np = make_last_daily_maps(daily_df, cutoff=cutoff)
# дальше вместо get_last_daily_prices(...) просто:
sp_d, bp_d = last_daily_all.get((sku, store), (None, None))
sp_np, bp_np = last_daily_np.get((sku, store), (None, None))



def make_last_bucket_maps(bucket_df: pd.DataFrame, cutoff: pd.Timestamp, bucket: str) -> dict:
    d = bucket_df.loc[(bucket_df["BUCKET"] == bucket) & (bucket_df["BUCKET_END"] <= pd.Timestamp(cutoff))].copy()
    d = d.sort_values("BUCKET_END")
    last = d.groupby(["PRODUCT_CODE","STORE"]).tail(1)

    return {
        (str(p), str(s)): (
            float(pr) if np.isfinite(pr) else None,
            float(bp) if np.isfinite(bp) else None,
            float(st) if np.isfinite(st) else None
        )
        for p, s, pr, bp, st in zip(last["PRODUCT_CODE"], last["STORE"], last["PRICE"], last["BASE_PRICE"], last["STOCK_END"])
    }


last_bucket_map = make_last_bucket_maps(train_all, cutoff=cutoff, bucket=bucket)
# usage:
last_sp_bucket, last_bp_bucket, stock_end = last_bucket_map.get((sku, store), (None, None, None))



def aggregate_to_buckets_fast(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Быстрая агрегация дневных данных в бакеты:
    - без groupby.apply (только groupby.agg + idxmax)
    - корректно считает:
        * QTY
        * PRICE (взвешенная SALE_PRICE)
        * BASE_PRICE (последняя в бакете)
        * PROMO_SHARE
        * STOCK_END
        * сезонность
        * флаг праздника (праздник или неделя до него)

    Ожидаемые колонки daily_df:
    - TRADE_DT
    - PRODUCT_CODE, STORE
    - SALE_QTY (+ SALE_QTY_ONLINE опц.)
    - SALE_PRICE
    - BASE_PRICE
    - END_STOCK
    - IS_PROMO
    """

    df = daily_df.copy()

    # -----------------------------
    # 1. Дата, количество, бакет
    # -----------------------------
    df["TRADE_DT"] = pd.to_datetime(df["TRADE_DT"]).dt.normalize()

    qty = pd.to_numeric(df["SALE_QTY"], errors="coerce").fillna(0.0)
    if "SALE_QTY_ONLINE" in df.columns:
        qty += pd.to_numeric(df["SALE_QTY_ONLINE"], errors="coerce").fillna(0.0)
    df["QTY_TOTAL"] = qty

    dow = df["TRADE_DT"].dt.weekday
    df["BUCKET"] = np.where(
        dow <= 3, "MON_THU",
        np.where(dow == 4, "FRI", "SAT_SUN")
    )

    df["WEEK"] = df["TRADE_DT"].dt.to_period("W-MON").dt.start_time

    # -----------------------------
    # 2. Взвешенная цена
    # -----------------------------
    df["SALE_PRICE"] = pd.to_numeric(df["SALE_PRICE"], errors="coerce")
    df["BASE_PRICE"] = pd.to_numeric(df["BASE_PRICE"], errors="coerce")

    df["PRICE_X_Q"] = df["SALE_PRICE"] * df["QTY_TOTAL"]

    # -----------------------------
    # 3. Основная агрегация
    # -----------------------------
    gcols = ["PRODUCT_CODE", "STORE", "WEEK", "BUCKET"]

    agg = df.groupby(gcols, as_index=False).agg(
        QTY=("QTY_TOTAL", "sum"),
        PRICE_X_Q=("PRICE_X_Q", "sum"),
        QTY_POS=("QTY_TOTAL", lambda x: float((x > 0).sum())),
        PROMO_DAYS=("IS_PROMO", "sum"),
        DAYS=("TRADE_DT", "nunique"),
    )

    # Взвешенная цена
    agg["PRICE"] = agg["PRICE_X_Q"] / agg["QTY"].replace(0.0, np.nan)
    agg["PRICE"] = agg["PRICE"].replace([np.inf, -np.inf], np.nan)

    # PROMO_SHARE
    agg["PROMO_SHARE"] = (agg["PROMO_DAYS"] / agg["DAYS"]).fillna(0.0)

    agg = agg.drop(columns=["PRICE_X_Q", "PROMO_DAYS", "DAYS"])

    # -----------------------------
    # 4. Последние BASE_PRICE и STOCK_END в бакете
    # -----------------------------
    idx = (
        df.sort_values("TRADE_DT")
          .groupby(gcols, as_index=False)
          .tail(1)
          .set_index(gcols)
    )

    agg = agg.merge(
        idx[["BASE_PRICE", "END_STOCK"]],
        left_on=gcols,
        right_index=True,
        how="left"
    )

    agg = agg.rename(columns={"END_STOCK": "STOCK_END"})

    # -----------------------------
    # 5. Границы бакета
    # -----------------------------
    bucket_days = {
        "MON_THU": (0, 3),
        "FRI": (4, 4),
        "SAT_SUN": (5, 6),
    }

    def bucket_bounds(row):
        wd0, wd1 = bucket_days[row["BUCKET"]]
        start = row["WEEK"] + pd.Timedelta(days=wd0)
        end = row["WEEK"] + pd.Timedelta(days=wd1)
        return pd.Series([start, end])

    agg[["BUCKET_START", "BUCKET_END"]] = agg.apply(bucket_bounds, axis=1)

    # -----------------------------
    # 6. Сезонность
    # -----------------------------
    woy = agg["WEEK"].dt.isocalendar().week.astype(int)
    agg["SIN_WOY"] = np.sin(2 * np.pi * woy / 52.0)
    agg["COS_WOY"] = np.cos(2 * np.pi * woy / 52.0)

    month = agg["WEEK"].dt.month.astype(int)
    agg["SIN_MONTH"] = np.sin(2 * np.pi * month / 12.0)
    agg["COS_MONTH"] = np.cos(2 * np.pi * month / 12.0)

    agg["TREND_W"] = (agg["WEEK"] - agg["WEEK"].min()).dt.days / 7.0

    # -----------------------------
    # 7. Праздники РФ (бакет)
    # -----------------------------
    agg["HOLIDAY_FLAG"] = agg.apply(
        lambda r: holiday_flag_for_bucket(r["BUCKET_START"], r["BUCKET_END"]),
        axis=1
    )

    # -----------------------------
    # 8. Финальная чистка
    # -----------------------------
    agg["PRODUCT_CODE"] = agg["PRODUCT_CODE"].astype(str)
    agg["STORE"] = agg["STORE"].astype(str)
    agg["BUCKET"] = agg["BUCKET"].astype(str)

    return agg.sort_values(
        ["PRODUCT_CODE", "STORE", "WEEK", "BUCKET"]
    ).reset_index(drop=True)
