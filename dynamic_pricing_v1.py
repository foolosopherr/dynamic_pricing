# Parse PROMO_PERIOD ("01-01-2024 - 03-01-2024")
def parse_promo_period(x):
    try:
        start, end = x.split(" - ")
        return pd.to_datetime(start, dayfirst=True), pd.to_datetime(end, dayfirst=True)
    except:
        return pd.NaT, pd.NaT

df[["promo_start","promo_end"]] = df["PROMO_PERIOD"].apply(lambda x: pd.Series(parse_promo_period(str(x))))

# Mark if promo active
df["promo_active"] = (df["TRADE_DT"] >= df["promo_start"]) & (df["TRADE_DT"] <= df["promo_end"])

# ======================
# Create features
# ======================
df["dow"] = df["TRADE_DT"].dt.dayofweek

# 3 buckets: Mon-Thu=0, Fri=1, Sat-Sun=2
df["dow_bucket"] = pd.cut(df["dow"], bins=[-1,3,4,6], labels=[0,1,2]).astype(int)

# Simple encodings for hierarchy
for col in ["PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE","STORE","REGION_NAME","STORE_TYPE","PLACE_TYPE"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Basic numeric features
features = ["IS_PROMO","BASE_PRICE","START_STOCK","END_STOCK",
            "LOSS_QTY","RETURN_QTY","DELIVERY_QTY","promo_active",
            "PRODUCT_CODE","FAMILY_CODE","CATEGORY_CODE","SEGMENT_CODE",
            "STORE","REGION_NAME","STORE_TYPE","PLACE_TYPE","dow_bucket"]

X = df[features].fillna(0).values
y = df["SALE_QTY"].fillna(0).values

# ======================
# Bandit model
# ======================
# 4 arms = e.g. base price, +5%, -5%, +10%
def get_arms(base_price):
    return [base_price,
            base_price*0.95,
            base_price*1.05,
            base_price*1.10]

arms = 4
model = LinUCB(nchoices=arms, alpha=0.5)

# ======================
# Train / Evaluate
# ======================
# Split: last 12 weeks = eval
last_date = df["TRADE_DT"].max()
eval_cut = last_date - pd.Timedelta(weeks=12)

train_idx = df["TRADE_DT"] < eval_cut
eval_idx = df["TRADE_DT"] >= eval_cut

X_train, y_train = X[train_idx], y[train_idx]
X_eval, y_eval = X[eval_idx], y[eval_idx]
prices_eval = df.loc[eval_idx,"BASE_PRICE"].values

# Training loop
for i in range(len(X_train)):
    base_price = df.loc[train_idx].iloc[i]["BASE_PRICE"]
    chosen_arm = model.predict_one(X_train[i])
    arm_prices = get_arms(base_price)
    reward = -abs(arm_prices[chosen_arm] - base_price)  # simple proxy
    model.partial_fit(X_train[i], chosen_arm, reward)

# Evaluation
rewards = []
for i in range(len(X_eval)):
    base_price = prices_eval[i]
    chosen_arm = model.predict_one(X_eval[i])
    arm_prices = get_arms(base_price)
    # reward = sales revenue approx
    reward = y_eval[i] * arm_prices[chosen_arm]
    rewards.append(reward)

print("Mean eval reward:", np.mean(rewards))

# ======================
# Prediction for NEXT week
# ======================
next_week_start = last_date + pd.Timedelta(days=1)
next_week_end   = next_week_start + pd.Timedelta(days=7)

# create skeleton for next week (minimal info known)
products = df["PRODUCT_CODE"].unique()
future_rows = []
for p in products:
    for d in pd.date_range(next_week_start, next_week_end, freq="D"):
        future_rows.append({
            "PRODUCT_CODE": p,
            "TRADE_DT": d,
            "dow_bucket": pd.cut([d.weekday()], bins=[-1,3,4,6], labels=[0,1,2]).astype(int)[0],
            "BASE_PRICE": df[df["PRODUCT_CODE"]==p]["BASE_PRICE"].median()
        })

future_df = pd.DataFrame(future_rows)

# Fill missing with zeros
for col in set(features)-set(future_df.columns):
    future_df[col] = 0

X_future = future_df[features].fillna(0).values

future_df["chosen_arm"] = [model.predict_one(x) for x in X_future]
future_df["new_price"] = [
    get_arms(bp)[arm] for bp,arm in zip(future_df["BASE_PRICE"], future_df["chosen_arm"])
]

print(future_df.head())