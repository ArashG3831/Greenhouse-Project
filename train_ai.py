# -----------------------------------------------------------------
# 0) CONFIG -- tweak these two knobs only
MAX_LOOKBACK        = 24    # what you want when you have ≥ 25 hourly rows
MIN_SEQS_FOR_LSTM   = 10    # need this many training samples to bother
# -----------------------------------------------------------------

available_rows = len(df_hourly)
if available_rows < 2:                       # 1 hourly row ⇒ nonsense
    raise ValueError("Need ≥ 2 hourly rows, have {available_rows}")

# 1) Dynamically shrink look-back so we always have at least 1 training sample
LOOKBACK = max(1, min(MAX_LOOKBACK, available_rows - 1))

sensor_cols = ["temperature", "humidity", "oxygen_level",
               "co2_level", "light_illumination"]
data   = df_hourly[sensor_cols].to_numpy()

# 2) Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 3) Build (X, Y) sequences
def make_xy(arr, lkbk):
    X, Y = [], []
    for i in range(len(arr) - lkbk):
        X.append(arr[i:i+lkbk])
        Y.append(arr[i+lkbk])
    return np.asarray(X), np.asarray(Y)

X, Y = make_xy(scaled, LOOKBACK)

# 4) Decide: train real model or fall back to “repeat last value”
if len(X) < MIN_SEQS_FOR_LSTM:
    # ---------- baseline ----------
    print(
        f"⚠️  Only {len(X)} training samples. "
        "Skipping LSTM; using persistence baseline."
    )
    last_obs    = df_hourly.iloc[-1][sensor_cols].to_numpy()
    predictions = np.tile(last_obs, (24, 1))          # 24 identical rows
else:
    # ---------- LSTM ----------
    split = int(0.8 * len(X))
    X_train, Y_train = X[:split] or X, Y[:split] or Y   # if split==0, use all
    X_val,   Y_val   = (X[split:], Y[split:]) if split else (None, None)

    model = Sequential([
        LSTM(64, activation="relu",
             input_shape=(LOOKBACK, len(sensor_cols))),
        Dense(len(sensor_cols))
    ])
    model.compile(optimizer="adam", loss="mse")

    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val) if split else None,
        epochs=30, batch_size=min(8, len(X_train)),
        verbose=1
    )

    # 5) Roll the forecast
    seq          = scaled[-LOOKBACK:].copy()
    pred_scaled  = []
    for _ in range(24):
        p = model.predict(seq[np.newaxis], verbose=0)[0]
        pred_scaled.append(p)
        seq = np.concatenate([seq[1:], p[None, :]], axis=0)

    predictions = scaler.inverse_transform(np.asarray(pred_scaled))

    # shift so the first forecast matches the last real observation
    predictions += df_hourly.iloc[-1].to_numpy() - predictions[0]

# 6) Clamp temperature
predictions[:, 0] = np.clip(predictions[:, 0], 15, 40)
