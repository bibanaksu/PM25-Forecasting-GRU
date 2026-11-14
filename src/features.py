# Daily series per station, 7-day windows, cyclic variables, saving of arrays
from src.preprocessing import *

CONT  = ["temperature_celsius", "pressure_mb", "wind_u", "wind_v", "wind_speed",
         "NO2", "CO", "O3", "AOD", "pm25_lag"]
FLAGS = ["miss_NO2", "miss_CO", "miss_O3", "obs"]
FEATURES = CONT + FLAGS
STATIC = ["doy_sin_target", "doy_cos_target"]
IDX = {n: i for i, n in enumerate(FEATURES)}

X_list, S_list, y_list, meta_rows = [], [], [], []

for sid, g in dfk.groupby("id"):
    info = g.iloc[0]
    g = g.set_index("date").sort_index()
    grid = pd.date_range(g.index.min(), g.index.max(), freq="D")
    gg = g.reindex(grid)                       # regular daily timeline, NaN on missing days
    obs = gg["pm25"].notna().to_numpy()

    F = np.full((len(grid), len(FEATURES)), np.nan, dtype=np.float32)
    F[:, IDX["temperature_celsius"]] = gg["temperature_celsius"]
    F[:, IDX["pressure_mb"]] = gg["pressure_mb"]
    F[:, IDX["wind_u"]] = gg["wind_u"]
    F[:, IDX["wind_v"]] = gg["wind_v"]
    F[:, IDX["wind_speed"]] = np.hypot(gg["wind_u"], gg["wind_v"])
    for c in gas:
        F[:, IDX[c]] = gg[c]
        F[:, IDX["miss_" + c]] = gg[c].isna().astype(np.float32)
    F[:, IDX["AOD"]] = gg["AOD"]
    F[:, IDX["pm25_lag"]] = gg["pm25"]         # past PM2.5 (only ever read for days < target)
    F[:, IDX["obs"]] = obs.astype(np.float32)

    for i in range(WINDOW, len(grid)):
        if not (obs[i] and obs[i - 1]):        # target observed and yesterday observed
            continue
        n_obs = int(obs[i - WINDOW:i].sum())
        if n_obs < MIN_OBS_IN_WINDOW:
            continue
        doy = grid[i].dayofyear
        X_list.append(F[i - WINDOW:i])         # days t-7 ... t-1 only
        S_list.append([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
        y_list.append(gg["pm25"].iloc[i])
        meta_rows.append((sid, info["sensor_name"], info["latitude"], info["longitude"],
                          grid[i], n_obs))

X = np.stack(X_list).astype(np.float32)        # (n_samples, 7, 14), NaN kept (imputed per fold later)
S = np.array(S_list, dtype=np.float32)         # (n_samples, 2)
y_arr = np.array(y_list, dtype=np.float32)
meta = pd.DataFrame(meta_rows, columns=["id", "sensor_name", "latitude", "longitude",
                                        "target_date", "n_obs_window"])
meta.insert(0, "sample_id", np.arange(len(meta)))
meta["y"] = y_arr
meta["split"] = np.where(meta["target_date"] >= TEST_START, "test", "dev")

print("X:", X.shape, "| S:", S.shape, "| y:", y_arr.shape)
print(meta["split"].value_counts().to_string())
print(f"\nSamples kept: {len(meta):,} of {len(dfk):,} observed rows")
print("Observed days in window (n_obs_window):")
print(meta["n_obs_window"].value_counts().sort_index().to_string())

lookup = df.set_index(["id", "date"])["pm25"]
rng = np.random.default_rng(SEED)
pick = rng.choice(len(meta), size=1000, replace=False)
ok_prev, ok_obs = 0, 0
for j in pick:
    r = meta.iloc[j]
    prev_val = lookup.get((r["id"], r["target_date"] - pd.Timedelta(days=1)), np.nan)
    ok_prev += bool(np.isclose(X[j, -1, IDX["pm25_lag"]], prev_val))   # last step = day t-1
    ok_obs  += bool(X[j, -1, IDX["obs"]] == 1)
assert ok_prev == len(pick) and ok_obs == len(pick)
assert not np.isclose(X[pick, -1, IDX["pm25_lag"]], y_arr[pick]).all()  # window does not contain the target
print(f"{len(pick)}/{len(pick)} windows end exactly at t-1 with the real PM2.5 of that day.")

# missingness left in the raw array (imputed later, per fold)
nan_share = pd.Series(np.isnan(X).reshape(-1, X.shape[2]).mean(0) * 100, index=FEATURES).round(2)
print("\n% NaN per feature in raw windows (unobserved days + missing S5P):")
print(nan_share.to_string())

np.save(f"{OUT_DIR}/X_raw.npy", X)            # (n, 7, 14) raw windows, NaN kept (impute per fold)
np.save(f"{OUT_DIR}/X_static.npy", S)         # (n, 2)
np.save(f"{OUT_DIR}/y.npy", y_arr)            # (n,) PM2.5 in µg/m³ (original scale)
meta.to_csv(f"{OUT_DIR}/meta.csv", index=False)

with open(f"{OUT_DIR}/feature_names.json", "w") as f:
    json.dump({"sequence_features": FEATURES, "static_features": STATIC, "window": WINDOW,
               "continuous_idx": list(range(len(CONT))), "obs_idx": IDX["obs"]}, f, indent=2)

config = {"seed": SEED, "window_days": WINDOW, "min_obs_in_window": MIN_OBS_IN_WINDOW,
          "min_station_obs": MIN_STATION_OBS, "test_start": str(TEST_START.date()),
          "forecast_horizon_days": 1, "temporal_gap_days": GAP_DAYS,
          "n_samples": int(len(meta)), "n_dev": int((meta.split == "dev").sum()),
          "n_test": int((meta.split == "test").sum()), "n_stations": int(meta["id"].nunique()),
          "input_file": os.path.basename(DATA_PATH)}
with open(f"{OUT_DIR}/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(json.dumps(config, indent=2))
print("\nFiles:", sorted(os.listdir(OUT_DIR)))
