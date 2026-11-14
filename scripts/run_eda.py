# Runs the EDA / preprocessing checks and generates the CSV + figures
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.splits import *

all_vars = ["temperature_celsius", "pressure_mb", "wind_u", "wind_v", "AOD"] + gas
miss_var = dev_df[all_vars].isna().mean().mul(100).round(2).rename("pct_missing")
print(miss_var.to_string())

# by year-month
tmp = dev_df.assign(year_month=dev_df["date"].dt.to_period("M").astype(str))
miss_month = tmp.groupby("year_month")[gas].agg(lambda s: s.isna().mean() * 100).round(2)
miss_month["n_obs"] = tmp.groupby("year_month").size()
miss_month.to_csv(f"{OUT_DIR}/missingness_by_month.csv")

# by station
miss_station = (dev_df.groupby(["id", "sensor_name"])[gas]
                .agg(lambda s: s.isna().mean() * 100).round(2))
miss_station["n_obs"] = dev_df.groupby(["id", "sensor_name"]).size()
miss_station = miss_station.sort_values("NO2", ascending=False)
miss_station.to_csv(f"{OUT_DIR}/missingness_by_station.csv")

print("\nMonths with >50% missing NO2:", int((miss_month["NO2"] > 50).sum()), "of", len(miss_month))
print("Stations with >50% missing NO2:", int((miss_station["NO2"] > 50).sum()), "of", len(miss_station))
print("\nWorst 8 stations (NO2 % missing):")
print(miss_station.head(8).to_string())

fig, ax = plt.subplots(figsize=(12, 4))
for g in gas:
    ax.plot(miss_month.index, miss_month[g], marker="o", ms=3, label=g)
ax.set_title("Missing Sentinel-5P values by month (development period)")
ax.set_ylabel("% missing"); ax.set_xlabel("Year-month")
step = max(1, len(miss_month) // 15)
ax.set_xticks(range(0, len(miss_month), step))
ax.set_xticklabels(miss_month.index[::step], rotation=45, ha="right")
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/missingness_by_month.png", dpi=150); plt.show()

y = dev_df["pm25"]
q1, q3 = y.quantile([0.25, 0.75]); iqr = q3 - q1
fence = q3 + 1.5 * iqr
summary = {
    "n_dev": len(y), "mean": y.mean(), "median": y.median(), "std": y.std(),
    "skewness": y.skew(), "kurtosis": y.kurt(),
    "q1": q1, "q3": q3, "iqr_upper_fence": fence,
    "n_above_fence": int((y > fence).sum()), "pct_above_fence": (y > fence).mean() * 100,
    "pct_above_25": (y > 25).mean() * 100, "pct_above_50": (y > 50).mean() * 100,
    "max": y.max(),
}
outlier_summary = pd.Series(summary).round(3)
outlier_summary.to_csv(f"{OUT_DIR}/outlier_summary.csv", header=["value"])
print(outlier_summary.to_string())

from src.preprocessing import FoldPreprocessor

rows = []
for name, folds in schemes.items():
    for f in folds:
        pre = FoldPreprocessor().fit(X, f["train"])
        Xtr, Str = pre.transform(X, S, f["train"])
        Xva, Sva = pre.transform(X, S, f["val"])

        # the transformed arrays contain no NaN
        assert not np.isnan(Xtr).any() and not np.isnan(Xva).any()

        # statistics come from the TRAIN fold only: recompute them independently and compare
        tr_idx = np.array(f["train"])
        obs_tr = X[tr_idx][:, :, IDX["obs"]] == 1
        train_mean_lag = np.nanmean(X[tr_idx][:, :, IDX["pm25_lag"]][obs_tr])
        assert np.isclose(pre.scaler_.mean_[IDX["pm25_lag"]], train_mean_lag, rtol=1e-4)

        # on observed steps the TRAIN set is centered (~0); the validation set is not forced to be
        obs_va = X[np.array(f["val"])][:, :, IDX["obs"]] == 1
        rows.append({
            "scheme": name, "fold": f["fold"],
            "n_train": len(f["train"]), "n_val": len(f["val"]),
            "train_pm25_mean": float(y_arr[tr_idx].mean()),
            "val_pm25_mean": float(y_arr[np.array(f["val"])].mean()),
            "scaler_mean_pm25_lag(train)": float(pre.scaler_.mean_[IDX["pm25_lag"]]),
            "train_scaled_mean": float(Xtr[:, :, IDX["pm25_lag"]][obs_tr].mean()),
            "val_scaled_mean": float(Xva[:, :, IDX["pm25_lag"]][obs_va].mean()),
        })
fold_summary = pd.DataFrame(rows).round(3)
fold_summary.to_csv(f"{OUT_DIR}/fold_preprocessing_summary.csv", index=False)
print(fold_summary.to_string(index=False))
print("\nFitted on train only: train scaled mean ~ 0, validation scaled mean differs (no information leaked).")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

# (a) PM2.5 distribution
ax = axes[0]
ax.hist(meta.loc[meta.split == "dev", "y"], bins=80, color="#4C72B0")
for t in (25, 50):
    ax.axvline(t, color="red", ls="--", lw=1)
ax.set_yscale("log"); ax.set_title("PM2.5 target (dev samples, log count)")
ax.set_xlabel("PM2.5 (µg/m³)")

# (b) samples per year by split
ax = axes[1]
yy = meta.assign(year=meta["target_date"].dt.year).groupby(["year", "split"]).size().unstack(fill_value=0)
yy.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452"])
ax.set_title("GRU samples per year"); ax.set_xlabel("")

# (c) temporal folds
ax = axes[2]
for f in folds_temporal:
    k = f["fold"]
    tr_d = dd.loc[f["train"], "target_date"]; va_d = dd.loc[f["val"], "target_date"]
    ax.barh(k, (tr_d.max() - tr_d.min()).days, left=tr_d.min(), color="#4C72B0", label="train" if k == 0 else None)
    ax.barh(k, (va_d.max() - va_d.min()).days, left=va_d.min(), color="#C44E52", label="validation" if k == 0 else None)
ax.barh(5, (meta.target_date.max() - TEST_START).days, left=TEST_START, color="#55A868", label="final test")
ax.set_yticks(range(6)); ax.set_yticklabels([f"Fold {i}" for i in range(5)] + ["Test"])
ax.set_title("Temporal CV (expanding window, 1-day gap)"); ax.legend(loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT_DIR}/part4_overview.png", dpi=150); plt.show()
