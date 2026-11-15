# Loading, test isolation, station selection, FoldPreprocessor (imputation + scaling)
from src.config import *

df = (pd.read_csv(DATA_PATH, parse_dates=["date"])
        .sort_values(["id", "date"]).reset_index(drop=True))

# sanity checks on the Part 1 output
assert df.duplicated(["id", "date"]).sum() == 0
assert df["pm25"].gt(0).all() and df["pm25"].lt(500).all()
print("Shape:", df.shape, "| stations:", df["id"].nunique())
print("Dates:", df["date"].min().date(), "->", df["date"].max().date())

dev_df  = df[df["date"] <  TEST_START].copy()
test_df = df[df["date"] >= TEST_START].copy()
print(f"Development rows: {len(dev_df):,}  ({dev_df['date'].min().date()} -> {dev_df['date'].max().date()})")
print(f"Test rows       : {len(test_df):,}  ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")

cnt = dev_df.groupby("id").size()
keep_ids = cnt[cnt >= MIN_STATION_OBS].index
st = (dev_df.groupby(["id", "sensor_name", "latitude", "longitude"])
      .agg(n_obs_dev=("pm25", "size"), first=("date", "min"), last=("date", "max")).reset_index())
st["kept"] = st["id"].isin(keep_ids)
st.to_csv(f"{OUT_DIR}/stations_selection.csv", index=False)

print(f"Stations in dev data : {cnt.size}")
print(f"Kept (>= {MIN_STATION_OBS} obs) : {len(keep_ids)}")
print(f"Dropped              : {cnt.size - len(keep_ids)}")
print(f"Test-only stations (absent from dev) are dropped automatically: "
      f"{test_df.loc[~test_df['id'].isin(cnt.index), 'id'].nunique()}")

dfk = df[df["id"].isin(keep_ids)].copy()
print(f"Rows kept: {len(dfk):,} of {len(df):,}")


# Fold-safe preprocessing for the GRU (Part 4). Fit on the training fold only.
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

N_CONT = 10        # first 10 features are continuous (see feature_names.json)
OBS_IDX = 13       # index of the 'obs' channel


class FoldPreprocessor:
    def __init__(self, n_cont=N_CONT, obs_idx=OBS_IDX, clip_q=None):
        self.n_cont = n_cont
        self.obs_idx = obs_idx
        self.clip_q = clip_q          # e.g. 0.005 -> clip features to [0.5%, 99.5%] learned on train

    def fit(self, X, train_idx):
        Xt = X[np.asarray(train_idx)]
        obs = Xt[:, :, self.obs_idx] == 1
        C = Xt[:, :, :self.n_cont][obs]                 # observed timesteps only
        self.median_ = np.nanmedian(C, axis=0)
        C = np.where(np.isnan(C), self.median_, C)
        if self.clip_q:
            self.lo_, self.hi_ = np.quantile(C, [self.clip_q, 1 - self.clip_q], axis=0)
            C = np.clip(C, self.lo_, self.hi_)
        self.scaler_ = StandardScaler().fit(C)
        return self

    def transform(self, X, S, idx):
        idx = np.asarray(idx)
        Xi = X[idx].copy()
        obs = Xi[:, :, self.obs_idx] == 1
        C = Xi[:, :, :self.n_cont]
        C = np.where(np.isnan(C), self.median_, C)      # median from TRAIN fold
        if self.clip_q:
            C = np.clip(C, self.lo_, self.hi_)
        C = (C - self.scaler_.mean_) / self.scaler_.scale_
        C[~obs] = 0.0                                   # padding for unobserved days
        Xi[:, :, :self.n_cont] = C
        return Xi.astype(np.float32), S[idx].astype(np.float32)


def load_part4(out_dir="results/04", folds_dir="folds"):
    # Load everything produced by the Part 4 notebook.
    X = np.load(f"{out_dir}/X_raw.npy")
    S = np.load(f"{out_dir}/X_static.npy")
    y = np.load(f"{out_dir}/y.npy")
    meta = pd.read_csv(f"{out_dir}/meta.csv", parse_dates=["target_date"])
    folds = {k: json.load(open(f"{folds_dir}/folds_{k}.json"))["folds"]
             for k in ["temporal", "spatial", "standard"]}
    test_idx = json.load(open(f"{folds_dir}/test_idx.json"))["idx"]
    return X, S, y, meta, folds, test_idx
