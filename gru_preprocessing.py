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


def load_part4(out_dir="results/04"):
    # Load everything produced by the Part 4 notebook.
    X = np.load(f"{out_dir}/X_raw.npy")
    S = np.load(f"{out_dir}/X_static.npy")
    y = np.load(f"{out_dir}/y.npy")
    meta = pd.read_csv(f"{out_dir}/meta.csv", parse_dates=["target_date"])
    folds = {k: json.load(open(f"{out_dir}/folds_{k}.json"))["folds"]
             for k in ["temporal", "spatial", "standard"]}
    test_idx = json.load(open(f"{out_dir}/test_idx.json"))["idx"]
    return X, S, y, meta, folds, test_idx
