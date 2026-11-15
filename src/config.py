# Paths, constants and SEED = 42
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, KFold
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

WINDOW = 7                  # days of history given to the GRU
MIN_OBS_IN_WINDOW = 3       # at least 3 observed days among the 7 (t-1 must be one of them)
MIN_STATION_OBS = 150       # minimum observations per station in the development period
TEST_START = pd.Timestamp("2025-01-01")   # final future test period (never used in any fold)
OUT_DIR = "results/04"
os.makedirs(OUT_DIR, exist_ok=True)
FOLDS_DIR = "folds"         # fold JSON files (temporal / spatial / standard) + test_idx.json
os.makedirs(FOLDS_DIR, exist_ok=True)

CANDIDATES = [
    "data/processed/combined_gee_pm25_clean.csv",
    "Files/combined_gee_pm25_clean.csv",
    "/content/combined_gee_pm25_clean.csv",
    "/content/PM25-Forecasting-GRU/data/processed/combined_gee_pm25_clean.csv",
    "/content/PM25-Forecasting-GRU/Files/combined_gee_pm25_clean.csv",
    "/mnt/user-data/uploads/combined_gee_pm25_clean.csv",
]
DATA_PATH = next(p for p in CANDIDATES if os.path.exists(p))
print("Reading:", DATA_PATH)

gas = ["NO2", "CO", "O3"]
GAP_DAYS = 1
