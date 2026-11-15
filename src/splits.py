# Folds (temporal, spatial GroupKFold, standard KFold) + leakage checks, saved as JSON
from src.features import *

dev = meta[meta["split"] == "dev"].sort_values(["target_date", "id"]).reset_index(drop=True)
n = len(dev)
dates = dev["target_date"].to_numpy()

# ---- Temporal: 6 date-aligned chunks by row count; fold k trains on chunks 0..k, validates on chunk k+1
edges = [dev["target_date"].min()] + [pd.Timestamp(dates[int(n * k / 6)]) for k in range(1, 6)] \
        + [dev["target_date"].max() + pd.Timedelta(days=1)]
folds_temporal = []
for k in range(5):
    val_start, val_end = edges[k + 1], edges[k + 2]
    tr = dev[dev["target_date"] < val_start - pd.Timedelta(days=GAP_DAYS)]   # 1 day removed before validation
    va = dev[(dev["target_date"] >= val_start) & (dev["target_date"] < val_end)]
    folds_temporal.append({"fold": k, "train": tr["sample_id"].tolist(), "val": va["sample_id"].tolist(),
                           "train_end": str(tr["target_date"].max().date()),
                           "val_start": str(va["target_date"].min().date()),
                           "val_end": str(va["target_date"].max().date())})

# ---- Spatial: GroupKFold by station id (id is a group only, never a feature)
folds_spatial = []
for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(dev, groups=dev["id"])):
    folds_spatial.append({"fold": k, "train": dev.iloc[tr]["sample_id"].tolist(),
                          "val": dev.iloc[va]["sample_id"].tolist(),
                          "val_stations": sorted(dev.iloc[va]["id"].unique().tolist())})

# ---- Standard random KFold (diagnostic only)
folds_standard = []
for k, (tr, va) in enumerate(KFold(n_splits=5, shuffle=True, random_state=SEED).split(dev)):
    folds_standard.append({"fold": k, "train": dev.iloc[tr]["sample_id"].tolist(),
                           "val": dev.iloc[va]["sample_id"].tolist()})

schemes = {"temporal": folds_temporal, "spatial": folds_spatial, "standard": folds_standard}
for name, folds in schemes.items():
    with open(f"{FOLDS_DIR}/folds_{name}.json", "w") as f:
        json.dump({"scheme": name, "n_splits": 5, "seed": SEED, "folds": folds}, f)

test_idx = meta.loc[meta["split"] == "test", "sample_id"].tolist()
with open(f"{FOLDS_DIR}/test_idx.json", "w") as f:
    json.dump({"test_start": str(TEST_START.date()), "idx": test_idx}, f)

for name, folds in schemes.items():
    print(name, [(len(f["train"]), len(f["val"])) for f in folds])
for f in folds_temporal:
    print(f"temporal fold {f['fold']}: train up to {f['train_end']} | val {f['val_start']} -> {f['val_end']}")

checks = []
test_set = set(test_idx)
dd = meta.set_index("sample_id")
for name, folds in schemes.items():
    for f in folds:
        tr, va = set(f["train"]), set(f["val"])
        c = {"scheme": name, "fold": f["fold"],
             "no_train_val_overlap": tr.isdisjoint(va),
             "no_test_in_fold": tr.isdisjoint(test_set) and va.isdisjoint(test_set)}
        if name == "temporal":
            gap = (dd.loc[list(va), "target_date"].min() - dd.loc[list(tr), "target_date"].max()).days
            c["temporal_gap_days"] = gap
            c["train_strictly_before_val"] = gap >= 2     # >= 1 full day between last train and first val
        if name == "spatial":
            c["no_station_overlap"] = set(dd.loc[list(tr), "id"]).isdisjoint(set(dd.loc[list(va), "id"]))
        checks.append(c)
leak = pd.DataFrame(checks)
leak.to_csv(f"{OUT_DIR}/leakage_checks.csv", index=False)
print(leak.to_string(index=False))
bool_cols = [c for c in leak.columns if leak[c].dtype == bool]
assert leak[bool_cols].all().all(), "A leakage check failed"
print("\nAll leakage checks passed.")
