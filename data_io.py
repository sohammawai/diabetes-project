import pandas as pd, numpy as np, hashlib
from pathlib import Path
import subprocess, datetime, csv, os

TARGET = "Diabetes_binary"
DEDUP = Path("data/dedup.csv")
SPLITS = Path("splits.npz")

LEAKAGE_SUSPECT = ["GenHlth", "DiffWalk", "PhysHlth"]


DATA_HASH = None


def load():
    """Returns (df, train_idx, val_idx, test_idx). Asserts the data hash."""
    df = pd.read_csv(DEDUP)
    s = np.load(SPLITS, allow_pickle=True)
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
    expected = str(s["data_hash"])
    assert h == expected, f"DATA HASH MISMATCH\n got {h}\n want {expected}"
    globals()["DATA_HASH"] = h
    return df, s["train"], s["val"], s["test"]

def features(df, drop_leakage=False):
    cols = [c for c in df.columns if c != TARGET]
    if drop_leakage:
        cols = [c for c in cols if c not in LEAKAGE_SUSPECT]
    return cols
REGISTRY = Path("results/registry.csv")
FIELDS = ["timestamp", "git_commit", "data_hash", "machine", "script",
          "feature_set", "n_features", "model", "params", "seed", "split",
          "pr_auc", "roc_auc", "brier", "prevalence", "notes"]


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def log_result(**kw):
    """Append one row to the append-only results registry."""
    REGISTRY.parent.mkdir(exist_ok=True)
    row = {k: "" for k in FIELDS}
    row.update({
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "machine": os.environ.get("COMPUTERNAME", "?"),
    })
    row.update(kw)
    unknown = set(kw) - set(FIELDS)
    assert not unknown, f"unknown registry fields: {unknown}"
    new = not REGISTRY.exists()
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)