import pandas as pd, numpy as np, hashlib
from pathlib import Path
from sklearn.model_selection import train_test_split

TARGET = "Diabetes_binary"
RAW = Path("data/raw/diabetes_binary_health_indicators_BRFSS2015.csv")

df = pd.read_csv(RAW)
feats = [c for c in df.columns if c != TARGET]

print("rows:", len(df))
print("prevalence:", round(df[TARGET].mean(), 4))
print("exact dupes:", df.duplicated().sum())
print("feature-identical:", df[feats].duplicated().sum())

g = df.groupby(feats)[TARGET].nunique()
print("contradictory groups:", int((g > 1).sum()))

df = df.drop_duplicates().reset_index(drop=True)
print("rows after dedup:", len(df))
print("prevalence after dedup:", round(df[TARGET].mean(), 4))

h = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
print("DATA HASH:", h)

idx = np.arange(len(df))
tr, tmp = train_test_split(idx, test_size=0.30, stratify=df[TARGET], random_state=42)
va, te = train_test_split(tmp, test_size=0.50, stratify=df[TARGET].iloc[tmp], random_state=42)

np.savez("splits.npz", train=tr, val=va, test=te, data_hash=h)
df.to_csv("data/dedup.csv", index=False)
print("splits:", len(tr), len(va), len(te))