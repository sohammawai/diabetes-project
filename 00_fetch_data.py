from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

TARGET = "Diabetes_binary"
RAW = Path("data/raw/diabetes_binary_health_indicators_BRFSS2015.csv")

ds = fetch_ucirepo(id=891)

print("target cols:", ds.data.targets.columns.tolist())

df = pd.concat([ds.data.features, ds.data.targets], axis=1)

# canonical column order so everyone's data hash matches
feats = sorted(c for c in df.columns if c != TARGET)
df = df[feats + [TARGET]]

assert df.shape == (253680, 22), df.shape
assert df.isna().sum().sum() == 0

RAW.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(RAW, index=False)
print("wrote", RAW, df.shape)