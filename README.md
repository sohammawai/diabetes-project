# Non-Invasive Diabetes Risk Stratification

Soft-computing semester project: rough sets, neural networks and
fuzzy inference on the CDC Diabetes Health Indicators dataset
(UCI #891, BRFSS 2015).

## Status
- Stage 0 done: deduplication (253,680 to 229,474 rows),
  stratified 70/15/15 split, data hash.
- Baselines on validation: logistic regression PR-AUC 0.4191,
  random forest 0.4426 (prevalence baseline 0.1529).
- Rough-set engine (gamma_vprs) verified on a toy example.

## Reproduce
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python 00_fetch_data.py
    python 02_baselines.py

The raw data is not stored here; 00_fetch_data.py downloads it
from UCI. splits.npz is committed and must not be regenerated.

## Rules
- The test split is frozen and evaluated exactly once, at the end.
- results/registry.csv is append-only.