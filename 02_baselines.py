import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, accuracy_score)
from data_io import load, features, log_result, TARGET
import data_io

df, tr, va, te = load()

for drop_leak in [False, True]:
    cols = features(df, drop_leakage=drop_leak)
    Xtr, ytr = df.loc[tr, cols], df.loc[tr, TARGET]
    Xva, yva = df.loc[va, cols], df.loc[va, TARGET]

    prev = yva.mean()
    tag = "leakage-free" if drop_leak else "all features"
    print(f"\n=== {tag} ({len(cols)} features) ===")
    print(f"PR-AUC baseline (prevalence): {prev:.4f}")
    print(f"majority-class accuracy:      {1 - prev:.4f}   <- the accuracy trap")

    models = {
        "logistic regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "decision tree (d=6)": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42),
        "random forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight="balanced",
            n_jobs=4, random_state=42),
    }

    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        pr = average_precision_score(yva, p)
        roc = roc_auc_score(yva, p)
        br = brier_score_loss(yva, p)
        print(f"{name:22s} PR-AUC {pr:.4f}  ROC-AUC {roc:.4f}  Brier {br:.4f}")

        log_result(
            data_hash=data_io.DATA_HASH,
            script="02_baselines.py",
            feature_set="leakage_free" if drop_leak else "all_21",
            n_features=len(cols),
            model=name,
            params=str(m.get_params()),
            seed=42,
            split="val",
            pr_auc=round(pr, 4),
            roc_auc=round(roc, 4),
            brier=round(br, 4),
            prevalence=round(prev, 4),
            notes="class_weight=balanced, uncalibrated",
        )