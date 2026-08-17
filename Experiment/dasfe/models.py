"""The seven lightweight classifiers benchmarked in the study.

Hyper-parameters match the original submission (500 estimators, lr 0.05,
max_depth 15, subsample 0.8, colsample 0.8, LightGBM num_leaves 64) so the
two-dataset results stay comparable with the numbers already reported.
"""
from __future__ import annotations

from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from . import config as C

# Classifiers that need feature scaling to behave.
NEEDS_SCALING = {"LogisticRegression"}
# Classifiers that accept `class_weight="balanced"` directly.
SUPPORTS_CLASS_WEIGHT = {
    "LightGBM", "RandomForest", "ExtraTrees", "LogisticRegression", "DecisionTree",
}


def build(name: str, strategy: str = "none", seed: int = C.SEED, n_jobs: int = C.N_JOBS):
    """Instantiate a classifier, wiring class weights only when requested."""
    balanced = strategy == "class_weight"
    cw = "balanced" if balanced else None
    p = C.BOOSTING_PARAMS

    if name == "LightGBM":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], num_leaves=C.LGBM_NUM_LEAVES,
            subsample=p["subsample"], subsample_freq=1,
            colsample_bytree=p["colsample_bytree"],
            class_weight=cw, random_state=seed, n_jobs=n_jobs, verbose=-1,
        )

    if name == "XGBoost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], subsample=p["subsample"],
            colsample_bytree=p["colsample_bytree"], tree_method="hist",
            random_state=seed, n_jobs=n_jobs, eval_metric="mlogloss",
        )

    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=p["n_estimators"], learning_rate=p["learning_rate"],
            max_depth=p["max_depth"], class_weight=cw, random_state=seed,
        )

    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=p["max_depth"], class_weight=cw,
            random_state=seed, n_jobs=n_jobs,
        )

    if name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=p["max_depth"], class_weight=cw,
            random_state=seed, n_jobs=n_jobs,
        )

    if name == "LogisticRegression":
        return LogisticRegression(
            solver="lbfgs", max_iter=1000, class_weight=cw,
            random_state=seed, n_jobs=n_jobs,
        )

    if name == "DecisionTree":
        return DecisionTreeClassifier(
            max_depth=p["max_depth"], class_weight=cw, random_state=seed,
        )

    raise KeyError(f"unknown model '{name}'")


def xgb_needs_sample_weight(name: str, strategy: str) -> bool:
    """XGBoost has no `class_weight`; cost-sensitive learning goes through
    `sample_weight` at fit time instead."""
    return name == "XGBoost" and strategy == "class_weight"


# XGBoost >= 2.0 dropped the internal label encoder from XGBClassifier, so it
# only accepts y already encoded as integers 0..n_classes-1.  `evaluate.
# fit_predict` encodes for these models and decodes the predictions back, which
# keeps the string event labels everywhere else in the pipeline.
NEEDS_ENCODED_LABELS = {"XGBoost"}


def needs_encoded_labels(name: str) -> bool:
    return name in NEEDS_ENCODED_LABELS


def build_cnn_baseline(n_classes: int, input_len: int = 2048):
    """The DAS_CNN reference architecture (Table 21) used for comparison."""
    import torch
    import torch.nn as nn

    class DASCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=7), nn.LeakyReLU(0.01),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 256, kernel_size=7), nn.LeakyReLU(0.01),
                nn.MaxPool1d(4),
            )
            with torch.no_grad():
                flat = self.features(torch.zeros(1, 1, input_len)).numel()
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(flat, 1024), nn.Sigmoid(),
                nn.Dropout(0.3), nn.Linear(1024, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return DASCNN()
