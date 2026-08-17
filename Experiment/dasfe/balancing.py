"""Class-balancing strategies, applied to the training fold only.

Resampling is wrapped in an `imblearn.pipeline.Pipeline` so that when the
pipeline is handed to `cross_validate` or `GridSearchCV`, the resampler runs
*inside* each fold and never touches validation rows.  Fitting SMOTE on data
that later becomes a validation fold is one of the most common silent leaks in
imbalanced-learning papers; the pipeline makes it structurally impossible.
"""
from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

from . import config as C

SAMPLERS = {
    "none": lambda seed: None,
    "class_weight": lambda seed: None,          # handled by the estimator, not a sampler
    "undersample": lambda seed: RandomUnderSampler(random_state=seed),
    # imbalanced-learn >= 0.12 dropped `n_jobs` from the SMOTE family; parallelism
    # now comes from the underlying nearest-neighbour estimator's own threading.
    "smote": lambda seed: SMOTE(random_state=seed, k_neighbors=5),
    "borderline_smote": lambda seed: BorderlineSMOTE(random_state=seed, k_neighbors=5),
    "smote_tomek": lambda seed: SMOTETomek(random_state=seed),
}

USES_CLASS_WEIGHT = {"class_weight"}


def make_pipeline(estimator, strategy: str, scale: bool = False,
                  seed: int = C.SEED) -> ImbPipeline:
    """Build `[scaler?] -> [sampler?] -> estimator` as one leak-safe pipeline."""
    if strategy not in SAMPLERS:
        raise KeyError(f"unknown balancing strategy '{strategy}'")
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))
    sampler = SAMPLERS[strategy](seed)
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", estimator))
    return ImbPipeline(steps)


def class_weights(y: np.ndarray) -> dict:
    """Inverse-frequency weights, as used for the CNN baseline."""
    classes, counts = np.unique(y, return_counts=True)
    w = len(y) / (len(classes) * counts)
    return dict(zip(classes, w))


def sample_weights(y: np.ndarray) -> np.ndarray:
    w = class_weights(y)
    return np.asarray([w[v] for v in y], dtype=float)


def describe(y_before: np.ndarray, y_after: np.ndarray):
    """Before/after class counts, for the balancing tables."""
    import pandas as pd

    b = pd.Series(y_before).value_counts().rename("before")
    a = pd.Series(y_after).value_counts().rename("after")
    df = pd.concat([b, a], axis=1).fillna(0).astype(int)
    df["ratio"] = (df["after"] / df["before"]).round(3)
    return df.sort_values("before", ascending=False)
