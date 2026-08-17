"""Ensemble feature selection: 8 methods + consensus voting.

**Leakage rule enforced here:** every selector receives training-fold data
only.  `fit_all` takes `X_train, y_train` and never sees validation or test
rows; the scaler used by the scale-sensitive methods is fitted inside this
function on the training fold as well.  Notebook 05 additionally re-fits the
whole ensemble inside each cross-validation fold, so the reported CV numbers
carry no selection bias - the original submission fitted selection once on the
full training set, which biases CV estimates upward even when the test set is
untouched.

Methods: Variance, one-vs-rest Pearson, Mutual Information, ReliefF, mRMR,
RFE (balanced logistic regression), Boruta (random forest), SHAP (TreeExplainer
on a random forest).
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config as C

warnings.filterwarnings("ignore", category=UserWarning)


def _stratified_subsample(X, y, n_max, seed=C.SEED):
    """Class-proportional subsample for the selectors that scale badly."""
    if n_max is None or len(y) <= n_max:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(classes, counts):
        take = max(1, int(round(n_max * cnt / len(y))))
        pool = np.flatnonzero(y == cls)
        idx.append(rng.choice(pool, size=min(take, pool.size), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


# ==========================================================================
# Individual selectors - each returns a score per feature (higher = better)
# ==========================================================================
def score_variance(X, y, **kw):
    return X.var(axis=0)


def score_pearson(X, y, **kw):
    """One-vs-rest mean absolute Pearson correlation across classes."""
    Xc = X - X.mean(axis=0)
    sx = Xc.std(axis=0) + 1e-12
    scores = np.zeros(X.shape[1])
    for cls in np.unique(y):
        t = (y == cls).astype(float)
        tc = t - t.mean()
        st = tc.std() + 1e-12
        scores += np.abs((Xc * tc[:, None]).mean(axis=0) / (sx * st))
    return scores / len(np.unique(y))


def score_mutual_info(X, y, seed=C.SEED, n_max=C.FS_SUBSAMPLE, **kw):
    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    return mutual_info_classif(Xs, ys, n_neighbors=3, random_state=seed)


def score_relieff(X, y, seed=C.SEED, n_max=6000, **kw):
    from skrebate import ReliefF

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    sel = ReliefF(n_neighbors=10, n_features_to_select=Xs.shape[1], n_jobs=C.N_JOBS)
    sel.fit(Xs.astype(np.float64), ys)
    return np.asarray(sel.feature_importances_, dtype=float)


def score_mrmr(X, y, k=None, n_max=C.FS_SUBSAMPLE, seed=C.SEED,
               mrmr_k=C.MRMR_K, **kw):
    """Maximum relevance, minimum redundancy; scored via mutual information.

    mRMR is greedy and costs O(mrmr_k * n_features) mutual-information
    evaluations, so ranking all 1,002 features is the slowest step in the
    ensemble.  We rank the top `mrmr_k` and give the remainder score 0 - they
    would not survive the `top_frac` cut anyway.
    """
    from mrmr import mrmr_classif

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    k = min(mrmr_k, Xs.shape[1])
    df = pd.DataFrame(Xs, columns=[str(i) for i in range(Xs.shape[1])])
    order = mrmr_classif(X=df, y=pd.Series(ys), K=k, show_progress=False)
    scores = np.zeros(X.shape[1])
    for rank, col in enumerate(order):
        scores[int(col)] = len(order) - rank        # first picked = highest score
    return scores


def score_rfe(X, y, n_max=C.FS_SUBSAMPLE, seed=C.SEED, step=0.1, **kw):
    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    est = LogisticRegression(solver="lbfgs", max_iter=300, class_weight="balanced",
                             n_jobs=C.N_JOBS)
    rfe = RFE(est, n_features_to_select=1, step=step)
    rfe.fit(StandardScaler().fit_transform(Xs), ys)
    return -rfe.ranking_.astype(float)              # rank 1 is best -> highest score


def score_boruta(X, y, n_max=8000, seed=C.SEED, max_iter=60, **kw):
    from boruta import BorutaPy

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    rf = RandomForestClassifier(n_estimators=800, class_weight="balanced",
                                max_depth=8, n_jobs=C.N_JOBS, random_state=seed)
    bor = BorutaPy(rf, n_estimators="auto", max_iter=max_iter, random_state=seed, verbose=0)
    bor.fit(Xs.astype(np.float64), ys.astype(str))
    return -bor.ranking_.astype(float)


def score_shap(X, y, n_max=6000, seed=C.SEED, **kw):
    import shap

    Xs, ys = _stratified_subsample(X, y, n_max, seed)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                n_jobs=C.N_JOBS, random_state=seed)
    rf.fit(Xs, ys)
    expl = shap.TreeExplainer(rf)
    vals = expl.shap_values(Xs[: min(2000, len(Xs))], check_additivity=False)
    vals = np.asarray(vals)
    # shape is (n_samples, n_features, n_classes) or a list per class
    axes = tuple(a for a in range(vals.ndim) if a != vals.ndim - 2) if vals.ndim == 3 else 0
    return np.abs(vals).mean(axis=axes) if vals.ndim == 3 else np.abs(vals).mean(axis=0)


SCORERS = {
    "variance": score_variance,
    "pearson": score_pearson,
    "mutual_info": score_mutual_info,
    "relieff": score_relieff,
    "mrmr": score_mrmr,
    "rfe": score_rfe,
    "boruta": score_boruta,
    "shap": score_shap,
}


# ==========================================================================
# Ensemble
# ==========================================================================
def fit_all(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str],
            methods=C.FS_METHODS, top_frac: float = 0.5,
            seed: int = C.SEED, verbose: bool = True, fast: bool = False) -> dict:
    """Score every feature with every method, using training data only.

    `top_frac` is the fraction of features each method is allowed to keep, so
    all methods vote on a comparably sized subset and the consensus is not
    dominated by whichever method happens to be least selective.

    `fast=True` shrinks the subsample budgets (see `config.FS_FAST_KW`) without
    changing which methods run.  It is used for the in-fold refits, where the
    ensemble is evaluated `n_folds` times.
    """
    n_feat = X_train.shape[1]
    k = max(1, int(round(top_frac * n_feat)))
    scores, selected, timings = {}, {}, {}

    for name in methods:
        t0 = time.perf_counter()
        kw = dict(C.FS_FAST_KW.get(name, {})) if fast else {}
        try:
            s = np.asarray(SCORERS[name](X_train, y_train, seed=seed, **kw), dtype=float)
            s = np.nan_to_num(s, nan=-np.inf)
        except Exception as exc:                      # keep the ensemble alive
            if verbose:
                print(f"  [{name}] FAILED: {type(exc).__name__}: {exc}")
            continue
        if s.size != n_feat:
            if verbose:
                print(f"  [{name}] wrong score length {s.size} != {n_feat}, skipped")
            continue
        scores[name] = s
        mask = np.zeros(n_feat, dtype=bool)
        mask[np.argsort(s)[::-1][:k]] = True
        selected[name] = mask
        timings[name] = time.perf_counter() - t0
        if verbose:
            print(f"  [{name}] kept {mask.sum()}/{n_feat} in {timings[name]:.1f}s")

    votes = np.sum(list(selected.values()), axis=0) if selected else np.zeros(n_feat, int)
    ranks = {m: pd.Series(-s).rank(method="average").to_numpy() for m, s in scores.items()}

    table = pd.DataFrame({"feature": feature_names, "votes": votes})
    for m, s in scores.items():
        table[f"score_{m}"] = s
        table[f"rank_{m}"] = ranks[m]
    rank_cols = [c for c in table.columns if c.startswith("rank_")]
    table["avg_rank"] = table[rank_cols].mean(axis=1)
    table["selection_pct"] = 100 * table["votes"] / max(len(selected), 1)
    table = table.sort_values(["votes", "avg_rank"], ascending=[False, True])

    return {
        "scores": scores,
        "selected": selected,
        "votes": votes,
        "table": table.reset_index(drop=True),
        "timings": timings,
        "n_methods": len(selected),
        "top_k": k,
    }


def consensus_mask(result: dict, threshold: int | None = None) -> np.ndarray:
    """Features chosen by at least `threshold` methods (default: all of them)."""
    threshold = threshold if threshold is not None else result["n_methods"]
    return result["votes"] >= threshold


def jaccard_matrix(result: dict) -> pd.DataFrame:
    """Pairwise agreement between selectors (Figure 6)."""
    methods = list(result["selected"])
    M = np.zeros((len(methods), len(methods)))
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            A, B = result["selected"][a], result["selected"][b]
            union = (A | B).sum()
            M[i, j] = (A & B).sum() / union if union else 1.0
    return pd.DataFrame(M, index=methods, columns=methods)


def save(result: dict, out_dir: Path, tag: str = "multi") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result["table"].to_csv(out_dir / f"fs_table_{tag}.csv", index=False)
    jaccard_matrix(result).to_csv(out_dir / f"fs_jaccard_{tag}.csv")
    np.save(out_dir / f"fs_votes_{tag}.npy", result["votes"])
    np.savez(out_dir / f"fs_masks_{tag}.npz", **{m: v for m, v in result["selected"].items()})
    with open(out_dir / f"fs_timings_{tag}.json", "w") as fh:
        json.dump(result["timings"], fh, indent=1)
