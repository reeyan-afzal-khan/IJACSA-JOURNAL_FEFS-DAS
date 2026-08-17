"""Metrics, benchmarking loop, and the Demsar statistical-comparison framework.

`benchmark` is the workhorse: it trains one (model x balancing strategy) on the
training split, selects on validation, and reports on the held-out test split -
in that order, and never the other way round.  Grouped cross-validation uses
the same group column as the split, so CV folds inherit the leakage guarantees
of the split itself.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

from . import balancing, models
from . import config as C


# ==========================================================================
# Metrics
# ==========================================================================
def score_all(y_true, y_pred, y_proba=None, labels=None) -> dict:
    labels = labels if labels is not None else np.unique(y_true)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None and len(labels) > 1:
        try:
            Y = label_binarize(y_true, classes=list(labels))
            out["auc_ovr_macro"] = roc_auc_score(Y, y_proba, average="macro", multi_class="ovr")
        except ValueError:
            out["auc_ovr_macro"] = np.nan
    return out


def per_class_report(y_true, y_pred, labels=None) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    df = pd.DataFrame(rep).T
    df.index.name = "class"
    return df.reset_index()


def confusion(y_true, y_pred, labels=None, normalize: str | None = None) -> pd.DataFrame:
    labels = labels if labels is not None else sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return pd.DataFrame(cm, index=labels, columns=labels)


# ==========================================================================
# Training / benchmarking
# ==========================================================================
def fit_predict(model_name: str, strategy: str, X_tr, y_tr, X_eval, y_eval,
                seed: int = C.SEED) -> dict:
    """Fit one configuration and score it on one evaluation split."""
    est = models.build(model_name, strategy, seed=seed)
    pipe = balancing.make_pipeline(
        est, strategy, scale=model_name in models.NEEDS_SCALING, seed=seed
    )

    # XGBoost >= 2.0 removed the internal label encoder from the scikit-learn
    # wrapper: `fit` now requires y to already be integers 0..n_classes-1 and
    # raises "Invalid classes inferred from unique values of `y`" on the string
    # labels every other estimator here accepts.  Encode for XGBoost only and
    # decode the predictions immediately, so all metrics stay in label space.
    encoder = None
    y_fit = y_tr
    if models.needs_encoded_labels(model_name):
        encoder = LabelEncoder().fit(y_tr)
        y_fit = encoder.transform(y_tr)

    fit_kw = {}
    if models.xgb_needs_sample_weight(model_name, strategy):
        fit_kw["clf__sample_weight"] = balancing.sample_weights(y_tr)

    t0 = time.perf_counter()
    pipe.fit(X_tr, y_fit, **fit_kw)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipe.predict(X_eval)
    infer_s = time.perf_counter() - t0
    if encoder is not None:
        y_pred = encoder.inverse_transform(np.asarray(y_pred).astype(int))

    proba = pipe.predict_proba(X_eval) if hasattr(pipe, "predict_proba") else None
    metrics = score_all(y_eval, y_pred, proba, labels=np.unique(y_tr))
    metrics.update(
        model=model_name, strategy=strategy,
        train_seconds=train_s, infer_seconds=infer_s,
        ms_per_sample=1000 * infer_s / max(len(y_eval), 1),
        n_train=len(y_tr), n_eval=len(y_eval), n_features=X_tr.shape[1],
    )
    return {"metrics": metrics, "pipeline": pipe, "y_pred": y_pred, "y_proba": proba}


def benchmark(X, y, split, model_names=C.MODEL_NAMES,
              strategies=C.BALANCING_STRATEGIES, seed: int = C.SEED,
              verbose: bool = True) -> pd.DataFrame:
    """Every (model x strategy) evaluated on validation *and* test.

    `split` is an array of 'train'/'val'/'test' tags aligned with X and y.
    Model selection must be made on the validation columns; the test columns
    are reported once, at the end, for the chosen configuration only.
    """
    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    rows = []
    failures: dict[str, list[str]] = {}
    for name in model_names:
        for strat in strategies:
            try:
                val = fit_predict(name, strat, X[tr], y[tr], X[va], y[va], seed)
                tst = fit_predict(name, strat, X[tr], y[tr], X[te], y[te], seed)
            except Exception as exc:
                failures.setdefault(name, []).append(
                    f"{strat}: {type(exc).__name__}: {exc}"
                )
                if verbose:
                    print(f"  {name:22s} {strat:18s} FAILED: {type(exc).__name__}: {exc}")
                continue
            row = {"model": name, "strategy": strat,
                   "train_seconds": val["metrics"]["train_seconds"]}
            row |= {f"val_{k}": v for k, v in val["metrics"].items() if k.startswith(("acc", "bal", "f1", "prec", "rec", "auc"))}
            row |= {f"test_{k}": v for k, v in tst["metrics"].items() if k.startswith(("acc", "bal", "f1", "prec", "rec", "auc"))}
            row["test_ms_per_sample"] = tst["metrics"]["ms_per_sample"]
            rows.append(row)
            if verbose:
                print(f"  {name:22s} {strat:18s} "
                      f"val_F1={row['val_f1_macro']:.4f}  test_F1={row['test_f1_macro']:.4f}  "
                      f"({row['train_seconds']:.1f}s)")

    # A model that fails for every strategy vanishes from the results table.
    # That is a silent hole in a benchmark, so it is reported regardless of
    # `verbose` - the domain-comparison section runs with verbose=False and
    # would otherwise drop a classifier without a single line of output.
    dropped = [m for m in model_names if m in failures and len(failures[m]) == len(strategies)]
    if dropped:
        print(f"  !! {len(dropped)} model(s) produced NO results and are absent from the "
              f"benchmark: {', '.join(dropped)}")
        for m in dropped:
            print(f"     {m}: {failures[m][0]}")

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(
            "every (model x strategy) configuration failed; first error: "
            + next(iter(next(iter(failures.values()), ["<none>"])), "<none>")
        )
    out.attrs["failures"] = failures
    return out.sort_values("val_f1_macro", ascending=False).reset_index(drop=True)


def grouped_cv(X, y, groups, model_name: str, strategy: str,
               n_splits: int = C.CV_FOLDS, seed: int = C.SEED,
               selector=None) -> pd.DataFrame:
    """Grouped stratified CV.  If `selector` is given it is re-fitted per fold.

    `selector(X_tr, y_tr) -> boolean mask` lets feature selection live inside
    the fold, which is what makes the CV estimate honest.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        Xtr, Xte = X[tr], X[te]
        if selector is not None:
            mask = selector(Xtr, y[tr])
            Xtr, Xte = Xtr[:, mask], Xte[:, mask]
        res = fit_predict(model_name, strategy, Xtr, y[tr], Xte, y[te], seed)
        rows.append({"fold": fold, **res["metrics"]})
    df = pd.DataFrame(rows)
    return df


def ci95(values) -> tuple:
    v = np.asarray(values, dtype=float)
    m = v.mean()
    half = 1.96 * v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    return float(m), float(m - half), float(m + half)


# ==========================================================================
# Demsar (2006) multi-classifier comparison
# ==========================================================================
def friedman_nemenyi(pivot: pd.DataFrame, alpha: float = 0.05) -> dict:
    """`pivot`: rows = conditions (strategies), columns = classifiers, values = score."""
    from scipy.stats import friedmanchisquare, rankdata, studentized_range

    scores = pivot.to_numpy(dtype=float)
    n, k = scores.shape
    stat, p = friedmanchisquare(*[scores[:, j] for j in range(k)])

    # Iman-Davenport correction
    f_stat = (n - 1) * stat / (n * (k - 1) - stat) if (n * (k - 1) - stat) != 0 else np.inf

    ranks = np.array([rankdata(-row) for row in scores])   # rank 1 = best
    avg_rank = ranks.mean(axis=0)
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    order = np.argsort(avg_rank)
    ranking = pd.DataFrame(
        {"model": pivot.columns[order], "avg_rank": avg_rank[order]}
    ).reset_index(drop=True)

    return {
        "friedman_chi2": float(stat), "friedman_p": float(p),
        "iman_davenport_F": float(f_stat),
        "df1": k - 1, "df2": (k - 1) * (n - 1),
        "critical_difference": float(cd),
        "ranking": ranking, "n_conditions": n, "n_models": k,
    }


def cd_groups(ranking: pd.DataFrame, cd: float) -> list[list[str]]:
    """Cliques of models that are not significantly different (Figure 10)."""
    models_ = ranking["model"].tolist()
    ranks = ranking["avg_rank"].to_numpy()
    groups = []
    for i in range(len(models_)):
        grp = [models_[j] for j in range(len(models_)) if abs(ranks[j] - ranks[i]) <= cd]
        if grp not in groups:
            groups.append(grp)
    return groups
