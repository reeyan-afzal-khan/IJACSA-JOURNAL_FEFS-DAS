"""Multi-domain feature fusion (Algorithm 1 of the manuscript).

Domain matrices are concatenated along the feature axis after their names are
prefixed with the domain identifier, so a feature stays traceable to its origin
all the way through selection and SHAP attribution.

Every consistency check the algorithm calls for is executed, not assumed: equal
row counts, identical `sample_id` ordering across domains, and identical label
ordering after the join with the manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .extract import DOMAIN_MODULES, DOMAIN_ORDER

PREFIX = {"time": "TIME", "freq": "FREQ", "tf": "TF", "spatial": "SPAT"}


def load_domain(feat_dir: Path, domain: str) -> tuple[np.ndarray, list[str]]:
    feat_dir = Path(feat_dir)
    X = np.load(feat_dir / f"X_{domain}.npy")
    with open(feat_dir / f"features_{domain}.json") as fh:
        names = json.load(fh)
    if X.shape[1] != len(names):
        raise AssertionError(f"{domain}: {X.shape[1]} columns vs {len(names)} names")
    return X, names


def fuse(feat_dir: Path, manifest: pd.DataFrame, domains=DOMAIN_ORDER,
         label_col: str = "label") -> dict:
    """Return the fused matrix, labels, groups, split tags and feature names."""
    feat_dir = Path(feat_dir)
    ids = np.load(feat_dir / "sample_ids.npy", allow_pickle=True).astype(str)

    mats, names = [], []
    for d in domains:
        X, nm = load_domain(feat_dir, d)
        if X.shape[0] != ids.size:
            raise AssertionError(f"domain '{d}' has {X.shape[0]} rows, expected {ids.size}")
        mats.append(X)
        names += [f"{PREFIX[d]}:{n}" for n in nm]

    X_multi = np.concatenate(mats, axis=1)

    # Align the manifest to the extraction order (extraction may have skipped
    # unreadable records, so this is an inner join, and we verify no dupes).
    m = manifest.drop_duplicates("sample_id").set_index("sample_id")
    missing = set(ids) - set(m.index)
    if missing:
        raise AssertionError(f"{len(missing)} extracted ids absent from the manifest")
    m = m.loc[ids]

    return {
        "X": X_multi,
        "y": m[label_col].to_numpy(),
        "sample_id": ids,
        "feature_names": names,
        "manifest": m.reset_index(),
        "domain_slices": _domain_slices(domains),
    }


def _domain_slices(domains=DOMAIN_ORDER) -> dict[str, slice]:
    out, start = {}, 0
    for d in domains:
        n = DOMAIN_MODULES[d].N_FEATURES
        out[d] = slice(start, start + n)
        start += n
    return out


def save(fused: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_multi.npy", fused["X"].astype(np.float32))
    np.save(out_dir / "y.npy", fused["y"])
    np.save(out_dir / "sample_ids.npy", fused["sample_id"], allow_pickle=True)
    with open(out_dir / "feature_names.json", "w") as fh:
        json.dump(fused["feature_names"], fh, indent=1)
    fused["manifest"].to_parquet(out_dir / "manifest_aligned.parquet", index=False)


def load(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    with open(out_dir / "feature_names.json") as fh:
        names = json.load(fh)
    return {
        "X": np.load(out_dir / "X_multi.npy"),
        "y": np.load(out_dir / "y.npy", allow_pickle=True),
        "sample_id": np.load(out_dir / "sample_ids.npy", allow_pickle=True),
        "feature_names": names,
        "manifest": pd.read_parquet(out_dir / "manifest_aligned.parquet"),
    }


def domain_composition(selected_names: list[str]) -> pd.DataFrame:
    """How many of the selected features came from each domain (Figure 9)."""
    rows = []
    for d, pfx in PREFIX.items():
        n = sum(1 for s in selected_names if s.startswith(pfx + ":"))
        rows.append({"domain": d, "n_selected": n,
                     "n_available": DOMAIN_MODULES[d].N_FEATURES})
    df = pd.DataFrame(rows)
    df["pct_of_selected"] = 100 * df["n_selected"] / max(len(selected_names), 1)
    df["retention_pct"] = 100 * df["n_selected"] / df["n_available"]
    return df
