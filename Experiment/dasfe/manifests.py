"""Sample-level manifests for both datasets.

A manifest is a pandas DataFrame with one row per *analysis window*.  Every row
carries the physical provenance of the window (recording, time offset, channel)
plus the `group` column that the leakage-safe splitter is allowed to cut on.

Nothing here reads bulk signal data: Tomasov manifests come from the small
boolean annotation masks (.npy) and Cao manifests from filenames, so building
both takes seconds rather than hours.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ==========================================================================
# Cao 2023
# ==========================================================================
_CAO_RE = re.compile(r"^(?P<date>\d{6})_(?P<operator>[^_]+)_(?P<rest>.+)_data_(?P<idx>\d+)\.mat$")


def parse_cao_filename(fname: str) -> dict:
    """Split `220112_cxm_background_01_single_data_1.mat` into its parts.

    The `rest` group holds the event token plus the session identifier, e.g.
    `background_01_single` or `walk30s_single`.  The session identifier is what
    makes two samples non-independent, so it becomes the split group.
    """
    m = _CAO_RE.match(fname)
    if m is None:
        return {"date": "unknown", "operator": "unknown", "session": fname, "sample_idx": -1}
    d = m.groupdict()
    return {
        "date": d["date"],
        "operator": d["operator"],
        "session": d["rest"],
        "sample_idx": int(d["idx"]),
    }


def build_cao_manifest(root: Path | None = None) -> pd.DataFrame:
    """One row per .mat file.  Records the official split for the audit."""
    root = Path(root or C.CAO_DIR)
    rows = []
    for official in ("Training", "Test"):
        for class_dir, label in C.CAO_CLASS_DIRS.items():
            d = root / official / class_dir
            if not d.is_dir():
                continue
            for path in sorted(d.iterdir()):
                if path.suffix.lower() != ".mat":
                    continue
                parts = parse_cao_filename(path.name)
                rows.append(
                    {
                        "dataset": "cao",
                        "label": label,
                        "class_dir": class_dir,
                        "official_split": official.lower().replace("training", "train"),
                        "filename": path.name,
                        "path": str(path),
                        "n_bytes": path.stat().st_size,
                        "date": parts["date"],
                        "operator": parts["operator"],
                        "session": parts["session"],
                        "sample_idx": parts["sample_idx"],
                    }
                )
    df = pd.DataFrame(rows)
    # A session is only well defined within a class + date + operator.
    df["session_id"] = (
        df["date"] + "|" + df["operator"] + "|" + df["label"] + "|" + df["session"]
    )
    df["date_id"] = df["date"]
    df["sample_id"] = "cao::" + df["class_dir"] + "::" + df["filename"]
    df["n_channels"] = C.CAO.n_channels
    df["fs"] = C.CAO.fs
    return df.reset_index(drop=True)


def audit_cao_official_split(df: pd.DataFrame) -> pd.DataFrame:
    """Quantify how far the released 8:2 split leaks at the session level."""
    g = (
        df.groupby("session_id")["official_split"]
        .agg(n_train=lambda s: (s == "train").sum(), n_test=lambda s: (s == "test").sum())
        .reset_index()
    )
    g["in_both"] = (g["n_train"] > 0) & (g["n_test"] > 0)
    return g


# ==========================================================================
# Tomasov 2024/2025
# ==========================================================================
def _tomasov_recordings(root: Path | None = None) -> list[dict]:
    root = Path(root or C.TOMASOV_DIR)
    out = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for h5 in sorted(class_dir.glob("*.h5")):
            stem = h5.with_suffix("").name
            npy = class_dir / f"{stem}.npy"
            js = class_dir / f"{stem}.json"
            if not npy.exists():
                continue
            out.append(
                {
                    "label": class_dir.name,
                    "stem": stem,
                    "h5": h5,
                    "npy": npy,
                    "json": js if js.exists() else None,
                }
            )
    return out


def build_tomasov_manifest(root: Path | None = None) -> pd.DataFrame:
    """One row per annotated (time-block, locus) cell.

    The released `.npy` is a boolean mask of shape (n_blocks, n_loci) where a
    block is `WIN_HOP` raw samples.  A window of `WIN_LEN` samples starting at
    block *b* therefore spans blocks b .. b + WIN_LEN/WIN_HOP - 1, which is the
    fact the guard band in `splits.py` relies on.
    """
    blocks_per_window = C.WIN_LEN // C.WIN_HOP
    rows = []
    for rec in _tomasov_recordings(root):
        mask = np.load(rec["npy"], allow_pickle=False)
        n_blocks, n_loci = mask.shape
        # Only blocks that can still fit a full window are usable.
        usable = n_blocks - blocks_per_window + 1
        b_idx, c_idx = np.nonzero(mask[:usable])
        if b_idx.size == 0:
            continue
        zone = b_idx // C.ZONE_BLOCKS
        rows.append(
            pd.DataFrame(
                {
                    "dataset": "tomasov",
                    "label": rec["label"],
                    "stem": rec["stem"],
                    "h5_path": str(rec["h5"]),
                    "block": b_idx.astype(np.int32),
                    "locus": c_idx.astype(np.int32),
                    "zone": zone.astype(np.int32),
                    "n_blocks": n_blocks,
                    "n_loci": n_loci,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["t0"] = df["block"].astype(np.int64) * C.WIN_HOP
    df["t1"] = df["t0"] + C.WIN_LEN
    df["zone_id"] = df["stem"] + "|z" + df["zone"].astype(str)
    df["recording_id"] = df["stem"]
    df["sample_id"] = (
        "tomasov::" + df["stem"] + "::b" + df["block"].astype(str)
        + "::c" + df["locus"].astype(str)
    )
    df["fs"] = C.TOMASOV.fs
    return df.reset_index(drop=True)


def read_tomasov_acquisition(h5_path: str | Path) -> dict:
    """Pull the acquisition attributes so `fs` is read, never assumed."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        a = dict(f["Acquisition"].attrs)
    decode = lambda v: v.decode() if isinstance(v, (bytes, np.bytes_)) else v
    return {k: decode(v) for k, v in a.items()}


def load_tomasov_annotation_polygons(json_path: str | Path) -> list[dict]:
    """The .json holds the hand-drawn polygons the .npy mask was rasterised from."""
    with open(json_path) as f:
        j = json.load(f)
    curves = []
    for k in sorted(k for k in j if k.startswith("curve")):
        curves.append({"name": k, "x": j[k]["x"], "y": j[k]["y"]})
    return curves


# ==========================================================================
# Shared helpers
# ==========================================================================
def summarise(df: pd.DataFrame, group_key: str) -> pd.DataFrame:
    """Per-class window count plus the number of independent split groups."""
    out = (
        df.groupby("label")
        .agg(n_windows=("label", "size"), n_groups=(group_key, "nunique"))
        .reset_index()
        .sort_values("n_windows", ascending=False)
    )
    out["pct"] = 100 * out["n_windows"] / out["n_windows"].sum()
    return out.reset_index(drop=True)
