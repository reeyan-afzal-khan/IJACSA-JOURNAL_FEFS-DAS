"""Chunked, resumable, parallel feature extraction over a manifest.

Two departures from the original submission, both deliberate:

1. **One pass, four domains.**  The original extracted each domain in a
   separate run, so every window was read and preprocessed four times
   (4 x ~5 h of I/O).  Here a window is read once and all four extractors are
   applied to it, cutting the offline budget by roughly 4x.
2. **Shard-per-chunk on disk.**  A chunk is one HDF5 recording (Tomasov) or a
   fixed-size batch of .mat files (Cao).  Finished shards are skipped on
   re-run, so an interrupted extraction resumes instead of restarting.

Ordering is deterministic: shards are assembled in the sorted chunk order
recorded in `chunk_index.json`, so `sample_id` alignment across domains is
guaranteed and verified by `assemble`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config as C
from . import features_freq, features_spatial, features_tf, features_time
from . import preprocess as pp

DOMAIN_MODULES = {
    "time": features_time,
    "freq": features_freq,
    "tf": features_tf,
    "spatial": features_spatial,
}
DOMAIN_ORDER = ("time", "freq", "tf", "spatial")


# ==========================================================================
# Per-window driver
# ==========================================================================
def extract_window(x1d: np.ndarray, patch: np.ndarray, fs: float,
                   domains=DOMAIN_ORDER) -> dict[str, np.ndarray]:
    """Run the requested extractors on one already-preprocessed window."""
    out = {}
    for d in domains:
        if d == "spatial":
            out[d] = features_spatial.extract(patch, fs)
        elif d == "tf":
            out[d] = features_tf.extract(x1d, fs)
        elif d == "time":
            out[d] = features_time.extract(x1d, fs)
        elif d == "freq":
            out[d] = features_freq.extract(x1d, fs)
        else:
            raise KeyError(d)
    return out


# ==========================================================================
# Chunk workers
# ==========================================================================
def _chunk_paths(out_dir: Path, chunk: str) -> dict:
    return {
        d: out_dir / "shards" / d / f"{chunk}.npy" for d in DOMAIN_ORDER
    } | {"ids": out_dir / "shards" / "ids" / f"{chunk}.npy"}


def _chunk_done(out_dir: Path, chunk: str, domains) -> bool:
    p = _chunk_paths(out_dir, chunk)
    return p["ids"].exists() and all(p[d].exists() for d in domains)


def _write_chunk(out_dir: Path, chunk: str, ids: list[str],
                 feats: dict[str, list], domains) -> None:
    p = _chunk_paths(out_dir, chunk)
    for d in domains:
        p[d].parent.mkdir(parents=True, exist_ok=True)
        np.save(p[d], np.asarray(feats[d], dtype=np.float32))
    p["ids"].parent.mkdir(parents=True, exist_ok=True)
    np.save(p["ids"], np.asarray(ids, dtype=object), allow_pickle=True)


def _run_tomasov_chunk(rows: pd.DataFrame, out_dir: Path, spec, domains) -> dict:
    """One chunk = one HDF5 recording; the file is opened exactly once."""
    chunk = str(rows["stem"].iloc[0])
    if _chunk_done(out_dir, chunk, domains):
        return {"chunk": chunk, "n": len(rows), "seconds": 0.0, "skipped": True}

    t0 = time.perf_counter()
    feats = {d: [] for d in domains}
    ids: list[str] = []
    rows = rows.sort_values("t0")
    with pp.TomasovReader(rows["h5_path"].iloc[0]) as reader:
        fs = reader.fs
        for t_start, locus, sid in zip(rows["t0"], rows["locus"], rows["sample_id"]):
            raw = reader.read_patch(int(t_start), int(locus), spec.win_len, spec.patch_channels)
            patch = pp.preprocess_patch(raw, fs)
            x1d = patch[pp.center_channel_index(spec.patch_channels)]
            got = extract_window(x1d, patch, fs, domains)
            for d in domains:
                feats[d].append(got[d])
            ids.append(sid)
    _write_chunk(out_dir, chunk, ids, feats, domains)
    return {"chunk": chunk, "n": len(ids), "seconds": time.perf_counter() - t0, "skipped": False}


def _run_cao_chunk(rows: pd.DataFrame, out_dir: Path, spec, domains, chunk: str) -> dict:
    if _chunk_done(out_dir, chunk, domains):
        return {"chunk": chunk, "n": len(rows), "seconds": 0.0, "skipped": True}

    t0 = time.perf_counter()
    feats = {d: [] for d in domains}
    ids: list[str] = []
    for path, sid, n_bytes in zip(rows["path"], rows["sample_id"], rows["n_bytes"]):
        if n_bytes == 0:      # the one corrupt zero-byte record in the release
            continue
        raw = pp.load_cao_window(path, spec.win_len)
        patch = pp.preprocess_patch(raw, spec.fs)
        x1d = patch[pp.cao_reference_channel(raw)]
        got = extract_window(x1d, patch, spec.fs, domains)
        for d in domains:
            feats[d].append(got[d])
        ids.append(sid)
    _write_chunk(out_dir, chunk, ids, feats, domains)
    return {"chunk": chunk, "n": len(ids), "seconds": time.perf_counter() - t0, "skipped": False}


# ==========================================================================
# Public API
# ==========================================================================
def run(manifest: pd.DataFrame, dataset: str, out_dir: Path,
        domains=DOMAIN_ORDER, n_jobs: int = C.N_JOBS,
        cao_chunk_size: int = 500, verbose: int = 10) -> pd.DataFrame:
    """Extract features for every row of `manifest`.  Returns a timing log."""
    spec = C.DATASETS[dataset]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "tomasov":
        chunks = [(str(stem), sub) for stem, sub in manifest.groupby("stem", sort=True)]
        jobs = [delayed(_run_tomasov_chunk)(sub, out_dir, spec, domains) for _, sub in chunks]
    elif dataset == "cao":
        m = manifest.sort_values("sample_id").reset_index(drop=True)
        chunks, jobs = [], []
        for i in range(0, len(m), cao_chunk_size):
            sub = m.iloc[i:i + cao_chunk_size]
            name = f"chunk_{i // cao_chunk_size:05d}"
            chunks.append((name, sub))
            jobs.append(delayed(_run_cao_chunk)(sub, out_dir, spec, domains, name))
    else:
        raise KeyError(dataset)

    with open(out_dir / "chunk_index.json", "w") as fh:
        json.dump([name for name, _ in chunks], fh, indent=1)

    log = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    return pd.DataFrame(log)


def assemble(out_dir: Path, domains=DOMAIN_ORDER) -> dict:
    """Concatenate shards in the recorded chunk order and verify alignment."""
    out_dir = Path(out_dir)
    with open(out_dir / "chunk_index.json") as fh:
        chunk_order = json.load(fh)

    ids_parts, parts = [], {d: [] for d in domains}
    for chunk in chunk_order:
        p = _chunk_paths(out_dir, chunk)
        if not p["ids"].exists():
            raise FileNotFoundError(f"missing shard for chunk '{chunk}' - re-run extract.run")
        ids_parts.append(np.load(p["ids"], allow_pickle=True))
        for d in domains:
            parts[d].append(np.load(p[d]))

    ids = np.concatenate(ids_parts)
    result = {"sample_id": ids}
    np.save(out_dir / "sample_ids.npy", ids, allow_pickle=True)

    for d in domains:
        X = np.concatenate(parts[d], axis=0)
        if X.shape[0] != ids.size:
            raise AssertionError(f"domain '{d}': {X.shape[0]} rows vs {ids.size} ids")
        if X.shape[1] != DOMAIN_MODULES[d].N_FEATURES:
            raise AssertionError(
                f"domain '{d}': {X.shape[1]} cols vs expected {DOMAIN_MODULES[d].N_FEATURES}"
            )
        np.save(out_dir / f"X_{d}.npy", X)
        with open(out_dir / f"features_{d}.json", "w") as fh:
            json.dump(DOMAIN_MODULES[d].feature_names(), fh, indent=1)
        result[d] = X.shape

    if pd.Series(ids).duplicated().any():
        raise AssertionError("duplicated sample_id after assembly")
    return result


def estimate_runtime(dataset: str, n_windows: int, domains=DOMAIN_ORDER,
                     n_trials: int = 20, n_jobs: int = C.N_JOBS) -> pd.DataFrame:
    """Time the extractors on synthetic windows to size the full run."""
    spec = C.DATASETS[dataset]
    rng = np.random.default_rng(C.SEED)
    patch = rng.standard_normal((spec.patch_channels, spec.win_len))
    x1d = patch[spec.patch_channels // 2]

    rows = []
    for d in domains:
        t0 = time.perf_counter()
        for _ in range(n_trials):
            extract_window(x1d, patch, spec.fs, [d])
        per = (time.perf_counter() - t0) / n_trials
        rows.append(
            {
                "domain": d,
                "ms_per_window": 1000 * per,
                "serial_hours": per * n_windows / 3600,
                f"parallel_hours_{n_jobs}j": per * n_windows / 3600 / n_jobs,
            }
        )
    df = pd.DataFrame(rows)
    df.loc[len(df)] = ["TOTAL", df["ms_per_window"].sum(), df["serial_hours"].sum(),
                       df[f"parallel_hours_{n_jobs}j"].sum()]
    return df
