"""Spatial and spatio-temporal descriptors: 61 features per window.

Groups follow Table 6 of the manuscript:
    spatial       - localisation 7 | distribution 10 | gradient 7 | clustering 5
    spatio-temporal - global 2D 6 | temporal activity 7 | motion 7 |
                      gradient/texture 8 | coarse texture 4

Input is a preprocessed patch X of shape (C, T).  Tomasov supplies C = 32 loci
around the annotated channel; Cao supplies all C = 12 loci of the record.  The
descriptors are all normalised by C, so they remain comparable between the two
systems - but the Cao values rest on only 12 channels and notebook 08 reports a
spatial-free variant alongside, because 12 channels is thin support for
propagation statistics.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

EPS = 1e-12
N_TEX_BINS = 16


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _gini(v: np.ndarray) -> float:
    v = np.sort(np.abs(np.asarray(v, dtype=np.float64)))
    n = v.size
    if n == 0 or v.sum() <= EPS:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(v) / (n * v.sum()))


def _clusters(mask: np.ndarray):
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return ends - starts


def feature_names() -> list[str]:
    return [
        # ---------- spatial: localisation & extent (7) ----------
        "sp_centroid", "sp_spread", "sp_width", "sp_active10", "sp_active50",
        "sp_active_frac", "sp_argmax",
        # ---------- spatial: distribution & shape (10) ----------
        "sp_q25", "sp_q50", "sp_q75", "sp_skew", "sp_kurt", "sp_gini",
        "sp_entropy", "sp_flatness", "sp_max_frac", "sp_top3_frac",
        # ---------- spatial: gradient & roughness (7) ----------
        "sp_grad_mean_abs", "sp_grad_std", "sp_grad_energy", "sp_grad_max",
        "sp_edge_density", "sp_smoothness", "sp_tv_norm",
        # ---------- spatial: clustering (5) ----------
        "sp_n_clusters", "sp_cluster_mean", "sp_cluster_max", "sp_cluster_std",
        "sp_coverage",
        # ---------- spatio-temporal: global 2D (6) ----------
        "st_mean", "st_std", "st_skew", "st_kurt", "st_energy", "st_entropy",
        # ---------- spatio-temporal: temporal activity (7) ----------
        "st_active_dur_frac", "st_t_centroid", "st_t_spread", "st_burstiness",
        "st_t_max", "st_t_entropy", "st_t_skew",
        # ---------- spatio-temporal: motion & propagation (7) ----------
        "st_vel_mean", "st_vel_std", "st_direction", "st_linearity_r2",
        "st_traj_var", "st_traj_range", "st_traj_drift",
        # ---------- spatio-temporal: gradient & texture (8) ----------
        "st_grad_t_energy", "st_grad_c_energy", "st_anisotropy",
        "st_edge_density_t", "st_edge_density_c", "st_grad_t_mean",
        "st_grad_c_mean", "st_grad_corr",
        # ---------- spatio-temporal: coarse texture (4) ----------
        "st_hist_energy", "st_hist_entropy", "st_hist_contrast",
        "st_hist_uniformity",
    ]


N_FEATURES = len(feature_names())
assert N_FEATURES == 61, f"spatial domain must expose 61 features, got {N_FEATURES}"


def extract(patch: np.ndarray, fs: float, n_time_bins: int = 32) -> np.ndarray:
    """Compute all 61 spatial / spatio-temporal features for one (C, T) patch."""
    X = np.asarray(patch, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2D (C, T) patch, got shape {X.shape}")
    Cn, T = X.shape
    out: list[float] = []

    # ================= spatial: channel-energy profile =================
    E = (X ** 2).sum(axis=1)                       # (C,)
    Etot = float(E.sum()) + EPS
    w = E / Etot
    c_idx = np.arange(Cn)

    centroid = float((c_idx * w).sum())
    spread = float(np.sqrt(((c_idx - centroid) ** 2 * w).sum()))
    act10 = E >= 0.10 * E.max()
    act50 = E >= 0.50 * E.max()
    width = float(np.ptp(c_idx[act10])) if act10.any() else 0.0
    out += [
        centroid, spread, width,
        float(act10.sum()), float(act50.sum()),
        float(act10.mean()), float(np.argmax(E)),
    ]

    cum = np.cumsum(w)
    q = [float(np.searchsorted(cum, f)) for f in (0.25, 0.50, 0.75)]
    geo = float(np.exp(np.mean(np.log(E + EPS))))
    top3 = float(np.sort(E)[-3:].sum() / Etot) if Cn >= 3 else 1.0
    out += [
        q[0], q[1], q[2],
        float(skew(E)) if Cn > 2 else 0.0,
        float(kurtosis(E)) if Cn > 3 else 0.0,
        _gini(E), _entropy(w), float(geo / (E.mean() + EPS)),
        float(E.max() / Etot), top3,
    ]

    dE = np.diff(E) if Cn > 1 else np.array([0.0])
    gmax = float(np.abs(dE).max()) + EPS
    out += [
        float(np.abs(dE).mean()), float(dE.std()), float((dE ** 2).sum()),
        float(np.abs(dE).max()),
        float((np.abs(dE) > 0.10 * gmax).mean()),
        float(1.0 / (1.0 + np.abs(dE).mean())),
        float(np.abs(dE).sum() / Etot),
    ]

    sizes = _clusters(act10)
    out += [
        float(sizes.size),
        float(sizes.mean()) if sizes.size else 0.0,
        float(sizes.max()) if sizes.size else 0.0,
        float(sizes.std()) if sizes.size else 0.0,
        float(act10.sum() / Cn),
    ]

    # ================= spatio-temporal: 2D space-time patch =================
    P = X ** 2                                     # instantaneous power
    flat = X.ravel()
    out += [
        float(flat.mean()), float(flat.std()),
        float(skew(flat)), float(kurtosis(flat)),
        float(P.sum()), _entropy(P.ravel()),
    ]

    # Temporal activity profile: energy projected onto the time axis.
    t_prof = P.sum(axis=0)
    t_prof_n = t_prof / (t_prof.sum() + EPS)
    t_idx = np.arange(T)
    t_centroid = float((t_idx * t_prof_n).sum())
    thr = t_prof.mean() + t_prof.std()
    out += [
        float((t_prof > thr).mean()),
        t_centroid / T,
        float(np.sqrt(((t_idx - t_centroid) ** 2 * t_prof_n).sum())) / T,
        float(t_prof.std() / (t_prof.mean() + EPS)),
        float(t_prof.max() / (t_prof.sum() + EPS)),
        _entropy(t_prof_n),
        float(skew(t_prof)),
    ]

    # Motion & propagation: track the spatial centroid over coarse time bins.
    bins = min(n_time_bins, T)
    Pb = np.array([b.sum(axis=1) for b in np.array_split(P, bins, axis=1)])  # (bins, C)
    wb = Pb / (Pb.sum(axis=1, keepdims=True) + EPS)
    traj = (wb * c_idx).sum(axis=1)                                          # (bins,)
    vel = np.diff(traj) if traj.size > 1 else np.array([0.0])
    tb = np.arange(traj.size)
    if traj.size > 2 and traj.std() > EPS:
        slope, intercept = np.polyfit(tb, traj, 1)
        pred = slope * tb + intercept
        ss_res = float(((traj - pred) ** 2).sum())
        ss_tot = float(((traj - traj.mean()) ** 2).sum()) + EPS
        r2 = 1.0 - ss_res / ss_tot
    else:
        slope, r2 = 0.0, 0.0
    out += [
        float(vel.mean()), float(vel.std()), float(slope), float(r2),
        float(traj.var()), float(np.ptp(traj)), float(traj[-1] - traj[0]),
    ]

    # Gradient & texture along both axes.
    gt = np.diff(X, axis=1)
    gc = np.diff(X, axis=0) if Cn > 1 else np.zeros((1, T))
    et, ec = float((gt ** 2).sum()), float((gc ** 2).sum())
    m = min(gt.shape[0], gc.shape[0]), min(gt.shape[1], gc.shape[1])
    a, b = gt[:m[0], :m[1]].ravel(), gc[:m[0], :m[1]].ravel()
    corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > EPS and b.std() > EPS else 0.0
    out += [
        et, ec, float(et / (ec + EPS)),
        float((np.abs(gt) > np.abs(gt).mean()).mean()),
        float((np.abs(gc) > np.abs(gc).mean()).mean()),
        float(np.abs(gt).mean()), float(np.abs(gc).mean()), corr,
    ]

    # Coarse amplitude-histogram texture over the whole patch.
    hist, _ = np.histogram(flat, bins=N_TEX_BINS)
    hist = hist / (hist.sum() + EPS)
    lvl = np.arange(N_TEX_BINS)
    mean_lvl = float((lvl * hist).sum())
    out += [
        float((hist ** 2).sum()), _entropy(hist),
        float(((lvl - mean_lvl) ** 2 * hist).sum()),
        float((hist ** 2).sum() / (hist.sum() + EPS)),
    ]

    arr = np.asarray(out, dtype=np.float32)
    if arr.size != N_FEATURES:
        raise RuntimeError(f"spatial extractor produced {arr.size} values, expected {N_FEATURES}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against ratio features exploding on degenerate windows; the bound is
    # far outside the range of any physically meaningful descriptor.
    return np.clip(arr, -1e9, 1e9)
