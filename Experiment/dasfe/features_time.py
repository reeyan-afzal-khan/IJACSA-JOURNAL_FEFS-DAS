"""Time-domain descriptors: 112 features per window.

Groups follow Table 3 of the manuscript:
    basic statistical 26 | energy & power 13 + K(=10) | temporal 19 |
    slope & derivative 7 | nonlinear 7 | envelope 6 | burst 6 | histogram 18
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.stats import skew, kurtosis

EPS = 1e-12
N_SEGMENTS = 10
N_HIST_BINS = 16


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


def _katz_fd(x: np.ndarray) -> float:
    d1 = np.abs(np.diff(x))
    L = d1.sum()
    if L <= EPS:
        return 0.0
    d = np.max(np.abs(x - x[0]))
    n = x.size - 1
    if d <= EPS:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(d / L)))


def _higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    n = x.size
    lk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            length = np.abs(np.diff(x[idx])).sum() * (n - 1) / ((idx.size - 1) * k)
            lm.append(length)
        if lm:
            lk.append(np.mean(lm))
    if len(lk) < 2:
        return 0.0
    k = np.arange(1, len(lk) + 1)
    coef = np.polyfit(np.log(1.0 / k), np.log(np.array(lk) + EPS), 1)
    return float(coef[0])


def _tkeo(x: np.ndarray) -> np.ndarray:
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def _runs(mask: np.ndarray):
    d = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends


def feature_names() -> list[str]:
    names = [
        # --- basic statistical (26) ---
        "mean", "std", "var", "skew", "kurt", "min", "max", "p2p", "median",
        "mad", "iqr", "p1", "p5", "p10", "p25", "p75", "p90", "p95", "p99",
        "max_abs", "mean_abs", "rms_to_mad", "trimmed_mean10", "sem",
        "range_over_std", "zero_frac",
        # --- energy & power (13) ---
        "rms", "energy", "log_energy", "mean_power", "crest_factor",
        "impulse_factor", "margin_factor", "shape_factor", "clearance_factor",
        "energy_entropy", "energy_centroid", "energy_spread", "energy_gini",
    ]
    names += [f"seg_energy_{i}" for i in range(N_SEGMENTS)]          # 10
    names += [
        # --- temporal patterns (19) ---
        "zcr", "zcr_rate", "n_peaks", "n_valleys", "peak_rate",
        "mean_peak_amp", "std_peak_amp", "max_peak_amp",
        "mean_peak_interval", "std_peak_interval", "activity_factor",
        "above_1std_frac", "above_2std_frac", "above_3std_frac",
        "longest_run_above", "autocorr_lag1", "autocorr_lag10",
        "autocorr_first_zero", "autocorr_peak",
        # --- slope & derivative (7) ---
        "d1_mean_abs", "d1_std", "d1_max_abs", "d2_mean_abs", "d2_std",
        "ssc_rate", "waveform_length",
        # --- nonlinear (7) ---
        "hjorth_activity", "hjorth_mobility", "hjorth_complexity",
        "katz_fd", "higuchi_fd", "tkeo_mean", "tkeo_std",
        # --- envelope (6) ---
        "env_mean", "env_std", "env_skew", "env_kurt", "env_entropy",
        "env_crest",
        # --- burst structure (6) ---
        "burst_count", "burst_rate", "burst_mean_dur", "burst_std_dur",
        "burst_mean_gap", "burst_occupancy",
    ]
    names += [f"hist_b{i}" for i in range(N_HIST_BINS)]              # 16
    names += ["hist_entropy", "hist_contrast"]                       # 2
    return names


N_FEATURES = len(feature_names())
assert N_FEATURES == 112, f"time domain must expose 112 features, got {N_FEATURES}"


def extract(x: np.ndarray, fs: float) -> np.ndarray:
    """Compute all 112 time-domain features for one preprocessed window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    ax = np.abs(x)
    mu, sd = float(x.mean()), float(x.std())
    out: list[float] = []

    # ---------------- basic statistical (26) ----------------
    q = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    mad = float(np.median(np.abs(x - q[4])))
    # NOTE: the preprocessing chain z-scores each window, so `mean` is 0 and
    # `std`/`var`/`sem` are 1 by construction.  Those columns are constant and
    # get removed by the variance threshold in notebook 05 - they are kept here
    # only so the feature vector matches the published 112-dimensional layout.
    # A plain coefficient of variation (std/|mean|) would divide by zero, so the
    # scale-invariant peakedness ratio rms/MAD is used in that slot instead.
    out += [
        mu, sd, sd ** 2, float(skew(x)), float(kurtosis(x)),
        float(x.min()), float(x.max()), float(x.max() - x.min()), float(q[4]),
        mad, float(q[5] - q[3]),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        float(q[5]), float(q[6]), float(q[7]), float(q[8]),
        float(ax.max()), float(ax.mean()),
        float(np.sqrt((x ** 2).mean()) / (mad + 1e-6)),
        float(np.mean(np.sort(x)[int(0.1 * n):n - int(0.1 * n)])),
        float(sd / np.sqrt(n)),
        float((x.max() - x.min()) / (sd + EPS)),
        float(np.mean(ax < 1e-9)),
    ]

    # ---------------- energy & power (13 + 10) ----------------
    energy = float((x ** 2).sum())
    rms = float(np.sqrt((x ** 2).mean()))
    arv = float(ax.mean()) + EPS
    sqrt_mean = float(np.mean(np.sqrt(ax))) ** 2 + EPS
    seg = np.array_split(x, N_SEGMENTS)
    seg_e = np.array([float((s ** 2).sum()) for s in seg])
    w = seg_e / (seg_e.sum() + EPS)
    idx = np.arange(N_SEGMENTS)
    e_centroid = float((idx * w).sum())
    out += [
        rms, energy, float(np.log10(energy + EPS)), float((x ** 2).mean()),
        float(ax.max() / (rms + EPS)),
        float(ax.max() / arv),
        float(ax.max() / sqrt_mean),
        float(rms / arv),
        float(ax.max() / (rms + EPS) / (arv + EPS)),
        _entropy(seg_e), e_centroid,
        float(np.sqrt(((idx - e_centroid) ** 2 * w).sum())),
        _gini(seg_e),
    ]
    out += list(seg_e / (energy + EPS))

    # ---------------- temporal patterns (19) ----------------
    sgn = np.sign(x)
    zc = int((np.diff(sgn) != 0).sum())
    peaks, props = find_peaks(x, height=sd)
    valleys, _ = find_peaks(-x, height=sd)
    pk_amp = props["peak_heights"] if peaks.size else np.array([0.0])
    pk_int = np.diff(peaks) if peaks.size > 1 else np.array([0.0])
    above = ax > sd
    ac = np.correlate(x, x, mode="full")[n - 1:]
    ac = ac / (ac[0] + EPS)
    first_zero = int(np.argmax(ac <= 0)) if (ac <= 0).any() else n
    out += [
        float(zc), float(zc / (2 * (n - 1)) * fs),
        float(peaks.size), float(valleys.size), float(peaks.size / n * fs),
        float(pk_amp.mean()), float(pk_amp.std()), float(pk_amp.max()),
        float(pk_int.mean()), float(pk_int.std()),
        float((x[above] ** 2).sum() / (energy + EPS)),   # activity factor
        float(above.mean()), float((ax > 2 * sd).mean()), float((ax > 3 * sd).mean()),
        float(_longest_run(above)),
        float(ac[1]) if n > 1 else 0.0,
        float(ac[10]) if n > 10 else 0.0,
        float(first_zero),
        float(ac[1:].max()) if n > 1 else 0.0,
    ]

    # ---------------- slope & derivative (7) ----------------
    d1 = np.diff(x)
    d2 = np.diff(d1)
    ssc = int((np.diff(np.sign(d1)) != 0).sum())
    out += [
        float(np.abs(d1).mean()), float(d1.std()), float(np.abs(d1).max()),
        float(np.abs(d2).mean()), float(d2.std()),
        float(ssc / (n - 2)), float(np.abs(d1).sum()),
    ]

    # ---------------- nonlinear (7) ----------------
    v0, v1, v2 = x.var(), d1.var(), d2.var()
    mob = float(np.sqrt(v1 / (v0 + EPS)))
    tk = _tkeo(x)
    out += [
        float(v0), mob,
        float(np.sqrt(v2 / (v1 + EPS)) / (mob + EPS)),
        _katz_fd(x), _higuchi_fd(x),
        float(tk.mean()), float(tk.std()),
    ]

    # ---------------- envelope (6) ----------------
    env = np.abs(hilbert(x))
    out += [
        float(env.mean()), float(env.std()),
        float(skew(env)), float(kurtosis(env)),
        _entropy(env), float(env.max() / (env.mean() + EPS)),
    ]

    # ---------------- burst structure (6) ----------------
    thr = env.mean() + env.std()
    burst = env > thr
    starts, ends = _runs(burst)
    durs = ends - starts
    gaps = starts[1:] - ends[:-1] if starts.size > 1 else np.array([0])
    out += [
        float(starts.size), float(starts.size / n * fs),
        float(durs.mean()) if durs.size else 0.0,
        float(durs.std()) if durs.size else 0.0,
        float(gaps.mean()) if gaps.size else 0.0,
        float(burst.mean()),
    ]

    # ---------------- histogram (18) ----------------
    hist, _ = np.histogram(x, bins=N_HIST_BINS, range=(x.min(), x.max() + EPS))
    hist = hist / (hist.sum() + EPS)
    lvl = np.arange(N_HIST_BINS)
    out += list(hist)
    out += [_entropy(hist), float(((lvl - (lvl * hist).sum()) ** 2 * hist).sum())]

    arr = np.asarray(out, dtype=np.float32)
    if arr.size != N_FEATURES:
        raise RuntimeError(f"time extractor produced {arr.size} values, expected {N_FEATURES}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against ratio features exploding on degenerate windows; the bound is
    # far outside the range of any physically meaningful descriptor.
    return np.clip(arr, -1e9, 1e9)
