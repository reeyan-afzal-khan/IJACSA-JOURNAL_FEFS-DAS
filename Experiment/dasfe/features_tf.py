"""Time-frequency descriptors: 739 features per window.

Groups follow Table 5 of the manuscript:
    STFT global power 16 | temporal envelope 13 | spectral marginal 21 |
    spectral dynamics 8 | modulation spectrum 36 | band power 5 |
    DWT 140 | WPT 357 | GLCM 80 | LBP 32 | HOG 16 |
    TF geometry 8 | TF distribution 3 | DAS-specific 4

This is by far the most expensive domain (wavelet-packet decomposition over 16
sub-bands plus grey-level co-occurrence over 4 distances x 4 orientations), and
it dominates the offline feature-preparation budget reported in the paper.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.signal import stft, find_peaks
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from . import config as C

EPS = 1e-12

DWT_LEVELS = C.DWT_LEVELS            # -> levels 0..6 = 7 coefficient sets
WPT_LEVEL = C.WPT_LEVEL              # -> 16 sub-bands
N_MOD_BANDS = 12
TF_BANDS_HZ = [(0, 20), (20, 50), (50, 150), (150, 300), (300, 500)]
GLCM_DISTANCES = (1, 2, 3, 4)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_PROPS = ("contrast", "energy", "homogeneity", "correlation", "entropy")
N_LBP_BINS = 32
N_HOG_BINS = 16
_COEF_STATS = (
    "mean", "std", "var", "skew", "kurt", "min", "max", "range", "median",
    "iqr", "mad", "energy", "log_energy", "rms", "entropy", "zcr", "p2p",
    "tk_mean", "tk_std", "tk_max",
)   # 20 per DWT level


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.abs(p)
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0


def _gini(v: np.ndarray) -> float:
    v = np.sort(np.abs(np.asarray(v, dtype=np.float64).ravel()))
    n = v.size
    if n == 0 or v.sum() <= EPS:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(v) / (n * v.sum()))


def _rolloff(mag: np.ndarray, freqs: np.ndarray, frac: float) -> float:
    c = np.cumsum(mag)
    if c[-1] <= EPS:
        return 0.0
    return float(freqs[min(int(np.searchsorted(c, frac * c[-1])), freqs.size - 1)])


def _tkeo(x: np.ndarray) -> np.ndarray:
    return x[1:-1] ** 2 - x[:-2] * x[2:] if x.size > 2 else np.zeros(1)


def _coef_stats(c: np.ndarray) -> list[float]:
    """The 20 statistics computed for every DWT level and WPT sub-band."""
    c = np.asarray(c, dtype=np.float64).ravel()
    if c.size < 3:
        return [0.0] * len(_COEF_STATS)
    q25, q50, q75 = np.percentile(c, [25, 50, 75])
    e = float((c ** 2).sum())
    tk = _tkeo(c)
    return [
        float(c.mean()), float(c.std()), float(c.var()),
        float(skew(c)), float(kurtosis(c)),
        float(c.min()), float(c.max()), float(c.max() - c.min()),
        float(q50), float(q75 - q25), float(np.median(np.abs(c - q50))),
        e, float(np.log10(e + EPS)), float(np.sqrt((c ** 2).mean())),
        _entropy(c ** 2),
        float((np.diff(np.sign(c)) != 0).mean()),
        float(np.ptp(c)),
        float(tk.mean()), float(tk.std()), float(np.abs(tk).max()),
    ]


def feature_names() -> list[str]:
    names = [
        # ---------------- STFT global power (16) ----------------
        "stft_mean", "stft_std", "stft_min", "stft_max", "stft_range",
        "stft_p10", "stft_p25", "stft_p50", "stft_p75", "stft_p90",
        "stft_skew", "stft_kurt", "stft_entropy", "stft_crest", "stft_rms",
        "stft_gini",
        # ---------------- temporal envelope Et (13) ----------------
        "Et_mean", "Et_std", "Et_min", "Et_max", "Et_range", "Et_p25",
        "Et_p75", "Et_iqr", "Et_skew", "Et_kurt", "Et_entropy",
        "Et_centroid", "Et_spread",
        # ---------------- spectral marginal Ef (21) ----------------
        "Ef_centroid", "Ef_spread", "Ef_skew", "Ef_kurt", "Ef_flatness",
        "Ef_rolloff85", "Ef_rolloff95", "Ef_rolloff99", "Ef_mean", "Ef_std",
        "Ef_min", "Ef_max", "Ef_range", "Ef_p25", "Ef_p75", "Ef_iqr",
        "Ef_entropy", "Ef_peak_freq", "Ef_peak_amp", "Ef_n_peaks",
        "Ef_harmonic_ratio",
        # ---------------- spectral dynamics (8) ----------------
        "flux_mean", "flux_std", "flux_max", "divergence_mean",
        "divergence_std", "coherence_mean", "coherence_std", "novelty",
    ]
    for b in range(N_MOD_BANDS):                                    # 36
        names += [f"mod{b}_centroid", f"mod{b}_spread", f"mod{b}_entropy"]
    names += [f"tfband_{i}" for i in range(len(TF_BANDS_HZ))]       # 5
    for lvl in range(DWT_LEVELS + 1):                               # 140
        names += [f"DWT_L{lvl}_{s}" for s in _COEF_STATS]
    for node in _wpt_node_names():                                  # 352
        names += [f"WPT_{node}_{s}" for s in _COEF_STATS]
        names += [f"WPT_{node}_rel_energy", f"WPT_{node}_abs_mean"]
    names += [                                                      # 5
        "WPT_band_entropy", "WPT_band_centroid", "WPT_band_spread",
        "WPT_band_flatness", "WPT_low_high_ratio",
    ]
    idx = 0
    for d in GLCM_DISTANCES:                                        # 80
        for a in range(len(GLCM_ANGLES)):
            for p in GLCM_PROPS:
                names.append(f"glcm_{idx}_{p}_d{d}_a{a}")
                idx += 1
    names += [f"lbp_b{i}" for i in range(N_LBP_BINS)]               # 32
    names += [f"hog_b{i}" for i in range(N_HOG_BINS)]               # 16
    names += [                                                      # 8
        "tf_area", "tf_perimeter", "tf_bbox_extent", "tf_aspect_ratio",
        "tf_elongation", "tf_ridge_density", "tf_sparsity", "tf_fill_ratio",
    ]
    names += ["tf_occupancy", "tf_burstiness", "tf_freq_time_var_ratio"]  # 3
    names += [                                                      # 4
        "das_tonal_ratio", "das_transient_index",
        "das_cyclostationary_index", "das_low_high_ratio",
    ]
    return names


def _wpt_node_names() -> list[str]:
    """Natural-order wavelet-packet node paths at `WPT_LEVEL` (16 sub-bands)."""
    nodes = [""]
    for _ in range(WPT_LEVEL):
        nodes = [n + c for n in nodes for c in ("a", "d")]
    return nodes


N_FEATURES = len(feature_names())
assert N_FEATURES == 739, f"time-frequency domain must expose 739 features, got {N_FEATURES}"


def extract(x: np.ndarray, fs: float,
            nfft: int = C.NFFT_STFT, hop: int = C.HOP_STFT) -> np.ndarray:
    """Compute all 739 time-frequency features for one preprocessed window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    out: list[float] = []

    # ================= STFT =================
    f, t, Z = stft(x, fs=fs, window="hann", nperseg=nfft,
                   noverlap=nfft - hop, boundary=None, padded=False)
    S = np.abs(Z)                       # (n_freq, n_frames)
    P = S ** 2
    if S.shape[1] < 2:                  # degenerate window
        S = np.pad(S, ((0, 0), (0, 2 - S.shape[1])))
        P = S ** 2
        t = np.arange(S.shape[1]) * hop / fs

    flatP = P.ravel()
    q = np.percentile(flatP, [10, 25, 50, 75, 90])
    out += [
        float(flatP.mean()), float(flatP.std()), float(flatP.min()),
        float(flatP.max()), float(flatP.max() - flatP.min()),
        float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
        float(skew(flatP)), float(kurtosis(flatP)), _entropy(flatP),
        float(flatP.max() / (np.sqrt((flatP ** 2).mean()) + EPS)),
        float(np.sqrt((flatP ** 2).mean())), _gini(flatP),
    ]

    # ---------------- temporal envelope Et (13) ----------------
    Et = P.sum(axis=0)
    Etn = Et / (Et.sum() + EPS)
    ti = np.arange(Et.size)
    t_centroid = float((ti * Etn).sum())
    eq = np.percentile(Et, [25, 75])
    out += [
        float(Et.mean()), float(Et.std()), float(Et.min()), float(Et.max()),
        float(Et.max() - Et.min()), float(eq[0]), float(eq[1]),
        float(eq[1] - eq[0]), float(skew(Et)), float(kurtosis(Et)),
        _entropy(Etn), t_centroid,
        float(np.sqrt(((ti - t_centroid) ** 2 * Etn).sum())),
    ]

    # ---------------- spectral marginal Ef (21) ----------------
    Ef = P.sum(axis=1)
    Efn = Ef / (Ef.sum() + EPS)
    centroid = float((f * Efn).sum())
    geo = float(np.exp(np.mean(np.log(Ef + EPS))))
    pk, props = find_peaks(Ef, height=Ef.mean())
    peak_i = int(pk[np.argmax(props["peak_heights"])]) if pk.size else int(np.argmax(Ef))
    harm = float(Ef[pk].sum() / (Ef.sum() + EPS)) if pk.size else 0.0
    fq = np.percentile(Ef, [25, 75])
    out += [
        centroid,
        float(np.sqrt(((f - centroid) ** 2 * Efn).sum())),
        float(skew(Ef)), float(kurtosis(Ef)),
        float(geo / (Ef.mean() + EPS)),
        _rolloff(Ef, f, 0.85), _rolloff(Ef, f, 0.95), _rolloff(Ef, f, 0.99),
        float(Ef.mean()), float(Ef.std()), float(Ef.min()), float(Ef.max()),
        float(Ef.max() - Ef.min()), float(fq[0]), float(fq[1]),
        float(fq[1] - fq[0]), _entropy(Efn),
        float(f[peak_i]), float(Ef[peak_i]), float(pk.size), harm,
    ]

    # ---------------- spectral dynamics (8) ----------------
    dS = np.diff(S, axis=1)
    flux = np.sqrt((dS ** 2).sum(axis=0))
    Pn = P / (P.sum(axis=0, keepdims=True) + EPS)
    div = (Pn[:, 1:] * np.log2((Pn[:, 1:] + EPS) / (Pn[:, :-1] + EPS))).sum(axis=0)
    num = (S[:, 1:] * S[:, :-1]).sum(axis=0)
    den = np.sqrt((S[:, 1:] ** 2).sum(axis=0) * (S[:, :-1] ** 2).sum(axis=0)) + EPS
    coh = num / den
    out += [
        float(flux.mean()), float(flux.std()), float(flux.max()),
        float(div.mean()), float(div.std()),
        float(coh.mean()), float(coh.std()),
        float(np.abs(np.diff(Et)).mean() / (Et.mean() + EPS)),
    ]

    # ---------------- modulation spectrum (36) ----------------
    band_edges = np.linspace(0, S.shape[0], N_MOD_BANDS + 1).astype(int)
    for b in range(N_MOD_BANDS):
        lo, hi = band_edges[b], max(band_edges[b + 1], band_edges[b] + 1)
        env = P[lo:hi].sum(axis=0)
        M = np.abs(np.fft.rfft(env - env.mean()))
        Mn = M / (M.sum() + EPS)
        mi = np.arange(M.size)
        cen = float((mi * Mn).sum())
        out += [cen, float(np.sqrt(((mi - cen) ** 2 * Mn).sum())), _entropy(Mn)]

    # ---------------- DAS band power (5) ----------------
    nyq = fs / 2.0
    for lo, hi in TF_BANDS_HZ:
        m = (f >= min(lo, nyq)) & (f < min(hi, nyq))
        out.append(float(P[m].sum()))

    # ================= wavelets =================
    coeffs = pywt.wavedec(x, C.WAVELET, level=DWT_LEVELS)
    for lvl in range(DWT_LEVELS + 1):
        out += _coef_stats(coeffs[lvl] if lvl < len(coeffs) else np.zeros(4))

    wp = pywt.WaveletPacket(data=x, wavelet=C.WAVELET, mode="symmetric", maxlevel=WPT_LEVEL)
    node_names = _wpt_node_names()
    node_energy = []
    node_coeffs = []
    for name in node_names:
        try:
            c = np.asarray(wp[name].data, dtype=np.float64)
        except (KeyError, IndexError):
            c = np.zeros(4)
        node_coeffs.append(c)
        node_energy.append(float((c ** 2).sum()))
    node_energy = np.asarray(node_energy)
    tot_e = float(node_energy.sum()) + EPS
    for c, e in zip(node_coeffs, node_energy):
        out += _coef_stats(c)
        out += [float(e / tot_e), float(np.abs(c).mean())]

    we = node_energy / tot_e
    bi = np.arange(node_energy.size)
    b_centroid = float((bi * we).sum())
    geo_e = float(np.exp(np.mean(np.log(node_energy + EPS))))
    out += [
        _entropy(we), b_centroid,
        float(np.sqrt(((bi - b_centroid) ** 2 * we).sum())),
        float(geo_e / (node_energy.mean() + EPS)),
        float(node_energy[: len(node_energy) // 2].sum()
              / (node_energy[len(node_energy) // 2:].sum() + EPS)),
    ]

    # ================= texture & geometry on the log-spectrogram =================
    logS = np.log10(P + EPS)
    lo, hi = logS.min(), logS.max()
    Q = np.floor((logS - lo) / (hi - lo + EPS) * (C.SPEC_QUANT_LEVELS - 1)).astype(np.uint8)

    glcm = graycomatrix(Q, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES),
                        levels=C.SPEC_QUANT_LEVELS, symmetric=True, normed=True)
    prop_vals = {p: graycoprops(glcm, p) for p in ("contrast", "energy", "homogeneity", "correlation")}
    glcm_ent = -(glcm * np.log2(glcm + EPS)).sum(axis=(0, 1))
    for di in range(len(GLCM_DISTANCES)):
        for ai in range(len(GLCM_ANGLES)):
            for p in GLCM_PROPS:
                v = glcm_ent[di, ai] if p == "entropy" else prop_vals[p][di, ai]
                out.append(float(v))

    lbp = local_binary_pattern(Q, P=8, R=1, method="default")
    lbp_hist, _ = np.histogram(lbp, bins=N_LBP_BINS, range=(0, 256))
    out += list(lbp_hist / (lbp_hist.sum() + EPS))

    gy, gx = np.gradient(logS)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.arctan2(gy, gx) + np.pi) % np.pi
    hog_hist, _ = np.histogram(ang, bins=N_HOG_BINS, range=(0, np.pi), weights=mag)
    out += list(hog_hist / (hog_hist.sum() + EPS))

    # ---------------- geometry of the binarised TF mask (8) ----------------
    mask = P > (P.mean() + P.std())
    area = float(mask.sum())
    if mask.any():
        rows, cols = np.nonzero(mask)
        h = float(rows.max() - rows.min() + 1)
        wdt = float(cols.max() - cols.min() + 1)
        perim = float(np.abs(np.diff(mask.astype(np.int8), axis=0)).sum()
                      + np.abs(np.diff(mask.astype(np.int8), axis=1)).sum())
    else:
        h = wdt = perim = 0.0
    total_px = float(mask.size)
    out += [
        area / total_px, perim / total_px,
        (h * wdt) / total_px,
        float(wdt / (h + EPS)),
        float(max(h, wdt) / (min(h, wdt) + EPS)),
        float(mask.sum(axis=0).astype(bool).mean()),
        float(1.0 - area / total_px),
        float(area / (h * wdt + EPS)),
    ]

    # ---------------- TF distribution (3) ----------------
    out += [
        float(mask.mean()),
        float(Et.std() / (Et.mean() + EPS)),
        float(Ef.var() / (Et.var() + EPS)),
    ]

    # ---------------- DAS-specific metrics (4) ----------------
    smooth = np.convolve(Ef, np.ones(5) / 5.0, mode="same")
    rough = np.abs(Ef - smooth)
    ac = np.correlate(Et - Et.mean(), Et - Et.mean(), mode="full")[Et.size - 1:]
    ac = ac / (ac[0] + EPS)
    low_m = f < 100
    high_m = f > 300
    out += [
        float(smooth.sum() / (rough.sum() + EPS)),
        float(Et.var() / (Et.mean() ** 2 + EPS)),
        float(ac[1:].max()) if ac.size > 1 else 0.0,
        float(Ef[low_m].sum() / (Ef[high_m].sum() + EPS)),
    ]

    arr = np.asarray(out, dtype=np.float32)
    if arr.size != N_FEATURES:
        raise RuntimeError(f"tf extractor produced {arr.size} values, expected {N_FEATURES}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against ratio features exploding on degenerate windows; the bound is
    # far outside the range of any physically meaningful descriptor.
    return np.clip(arr, -1e9, 1e9)
