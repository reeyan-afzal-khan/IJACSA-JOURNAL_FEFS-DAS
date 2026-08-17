"""Frequency-domain descriptors: 90 features per window.

Groups follow Table 4 of the manuscript:
    basic spectral stats 8 | spectral shape 8 | peak-related 6 |
    band powers 14 | DAS band ratios 7 | spectral entropy 3 |
    PSD statistics 3 | histogram 33 | real cepstral coefficients 8

Band edges are expressed in Hz and clipped to the Nyquist frequency of the
dataset, so the same code is valid for the 20 kHz Tomasov stream and the much
slower Cao acquisition.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis

EPS = 1e-12
BANDS_HZ = [(0, 50), (50, 100), (100, 200), (200, 400), (400, 800),
            (800, 1600), (1600, None)]     # None -> fs/2
N_HIST_BINS = 32
N_CEPSTRAL = 8


def _entropy(p: np.ndarray, base: float = 2.0) -> float:
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    return float(-(p * (np.log(p) / np.log(base))).sum()) if p.size else 0.0


def _renyi(p: np.ndarray, alpha: float = 2.0) -> float:
    p = p / (p.sum() + EPS)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(np.log2((p ** alpha).sum()) / (1.0 - alpha))


def _rolloff(mag: np.ndarray, freqs: np.ndarray, frac: float) -> float:
    c = np.cumsum(mag)
    if c[-1] <= EPS:
        return 0.0
    return float(freqs[int(np.searchsorted(c, frac * c[-1]))])


def feature_names() -> list[str]:
    names = [
        # --- basic spectral stats (8) ---
        "mag_mean", "mag_std", "mag_min", "mag_max", "mag_range",
        "pow_mean", "pow_std", "pow_max",
        # --- spectral shape (8) ---
        "spec_centroid", "spec_spread", "spec_rolloff85", "spec_rolloff",
        "spec_flatness", "spec_skew", "spec_kurt", "spec_slope",
        # --- peak-related (6) ---
        "peak1_freq", "peak1_amp", "peak2_freq", "peak2_amp",
        "peak_count", "peak_ratio",
    ]
    names += [f"bandP_{i}" for i in range(len(BANDS_HZ))]                 # 7
    names += [f"bandP_rel_{i}" for i in range(len(BANDS_HZ))]             # 7
    names += [
        # --- DAS band ratios (7) ---
        "ratio_low_mid", "ratio_mid_high", "ratio_low_high", "ratio_high_tail",
        "frac_low", "frac_mid", "frac_high",
        # --- spectral entropy (3) ---
        "spec_entropy_shannon", "spec_entropy_renyi", "spec_decrease",
        # --- PSD statistics (3) ---
        "psd_max", "psd_min", "psd_range",
    ]
    names += [f"hist_b{i}" for i in range(N_HIST_BINS)] + ["hist_entropy"]  # 33
    names += [f"cepstral_{i}" for i in range(1, N_CEPSTRAL + 1)]           # 8
    return names


N_FEATURES = len(feature_names())
assert N_FEATURES == 90, f"frequency domain must expose 90 features, got {N_FEATURES}"


def extract(x: np.ndarray, fs: float) -> np.ndarray:
    """Compute all 90 frequency-domain features for one preprocessed window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    spec = np.fft.rfft(x)
    mag = np.abs(spec)
    power = mag ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    nyq = fs / 2.0
    msum = mag.sum() + EPS
    p_norm = power / (power.sum() + EPS)
    out: list[float] = []

    # ---------------- basic spectral stats (8) ----------------
    out += [
        float(mag.mean()), float(mag.std()), float(mag.min()), float(mag.max()),
        float(mag.max() - mag.min()),
        float(power.mean()), float(power.std()), float(power.max()),
    ]

    # ---------------- spectral shape (8) ----------------
    centroid = float((freqs * mag).sum() / msum)
    spread = float(np.sqrt((((freqs - centroid) ** 2) * mag).sum() / msum))
    geo = float(np.exp(np.mean(np.log(mag + EPS))))
    slope = float(np.polyfit(freqs, mag, 1)[0]) if freqs.size > 2 else 0.0
    out += [
        centroid, spread,
        _rolloff(mag, freqs, 0.85), _rolloff(mag, freqs, 0.95),
        float(geo / (mag.mean() + EPS)),
        float(skew(mag)), float(kurtosis(mag)), slope,
    ]

    # ---------------- peak-related (6) ----------------
    pk, props = find_peaks(mag, height=mag.mean())
    if pk.size:
        order = np.argsort(props["peak_heights"])[::-1]
        top = pk[order[:2]]
        amps = props["peak_heights"][order[:2]]
    else:
        top, amps = np.array([0]), np.array([0.0])
    f1, a1 = float(freqs[top[0]]), float(amps[0])
    f2 = float(freqs[top[1]]) if top.size > 1 else 0.0
    a2 = float(amps[1]) if amps.size > 1 else 0.0
    out += [f1, a1, f2, a2, float(pk.size), float(a2 / (a1 + EPS))]

    # ---------------- band powers (14) ----------------
    total = float(power.sum()) + EPS
    band_abs = []
    for lo, hi in BANDS_HZ:
        hi_hz = nyq if hi is None else min(hi, nyq)
        lo_hz = min(lo, nyq)
        m = (freqs >= lo_hz) & (freqs < hi_hz)
        band_abs.append(float(power[m].sum()))
    band_abs = np.asarray(band_abs)
    out += list(band_abs)
    out += list(band_abs / total)

    # ---------------- DAS band ratios (7) ----------------
    low = band_abs[:2].sum()          # < 100 Hz
    mid = band_abs[2:4].sum()         # 100-400 Hz
    high = band_abs[4:].sum()         # > 400 Hz
    out += [
        float(low / (mid + EPS)), float(mid / (high + EPS)),
        float(low / (high + EPS)), float(band_abs[-1] / (total + EPS)),
        float(low / total), float(mid / total), float(high / total),
    ]

    # ---------------- spectral entropy (3) ----------------
    k = np.arange(mag.size)
    decrease = float(((mag[1:] - mag[0]) / (k[1:] + EPS)).sum() / (mag[1:].sum() + EPS))
    out += [_entropy(p_norm), _renyi(p_norm), decrease]

    # ---------------- PSD statistics (3) ----------------
    psd = power / (fs * n)
    out += [float(psd.max()), float(psd.min()), float(psd.max() - psd.min())]

    # ---------------- histogram (33) ----------------
    logmag = np.log10(mag + EPS)
    hist, _ = np.histogram(logmag, bins=N_HIST_BINS)
    hist = hist / (hist.sum() + EPS)
    out += list(hist) + [_entropy(hist)]

    # ---------------- real cepstrum (8) ----------------
    cep = np.fft.irfft(np.log(mag + EPS), n=n).real
    out += list(cep[1:N_CEPSTRAL + 1])

    arr = np.asarray(out, dtype=np.float32)
    if arr.size != N_FEATURES:
        raise RuntimeError(f"freq extractor produced {arr.size} values, expected {N_FEATURES}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard against ratio features exploding on degenerate windows; the bound is
    # far outside the range of any physically meaningful descriptor.
    return np.clip(arr, -1e9, 1e9)
