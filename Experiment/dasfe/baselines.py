"""Published CNN baselines, reproduced as faithfully as the sources allow.

Two architectures, each taken from its dataset's own authors:

**Cao et al. 2023** - `models.py` / `mydataset.py` / `das_data_cnn.py` from
https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes (shipped in
`Dataset/Cao_2023/Phi-OTDR_dataset_and_codes-main/`).  A *2D* CNN over the whole
10000x12 space-time record, 7,421 parameters.  Input is min-max scaled to
0-255 per sample - not band-passed, not z-scored.

**Tomasov et al. 2025** - "Advancing Perimeter Security: Integrating DAS and CNN
for Object Classification in Fiber Vicinity", IEEE Access 13, Section III-E.  No
code was released, so the architecture is transcribed from the text:

    two Conv1D layers, 64 then 256 filters, LeakyReLU after each, max pooling
    with pool size 4 after each, flatten, dense layer with 256 neurons and
    sigmoid activation, output layer with softmax.

Kernel size 7 is taken from Table 21 of the DAS feature-engineering manuscript,
which tabulated this same network.  Note that table used a 1024-unit dense
layer; the Access paper says 256, and 256 is what we use here.

Crucially, Tomasov et al. do **not** feed raw signals to the CNN - Section III-C
states plainly that raw time-domain input is inadequate, and applies RDFT, DWT or
MFCC first.  RDFT gave their best F1 (84.84%) and MFCC their best accuracy
(85.61%).  All three transforms are implemented in `tomasov_transform`.

WHERE WE HAD TO INFER
  The Access paper's Table 2 (hyper-parameters) does not survive text
  extraction, so the RDFT redundancy factor is inferred from two stated facts:
  the output is truncated to 2048 bins and that corresponds to 0-833 Hz.  With
  fs = 20 kHz this requires an FFT length of 20000 * 2048 / 833 ~ 49152 = 6 x
  8192, i.e. a redundancy factor of 6.  `RDFT_REDUNDANCY` is exposed so the
  choice can be varied.
"""
from __future__ import annotations

import numpy as np

from . import config as C

# --------------------------------------------------------------------------
# Cao et al. 2023
# --------------------------------------------------------------------------
CAO_TIME = 10000          # their records are 10000 samples; the conv arithmetic
CAO_CHANNELS = 12         # below only produces their stated shapes at 10000x12


def cao_normalise(record: np.ndarray) -> np.ndarray:
    """Per-sample min-max scaling to 0-255, as in their `mydataset.normalize`.

    Their loop rounds to integers; we keep float32 (numerically identical after
    the first conv, and ~1000x faster than the nested Python loop).
    """
    x = np.asarray(record, dtype=np.float64)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return np.round(255.0 * (x - lo) / (hi - lo)).astype(np.float32)


def build_cao_cnn(n_classes: int = 6):
    """Their 2D CNN verbatim.  Layer shapes match the comments in `models.py`:
    (1,10000,12) -> (5,197,12) -> pool (5,99,7) -> (10,21,8) -> pool (10,10,4).
    """
    import torch.nn as nn

    class CaoCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 5, kernel_size=(200, 3), stride=(50, 1), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, padding=1),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(5, 10, kernel_size=(20, 2), stride=(4, 1), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            )
            self.out = nn.Linear(10 * 10 * 4, n_classes)

        def forward(self, x):
            x = self.conv2(self.conv1(x))
            feat = x.flatten(1)
            return self.out(feat)

    return CaoCNN()


# --------------------------------------------------------------------------
# Tomasov et al. 2025
# --------------------------------------------------------------------------
RDFT_REDUNDANCY = 6       # inferred: 2048 bins <-> 0-833 Hz at fs = 20 kHz
RDFT_BINS = 2048
DWT_LEVEL = 3             # "decomposition depth of 3 with the Daubechies 4 wavelet"
DWT_WAVELET = "db4"
MFCC_N = 40
MFCC_HOP = 512


def tomasov_transform(x: np.ndarray, fs: float, kind: str = "rdft") -> np.ndarray:
    """Section III-C of the Access paper.  Returns a 1D float32 feature vector.

    rdft - zero-pad by RDFT_REDUNDANCY, FFT, |.|, log10, subtract the mean,
           truncate to the first RDFT_BINS bins (0-833 Hz).
    dwt  - level-3 db4 decomposition, coefficients concatenated.
    mfcc - librosa MFCCs, flattened.

    DWT and MFCC are additionally standardised by the caller across the training
    set ("normalized using a Standard Scaler"), which is why no scaling happens
    here - doing it per window would not be what the paper describes.
    """
    x = np.asarray(x, dtype=np.float64).ravel()

    if kind == "rdft":
        n_fft = len(x) * RDFT_REDUNDANCY
        mag = np.abs(np.fft.rfft(x, n=n_fft)).astype(np.float32)
        spec = np.log10(mag + 1e-12)
        spec = spec - spec.mean()
        out = spec[:RDFT_BINS]
        if out.size < RDFT_BINS:
            out = np.pad(out, (0, RDFT_BINS - out.size))
        return np.ascontiguousarray(out, dtype=np.float32)

    if kind == "dwt":
        import pywt

        coeffs = pywt.wavedec(x, DWT_WAVELET, level=DWT_LEVEL)
        return np.ascontiguousarray(np.concatenate(coeffs), dtype=np.float32)

    if kind == "mfcc":
        try:
            import librosa

            m = librosa.feature.mfcc(y=x.astype(np.float32), sr=int(fs),
                                     n_mfcc=MFCC_N, hop_length=MFCC_HOP)
        except Exception:
            m = _mfcc_fallback(x, fs, n_mfcc=MFCC_N, hop=MFCC_HOP)
        return np.ascontiguousarray(np.asarray(m).ravel(), dtype=np.float32)

    raise KeyError(f"unknown transform '{kind}'")


def _mel_filterbank(n_filters: int, n_fft: int, fs: float) -> np.ndarray:
    """Slaney-style triangular mel filterbank, so MFCC does not require librosa.

    librosa is the reference implementation used by Tomasov et al., but it is not
    always installable (it pulls in msgpack/soundfile).  This fallback matches
    librosa's default `htk=False` mel scale closely enough that the CNN sees an
    equivalent representation; the notebook records which path was taken.
    """
    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        mel = 3.0 * f / 200.0
        log_t = f >= 1000.0
        mel[log_t] = 15.0 + np.log(f[log_t] / 1000.0) / (np.log(6.4) / 27.0)
        return mel

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        f = 200.0 * m / 3.0
        log_t = m >= 15.0
        f[log_t] = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m[log_t] - 15.0))
        return f

    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    edges = mel_to_hz(np.linspace(hz_to_mel(np.array([0.0]))[0],
                                  hz_to_mel(np.array([fs / 2.0]))[0],
                                  n_filters + 2))
    fb = np.zeros((n_filters, fft_freqs.size))
    for i in range(n_filters):
        lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        falling = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(rising, falling), 0, None)
        norm = fb[i].sum()
        if norm > 0:
            fb[i] /= norm
    return fb


def _mfcc_fallback(x: np.ndarray, fs: float, n_mfcc: int = 40,
                   hop: int = 512, n_fft: int = 2048) -> np.ndarray:
    from scipy.fftpack import dct
    from scipy.signal import stft as _stft

    _, _, Z = _stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                    boundary=None, padded=False)
    power = np.abs(Z) ** 2
    mel = _mel_filterbank(n_mfcc * 2, n_fft, fs) @ power
    log_mel = np.log(mel + 1e-10)
    return dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]


def tomasov_input_length(kind: str, win_len: int = C.WIN_LEN, fs: float = 20000.0) -> int:
    """Length of the vector `tomasov_transform` returns, for building the net."""
    probe = np.zeros(win_len)
    return int(tomasov_transform(probe, fs, kind).size)


def build_tomasov_cnn(n_classes: int, input_len: int, dense_units: int = 256,
                      kernel: int = 7, dropout: float = 0.3):
    """Access paper Section III-E: 64 then 256 Conv1D filters, LeakyReLU after
    each, max pooling 4 after each, flatten, dense 256 + sigmoid, softmax out.

    `dense_units=1024` reproduces Table 21 of the feature-engineering manuscript
    instead, for an explicit like-for-like with the previously reported figure.
    """
    import torch
    import torch.nn as nn

    class TomasovCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=kernel), nn.LeakyReLU(0.01),
                nn.MaxPool1d(4),
                nn.Conv1d(64, 256, kernel_size=kernel), nn.LeakyReLU(0.01),
                nn.MaxPool1d(4),
            )
            with torch.no_grad():
                flat = self.features(torch.zeros(1, 1, input_len)).numel()
            self.flat = flat
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, dense_units), nn.Sigmoid(),
                nn.Dropout(dropout),
                nn.Linear(dense_units, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return TomasovCNN()


# --------------------------------------------------------------------------
# Metrics the source papers report but our pipeline does not
# --------------------------------------------------------------------------
def nar_fnr(y_true, y_pred, background_label) -> dict:
    """Nuisance Alarm Rate and False Negative Rate, as defined in the Cao README.

    NAR = false alarms / total alarms          (background predicted as an event)
    FNR = missed events / total real events    (event predicted as background)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    is_bg_true = y_true == background_label
    is_bg_pred = y_pred == background_label

    total_alarms = int((~is_bg_pred).sum())
    false_alarms = int((is_bg_true & ~is_bg_pred).sum())
    total_events = int((~is_bg_true).sum())
    missed = int((~is_bg_true & is_bg_pred).sum())
    return {
        "NAR": false_alarms / total_alarms if total_alarms else float("nan"),
        "FNR": missed / total_events if total_events else float("nan"),
        "n_alarms": total_alarms, "n_events": total_events,
    }
