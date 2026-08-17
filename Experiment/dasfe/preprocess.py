"""Window loading and the shared preprocessing chain.

Both datasets are reduced to the same two objects before any feature is
computed, so a single extractor serves both systems:

* `x`     - 1D preprocessed strain/intensity window, shape (win_len,)
* `patch` - 2D preprocessed space-time patch, shape (n_channels, win_len)

Preprocessing follows the original submission: DC removal, 4th-order
Butterworth band-pass, z-score.  It is stateless and per-window, so it cannot
leak information between samples or between splits.
"""
from __future__ import annotations

import numpy as np
import scipy.io as scio
from scipy.signal import butter, sosfiltfilt

from . import config as C

_SOS_CACHE: dict = {}


def _sos(fs: float, low: float = C.BP_LOW, high: float = C.BP_HIGH, order: int = C.BP_ORDER):
    key = (fs, low, high, order)
    if key not in _SOS_CACHE:
        nyq = fs / 2.0
        hi = min(high, 0.99 * nyq)
        _SOS_CACHE[key] = butter(order, [low / nyq, hi / nyq], btype="band", output="sos")
    return _SOS_CACHE[key]


def preprocess_1d(x: np.ndarray, fs: float) -> np.ndarray:
    """Demean -> band-pass -> z-score.  Equation (1) of the paper."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x - x.mean()
    if x.size > 3 * (C.BP_ORDER * 2 + 1):
        x = sosfiltfilt(_sos(fs), x)
    sd = x.std()
    return (x - x.mean()) / (sd + 1e-12)


def preprocess_patch(patch: np.ndarray, fs: float) -> np.ndarray:
    """Apply the same chain channel-wise to a (n_channels, win_len) patch."""
    patch = np.asarray(patch, dtype=np.float64)
    patch = patch - patch.mean(axis=1, keepdims=True)
    if patch.shape[1] > 3 * (C.BP_ORDER * 2 + 1):
        patch = sosfiltfilt(_sos(fs), patch, axis=1)
    sd = patch.std(axis=1, keepdims=True)
    return (patch - patch.mean(axis=1, keepdims=True)) / (sd + 1e-12)


# ==========================================================================
# Cao loader
# ==========================================================================
def load_cao_window(path: str, win_len: int = C.CAO.win_len) -> np.ndarray:
    """Return the raw (n_channels, win_len) patch from one .mat record.

    The release stores each sample as uint16 (10000, 12).  We transpose to
    channel-major and centre-crop to `win_len` so that every window in the
    study has identical length regardless of dataset.
    """
    raw = scio.loadmat(path)["data"]            # (10000, 12) uint16
    patch = np.asarray(raw, dtype=np.float64).T  # (12, 10000)
    n = patch.shape[1]
    if n > win_len:
        start = (n - win_len) // 2
        patch = patch[:, start:start + win_len]
    elif n < win_len:
        patch = np.pad(patch, ((0, 0), (0, win_len - n)), mode="reflect")
    return patch


def cao_reference_channel(patch: np.ndarray) -> int:
    """Index of the highest-energy channel - the disturbance location.

    Cao et al. clipped each record around the disturbance ("mostly at the
    centre"), so the strongest channel is the physically meaningful one to feed
    the single-channel descriptors.
    """
    energy = ((patch - patch.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return int(np.argmax(energy))


# ==========================================================================
# Tomasov loader
# ==========================================================================
class TomasovReader:
    """Keeps one HDF5 file open while all of its windows are extracted.

    Opening a 3 GB HDF5 per window would dominate runtime, so the extraction
    notebooks sort the manifest by recording and reuse a single reader.
    """

    def __init__(self, h5_path: str):
        import h5py

        self._f = h5py.File(h5_path, "r")
        self.dset = self._f["Acquisition/Raw[0]/RawData"]     # (n_time, n_loci)
        attrs = self._f["Acquisition"].attrs
        self.fs = float(attrs["PulseRate"])
        self.n_loci = int(attrs["NumberOfLoci"])
        self.n_time = self.dset.shape[0]

    def read_patch(self, t0: int, locus: int, win_len: int, patch_channels: int) -> np.ndarray:
        """Raw (patch_channels, win_len) patch centred on `locus`."""
        half = patch_channels // 2
        c0 = int(np.clip(locus - half, 0, self.n_loci - patch_channels))
        t0 = int(np.clip(t0, 0, self.n_time - win_len))
        block = self.dset[t0:t0 + win_len, c0:c0 + patch_channels]   # (T, C)
        return np.asarray(block, dtype=np.float64).T                  # (C, T)

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def center_channel_index(patch_channels: int) -> int:
    return patch_channels // 2
