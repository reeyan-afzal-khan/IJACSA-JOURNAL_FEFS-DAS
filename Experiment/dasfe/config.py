"""Global configuration for the two-dataset DAS event-classification study.

Every notebook imports from here so that paths, seeds, window geometry and
split policy are defined in exactly one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan")
DATASET_DIR = PROJECT_ROOT / "Dataset"
EXPERIMENT_DIR = PROJECT_ROOT / "Experiment"
RESULTS_DIR = PROJECT_ROOT / "Results"

CAO_DIR = DATASET_DIR / "Cao_2023"
TOMASOV_DIR = DATASET_DIR / "Tomasov_2024" / "data"

# Results are organised per dataset so the two benchmarks never overwrite
# each other.
def results_dir(dataset: str, *parts: str) -> Path:
    p = RESULTS_DIR / dataset
    for part in parts:
        p = p / part
    p.mkdir(parents=True, exist_ok=True)
    return p


STAGES = (
    "00_inventory",
    "01_splits",
    "02_windows",
    "03_features",
    "04_fusion",
    "05_selection",
    "06_benchmark",
    "07_final",
    "08_crossdataset",
    "09_report",
)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
N_JOBS = max(1, (os.cpu_count() or 4) - 2)

# --------------------------------------------------------------------------
# Signal / window geometry
# --------------------------------------------------------------------------
WIN_LEN = 8192          # samples per analysis window (paper: T = 8192)
WIN_HOP = 2048          # hop between windows (paper: shift = 2048)
BP_LOW = 20.0           # Butterworth band-pass low cut  [Hz]
BP_HIGH = 2000.0        # Butterworth band-pass high cut [Hz]
BP_ORDER = 4

# STFT / wavelet settings used by the time-frequency extractor.
# The original submission used nfft=4096 / hop=2048, which yields only ~5 STFT
# frames per 8192-sample window - too thin for the GLCM/LBP/HOG texture
# descriptors computed on the spectrogram.  We default to a denser STFT
# (257 x 61 image) and keep the original values available for an ablation.
NFFT_STFT = 512
HOP_STFT = 128
PAPER_NFFT_STFT = 4096
PAPER_HOP_STFT = 2048
WAVELET = "db4"
DWT_LEVELS = 6          # levels 0..6 inclusive -> 7 coefficient sets
WPT_LEVEL = 4           # 16 sub-bands
SPEC_QUANT_LEVELS = 32  # grey levels for GLCM / LBP on the log-spectrogram

# --------------------------------------------------------------------------
# Per-dataset acquisition parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Everything that differs between the two DAS systems."""

    name: str
    fs: float                     # sampling / pulse rate [Hz]
    n_channels: int               # channels available per sample or patch
    patch_channels: int           # channels kept for the spatial descriptors
    win_len: int
    win_hop: int
    classes: tuple
    group_key: str                # column in the manifest used as split group
    notes: str = ""
    fs_alternatives: tuple = field(default_factory=tuple)


# Tomasov et al. 2025 (Scientific Data 12:793) - OptaSense ODH-F, phase-based.
# PulseRate is read straight from the HDF5 attribute Acquisition/PulseRate.
TOMASOV = DatasetSpec(
    name="tomasov",
    fs=20000.0,
    n_channels=1700,
    patch_channels=32,
    win_len=WIN_LEN,
    win_hop=WIN_HOP,
    classes=(
        "car", "construction", "fence", "longboard", "manipulation",
        "openclose", "regular", "running", "walk",
    ),
    group_key="zone_id",
    notes="Phase-sensitive OTDR, 1700 loci @ 1.021 m, 20 kHz pulse rate.",
)

# Cao et al. 2023 (Results in Optics 10:100372) - intensity-based Phi-OTDR.
# Each .mat holds one pre-segmented sample of 10000 time points x 12 channels.
# The paper reports 0.8 s for the 5.1 km fibre (fs = 12500 Hz) and 1.25 s for
# the 10.1 km fibre (fs = 8000 Hz), but the release carries no per-file marker
# telling the two campaigns apart.  We therefore use a nominal fs and run an
# explicit sensitivity check over `fs_alternatives` in notebook 08.
CAO = DatasetSpec(
    name="cao",
    fs=10000.0,
    n_channels=12,
    patch_channels=12,
    win_len=8192,           # centre crop of the 10000-sample record
    win_hop=8192,           # one window per .mat file: no overlap by construction
    classes=("background", "dig", "knock", "water", "shake", "walk"),
    group_key="session_id",
    notes=(
        "Intensity-based Phi-OTDR, 12 adjacent loci @ 10 m, samples are "
        "pre-segmented 10000x12 .mat records."
    ),
    fs_alternatives=(8000.0, 12500.0),
)

DATASETS = {"tomasov": TOMASOV, "cao": CAO}

CAO_CLASS_DIRS = {
    "01_background": "background",
    "02_dig": "dig",
    "03_knock": "knock",
    "04_water": "water",
    "05_shake": "shake",
    "06_walk": "walk",
}

# --------------------------------------------------------------------------
# Leakage-control policy
# --------------------------------------------------------------------------
# Tomasov: each recording's timeline is cut into N_SEGMENTS contiguous
# equal-*mass* segments (one segment ~ 1/N of that recording's windows).  Whole
# segments are assigned to splits, and windows falling within GUARD_BLOCKS of a
# boundary with a differently-assigned segment are dropped, so no two windows in
# different splits can share a single raw sample.
#
# Segments are cut by window mass rather than by group count because a zone of
# fixed duration can hold anything from 30 to 8,000 annotated windows depending
# on how many loci the event touches - balancing group counts would give 92/5/2
# splits and leave sparse classes with no test data at all.
N_SEGMENTS = 10                        # -> 8 train / 1 val / 1 test per recording
ZONE_BLOCKS = 21                       # coarse zone id, kept for reporting only
GUARD_BLOCKS = WIN_LEN // WIN_HOP      # = 4 -> one full window length

# Cao: the split group is the recording session parsed from the filename
# (date_operator_event_session).  LODO = leave-one-date-out robustness split.
SPLIT_FRACTIONS = (0.8, 0.1, 0.1)      # train / val / test

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
BALANCING_STRATEGIES = (
    "none", "class_weight", "undersample",
    "smote", "borderline_smote", "smote_tomek",
)

MODEL_NAMES = (
    "LightGBM", "XGBoost", "HistGradientBoosting", "RandomForest",
    "ExtraTrees", "LogisticRegression", "DecisionTree",
)

FS_METHODS = (
    "variance", "pearson", "mutual_info", "relieff",
    "mrmr", "rfe", "boruta", "shap",
)

CONSENSUS_THRESHOLD = 8   # feature must be picked by all 8 methods
FS_SUBSAMPLE = 20000      # stratified subsample for the expensive selectors
MRMR_K = 400              # mRMR is greedy: rank this many, score the rest 0
CV_FOLDS = 5

# Reduced budgets used when the whole ensemble is re-fitted inside every CV fold.
# The methods are identical; only the subsample sizes shrink, which keeps the
# in-fold refit tractable (the honest CV estimate costs n_folds x the ensemble).
FS_FAST_KW = dict(
    mutual_info=dict(n_max=4000),
    relieff=dict(n_max=1500),
    mrmr=dict(n_max=4000, mrmr_k=120),
    rfe=dict(n_max=4000, step=0.25),
    boruta=dict(n_max=2500, max_iter=20),
    shap=dict(n_max=2000),
)

# Shared hyper-parameters (identical to the original submission so the two
# studies stay comparable).
BOOSTING_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=15,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)
LGBM_NUM_LEAVES = 64
