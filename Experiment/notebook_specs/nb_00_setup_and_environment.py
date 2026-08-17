CELLS = [
    ("md", """
# 00 - Setup and Environment

Establishes the runtime, verifies every dependency the pipeline needs, creates the
`Results/` tree, and records the exact software versions for the paper's
reproducibility table.

**Run this notebook first.** Every later notebook assumes `Experiment/` is importable
and that `Results/<dataset>/<stage>/` exists.

### Study design in one paragraph

The original submission benchmarked a multi-domain handcrafted feature pipeline
(1,002 features across time / frequency / time-frequency / spatial) with lightweight
classifiers, on a single DAS dataset, using a random stratified 80/10/10 split.
This revision keeps the methodology and changes three things: **two independent
datasets** instead of one, **group-aware splitting** that makes leakage structurally
impossible, and **feature selection re-fitted inside every cross-validation fold**
rather than once on the full training set.
"""),
    ("code", """
import sys, platform, importlib, json, os
from pathlib import Path

EXPERIMENT_DIR = Path(r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan/Experiment")
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import dasfe
from dasfe import config as C

print("python  :", platform.python_version())
print("platform:", platform.platform())
print("cpu cores:", os.cpu_count(), "-> n_jobs =", C.N_JOBS)
print("dasfe   :", dasfe.__version__)
"""),
    ("md", """
## 1. Dependency check

Anything reported as MISSING here must be installed before continuing - the pipeline
has no fallbacks, because a silently-skipped selector would change the consensus
subset without warning.
"""),
    ("code", """
REQUIRED = [
    "numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn",
    "h5py", "pywt", "skimage", "joblib", "tqdm", "pyarrow",
    "lightgbm", "xgboost", "imblearn", "shap", "boruta", "skrebate", "mrmr",
]
OPTIONAL = ["torch"]

rows = []
for name in REQUIRED + OPTIONAL:
    try:
        m = importlib.import_module(name)
        rows.append((name, getattr(m, "__version__", "n/a"), "ok"))
    except ImportError as exc:
        rows.append((name, "-", f"MISSING ({exc.name})"))

import pandas as pd
env = pd.DataFrame(rows, columns=["package", "version", "status"])
missing = env[env.status.str.startswith("MISSING")]
display(env)
if len(missing):
    print("\\nInstall the missing packages, e.g.:")
    print("  pip install " + " ".join(missing.package.replace(
        {"sklearn": "scikit-learn", "skimage": "scikit-image",
         "pywt": "PyWavelets", "imblearn": "imbalanced-learn",
         "mrmr": "mrmr-selection", "boruta": "Boruta"})))
else:
    print("\\nAll required packages present.")
"""),
    ("md", """
## 2. Dataset presence

We only check that the two dataset roots exist and are populated. No bulk data is
read here - notebook 01 builds the manifests.
"""),
    ("code", """
print("Cao_2023    :", C.CAO_DIR, "->", "OK" if C.CAO_DIR.is_dir() else "NOT FOUND")
for split in ("Training", "Test"):
    d = C.CAO_DIR / split
    n = sum(1 for _ in d.rglob("*.mat")) if d.is_dir() else 0
    print(f"   {split:9s} {n:6d} .mat files")

print("\\nTomasov_2024:", C.TOMASOV_DIR, "->", "OK" if C.TOMASOV_DIR.is_dir() else "NOT FOUND")
if C.TOMASOV_DIR.is_dir():
    total_gb = 0
    for cls in sorted(p for p in C.TOMASOV_DIR.iterdir() if p.is_dir()):
        h5 = list(cls.glob("*.h5"))
        gb = sum(p.stat().st_size for p in h5) / 1e9
        total_gb += gb
        print(f"   {cls.name:14s} {len(h5)} recording(s)  {gb:6.1f} GB")
    print(f"   {'TOTAL':14s} {total_gb:.1f} GB")
"""),
    ("md", """
## 3. Results tree

One directory per dataset, one sub-directory per pipeline stage. Every notebook
writes only into its own stage directory, so a stage can be deleted and re-run
without disturbing anything upstream or downstream.
"""),
    ("code", """
for ds in ("cao", "tomasov"):
    for stage in C.STAGES:
        C.results_dir(ds, stage)
C.results_dir("shared", "figures")
C.results_dir("shared", "tables")

for p in sorted(C.RESULTS_DIR.rglob("*")):
    if p.is_dir():
        print(p.relative_to(C.RESULTS_DIR))
"""),
    ("md", """
## 4. Feature-space contract

The four extractors expose a fixed dimensionality that is asserted at import time.
If any of these numbers drift, `import dasfe` fails immediately rather than
producing a silently different feature space.
"""),
    ("code", """
import pandas as pd
counts = pd.DataFrame(
    [{"domain": d, "n_features": n} for d, n in dasfe.FEATURE_COUNTS.items()]
)
counts.loc[len(counts)] = ["MULTI-DOMAIN (fused)", dasfe.TOTAL_FEATURES]
display(counts)

print("window length :", C.WIN_LEN, "samples")
print("window hop    :", C.WIN_HOP, "samples  ->", 100 * (1 - C.WIN_HOP / C.WIN_LEN), "% overlap")
print("band-pass     :", C.BP_LOW, "-", C.BP_HIGH, "Hz, order", C.BP_ORDER)
print("STFT          : nfft", C.NFFT_STFT, "hop", C.HOP_STFT)
print("wavelet       :", C.WAVELET, "| DWT levels", C.DWT_LEVELS, "| WPT level", C.WPT_LEVEL)
"""),
    ("md", """
### Why the window overlap matters

`WIN_HOP < WIN_LEN` means consecutive windows share 75% of their raw samples. That
is fine for *extraction* - it is how the annotation masks are laid out - but it makes
a random train/test split invalid, because a test window can share three quarters of
its samples with a training window. Notebook 02 handles this with guard bands.
"""),
    ("code", """
import numpy as np, time
from dasfe import preprocess as pp, extract as ex

rng = np.random.default_rng(C.SEED)
patch = pp.preprocess_patch(rng.standard_normal((32, C.WIN_LEN)), C.TOMASOV.fs)
x1d = patch[16]

t0 = time.perf_counter()
feats = ex.extract_window(x1d, patch, C.TOMASOV.fs)
dt = time.perf_counter() - t0

for d, v in feats.items():
    print(f"  {d:8s} {v.shape[0]:4d} features   finite={np.isfinite(v).all()}")
print(f"\\nall four domains, one window: {1000*dt:.1f} ms (single core)")
"""),
    ("md", """
## 5. Environment record

Saved to `Results/shared/tables/environment.json` and reproduced verbatim in the
paper's specifications table.
"""),
    ("code", """
record = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "cpu_count": os.cpu_count(),
    "n_jobs": C.N_JOBS,
    "seed": C.SEED,
    "dasfe_version": dasfe.__version__,
    "packages": {r[0]: r[1] for r in rows if r[2] == "ok"},
    "feature_counts": dasfe.FEATURE_COUNTS,
    "window": {"len": C.WIN_LEN, "hop": C.WIN_HOP,
               "bp_low": C.BP_LOW, "bp_high": C.BP_HIGH, "bp_order": C.BP_ORDER},
}
out = C.results_dir("shared", "tables") / "environment.json"
out.write_text(json.dumps(record, indent=2), encoding="utf-8")
print("wrote", out)
"""),
    ("md", """
---
**Next:** `01_dataset_inventory_and_leakage_audit.ipynb` - build the manifests for both
datasets and quantify the leakage present in the published Cao train/test split.
"""),
]
