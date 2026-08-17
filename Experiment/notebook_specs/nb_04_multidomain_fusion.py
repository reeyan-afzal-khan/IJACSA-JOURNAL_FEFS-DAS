CELLS = [
    ("md", """
# 04 - Multi-Domain Feature Fusion

Concatenates the four domain matrices into the 1,002-dimensional multi-domain
representation (Algorithm 1 of the manuscript), prefixing every feature name with its
domain so it stays traceable through selection and SHAP attribution.

Run once per dataset by setting `DATASET`.

Every consistency check the algorithm specifies is **executed**, not assumed: equal row
counts across domains, identical `sample_id` ordering, and an exact join against the
frozen split manifest.
"""),
    ("code", """
import sys, json
from pathlib import Path
EXPERIMENT_DIR = Path(r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan/Experiment")
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dasfe import config as C, fusion as F, extract as EX

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
DATASET    = "cao"        # "cao" or "tomasov"
SMOKE_TEST = True         # must match notebook 03
# ------------------------------------------------------------------------

feat_dir  = C.results_dir(DATASET, "03_features" + ("_smoke" if SMOKE_TEST else ""))
fuse_dir  = C.results_dir(DATASET, "04_fusion"   + ("_smoke" if SMOKE_TEST else ""))
split_dir = C.results_dir(DATASET, "01_splits")

manifest = pd.read_parquet(split_dir / "split_manifest.parquet")
print(f"dataset : {DATASET}\\nfeatures: {feat_dir}\\noutput  : {fuse_dir}")
"""),
    ("md", """
## 1. Per-domain matrices
"""),
    ("code", """
rows = []
for d in EX.DOMAIN_ORDER:
    X, names = F.load_domain(feat_dir, d)
    rows.append({"domain": d, "prefix": F.PREFIX[d], "n_windows": X.shape[0],
                 "n_features": X.shape[1], "example_feature": names[0],
                 "MB": round(X.nbytes / 1e6, 1)})
domains = pd.DataFrame(rows)
domains.loc[len(domains)] = ["MULTI", "-", domains.n_windows.iloc[0],
                             domains.n_features.sum(), "-", domains.MB.sum()]
display(domains)

assert domains.n_windows.iloc[:-1].nunique() == 1, "domains disagree on the number of windows"
assert domains.n_features.iloc[-1] == 1002, "fused dimensionality is not 1002"
print("\\nPASS: all four domains aligned; fused dimensionality = 1002")
"""),
    ("md", """
## 2. Fuse and join to the split manifest
"""),
    ("code", """
fused = F.fuse(feat_dir, manifest)
X, y = fused["X"], fused["y"]
mani = fused["manifest"]
split = mani["split"].to_numpy()

print(f"X {X.shape}   y {y.shape}")
print("\\nsplit sizes:")
print(pd.Series(split).value_counts().to_string())
print("\\nclass x split:")
display(pd.crosstab(mani.label, mani.split, margins=True))

# Order integrity: the manifest row i must describe feature row i.
assert (mani.sample_id.to_numpy() == fused["sample_id"]).all(), "manifest/feature misalignment"
print("\\nPASS: manifest rows are aligned 1:1 with feature rows.")
"""),
    ("code", """
print("domain slices in the fused matrix:")
for d, sl in fused["domain_slices"].items():
    print(f"  {d:8s} columns {sl.start:4d}..{sl.stop-1:4d}")

print("\\nfirst feature of each domain:")
for d, sl in fused["domain_slices"].items():
    print(f"  {fused['feature_names'][sl.start]}")
"""),
    ("md", """
## 3. Distributional sanity

The fused matrix mixes descriptors whose natural scales differ by many orders of
magnitude (probabilities near 0-1 alongside raw spectral energies). This is harmless for
tree ensembles, and the scale-sensitive models get a `StandardScaler` **inside** their
pipeline, fitted on training folds only.
"""),
    ("code", """
scales = pd.DataFrame({
    "feature": fused["feature_names"],
    "domain": sum([[d] * (sl.stop - sl.start) for d, sl in fused["domain_slices"].items()], []),
    "std": X.std(axis=0),
    "abs_max": np.abs(X).max(axis=0),
})
display(scales.groupby("domain")["std"].describe().round(4))

const = scales[scales["std"] == 0]
print(f"\\nconstant features: {len(const)} / {X.shape[1]}")
if len(const):
    display(const.head(15)[["feature", "domain"]])
print("\\nThese are removed by the variance threshold in notebook 05. Several are constant")
print("by construction: per-window z-scoring fixes mean=0 and std=1.")
"""),
    ("code", """
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

sns.boxplot(data=scales, x="domain", y="std", ax=axes[0], showfliers=False)
axes[0].set_yscale("symlog"); axes[0].set_title("feature standard deviation by domain")

corr_sample = np.corrcoef(X[np.random.default_rng(C.SEED).choice(
    len(X), size=min(3000, len(X)), replace=False)].T)
corr_sample = np.nan_to_num(corr_sample)
tri = np.abs(corr_sample[np.triu_indices_from(corr_sample, k=1)])
axes[1].hist(tri, bins=60, color="#C44E52")
axes[1].set_title("pairwise |correlation| between fused features")
axes[1].set_xlabel("|r|")
axes[1].axvline(0.95, color="k", ls="--", label=f"|r|>0.95: {100*(tri>0.95).mean():.1f}%")
axes[1].legend()

plt.tight_layout()
plt.savefig(fuse_dir / "feature_scales_and_redundancy.png", dpi=200)
plt.show()

print(f"redundancy: {100*(tri > 0.95).mean():.2f}% of feature pairs have |r| > 0.95")
print("-> this is exactly the redundancy the ensemble feature selection targets.")
"""),
    ("md", """
## 4. Save

`X_multi.npy`, `y.npy`, `feature_names.json` and `manifest_aligned.parquet` are the only
inputs every downstream notebook needs.
"""),
    ("code", """
F.save(fused, fuse_dir)
scales.to_csv(fuse_dir / "feature_scales.csv", index=False)

check = F.load(fuse_dir)
assert check["X"].shape == X.shape and (check["y"] == y).all()
print("wrote and verified:", fuse_dir)
for p in sorted(fuse_dir.iterdir()):
    print(f"  {p.name:28s} {p.stat().st_size/1e6:8.2f} MB")
"""),
    ("md", """
---
**Next:** `05_feature_selection.ipynb` - the 8-method ensemble, fitted on training data
only and re-fitted inside every CV fold.
"""),
]
