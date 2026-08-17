CELLS = [
    ("md", """
# 05 - Ensemble Feature Selection

Eight selectors spanning filter, wrapper, embedded and explainability families -
Variance, one-vs-rest Pearson, Mutual Information, ReliefF, mRMR, RFE, Boruta and
SHAP - are combined by consensus voting.

## The leakage rule this notebook enforces

> **Selectors never see validation or test rows.**

Two levels of protection:

1. The **reported subset** is fitted on the training split only, then applied unchanged
   to val and test.
2. The **CV estimate** re-fits the entire ensemble inside every fold
   (`selector` callback in `grouped_cv`). The submitted version fitted selection once on
   the full training set and then cross-validated on that same set, which inflates the
   CV number even though the test set stayed clean. Section 4 quantifies the gap.
"""),
    ("code", """
import sys, json, time
from pathlib import Path
EXPERIMENT_DIR = Path(r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan/Experiment")
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dasfe import config as C, fusion as F, selection as FS, evaluate as EV

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
DATASET    = "cao"
SMOKE_TEST = True
TOP_FRAC   = 0.50      # fraction of features each method may keep
# ------------------------------------------------------------------------

suffix   = "_smoke" if SMOKE_TEST else ""
fuse_dir = C.results_dir(DATASET, "04_fusion" + suffix)
sel_dir  = C.results_dir(DATASET, "05_selection" + suffix)

data  = F.load(fuse_dir)
X, y  = data["X"], data["y"]
names = data["feature_names"]
mani  = data["manifest"]
split = mani["split"].to_numpy()
group_key = C.DATASETS[DATASET].group_key
groups = mani[group_key if group_key in mani.columns else "group_id"].to_numpy()

tr = split == "train"
print(f"{DATASET}: X {X.shape}")
print(f"train {tr.sum():,} | val {(split=='val').sum():,} | test {(split=='test').sum():,}")
print(f"split groups: {pd.Series(groups).nunique()}")
"""),
    ("md", """
## 1. Fit all eight selectors on the training split

`variance` and `pearson` are cheap and run on the full training split. `mutual_info`,
`relieff`, `mrmr`, `rfe`, `boruta` and `shap` use a class-proportional subsample for
tractability - the subsample is drawn from the training split only.
"""),
    ("code", """
t0 = time.perf_counter()
result = FS.fit_all(X[tr], y[tr], names, top_frac=TOP_FRAC, seed=C.SEED, verbose=True)
print(f"\\nensemble selection: {time.perf_counter()-t0:.1f} s "
      f"({result['n_methods']}/8 methods succeeded, each keeping {result['top_k']} features)")

timings = pd.DataFrame([{"method": k, "seconds": round(v, 2)}
                        for k, v in result["timings"].items()]).sort_values("seconds", ascending=False)
display(timings)
"""),
    ("md", """
## 2. Consensus subsets

A feature retained by all eight methods has support across fundamentally different
criteria - linear correlation, information content, neighbourhood separability,
redundancy penalty, tree importance and model attribution.
"""),
    ("code", """
vote_hist = pd.Series(result["votes"]).value_counts().sort_index(ascending=False)
vote_table = pd.DataFrame({
    "votes": vote_hist.index,
    "n_features": vote_hist.values,
    "cumulative": vote_hist.values.cumsum(),
})
vote_table["reduction_pct"] = (100 * (1 - vote_table.cumulative / X.shape[1])).round(2)
display(vote_table)

for thr in range(result["n_methods"], 0, -1):
    n = int((result["votes"] >= thr).sum())
    print(f"  >= {thr}/8 methods: {n:5d} features  ({100*n/X.shape[1]:5.1f}% kept)")
"""),
    ("code", """
mask_consensus = FS.consensus_mask(result)             # unanimous 8/8
selected = [n for n, m in zip(names, mask_consensus) if m]
print(f"unanimous consensus subset: {mask_consensus.sum()} / {X.shape[1]} features "
      f"({100*(1-mask_consensus.sum()/X.shape[1]):.2f}% reduction)")

comp = F.domain_composition(selected)
display(comp)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].bar(vote_table.votes.astype(str), vote_table.n_features, color="#4C72B0")
axes[0].set_xlabel("number of methods selecting the feature")
axes[0].set_ylabel("features"); axes[0].set_title("consensus vote distribution")

axes[1].bar(comp.domain, comp.n_selected, color="#55A868", label="selected")
axes[1].bar(comp.domain, comp.n_available - comp.n_selected, bottom=comp.n_selected,
            color="#DDDDDD", label="dropped")
axes[1].set_title(f"domain composition of the {mask_consensus.sum()}-feature consensus subset")
axes[1].legend()
for i, r in comp.iterrows():
    axes[1].text(i, r.n_available + 8, f"{r.retention_pct:.0f}%", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(sel_dir / "consensus_votes_and_domains.png", dpi=200)
plt.show()
"""),
    ("md", """
## 3. Method agreement (Jaccard) and top-ranked features
"""),
    ("code", """
J = FS.jaccard_matrix(result)
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(J, annot=True, fmt=".2f", cmap="Reds", vmin=0, vmax=1, ax=ax,
            cbar_kws={"label": "Jaccard index"})
ax.set_title(f"{DATASET} - agreement between feature-selection methods")
plt.tight_layout()
plt.savefig(sel_dir / "jaccard_agreement.png", dpi=200)
plt.show()

off = J.where(~np.eye(len(J), dtype=bool))
print(f"mean pairwise agreement: {off.stack().mean():.3f}")
print(f"strongest pair : {off.stack().idxmax()} = {off.stack().max():.3f}")
print(f"weakest pair   : {off.stack().idxmin()} = {off.stack().min():.3f}")
"""),
    ("code", """
top20 = result["table"].head(20)[["feature", "votes", "selection_pct", "avg_rank"]]
top20 = top20.copy()
top20["domain"] = top20.feature.str.split(":").str[0]
display(top20.round(2))

fig, ax = plt.subplots(figsize=(9, 6))
colors = {"TIME": "#4C72B0", "FREQ": "#DD8452", "TF": "#55A868", "SPAT": "#C44E52"}
ax.barh(range(len(top20)), top20.selection_pct,
        color=[colors.get(d, "grey") for d in top20.domain])
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.feature, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("selection rate across methods (%)")
ax.set_title(f"{DATASET} - top 20 features by consolidated ranking")
handles = [plt.Rectangle((0,0),1,1, color=c) for c in colors.values()]
ax.legend(handles, colors.keys(), loc="lower right")
plt.tight_layout()
plt.savefig(sel_dir / "top20_features.png", dpi=200)
plt.show()
"""),
    ("md", """
## 4. How much does fitting selection outside the CV loop inflate the estimate?

Both configurations use the identical model and the identical grouped folds. The only
difference is *when* selection happens.
"""),
    ("code", """
MODEL, STRATEGY = "LightGBM", "class_weight"
BIAS_FOLDS = 3          # the in-fold refit runs the whole ensemble BIAS_FOLDS times
X_tr, y_tr, g_tr = X[tr], y[tr], groups[tr]

# (a) leaky: selection fitted once on the whole training split, then cross-validated
cv_leaky = EV.grouped_cv(X_tr[:, mask_consensus], y_tr, g_tr, MODEL, STRATEGY,
                         n_splits=BIAS_FOLDS, seed=C.SEED)

# (b) honest: the whole 8-method ensemble re-fitted inside every fold.
# `fast=True` shrinks the subsample budgets only - the same 8 methods run.
def fold_selector(Xf, yf):
    r = FS.fit_all(Xf, yf, names, top_frac=TOP_FRAC, seed=C.SEED,
                   verbose=False, fast=True)
    m = FS.consensus_mask(r)
    return m if m.sum() >= 10 else (r["votes"] >= max(1, r["n_methods"] - 1))

t0 = time.perf_counter()
cv_honest = EV.grouped_cv(X_tr, y_tr, g_tr, MODEL, STRATEGY,
                          n_splits=BIAS_FOLDS, seed=C.SEED, selector=fold_selector)
print(f"in-fold ensemble refit x{BIAS_FOLDS}: {time.perf_counter()-t0:.0f}s")

m_l, lo_l, hi_l = EV.ci95(cv_leaky.f1_macro)
m_h, lo_h, hi_h = EV.ci95(cv_honest.f1_macro)
bias = pd.DataFrame([
    {"protocol": "selection fitted once on full train (submitted)",
     "cv_f1_macro": round(100*m_l, 2), "ci95_low": round(100*lo_l, 2), "ci95_high": round(100*hi_l, 2)},
    {"protocol": "selection re-fitted inside each fold (revised)",
     "cv_f1_macro": round(100*m_h, 2), "ci95_low": round(100*lo_h, 2), "ci95_high": round(100*hi_h, 2)},
])
bias.loc[len(bias)] = ["optimism (pp)", round(100*(m_l - m_h), 2), "", ""]
display(bias)
bias.to_csv(sel_dir / "selection_bias_analysis.csv", index=False)
"""),
    ("md", """
## 5. Dimensionality-performance trade-off

How far can the feature space be cut before accuracy falls off? Subsets are taken from
the consolidated consensus ranking (training data only) and evaluated on validation.
"""),
    ("code", """
order = np.argsort(-result["table"].set_index("feature").loc[names, "votes"].to_numpy()
                   + result["table"].set_index("feature").loc[names, "avg_rank"].to_numpy() / 1e6)
K_GRID = [k for k in (10, 20, 40, 80, 120, 160, 200, 271, 400, 600, 800, X.shape[1])
          if k <= X.shape[1]]

curve = []
va = split == "val"
for k in K_GRID:
    cols = order[:k]
    r = EV.fit_predict(MODEL, STRATEGY, X[tr][:, cols], y[tr], X[va][:, cols], y[va], C.SEED)
    curve.append({"k": k, **{m: r["metrics"][m] for m in
                             ("accuracy", "balanced_accuracy", "f1_macro")},
                  "train_seconds": r["metrics"]["train_seconds"]})
curve = pd.DataFrame(curve)
display(curve.round(4))
"""),
    ("code", """
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(curve.k, 100*curve.f1_macro, "o-", label="macro F1", color="#4C72B0")
ax.plot(curve.k, 100*curve.accuracy, "s--", label="accuracy", color="#DD8452")
ax.axvline(int(mask_consensus.sum()), color="k", ls=":",
           label=f"8/8 consensus ({int(mask_consensus.sum())})")
ax.set_xscale("log"); ax.set_xlabel("number of features (log)"); ax.set_ylabel("validation %")
ax.set_title(f"{DATASET} - feature-reduction curve")
ax.legend(); plt.tight_layout()
plt.savefig(sel_dir / "feature_reduction_curve.png", dpi=200)
plt.show()

best = curve.loc[curve.f1_macro.idxmax()]
full = curve.loc[curve.k.idxmax()]
print(f"best   : k={int(best.k):4d}  F1={100*best.f1_macro:.2f}%")
print(f"full   : k={int(full.k):4d}  F1={100*full.f1_macro:.2f}%")
print(f"consensus subset keeps {100*mask_consensus.sum()/X.shape[1]:.1f}% of the features")
"""),
    ("md", """
## 6. Save the selection artefacts

`fs_masks_multi.npz` holds one boolean mask per method; downstream notebooks load it and
never re-run selection, so the reported subsets are frozen.
"""),
    ("code", """
FS.save(result, sel_dir, tag="multi")
np.save(sel_dir / "mask_consensus.npy", mask_consensus)
np.save(sel_dir / "rank_order.npy", order)
curve.to_csv(sel_dir / "reduction_curve.csv", index=False)
comp.to_csv(sel_dir / "domain_composition.csv", index=False)
timings.to_csv(sel_dir / "method_timings.csv", index=False)
json.dump({"selected_features": selected}, open(sel_dir / "consensus_features.json", "w"), indent=1)

print("wrote", sel_dir)
for p in sorted(sel_dir.iterdir()):
    print("  ", p.name)
"""),
    ("md", """
---
**Next:** `06_balancing_and_model_benchmark.ipynb` - seven classifiers x six balancing
strategies, with every resampler confined inside the training fold.
"""),
]
