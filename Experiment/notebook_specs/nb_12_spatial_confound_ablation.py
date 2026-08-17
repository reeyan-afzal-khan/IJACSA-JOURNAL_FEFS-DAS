CELLS = [
    ("md", """
# 12 - Spatial-Domain Confound Ablation

**This notebook decides what the paper is allowed to claim.**

On Tomasov, the 61 spatial features (6% of the feature space) carry **55.3%** of the
model's importance mass, and the top 15 features are *all* spatial. In the original
submission the time-frequency domain was strongest and spatial was third. That inversion
appeared the moment the leak was removed - which is suspicious in a specific way.

## The hypothesis being tested

Five of the nine Tomasov classes come from **exactly one recording each**
(`construction`, `manipulation`, `openclose`, `regular`, `running`). For those classes
*recording identity is class identity*. Spatial descriptors summarise fibre geometry,
coupling and channel-to-channel energy structure - all of which are fixed properties of a
recording. So a model can score well by answering "which recording is this?" instead of
"what event is this?".

The blocked split cannot prevent that: it splits *within* each recording, so the
recording is present in train, val and test by construction.

## Four experiments, in order of how directly they test the hypothesis

1. **Leave-one-domain-out** - how much does each domain contribute?
2. **Recording-identity probe** - *the decisive test*. Fix the class, then ask the
   features to predict **which recording** a window came from. A domain that does this
   well is a recording fingerprint, whatever else it may also encode.
3. **Positional leakage check** - do spatial features correlate with the absolute locus
   index rather than with the event?
4. **Leave-one-recording-out, with and without spatial** - if spatial features are a
   fingerprint, removing them should *help* cross-recording generalisation.

## Reading the outcome

| Finding | What the paper says |
|---|---|
| Dropping spatial costs little | Concern dissolves; keep the current framing |
| Dropping spatial costs a lot **and** spatial predicts recording identity | The headline must become the non-spatial number, with the confound reported |
| Dropping spatial costs a lot but spatial does **not** predict recording identity | Spatial genuinely carries event information; report it as a finding |
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
from dasfe import config as C, fusion as F, evaluate as EV

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
DATASETS   = ("tomasov", "cao")
MODEL      = "LightGBM"
STRATEGY   = "class_weight"     # fixed across all arms so only features vary
# The expensive sweeps (recording probe, LORO) are subsampled per class to keep
# runtime sane; the headline leave-one-domain-out arm uses everything.
PROBE_PER_CLASS = 4000
LORO_PER_CLASS  = 6000
# ------------------------------------------------------------------------

DOMAIN_SLICES = {"time": slice(0, 112), "freq": slice(112, 202),
                 "tf": slice(202, 941), "spatial": slice(941, 1002)}
SIGNAL_AGNOSTIC = slice(0, 941)     # time + freq + tf

def cols(*keep):
    idx = []
    for k in keep:
        sl = DOMAIN_SLICES[k]
        idx.extend(range(sl.start, sl.stop))
    return np.array(idx)

DATA = {}
for ds in DATASETS:
    d = F.load(C.results_dir(ds, "04_fusion"))
    d["split"] = d["manifest"]["split"].to_numpy()
    DATA[ds] = d
    print(f"{ds:8s} X {d['X'].shape}  classes {len(set(d['y']))}")

out_dir = C.results_dir("shared", "12_spatial_ablation")
print("output:", out_dir)
"""),
    ("md", """
## 1. Leave-one-domain-out

Same model, same balancing, same split. Only the feature columns change.
"""),
    ("code", """
ARMS = {
    "all 1002 (reference)":      list(DOMAIN_SLICES),
    "no spatial (941)":          ["time", "freq", "tf"],
    "no time-frequency (263)":   ["time", "freq", "spatial"],
    "no time (890)":             ["freq", "tf", "spatial"],
    "no frequency (912)":        ["time", "tf", "spatial"],
    "spatial ONLY (61)":         ["spatial"],
    "time-frequency ONLY (739)": ["tf"],
}

lodo = []
for ds in DATASETS:
    d = DATA[ds]
    X, y, sp = d["X"], d["y"], d["split"]
    tr, te = sp == "train", sp == "test"
    print(f"\\n=== {ds} ===")
    for name, keep in ARMS.items():
        c = cols(*keep)
        t0 = time.perf_counter()
        r = EV.fit_predict(MODEL, STRATEGY, X[tr][:, c], y[tr], X[te][:, c], y[te], C.SEED)
        m = r["metrics"]
        lodo.append({"dataset": ds, "arm": name, "n_features": len(c),
                     "accuracy": 100*m["accuracy"],
                     "balanced_accuracy": 100*m["balanced_accuracy"],
                     "f1_macro": 100*m["f1_macro"],
                     "train_seconds": round(m["train_seconds"], 1)})
        print(f"  {name:28s} {len(c):4d} feat  acc {100*m['accuracy']:6.2f}  "
              f"F1 {100*m['f1_macro']:6.2f}  ({time.perf_counter()-t0:.0f}s)", flush=True)

lodo = pd.DataFrame(lodo)
display(lodo.round(2))
lodo.to_csv(out_dir / "leave_one_domain_out.csv", index=False)
"""),
    ("code", """
# Cost of removing each domain, relative to the full feature set.
cost = []
for ds in DATASETS:
    sub = lodo[lodo.dataset == ds].set_index("arm")
    ref = sub.loc["all 1002 (reference)", "f1_macro"]
    for arm in sub.index:
        if arm.startswith("no "):
            cost.append({"dataset": ds, "removed": arm.replace("no ", "").split(" (")[0],
                         "f1_without": sub.loc[arm, "f1_macro"],
                         "f1_full": ref,
                         "cost_pp": round(ref - sub.loc[arm, "f1_macro"], 2)})
cost = pd.DataFrame(cost).sort_values(["dataset", "cost_pp"], ascending=[True, False])
display(cost)
cost.to_csv(out_dir / "domain_removal_cost.csv", index=False)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(7*len(DATASETS), 4.6))
axes = np.atleast_1d(axes)
for ax, ds in zip(axes, DATASETS):
    s = cost[cost.dataset == ds]
    ax.barh(s.removed, s.cost_pp, color=["#C44E52" if r == "spatial" else "#4C72B0"
                                          for r in s.removed])
    ax.set_xlabel("macro-F1 lost when the domain is removed (pp)")
    ax.set_title(f"{ds} - domain contribution")
    ax.invert_yaxis()
    for i, v in enumerate(s.cost_pp):
        ax.text(v, i, f" {v:+.2f}", va="center", fontsize=9)
plt.tight_layout(); plt.savefig(out_dir / "domain_removal_cost.png", dpi=200); plt.show()
"""),
    ("md", """
## 2. Recording-identity probe - the decisive test

For each class that has **two or more recordings**, hold the class fixed and ask the
features to predict *which recording* the window came from.

The class label is constant within each probe, so any accuracy above chance is the
features identifying nuisance structure - fibre state, coupling, time of day - rather
than event type. The split is the same blocked split, so this is measured under exactly
the conditions the main model enjoys.

A domain that identifies the recording almost perfectly is a fingerprint. That is the
mechanism by which a within-recording split can flatter a model.
"""),
    ("code", """
PROBE_ARMS = {"time": ["time"], "freq": ["freq"], "tf": ["tf"], "spatial": ["spatial"],
              "all": list(DOMAIN_SLICES)}

def group_column(ds, mani):
    if ds == "tomasov":
        return "recording_id" if "recording_id" in mani.columns else "stem"
    return "date"          # Cao: acquisition date is the analogous nuisance variable

probe = []
for ds in DATASETS:
    d = DATA[ds]
    mani = d["manifest"]
    gcol = group_column(ds, mani)
    groups = mani[gcol].to_numpy()
    y, sp = d["y"], d["split"]

    counts = pd.DataFrame({"label": y, "g": groups}).groupby("label").g.nunique()
    multi = counts[counts >= 2].index.tolist()
    print(f"\\n=== {ds}: classes with >=2 {gcol} values: {multi} ===")

    for cls in multi:
        m = y == cls
        idx = np.flatnonzero(m)
        if len(idx) > PROBE_PER_CLASS:
            rng = np.random.default_rng(C.SEED)
            idx = np.sort(rng.choice(idx, PROBE_PER_CLASS, replace=False))
        gy, gsp = groups[idx], sp[idx]
        tr, te = gsp == "train", gsp == "test"
        if te.sum() < 30 or len(np.unique(gy[tr])) < 2 or len(np.unique(gy[te])) < 2:
            print(f"  {cls:14s} skipped (insufficient coverage)"); continue
        chance = 100 * pd.Series(gy[te]).value_counts(normalize=True).max()

        for aname, keep in PROBE_ARMS.items():
            c = cols(*keep)
            r = EV.fit_predict(MODEL, STRATEGY, d["X"][idx][tr][:, c], gy[tr],
                               d["X"][idx][te][:, c], gy[te], C.SEED)
            probe.append({"dataset": ds, "class": cls, "domain": aname,
                          "n_groups": int(len(np.unique(gy))),
                          "recording_id_accuracy": 100*r["metrics"]["accuracy"],
                          "majority_chance": round(chance, 2),
                          "above_chance_pp": round(100*r["metrics"]["accuracy"] - chance, 2)})
        row = [p for p in probe if p["class"] == cls and p["dataset"] == ds]
        got = {p["domain"]: p["recording_id_accuracy"] for p in row}
        print(f"  {cls:14s} chance {chance:5.1f}%  " +
              "  ".join(f"{k}={v:5.1f}" for k, v in got.items()), flush=True)

probe = pd.DataFrame(probe)
display(probe.round(2))
probe.to_csv(out_dir / "recording_identity_probe.csv", index=False)
"""),
    ("code", """
if len(probe):
    summ = (probe.groupby(["dataset", "domain"])
                 .agg(mean_id_accuracy=("recording_id_accuracy", "mean"),
                      mean_above_chance=("above_chance_pp", "mean"))
                 .reset_index().sort_values(["dataset", "mean_above_chance"], ascending=[True, False]))
    display(summ.round(2))
    summ.to_csv(out_dir / "recording_probe_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7*len(DATASETS), 4.4))
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, DATASETS):
        s = summ[summ.dataset == ds]
        ax.barh(s.domain, s.mean_above_chance,
                color=["#C44E52" if d == "spatial" else "#4C72B0" for d in s.domain])
        ax.set_xlabel("recording-ID accuracy above chance (pp)")
        ax.set_title(f"{ds} - can each domain identify the recording?\\n(class held fixed)")
        ax.invert_yaxis()
        for i, v in enumerate(s.mean_above_chance):
            ax.text(v, i, f" {v:+.1f}", va="center", fontsize=9)
    plt.tight_layout(); plt.savefig(out_dir / "recording_identity_probe.png", dpi=200); plt.show()
"""),
    ("md", """
## 3. Positional leakage check

`read_patch` centres the 32-channel window on the annotated locus, so within-patch
position should carry no absolute information. The exception is loci near the fibre ends,
where the window is clipped and the event is no longer centred. This checks whether
spatial feature values track the absolute locus index.
"""),
    ("code", """
if "tomasov" in DATASETS:
    d = DATA["tomasov"]
    mani = d["manifest"]
    if "locus" in mani.columns:
        locus = mani["locus"].to_numpy().astype(float)
        names = np.array(d["feature_names"])
        sl = DOMAIN_SLICES["spatial"]
        Xs = d["X"][:, sl]
        r = np.array([abs(np.corrcoef(Xs[:, j], locus)[0, 1]) if Xs[:, j].std() > 0 else 0.0
                      for j in range(Xs.shape[1])])
        r = np.nan_to_num(r)
        top = pd.DataFrame({"feature": names[sl], "abs_corr_with_locus": r}) \
                .sort_values("abs_corr_with_locus", ascending=False)
        display(top.head(12).round(3))
        top.to_csv(out_dir / "spatial_vs_locus_correlation.csv", index=False)

        n_loci = int(mani["n_loci"].iloc[0]) if "n_loci" in mani.columns else int(locus.max()) + 1
        half = C.TOMASOV.patch_channels // 2
        n_edge = int(((locus < half) | (locus > n_loci - half)).sum())
        print(f"\\nmax |corr| with absolute locus: {r.max():.3f}")
        print(f"windows within 16 loci of a fibre end (patch not centred): "
              f"{n_edge:,} / {len(locus):,} ({100*n_edge/len(locus):.2f}%)")
        print("A high correlation would mean spatial features encode WHERE on the fibre")
        print("the event sits, which for single-recording classes is the class itself.")
"""),
    ("md", """
## 4. Leave-one-recording-out, with and without spatial

The generalisation test that matters. Restricted to classes with >= 2 recordings, so a
held-out recording still leaves training data for its class.

If spatial features are a recording fingerprint, dropping them should **improve**
cross-recording accuracy - the model loses a shortcut it could not have used anyway.
"""),
    ("code", """
loro = []
if "tomasov" in DATASETS:
    d = DATA["tomasov"]
    mani = d["manifest"]
    gcol = group_column("tomasov", mani)
    rec = mani[gcol].to_numpy()
    y = d["y"]

    counts = pd.DataFrame({"label": y, "g": rec}).groupby("label").g.nunique()
    multi = counts[counts >= 2].index.tolist()
    keep_rows = np.isin(y, multi)
    idx = np.flatnonzero(keep_rows)

    rng = np.random.default_rng(C.SEED)
    balanced = []
    for cls in multi:
        ci = idx[y[idx] == cls]
        balanced.append(rng.choice(ci, min(LORO_PER_CLASS, len(ci)), replace=False))
    idx = np.sort(np.concatenate(balanced))
    ys, recs = y[idx], rec[idx]
    print(f"LORO subset: {len(idx):,} windows, {len(multi)} classes, "
          f"{len(np.unique(recs))} recordings")

    for arm, keep in [("with spatial (1002)", list(DOMAIN_SLICES)),
                      ("without spatial (941)", ["time", "freq", "tf"])]:
        c = cols(*keep)
        accs = []
        for held in sorted(pd.unique(recs)):
            te = recs == held
            tr = ~te
            if te.sum() < 30 or len(np.unique(ys[tr])) < 2:
                continue
            r = EV.fit_predict(MODEL, STRATEGY, d["X"][idx][tr][:, c], ys[tr],
                               d["X"][idx][te][:, c], ys[te], C.SEED)
            a = 100*r["metrics"]["accuracy"]
            accs.append(a)
            loro.append({"arm": arm, "held_out": held[:36], "true_class": ys[te][0],
                         "n_test": int(te.sum()), "accuracy": a})
        print(f"  {arm:24s} mean {np.mean(accs):6.2f}%  sd {np.std(accs):5.2f}", flush=True)

loro = pd.DataFrame(loro)
if len(loro):
    display(loro.round(2))
    loro.to_csv(out_dir / "loro_with_without_spatial.csv", index=False)
    piv = loro.pivot(index="held_out", columns="arm", values="accuracy")
    piv["delta_pp"] = (piv["without spatial (941)"] - piv["with spatial (1002)"]).round(2)
    display(piv.round(2))
    print(f"\\nmean change from dropping spatial: {piv.delta_pp.mean():+.2f} pp")
    print("A POSITIVE mean means spatial features were hurting cross-recording")
    print("generalisation, i.e. acting as a within-recording shortcut.")
"""),
    ("code", """
if len(loro):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(piv))
    ax.bar(x - 0.2, piv["with spatial (1002)"], 0.4, label="with spatial", color="#C44E52")
    ax.bar(x + 0.2, piv["without spatial (941)"], 0.4, label="without spatial", color="#4C72B0")
    ax.set_xticks(x); ax.set_xticklabels(piv.index, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("leave-one-recording-out accuracy (%)")
    ax.set_title("Cross-recording generalisation, with and without the spatial domain")
    ax.legend(); plt.tight_layout()
    plt.savefig(out_dir / "loro_with_without_spatial.png", dpi=200, bbox_inches="tight")
    plt.show()
"""),
    ("md", """
## 5. Verdict

Mechanical reading of the four experiments. The thresholds are deliberate and stated so
the conclusion is not a judgement call.
"""),
    ("code", """
# The verdict must not rest on a raw mean of above-chance accuracy. Two problems
# showed up on the first run:
#
#   1. A probe with 2 groups and a 78% majority (Tomasov `walk`) is degenerate:
#      the class-weighted model predicts both groups, which scores BELOW the
#      majority baseline. That -16 pp dragged the mean down and mislabelled a
#      clearly confounded dataset as clean.
#   2. Absolute above-chance is the wrong statistic anyway. What matters is
#      whether spatial identifies the recording MORE than the other domains do.
#      On Cao every domain identifies the acquisition date (+17 to +38 pp), so a
#      high spatial value there says something about the dataset, not the domain.
#
# So: drop degenerate probes, and score the DIFFERENTIAL against the best
# non-spatial domain on the same probe.

MIN_GROUPS = 3            # 2-group probes with a lopsided majority are unstable
MAX_CHANCE = 0.70         # skip probes where the majority baseline already dominates

def probe_diagnostics(ds):
    p = probe[probe.dataset == ds]
    if not len(p):
        return {}
    wide = p.pivot_table(index="class", columns="domain",
                         values="above_chance_pp", aggfunc="first")
    meta = p.groupby("class").agg(n_groups=("n_groups", "first"),
                                  chance=("majority_chance", "first"))
    ok = (meta.n_groups >= MIN_GROUPS) & (meta.chance <= 100*MAX_CHANCE)
    kept, dropped = list(meta.index[ok]), list(meta.index[~ok])
    if not kept:
        return {"n_usable_probes": 0, "dropped_probes": dropped}

    others = [c for c in wide.columns if c not in ("spatial", "all")]
    diff = (wide.loc[kept, "spatial"] - wide.loc[kept, others].max(axis=1))
    return {
        "n_usable_probes": len(kept), "usable_probes": kept,
        "dropped_probes": dropped,
        "spatial_above_chance_pp": round(float(wide.loc[kept, "spatial"].mean()), 2),
        "best_other_above_chance_pp": round(float(wide.loc[kept, others].max(axis=1).mean()), 2),
        "spatial_differential_pp": round(float(diff.mean()), 2),
        "all_domains_confounded": bool(wide.loc[kept, others].max(axis=1).mean() > 15),
    }

verdict = {}
for ds in DATASETS:
    sub = lodo[lodo.dataset == ds].set_index("arm")
    ref = float(sub.loc["all 1002 (reference)", "f1_macro"])
    without = float(sub.loc["no spatial (941)", "f1_macro"])
    spatial_cost = ref - without
    diag = probe_diagnostics(ds)
    dfl = diag.get("spatial_differential_pp", float("nan"))

    if spatial_cost < 2.0:
        v = "CLEAN - spatial contributes little; keep the current framing"
    elif diag.get("all_domains_confounded"):
        v = ("DATASET-WIDE CONFOUND - every domain identifies the acquisition group, "
             "so this is not a spatial-domain problem; report the grouped and "
             "leave-one-campaign-out numbers side by side")
    elif not np.isnan(dfl) and dfl > 10:
        v = ("MIXED - spatial is load-bearing AND identifies the recording far better "
             "than any other domain; report both the full and no-spatial numbers and "
             "state that this dataset cannot separate the two contributions")
    elif np.isnan(dfl):
        v = "UNDETERMINED - no usable probe (too few multi-recording classes)"
    else:
        v = ("GENUINE - spatial is load-bearing and no better than other domains at "
             "identifying the recording; report it as a substantive finding")

    verdict[ds] = {
        "f1_full_1002": round(ref, 2),
        "f1_no_spatial_941": round(without, 2),
        "spatial_removal_cost_pp": round(spatial_cost, 2),
        "spatial_only_f1": round(float(sub.loc["spatial ONLY (61)", "f1_macro"]), 2),
        **diag,
        "verdict": v,
    }
    print(f"\\n=== {ds} ===")
    for k, val in verdict[ds].items():
        print(f"  {k:34s} {val}")

if len(loro):
    d = float(piv.delta_pp.mean())
    verdict["loro"] = {
        "mean_delta_pp_dropping_spatial": round(d, 2),
        "per_recording_delta_range": [round(float(piv.delta_pp.min()), 2),
                                      round(float(piv.delta_pp.max()), 2)],
        "reading": ("negative = spatial HELPS cross-recording generalisation, so it "
                    "carries transferable information as well as recording nuisance; "
                    "the wide per-recording spread is the signature of a mixture"),
    }
    print(f"\\n=== LORO ===\\n  mean delta dropping spatial: {d:+.2f} pp "
          f"(range {piv.delta_pp.min():+.2f} to {piv.delta_pp.max():+.2f})")

(out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
print("\\nwritten:", out_dir / "verdict.json")
"""),
    ("code", """
final = pd.DataFrame([
    {"dataset": ds,
     "full 1002 macro-F1": lodo[(lodo.dataset==ds) & (lodo.arm=="all 1002 (reference)")].f1_macro.iloc[0],
     "no-spatial 941 macro-F1": lodo[(lodo.dataset==ds) & (lodo.arm=="no spatial (941)")].f1_macro.iloc[0],
     "spatial-only macro-F1": lodo[(lodo.dataset==ds) & (lodo.arm=="spatial ONLY (61)")].f1_macro.iloc[0],
     "spatial above chance (pp)": verdict[ds].get("spatial_above_chance_pp"),
     "best other domain above chance (pp)": verdict[ds].get("best_other_above_chance_pp"),
     "spatial differential (pp)": verdict[ds].get("spatial_differential_pp"),
     "verdict": verdict[ds]["verdict"].split(" - ")[0]}
    for ds in DATASETS
])
display(final.round(2))
final.to_csv(C.results_dir("shared", "tables") / "spatial_ablation_summary.csv", index=False)
"""),
    ("md", """
---
### Sentence to paste into the paper, depending on the verdict

**CLEAN**
> Removing the spatial and spatio-temporal descriptors costs only X pp of macro-F1,
> confirming that the reported performance does not depend on recording-specific fibre
> geometry.

**CONFOUNDED**
> The spatial descriptors carry a disproportionate share of the model's decision, and a
> probe in which the event class is held fixed shows they identify the source recording
> X pp above chance. Because five of the nine classes originate from a single recording,
> these features can act as a proxy for class identity. We therefore report the
> 941-feature signal-agnostic configuration (X% macro-F1) as the operational result, and
> treat the full 1,002-feature figure as an upper bound attainable only when the
> deployment recording is represented in training.

**GENUINE**
> Although the spatial descriptors dominate feature importance, they do not identify the
> source recording above chance once the event class is fixed, indicating that they encode
> event-specific propagation structure rather than recording-specific nuisance.

After this runs, Sections IV-V can be written against a settled claim.
"""),
]
