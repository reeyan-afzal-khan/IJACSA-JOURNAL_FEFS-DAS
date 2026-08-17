CELLS = [
    ("md", """
# 10 - Cao et al. 2023 CNN Baseline (their architecture, our splits)

A **faithful** reproduction of the CNN that Cao et al. ship with their dataset, so the
comparison against the feature pipeline is like-for-like.

Source: `Dataset/Cao_2023/Phi-OTDR_dataset_and_codes-main/` -
`models.py` (architecture), `mydataset.py` (preprocessing), `das_data_cnn.py` (training).

## What their CNN actually is

| | their CNN | what notebook 09 ran before |
|---|---|---|
| input | full **10000 x 12** space-time record, 2D | one channel, decimated to 2048, 1D |
| scaling | per-sample **min-max to 0-255** | band-pass + z-score |
| architecture | Conv2d(1->5, 200x3, s=50x1) -> pool -> Conv2d(5->10, 20x2, s=4x1) -> pool -> Linear(400, 6) | Conv1d 64/256 + Dense(1024) |
| parameters | **7,421** | 33,152,774 |
| optimiser | Adam, lr 1e-4, weight_decay 1e-5 | Adam, lr 1e-4 |
| loss | CrossEntropy, **no class weights** | CrossEntropy with class weights |
| epochs / batch | 50 / 100 | 20 / 256 |

So the earlier baseline was not their model at all - it was ~4,500x larger and fed an
input format they never used. That comparison would not survive review.

## Two protocols, deliberately

1. **Their released 8:2 split** - reproduces the number they published.
2. **Our session-grouped split** - the same model under leakage control.

The gap between the two isolates how much of the published figure came from the fact
that 423 of 441 recording sessions appear in both their Training and Test folders.

Loading is on demand (their `MyDataset` does the same), so memory stays O(batch)
rather than holding 7 GB of records in RAM.
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
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dasfe import config as C, baselines as B, evaluate as EV

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
EPOCHS       = 50            # their das_data_cnn.py default
BATCH        = 100           # their default
LR           = 1e-4          # hardcoded in their main(), not the argparse default
WEIGHT_DECAY = 1e-5          # their default
PROTOCOLS    = ("official", "grouped")   # run both, compare
MAX_PER_CLASS = None         # None = use everything; set an int to subsample
NUM_WORKERS  = 4             # on-demand .mat loading parallelises well
# ------------------------------------------------------------------------

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")

out_dir = C.results_dir("cao", "10_cnn_baseline")
mani = pd.read_parquet(C.results_dir("cao", "01_splits") / "split_manifest.parquet")
mani = mani[mani.n_bytes > 0].reset_index(drop=True)
CLASSES = sorted(mani.label.unique())
print(f"{len(mani):,} records | {len(CLASSES)} classes: {CLASSES}")
print("output:", out_dir)
"""),
    ("md", """
## 1. Their dataset class and architecture
"""),
    ("code", """
import scipy.io as scio

class CaoRecords(Dataset):
    # Mirrors their mydataset.MyDataset: load the .mat on demand, min-max to
    # 0-255, return the full 10000x12 record. No band-pass, no z-score.
    def __init__(self, frame, classes):
        self.paths = frame["path"].tolist()
        self.labels = [classes.index(v) for v in frame["label"]]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rec = scio.loadmat(self.paths[i])["data"].astype(np.int64)
        x = B.cao_normalise(rec)                       # (10000, 12) float32, 0-255
        return torch.from_numpy(x).unsqueeze(0), self.labels[i]

net_probe = B.build_cao_cnn(len(CLASSES))
n_par = sum(p.numel() for p in net_probe.parameters())
with torch.no_grad():
    h1 = net_probe.conv1(torch.zeros(1, 1, B.CAO_TIME, B.CAO_CHANNELS))
    h2 = net_probe.conv2(h1)
print(f"architecture check: (1,{B.CAO_TIME},{B.CAO_CHANNELS}) -> {tuple(h1.shape[1:])} "
      f"-> {tuple(h2.shape[1:])} -> Linear({h2.numel()}, {len(CLASSES)})")
print(f"their stated shapes: (5, 99, 7) then (10, 10, 4)   [comments in models.py]")
print(f"parameters: {n_par:,}")
"""),
    ("md", """
## 2. Training loop, following `das_data_cnn.py`
"""),
    ("code", """
def run_protocol(frame, train_mask, test_mask, tag, epochs=None, quiet=False):
    tr_df, te_df = frame[train_mask], frame[test_mask]
    if MAX_PER_CLASS:
        tr_df = tr_df.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), MAX_PER_CLASS), random_state=C.SEED))
    n_ep = epochs or EPOCHS
    if not quiet:
        print(f"=== {tag}: train {len(tr_df):,} | test {len(te_df):,} ===")

    tl = DataLoader(CaoRecords(tr_df, CLASSES), batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, drop_last=False)
    vl = DataLoader(CaoRecords(te_df, CLASSES), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS)

    torch.manual_seed(C.SEED)
    net = B.build_cao_cnn(len(CLASSES)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()          # no class weights, as in their code

    hist, t0 = [], time.perf_counter()
    for ep in range(n_ep):
        net.train(); tot = 0.0; n = 0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
            tot += float(l) * len(yb); n += len(yb)
        hist.append(tot / max(n, 1))
        if not quiet and (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{n_ep} loss {hist[-1]:.4f}", flush=True)
    train_s = time.perf_counter() - t0

    net.eval(); preds = []
    with torch.no_grad():
        for xb, _ in vl:
            preds.append(net(xb.to(dev)).argmax(1).cpu().numpy())
    pred_i = np.concatenate(preds)
    y_true = np.array([CLASSES[i] for i in CaoRecords(te_df, CLASSES).labels])
    y_pred = np.array([CLASSES[i] for i in pred_i])

    met = EV.score_all(y_true, y_pred, labels=CLASSES)
    met.update(B.nar_fnr(y_true, y_pred, "background"))
    met.update(protocol=tag, n_train=len(tr_df), n_test=len(te_df),
               train_seconds=round(train_s, 1), n_params=n_par,
               epochs=n_ep, final_loss=round(hist[-1], 4))
    if not quiet:
        print(f"  accuracy {100*met['accuracy']:.2f}%  macro-F1 {100*met['f1_macro']:.2f}%  "
              f"NAR {met['NAR']:.4f}  FNR {met['FNR']:.4f}  ({train_s:.0f}s)")
    return met, y_true, y_pred, hist
"""),
    ("code", """
results, curves, confusions = [], {}, {}

if "official" in PROTOCOLS:
    m, yt, yp, h = run_protocol(mani,
                               mani.official_split == "train",
                               mani.official_split == "test",
                               "Cao released 8:2 split (leaky)")
    results.append(m); curves["official"] = h
    confusions["official"] = EV.confusion(yt, yp, labels=CLASSES, normalize="true")

if "grouped" in PROTOCOLS:
    m, yt, yp, h = run_protocol(mani,
                               mani.split == "train",
                               mani.split == "test",
                               "session-grouped split (ours)")
    results.append(m); curves["grouped"] = h
    confusions["grouped"] = EV.confusion(yt, yp, labels=CLASSES, normalize="true")
    per_class = EV.per_class_report(yt, yp, labels=CLASSES)

res = pd.DataFrame(results)
display(res[["protocol", "n_train", "n_test", "accuracy", "balanced_accuracy",
             "f1_macro", "NAR", "FNR", "train_seconds"]].round(4))
res.to_csv(out_dir / "cao_cnn_results.csv", index=False)
"""),
    ("md", """
## 3. How much of their published number was leakage?
"""),
    ("code", """
if len(res) == 2:
    off = res[res.protocol.str.contains("released")].iloc[0]
    grp = res[res.protocol.str.contains("grouped")].iloc[0]
    delta = pd.DataFrame([{
        "metric": k,
        "released 8:2 (%)": round(100*off[k], 2),
        "session-grouped (%)": round(100*grp[k], 2),
        "drop (pp)": round(100*(off[k] - grp[k]), 2),
    } for k in ("accuracy", "balanced_accuracy", "f1_macro")])
    display(delta)
    delta.to_csv(out_dir / "leakage_effect_on_cnn.csv", index=False)
    print("Any positive drop is attributable to the 423/441 recording sessions that")
    print("straddle their Training and Test folders - the model had already seen the")
    print("same continuous recording it is being tested on.")
"""),
    ("code", """
fig, axes = plt.subplots(1, len(confusions) + 1, figsize=(6*len(confusions) + 6, 5))
axes = np.atleast_1d(axes)
for ax, (tag, cm) in zip(axes, confusions.items()):
    sns.heatmap(100*cm, annot=True, fmt=".1f", cmap="Oranges", ax=ax,
                cbar_kws={"label": "% of true class"})
    ax.set_title(f"Cao CNN - {tag}"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
for tag, h in curves.items():
    axes[-1].plot(h, label=tag)
axes[-1].set_xlabel("epoch"); axes[-1].set_ylabel("training loss")
axes[-1].set_title("convergence"); axes[-1].legend()
plt.tight_layout()
plt.savefig(out_dir / "cao_cnn_confusion_and_loss.png", dpi=200)
plt.show()

if "grouped" in PROTOCOLS:
    display(per_class.round(3))
    per_class.to_csv(out_dir / "cao_cnn_per_class.csv", index=False)
    for tag, cm in confusions.items():
        cm.to_csv(out_dir / f"cao_cnn_confusion_{tag.split()[0]}.csv")
"""),
    ("md", """
## 4. Leave-one-date-out - the fair cross-campaign comparison

Section 3 showed this CNN barely notices the session-level leakage in the released split -
a 7,421-parameter model has no capacity to memorise individual recordings, so it gains
almost nothing from seeing the same recording twice.

That raises the question the paper has to answer. Our feature pipeline scores 98.90%
macro-F1 on the session-grouped split but only **57.82%** under leave-one-date-out
(notebook 08), because the engineered features encode acquisition-campaign structure. If
this low-capacity CNN is more robust across campaigns, it could win where it matters.

So we run the identical protocol on it: hold out one acquisition date, train on the rest.
This is the number to compare against the pipeline's 57.82%.
"""),
    ("code", """
# ---------------------------- CONFIGURE ME ----------------------------
RUN_LODO   = True
LODO_EPOCHS = 50        # 50 keeps it identical to the protocols above; 25 halves runtime
MIN_TEST    = 30        # skip dates with too little test data
MIN_CLASSES = 2         # a single-class test set makes macro-F1 meaningless
# -----------------------------------------------------------------------

lodo_rows = []
if RUN_LODO:
    dates = sorted(mani.date.unique())
    print(f"{len(dates)} acquisition dates; {LODO_EPOCHS} epochs each")
    print()
    for dt in dates:
        te_mask = mani.date == dt
        tr_mask = ~te_mask
        n_cls_te = mani.loc[te_mask, "label"].nunique()
        if te_mask.sum() < MIN_TEST or n_cls_te < MIN_CLASSES:
            print(f"  {dt}: skipped (n_test={int(te_mask.sum())}, classes={n_cls_te})")
            continue
        if mani.loc[tr_mask, "label"].nunique() < len(CLASSES):
            print(f"  {dt}: note - training set is missing a class")
        m, yt, yp, _ = run_protocol(mani, tr_mask, te_mask,
                                    f"LODO date={dt}", epochs=LODO_EPOCHS, quiet=True)
        lodo_rows.append({"held_out_date": dt, "n_test": int(te_mask.sum()),
                          "n_classes_in_test": int(n_cls_te),
                          "accuracy": 100*m["accuracy"],
                          "balanced_accuracy": 100*m["balanced_accuracy"],
                          "f1_macro": 100*m["f1_macro"],
                          "NAR": m["NAR"], "FNR": m["FNR"]})
        print(f"  {dt}: acc {100*m['accuracy']:6.2f}%  F1 {100*m['f1_macro']:6.2f}%  "
              f"({int(te_mask.sum())} test, {n_cls_te} classes)", flush=True)

    cnn_lodo = pd.DataFrame(lodo_rows)
    display(cnn_lodo.round(2))
    cnn_lodo.to_csv(out_dir / "cao_cnn_leave_one_date_out.csv", index=False)
    print()
    print(f"CNN leave-one-date-out: accuracy {cnn_lodo.accuracy.mean():.2f}% "
          f"+/- {cnn_lodo.accuracy.std():.2f}, "
          f"macro-F1 {cnn_lodo.f1_macro.mean():.2f}% +/- {cnn_lodo.f1_macro.std():.2f} "
          f"(n = {len(cnn_lodo)} dates)")
else:
    cnn_lodo = pd.DataFrame()
    print("RUN_LODO = False - skipping the cross-campaign arm.")
"""),
    ("code", """
# Same dates, same protocol: CNN vs the feature pipeline.
pipe_path = C.results_dir("shared", "08_crossdataset") / "cao_leave_one_date_out.csv"
if len(cnn_lodo) and pipe_path.exists():
    pipe = pd.read_csv(pipe_path).rename(
        columns={"accuracy": "pipe_accuracy", "f1_macro": "pipe_f1"})
    cmp = (cnn_lodo.rename(columns={"accuracy": "cnn_accuracy", "f1_macro": "cnn_f1"})
                   [["held_out_date", "n_test", "cnn_accuracy", "cnn_f1"]]
           .merge(pipe[["held_out_date", "pipe_accuracy", "pipe_f1"]],
                  on="held_out_date", how="inner"))
    cmp["acc_diff_pp"] = (cmp.pipe_accuracy - cmp.cnn_accuracy).round(2)
    cmp["f1_diff_pp"] = (cmp.pipe_f1 - cmp.cnn_f1).round(2)
    display(cmp.round(2))
    cmp.to_csv(out_dir / "cao_lodo_cnn_vs_pipeline.csv", index=False)

    print(f"dates compared            : {len(cmp)}")
    print(f"feature pipeline  mean F1 : {cmp.pipe_f1.mean():.2f}%")
    print(f"Cao CNN           mean F1 : {cmp.cnn_f1.mean():.2f}%")
    print(f"difference                : {cmp.pipe_f1.mean() - cmp.cnn_f1.mean():+.2f} pp "
          f"in favour of {'the feature pipeline' if cmp.pipe_f1.mean() > cmp.cnn_f1.mean() else 'the CNN'}")
    print(f"dates where the CNN wins  : {int((cmp.f1_diff_pp < 0).sum())}/{len(cmp)}")
    print()
    print("This is the comparison that decides whether the pipeline's advantage on the")
    print("session-grouped split survives a genuine change of acquisition campaign.")

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(cmp))
    ax.bar(x - 0.2, cmp.pipe_f1, 0.4, label="multi-domain features", color="#4C72B0")
    ax.bar(x + 0.2, cmp.cnn_f1, 0.4, label="Cao et al. CNN", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(cmp.held_out_date, rotation=45, ha="right")
    ax.set_ylabel("macro-F1 (%)"); ax.set_xlabel("held-out acquisition date")
    ax.set_title("Cao 2023 - leave-one-date-out: features vs the published CNN")
    ax.legend(); plt.tight_layout()
    plt.savefig(out_dir / "cao_lodo_cnn_vs_pipeline.png", dpi=200)
    plt.show()
elif len(cnn_lodo):
    print("Pipeline LODO results not found; run notebook 08 to produce")
    print(pipe_path)
"""),
    ("md", """
## 5. Head-to-head with the feature pipeline

Two protocols side by side: the session-grouped split (same recordings in train and
test, different samples) and leave-one-date-out (a genuinely unseen acquisition
campaign). A method that wins on the first but loses on the second is only usable where
the deployment site is already represented in training.
"""),
    ("code", """
rows = []
for variant in ("full_1002", "8_consensus"):
    p = C.results_dir("cao", "07_final") / variant / "model_card.json"
    if p.exists():
        card = json.loads(p.read_text())
        rows.append({"approach": f"Multi-domain features ({variant})",
                     "model": card["model"], "n_params_or_features": card["n_features"],
                     "accuracy": card["test"]["accuracy"],
                     "balanced_accuracy": card["test"]["balanced_accuracy"],
                     "f1_macro": card["test"]["f1_macro"],
                     "inference_ms": card["inference_ms_per_sample"]})
if "grouped" in PROTOCOLS:
    g = res[res.protocol.str.contains("grouped")].iloc[0]
    rows.append({"approach": "Cao et al. 2023 CNN (their architecture)",
                 "model": "2D CNN", "n_params_or_features": int(g.n_params),
                 "accuracy": 100*g.accuracy, "balanced_accuracy": 100*g.balanced_accuracy,
                 "f1_macro": 100*g.f1_macro, "inference_ms": np.nan})

if len(cnn_lodo):
    rows.append({"approach": "Cao et al. 2023 CNN - leave-one-date-out",
                 "model": "2D CNN", "n_params_or_features": n_par,
                 "accuracy": cnn_lodo.accuracy.mean(),
                 "balanced_accuracy": cnn_lodo.balanced_accuracy.mean(),
                 "f1_macro": cnn_lodo.f1_macro.mean(), "inference_ms": np.nan})
if pipe_path.exists():
    pl = pd.read_csv(pipe_path)
    rows.append({"approach": "Multi-domain features - leave-one-date-out",
                 "model": "LightGBM", "n_params_or_features": 1002,
                 "accuracy": pl.accuracy.mean(), "balanced_accuracy": np.nan,
                 "f1_macro": pl.f1_macro.mean(), "inference_ms": np.nan})

head = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
display(head.round(3))
head.to_csv(out_dir / "cao_head_to_head.csv", index=False)
print("\\nNote: n_params_or_features is engineered features for the pipeline and")
print("trainable parameters for the CNN - they are not directly comparable, but both")
print("bound the model's capacity and its deployment cost.")
"""),
    ("md", """
---
**Next:** `11_tomasov_cnn_baseline.ipynb` - the same treatment for the Tomasov CNN,
including the RDFT/DWT/MFCC front end their paper requires.
"""),
]
