CELLS = [
    ("md", """
# 11 - Tomasov et al. 2025 CNN Baseline (their architecture, our splits)

A faithful reproduction of the CNN from *"Advancing Perimeter Security: Integrating DAS
and CNN for Object Classification in Fiber Vicinity"*, IEEE Access 13 (2025) - the paper
that accompanies this dataset. No code was released, so everything below is transcribed
from the text and cited to a section.

## What their paper specifies

**Section III-C - the CNN does NOT see raw signals.** Verbatim: *"Directly classifying
raw signals poses significant challenges due to sampling rate incompatibility,
time-domain complexity, and inadequate normalization."* They apply one of three
transforms first:

| transform | their spec | their result |
|---|---|---|
| **RDFT** | zero-pad, FFT, log10 magnitude, subtract mean, truncate to 2048 bins (0-833 Hz) | 85.47% acc, **84.84% F1 (best)** |
| MFCC | librosa, then StandardScaler | **85.61% acc (best)**, high cost |
| DWT | level 3, Daubechies-4, then StandardScaler | 78.48% acc, 75.84% F1 |

**Section III-E - the architecture.** *"two convolutional layers... The first layer
contains 64 filters, while the second one 256. LeakyReLU is applied after each...
max pooling with a pool size of 4 after each... flattened... a dense layer with 256
neurons, which uses a sigmoid activation function. Finally, an output layer... softmax."*

**Section IV - training.** Adam optimiser; loss-function class weights to address
imbalance; 80/10/10 splits.

## Two corrections to the earlier baseline

1. It fed the CNN a **decimated raw waveform** - exactly the input their paper rules out.
2. It used **`Dense(1024)`** (33,155,849 parameters, matching Table 21 of our
   manuscript). The Access paper says **256 neurons** -> 8,375,561 parameters. We report
   both, so the previously published figure remains traceable.

## One inference we had to make

The paper's Table 2 does not survive text extraction from the PDF, so the RDFT
redundancy factor is derived from two stated facts: the output is truncated to 2048 bins
and that corresponds to 0-833 Hz. At fs = 20 kHz this needs an FFT length of
20000 x 2048 / 833 ~ 49152 = 6 x 8192, i.e. **redundancy factor 6**. This is recorded as
an assumption, and `B.RDFT_REDUNDANCY` makes it easy to vary.

> Their own protocol used an 8,192-sample window with a stride of **256** (96.9%
> overlap) and a random 10-fold split. We keep the window but use our leakage-safe
> blocked split - so this is their model under leakage control, not their number.
"""),
    ("code", """
import sys, json, time, gc
from pathlib import Path
EXPERIMENT_DIR = Path(r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan/Experiment")
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from dasfe import config as C, baselines as B, evaluate as EV, preprocess as pp

sns.set_theme(style="whitegrid", context="notebook")

# ----------------------------- CONFIGURE ME -----------------------------
TRANSFORMS   = ("rdft",)          # add "mfcc", "dwt" to reproduce their full Table 3
DENSE_UNITS  = (256, 1024)        # 256 = Access paper; 1024 = our old Table 21
EPOCHS       = 30
BATCH        = 256
LR           = 1e-4
# Windows PER CLASS PER SPLIT. Transformed vectors are small (2048 float32 = 8 kB),
# so the cost here is the raw HDF5 read, not RAM.
# Use EVERY window, so the CNN trains on the same data as the feature pipeline
# (148,229 train rows). The first run capped this at 2,000/class = 18,000 rows -
# 8x less than the pipeline it is being compared against, which made the
# comparison unfair in the CNN's disfavour. RDFT vectors are only 2048 float32
# (8 kB), so the full set is ~1.4 GB; the cost is HDF5 read time, not RAM.
PER_CLASS    = None              # None = all windows; or {"train": N, "val": N, "test": N}
MAX_GB       = 8.0
# ------------------------------------------------------------------------

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev, "|", torch.cuda.get_device_name(0) if dev == "cuda" else "")

spec = C.TOMASOV
out_dir = C.results_dir("tomasov", "10_cnn_baseline")
mani = pd.read_parquet(C.results_dir("tomasov", "01_splits") / "split_manifest.parquet")
CLASSES = sorted(mani.label.unique())
print(f"{len(mani):,} windows | {len(CLASSES)} classes")
print(f"RDFT: fft length {C.WIN_LEN*B.RDFT_REDUNDANCY:,}, "
      f"bin width {spec.fs/(C.WIN_LEN*B.RDFT_REDUNDANCY):.4f} Hz, "
      f"{B.RDFT_BINS} bins -> {B.RDFT_BINS*spec.fs/(C.WIN_LEN*B.RDFT_REDUNDANCY):.0f} Hz "
      f"(paper: 833 Hz)")
"""),
    ("md", """
## 1. Load raw windows and apply their transform

One HDF5 open per recording, one locus per window, results written straight into a
preallocated float32 array. This is the pattern that stopped notebook 09 from hanging
the Jetson: appending numpy *views* retained the whole parent patch (2.10 MB per row).
"""),
    ("code", """
def load_transformed(kind):
    if PER_CLASS is None:
        sub = mani.sort_values("h5_path").reset_index(drop=True)
    else:
        parts = []
        for (label, sp), g in mani.groupby(["label", "split"]):
            n = min(PER_CLASS.get(sp, 0), len(g))
            if n:
                parts.append(g.sample(n, random_state=C.SEED))
        sub = pd.concat(parts).sort_values("h5_path").reset_index(drop=True)

    dim = B.tomasov_input_length(kind, spec.win_len, spec.fs)
    gb = len(sub) * dim * 4 / 1e9
    print(f"  {kind}: {len(sub):,} windows x {dim} dims -> {gb:.2f} GB float32")
    if gb > MAX_GB:
        raise MemoryError(f"{gb:.2f} GB exceeds MAX_GB={MAX_GB}; lower PER_CLASS.")

    X = np.empty((len(sub), dim), dtype=np.float32)
    reader, cur, t0 = None, None, time.perf_counter()
    try:
        for i, r in enumerate(sub.itertuples()):
            if r.h5_path != cur:
                if reader is not None:
                    reader.close()
                reader = pp.TomasovReader(r.h5_path)
                cur = r.h5_path
                print(f"    opened {Path(cur).name}", flush=True)
            trace = reader.read_patch(int(r.t0), int(r.locus), spec.win_len, 1)[0]
            X[i] = B.tomasov_transform(trace, reader.fs, kind)
            if (i + 1) % 2500 == 0:
                print(f"    {i+1:,}/{len(sub):,}", flush=True)
    finally:
        if reader is not None:
            reader.close()
    print(f"  loaded in {time.perf_counter()-t0:.0f}s")
    return X, sub["label"].to_numpy(), sub["split"].to_numpy()
"""),
    ("md", """
## 2. Train, following Section IV

Adam, class-weighted cross-entropy. The `StandardScaler` for DWT and MFCC is fitted on
the **training split only** - their paper says the coefficients are standardised, and
fitting that on all data would reintroduce leakage through the scaler.
"""),
    ("code", """
@torch.no_grad()
def predict_batched(net, X, batch=512):
    net.eval(); out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).unsqueeze(1).to(dev)
        out.append(net(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train_one(X, y, sp, kind, dense_units):
    tr, va, te = sp == "train", sp == "val", sp == "test"
    yi = np.array([CLASSES.index(v) for v in y])

    Xw = X
    if kind in ("dwt", "mfcc"):                 # "normalized using a Standard Scaler"
        sc = StandardScaler().fit(X[tr])
        # Scale IN PLACE. sc.transform(X) allocates a second full copy, which for
        # DWT at full size (8212 dims x 172k windows) means ~11 GB peak instead of
        # ~5.7 GB - enough to swap-kill a Jetson. X is our own preallocated array.
        mu = sc.mean_.astype(np.float32)
        sd = np.where(sc.scale_ > 0, sc.scale_, 1.0).astype(np.float32)
        np.subtract(X, mu, out=X)
        np.divide(X, sd, out=X)
        Xw = X

    torch.manual_seed(C.SEED)
    net = B.build_tomasov_cnn(len(CLASSES), Xw.shape[1], dense_units=dense_units).to(dev)
    n_par = sum(p.numel() for p in net.parameters())

    counts = np.bincount(yi[tr], minlength=len(CLASSES)).astype(float)
    w = torch.tensor(len(yi[tr]) / (len(CLASSES) * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3)
    lossf = nn.CrossEntropyLoss(weight=w)

    dl = DataLoader(TensorDataset(torch.from_numpy(Xw[tr]).unsqueeze(1),
                                  torch.from_numpy(yi[tr])),
                    batch_size=BATCH, shuffle=True)
    hist, t0 = [], time.perf_counter()
    for ep in range(EPOCHS):
        net.train(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step()
            tot += float(l) * len(yb)
        hist.append(tot / max(tr.sum(), 1)); sched.step(hist[-1])
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{EPOCHS} loss {hist[-1]:.4f}", flush=True)
    train_s = time.perf_counter() - t0

    y_val = np.array([CLASSES[i] for i in predict_batched(net, Xw[va])])
    y_te = np.array([CLASSES[i] for i in predict_batched(net, Xw[te])])
    m_val = EV.score_all(y[va], y_val, labels=CLASSES)
    m_te = EV.score_all(y[te], y_te, labels=CLASSES)

    row = {"transform": kind, "dense_units": dense_units, "n_params": n_par,
           "input_dim": Xw.shape[1], "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "val_f1_macro": 100*m_val["f1_macro"],
           "accuracy": 100*m_te["accuracy"],
           "balanced_accuracy": 100*m_te["balanced_accuracy"],
           "f1_macro": 100*m_te["f1_macro"],
           "train_seconds": round(train_s, 1), "final_loss": round(hist[-1], 4)}
    print(f"  {kind}/dense{dense_units}: acc {row['accuracy']:.2f}%  "
          f"F1 {row['f1_macro']:.2f}%  ({n_par:,} params, {train_s:.0f}s)")
    del net, dl; gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return row, y[te], y_te, hist
"""),
    ("code", """
results, curves, confusions, per_class = [], {}, {}, {}
for kind in TRANSFORMS:
    X, y, sp = load_transformed(kind)
    for du in DENSE_UNITS:
        row, yt, yp, hist = train_one(X, y, sp, kind, du)
        results.append(row)
        key = f"{kind}_d{du}"
        curves[key] = hist
        confusions[key] = EV.confusion(yt, yp, labels=CLASSES, normalize="true")
        per_class[key] = EV.per_class_report(yt, yp, labels=CLASSES)
    del X; gc.collect()

res = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
display(res.round(3))
res.to_csv(out_dir / "tomasov_cnn_results.csv", index=False)
"""),
    ("md", """
## 3. Does the dense-layer size matter?

The Access paper says 256; our Table 21 used 1024. If the two are within noise, the
earlier figure stands and only the *input representation* was wrong.
"""),
    ("code", """
if len(DENSE_UNITS) > 1:
    piv = res.pivot(index="transform", columns="dense_units", values="f1_macro")
    piv["diff_pp"] = (piv[1024] - piv[256]).round(2)
    display(piv.round(2))
    print("Positive diff_pp means the larger dense layer helped despite 4x the parameters.")

fig, axes = plt.subplots(1, len(confusions) + 1,
                         figsize=(5.5*len(confusions) + 6, 5))
axes = np.atleast_1d(axes)
for ax, (tag, cm) in zip(axes, confusions.items()):
    sns.heatmap(100*cm, annot=True, fmt=".0f", cmap="Greens", ax=ax, cbar=False)
    ax.set_title(f"Tomasov CNN - {tag}", fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
for tag, h in curves.items():
    axes[-1].plot(h, label=tag)
axes[-1].set_xlabel("epoch"); axes[-1].set_ylabel("training loss")
axes[-1].set_title("convergence"); axes[-1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(out_dir / "tomasov_cnn_confusion_and_loss.png", dpi=200, bbox_inches="tight")
plt.show()

# NOTE: `res.iloc[0].transform` returns pandas' Series.transform METHOD, not the
# column. Always use bracket access for a column named after a DataFrame method.
best_key = f"{res.iloc[0]['transform']}_d{int(res.iloc[0]['dense_units'])}"
display(per_class[best_key].round(3))
for k, v in per_class.items():
    v.to_csv(out_dir / f"tomasov_cnn_per_class_{k}.csv", index=False)
for k, v in confusions.items():
    v.to_csv(out_dir / f"tomasov_cnn_confusion_{k}.csv")
"""),
    ("md", """
## 4. Head-to-head with the feature pipeline

Same dataset, same leakage-safe split. The `openclose` row is the one to read carefully:
the feature pipeline collapses on it (F1 0.044) once the leak is removed, so if the CNN
also collapses, that class is simply not learnable from this data - not a modelling
failure on either side.
"""),
    ("code", """
rows = []
for variant in ("full_1002", "8_consensus"):
    p = C.results_dir("tomasov", "07_final") / variant / "model_card.json"
    if p.exists():
        card = json.loads(p.read_text())
        rows.append({"approach": f"Multi-domain features ({variant})",
                     "detail": f"{card['model']} / {card['n_features']} feat",
                     "accuracy": card["test"]["accuracy"],
                     "balanced_accuracy": card["test"]["balanced_accuracy"],
                     "f1_macro": card["test"]["f1_macro"]})
for _, r in res.iterrows():   # r is a Series; use r['col']
    rows.append({"approach": f"Tomasov CNN ({r['transform']}, dense {int(r['dense_units'])})",
                 "detail": f"{int(r['n_params']):,} params",
                 "accuracy": r["accuracy"], "balanced_accuracy": r["balanced_accuracy"],
                 "f1_macro": r["f1_macro"]})

head = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
display(head.round(2))
head.to_csv(out_dir / "tomasov_head_to_head.csv", index=False)

ref = pd.DataFrame([
    {"source": "Tomasov et al. 2025 (their own split)", "preproc": "MFCC", "accuracy": 85.61, "f1_macro": 84.72},
    {"source": "Tomasov et al. 2025 (their own split)", "preproc": "RDFT", "accuracy": 85.47, "f1_macro": 84.84},
    {"source": "Tomasov et al. 2025 (their own split)", "preproc": "DWT",  "accuracy": 78.48, "f1_macro": 75.84},
])
print("\\nFor context - as published, on their own random split (8192 window, stride 256):")
display(ref)
ref.to_csv(out_dir / "tomasov_published_reference.csv", index=False)
print("Their split reused windows overlapping by 96.9%, so these are not directly")
print("comparable to the numbers above; they are the target our reproduction aims at.")
"""),
    ("md", """
## 5. Record the assumptions

Anything we inferred rather than read is written to disk, so the paper's methods section
can state it explicitly.
"""),
    ("code", """
assumptions = {
    "source": "Tomasov et al., IEEE Access 13, 2025, Sections III-C, III-E, IV",
    "read_from_paper": {
        "conv_filters": [64, 256], "activation": "LeakyReLU",
        "pool_size": 4, "dense_units": 256, "dense_activation": "sigmoid",
        "output_activation": "softmax", "optimiser": "Adam",
        "class_weighted_loss": True, "split": "80/10/10",
        "rdft_bins": B.RDFT_BINS, "rdft_freq_range_hz": [0, 833],
        "dwt_level": B.DWT_LEVEL, "dwt_wavelet": B.DWT_WAVELET,
        "window": C.WIN_LEN, "their_stride": 256,
    },
    "inferred_or_substituted": {
        "kernel_size": "7, from Table 21 of our manuscript (not stated in the Access paper)",
        "rdft_redundancy": f"{B.RDFT_REDUNDANCY}, derived from 2048 bins <-> 0-833 Hz at fs=20 kHz",
        "epochs": EPOCHS, "batch_size": BATCH, "learning_rate": LR,
        "mfcc_params": f"n_mfcc={B.MFCC_N}, hop={B.MFCC_HOP}; librosa if importable, "
                       "otherwise an equivalent scipy mel/DCT implementation",
        "split": "our blocked spatio-temporal split, NOT their random 10-fold",
    },
    "our_dense_1024_variant": "reproduces Table 21 (33,155,849 params) for traceability",
}
(out_dir / "tomasov_cnn_assumptions.json").write_text(
    json.dumps(assumptions, indent=2), encoding="utf-8")
print(json.dumps(assumptions, indent=2))
"""),
    ("md", """
---
### What to put in the paper

> The CNN baselines are the architectures published by each dataset's own authors. For
> Cao et al. we use their released implementation unchanged (2D CNN over the 10000x12
> record, 7,421 parameters, min-max scaling to 0-255). For Tomasov et al., who released
> no code, we transcribe the architecture from Section III-E of their paper and apply the
> RDFT front end of Section III-C, since that paper explicitly rejects raw time-domain
> input. Both are trained on the same leakage-safe splits as the feature pipeline, so the
> comparison isolates the model rather than the evaluation protocol.
"""),
]
