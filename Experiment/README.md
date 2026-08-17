# Multi-Domain DAS Event Classification — Two-Dataset Revision

Revision of the IEEE Access submission *"Multi-Domain Feature Engineering Approach with
Lightweight Classifiers for Event Classification in Distributed Acoustic Sensing Data"*.

The methodology is unchanged — 1,002 handcrafted features across four signal-processing
domains, ensemble feature selection, class balancing, and a benchmark of seven
lightweight classifiers. Three things are different, and they are the point of the
revision:

1. **Two independent datasets** instead of one.
2. **Group-aware splitting** that makes data leakage structurally impossible, and
   executable audits that prove it.
3. **Feature selection re-fitted inside every CV fold**, with the optimism of the
   original protocol quantified rather than assumed away.

---

## Why the original protocol leaked

### Tomasov et al. 2025

Windows are 8,192 samples with a 2,048-sample hop — **75% overlap**. The submitted
version used a random stratified 80/10/10 split, so a test window could share 6,144 of
its 8,192 raw samples with a training window. Notebook 02 measures it:

| protocol | test windows | sharing raw samples with train |
|---|---|---|
| random stratified 80/10/10 (submitted) | 43,976 | **43,976 (100.0%)** |
| blocked segment + guard band (revised) | 31,028 | **0 (0.0%)** |

Compounding factors: adjacent fibre channels are 1.02 m apart and see the same event, and
there are only **16 HDF5 recordings** for 9 classes — five classes come from a single
recording, so leave-one-recording-out is impossible for the full taxonomy.

### Cao et al. 2023

Filenames encode `<date>_<operator>_<event>_<session>_data_<n>.mat`. Parsing all 15,419
records gives 441 recording sessions — and **423 of them (95.9%) contribute samples to
both the released `Training/` and `Test/` folders**. The published 8:2 split leaks at the
session level, so the SVM and CNN baselines shipped with the dataset measure
within-recording memorisation as much as event recognition.

---

## Revised split protocol

| | Cao 2023 | Tomasov 2025 |
|---|---|---|
| Split group | recording session (441) | contiguous recording segment (154) |
| Mechanism | `StratifiedGroupKFold`, 10 folds merged 8/1/1 | 10 contiguous equal-**mass** segments per recording, rotated assignment |
| Guard band | not needed (samples are pre-segmented, no overlap) | 4 blocks = one full window at every split boundary |
| Robustness split | leave-one-date-out (notebook 08) | leave-one-recording-out, 4 classes with ≥2 recordings (notebook 08) |

Segments are cut by **window mass**, not duration. A first attempt using
`StratifiedGroupKFold` over fixed-duration zones balanced *group counts*, but a zone
holds anywhere from ~30 to ~8,000 windows — producing a 92/5/2 split in which the
`openclose` class received **no test samples at all**.

Guard-band removal costs 8.9% of the annotated windows and moves the realised split to
84.6 / 7.6 / 7.8. That is the price of the guarantee, and it is stated rather than hidden.

### Every leakage channel and its control

| Leak | Source | Control | Enforced in |
|---|---|---|---|
| Sample-level | 75% window overlap | guard bands + `assert_no_sample_overlap` | `splits.py`, nb 02 |
| Recording-level | many windows per recording | whole-group splitting | `splits.py`, nb 02 |
| Selection-level | selectors fitted on all data | training folds only; re-fitted per CV fold | `selection.py`, nb 05 |
| Resampling-level | SMOTE fitted before splitting | sampler inside `imblearn.Pipeline` | `balancing.py`, nb 06 |
| Scaling-level | scaler fitted on train+test | `StandardScaler` in the same pipeline | `balancing.py`, nb 06 |

---

## Notebook series

Run in order. Notebooks 03–07 are parameterised: set `DATASET = "cao"` or `"tomasov"` in
the configuration cell and run each twice.

| # | Notebook | Produces |
|---|---|---|
| 00 | `00_setup_and_environment` | dependency check, `Results/` tree, environment record |
| 01 | `01_dataset_inventory_and_leakage_audit` | manifests, the Cao official-split leakage audit, dataset comparison table |
| 02 | `02_leakage_safe_splits` | frozen split manifests + leakage scorecards (**these assert**) |
| 03 | `03_feature_extraction` | 1,002 features/window, chunked and resumable |
| 04 | `04_multidomain_fusion` | `X_multi.npy` (1,002 cols) + alignment checks |
| 05 | `05_feature_selection` | 8-method ensemble, consensus subset, Jaccard, selection-bias analysis |
| 06 | `06_balancing_and_model_benchmark` | 7 models × 6 strategies, domain comparison, grouped CV |
| 07 | `07_final_model_and_ablation` | per-class report, confusion matrices, class-removal ablation |
| 08 | `08_cross_dataset_generalisation` | parallel benchmark, cross-system transfer, LODO/LORO |
| 09 | `09_statistics_efficiency_and_report` | Friedman/Nemenyi, CNN baseline, efficiency, all export tables |

Set `SMOKE_TEST = True` for a fast end-to-end validation of the whole chain (~60 windows
per class), then `SMOKE_TEST = False` for the real run.

---

## The two-dataset argument

The submitted paper argued Cao was unusable for validation: different sensing principle,
one overlapping class, only 12 spatial channels. All three are true, and all three only
rule out *transferring a fitted model*. They do not rule out **re-running the
methodology** — which is the claim the paper actually needs.

Notebook 08 runs four experiments:

1. **Parallel benchmark** — identical pipeline on both datasets. If the domain ranking
   and classifier ranking agree across two interrogators, the conclusion is about the
   method, not the fibre. Reported as Spearman ρ.
2. **Cross-system transfer** on a harmonised 4-class taxonomy:

   | harmonised | Cao | Tomasov |
   |---|---|---|
   | `background` | background | regular |
   | `walk` | walk | walk |
   | `impact` | knock | fence (knocking) |
   | `excavation` | dig | construction |

   Spatial features are excluded — they depend on channel spacing (10 m vs 1.02 m) and
   count (12 vs 32), so transferring them would be meaningless rather than merely hard.
3. **Leave-one-date-out** (Cao) — robustness to day-to-day soil/temperature/operator drift.
4. **Leave-one-recording-out** (Tomasov) — the strictest test the data supports.

---

## Feature space (unchanged from the submission)

| Domain | Count | Prefix | Notes |
|---|---|---|---|
| Time | 112 | `TIME:` | statistics, energy, temporal, slope, nonlinear, envelope, burst, histogram |
| Frequency | 90 | `FREQ:` | spectral stats/shape, peaks, 7 band powers, entropy, PSD, histogram, cepstrum |
| Time-frequency | 739 | `TF:` | STFT, DWT (7 levels), WPT (16 sub-bands), GLCM, LBP, HOG, geometry, DAS metrics |
| Spatial / spatio-temporal | 61 | `SPAT:` | localisation, distribution, gradient, clustering, motion, texture |
| **Multi-domain** | **1,002** | | |

Dimensionality is asserted at import time — `import dasfe` fails if any extractor drifts.

Two deliberate deviations, both documented in code:

- **STFT resolution.** The original used `nfft=4096, hop=2048`, giving ~5 frames per
  window — too thin for GLCM/LBP/HOG texture descriptors. Default is now `512/128`
  (257 × 61 image); the original values are kept in `config.PAPER_NFFT_STFT` for ablation.
- **Coefficient of variation.** Per-window z-scoring forces `mean = 0`, so `std/|mean|`
  diverged to ~1e12. That slot now holds the scale-invariant `rms/MAD`. Nine features
  remain constant by construction (z-scoring fixes mean and std) and are removed by the
  variance selector — reported, not hidden.

---

## Efficiency

Extraction is **one pass over the raw data for all four domains**. The original ran four
separate passes, reading and preprocessing every window four times (~23 h total).
Measured on this machine (22 workers):

| dataset | ms/window (1 core) | throughput | est. full run |
|---|---|---|---|
| Cao (15.4k windows) | 36.8 | ~10 win/s | ~25 min |
| Tomasov (172k windows) | 66.1 | ~18 win/s | ~2.7 h |

Tomasov throughput is HDF5-read-bound, not CPU-bound. Chunks are one recording each, so
the file is opened once per chunk rather than once per window.

**Real-time framing.** A new window arrives every `2048 / 20000 = 102.4 ms`. Inference is
sub-millisecond, but *feature extraction* is the binding constraint — notebook 09
computes how many cores meet the budget. The honest claim is that the **classifier** is
lightweight, not the whole pipeline.

---

## Layout

```
Paper_Reeyan/
├── Dataset/
│   ├── Cao_2023/           Training/ Test/ + Phi-OTDR_dataset_and_codes-main/
│   └── Tomasov_2024/data/  9 class dirs, .h5 + .json + .npy per recording
├── Experiment/
│   ├── dasfe/              shared library (see below)
│   ├── notebook_specs/     notebook sources as reviewable .py
│   ├── _make_notebooks.py  regenerate .ipynb from specs
│   ├── _run_notebook.py    execute a notebook in-process (validation)
│   └── NN_*.ipynb          the notebook series
└── Results/
    ├── cao/                00_inventory … 07_final
    ├── tomasov/            00_inventory … 07_final
    └── shared/             tables/, figures/, 08_crossdataset/
```

### `dasfe/`

| Module | Responsibility |
|---|---|
| `config.py` | paths, seeds, window geometry, dataset specs, split policy, hyper-parameters |
| `manifests.py` | sample-level manifests + the Cao official-split audit |
| `splits.py` | grouped splitters, guard bands, and the **audits that raise** |
| `preprocess.py` | demean → band-pass → z-score; Cao `.mat` and Tomasov HDF5 loaders |
| `features_{time,freq,tf,spatial}.py` | the four extractors (112 / 90 / 739 / 61) |
| `extract.py` | chunked resumable parallel extraction + shard assembly |
| `fusion.py` | domain concatenation with prefixed names + alignment checks |
| `selection.py` | the 8 selectors, consensus voting, Jaccard |
| `balancing.py` | leak-safe resampling pipelines |
| `models.py` | the 7 classifiers + the DAS_CNN baseline |
| `evaluate.py` | metrics, benchmark loop, grouped CV, Friedman/Nemenyi |

Editing a notebook: change `notebook_specs/nb_*.py`, then `python _make_notebooks.py`.
Editing the `.ipynb` directly works too, but the change will be lost on the next
regeneration.

---

## Reproducing

```bash
python _make_notebooks.py          # regenerate all .ipynb from specs
python _run_notebook.py 02_        # validate one notebook headlessly
python _run_notebook.py 05_ DATASET=\"tomasov\"   # override a config constant
```

Seed is `42` throughout (`config.SEED`); splits are frozen to disk in notebook 02 and
never re-derived downstream.
