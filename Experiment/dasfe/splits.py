"""Leakage-safe train/validation/test splitting.

Two independent guarantees are enforced and then *verified*:

1. **Group disjointness** - a split group (Cao: recording session, Tomasov:
   spatio-temporal zone within a recording) lives in exactly one split.
2. **Raw-sample disjointness** - for Tomasov, where analysis windows overlap by
   75%, no raw sample of the interrogator stream may be read by windows in two
   different splits.  Guard bands around zone boundaries make this true and
   `assert_no_sample_overlap` proves it.

The audit functions are meant to be *run*, not trusted.  Every split notebook
calls them and fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config as C


# ==========================================================================
# Core grouped split
# ==========================================================================
def grouped_three_way_split(
    df: pd.DataFrame,
    group_key: str,
    label_col: str = "label",
    fractions: tuple = C.SPLIT_FRACTIONS,
    seed: int = C.SEED,
) -> pd.Series:
    """Assign each row to train/val/test, cutting only between whole groups.

    Uses `StratifiedGroupKFold` with 10 folds so class proportions are held as
    close as the group structure allows, then merges folds into 8/1/1.  When a
    class has fewer groups than folds the split is still valid (groups stay
    disjoint) but the class simply cannot appear in every split - the caller is
    warned by `split_report`.
    """
    n_train, n_val, n_test = [int(round(f * 10)) for f in fractions]
    if n_train + n_val + n_test != 10:
        raise ValueError(f"fractions {fractions} must be multiples of 0.1 summing to 1.0")

    y = df[label_col].to_numpy()
    groups = df[group_key].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_of_row = np.empty(len(df), dtype=int)
    for fold, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
        fold_of_row[test_idx] = fold

    assignment = np.where(
        fold_of_row < n_train, "train",
        np.where(fold_of_row < n_train + n_val, "val", "test"),
    )
    return pd.Series(assignment, index=df.index, name="split")


def blocked_split_by_recording(
    df: pd.DataFrame,
    recording_col: str = "recording_id",
    block_col: str = "block",
    n_segments: int = 10,
    fractions: tuple = C.SPLIT_FRACTIONS,
    seed: int = C.SEED,
) -> pd.DataFrame:
    """Contiguous-in-time ("blocked") split, applied inside every recording.

    `StratifiedGroupKFold` balances the *number* of groups per fold, which is the
    wrong objective here: a Tomasov zone can hold anything from 30 to 8,000
    windows, so equal group counts give wildly unequal split sizes, and a class
    represented by a single short recording can miss a split entirely.

    Instead each recording's timeline is cut into `n_segments` contiguous
    equal-*mass* segments (each holding ~1/n of that recording's windows), and
    whole segments are assigned to splits.  Because every recording contributes
    to all three splits, every class is guaranteed to appear in all three, and
    the 80/10/10 proportions hold per class rather than only in aggregate.

    Which segments become val/test is rotated per recording so the held-out data
    is not always the tail of the timeline.

    Returns a frame with `segment`, `group_id` and `split` columns.
    """
    n_train, n_val, n_test = [int(round(f * n_segments)) for f in fractions]
    if n_train + n_val + n_test != n_segments:
        raise ValueError(f"fractions {fractions} do not divide {n_segments} segments evenly")

    rng = np.random.default_rng(seed)
    seg_out = np.empty(len(df), dtype=int)
    split_out = np.empty(len(df), dtype=object)
    positions = np.arange(len(df))
    rec_values = df[recording_col].to_numpy()
    blocks = df[block_col].to_numpy()

    for r, rec in enumerate(pd.unique(rec_values)):
        pos = positions[rec_values == rec]
        b = blocks[pos]

        # Equal-mass contiguous segmentation: cut the block axis at the
        # quantiles of the window distribution, not at equal block spacing.
        order = np.argsort(b, kind="stable")
        ranks = np.empty(pos.size, dtype=float)
        ranks[order] = np.arange(pos.size)
        seg = np.minimum((ranks / pos.size * n_segments).astype(int), n_segments - 1)
        # Force segments to respect block boundaries so a block is never split.
        seg_of_block = pd.Series(seg).groupby(b).transform("min").to_numpy()
        seg_out[pos] = seg_of_block

        # Rotate which segments are held out, deterministically per recording.
        offset = int(rng.integers(0, n_segments))
        val_segs = {(offset + i) % n_segments for i in range(n_val)}
        test_segs = {(offset + n_val + i) % n_segments for i in range(n_test)}
        split_out[pos] = np.where(
            np.isin(seg_of_block, list(test_segs)), "test",
            np.where(np.isin(seg_of_block, list(val_segs)), "val", "train"),
        )

    out = pd.DataFrame(
        {
            "segment": seg_out,
            "split": pd.Series(split_out, index=df.index).astype(str),
        },
        index=df.index,
    )
    out["group_id"] = df[recording_col].astype(str) + "|s" + out["segment"].astype(str)
    return out


def leave_one_group_out_folds(df: pd.DataFrame, group_key: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Yield (held_out_group, train_idx, test_idx) for a LOGO robustness study."""
    folds = []
    groups = df[group_key].to_numpy()
    for g in pd.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        folds.append((str(g), train, test))
    return folds


# ==========================================================================
# Tomasov guard bands
# ==========================================================================
def apply_temporal_guard(
    df: pd.DataFrame,
    split_col: str = "split",
    recording_col: str = "recording_id",
    block_col: str = "block",
    guard_blocks: int = C.GUARD_BLOCKS,
) -> pd.Series:
    """Flag windows that must be dropped to keep splits sample-disjoint.

    A window starting at block *b* physically reads blocks
    `b .. b + WIN_LEN/WIN_HOP - 1`.  Any window whose read span touches a block
    that another split also reads is dropped.  We implement this by rejecting
    windows within `guard_blocks` of a block owned by a different split.

    Returns a boolean Series: True = keep.
    """
    blocks_per_window = C.WIN_LEN // C.WIN_HOP
    keep = np.ones(len(df), dtype=bool)

    split_codes, split_levels = pd.factorize(df[split_col].to_numpy())
    n_splits = len(split_levels)
    rec_codes = df[recording_col].to_numpy()
    all_blocks = df[block_col].to_numpy().astype(np.int64)
    positions = np.arange(len(df))

    for rec in pd.unique(rec_codes):
        pos = positions[rec_codes == rec]
        b = all_blocks[pos]
        codes = split_codes[pos]
        n_blocks = int(b.max()) + blocks_per_window + guard_blocks + 1

        # owner[s, k] = split s reads block k
        owner = np.zeros((n_splits, n_blocks), dtype=bool)
        for off in range(blocks_per_window):
            owner[codes, b + off] = True

        # Dilate each split's footprint by the guard band.
        dilated = owner.copy()
        for shift in range(1, guard_blocks + 1):
            dilated[:, shift:] |= owner[:, :-shift]
            dilated[:, :-shift] |= owner[:, shift:]

        # Drop a window if any block it reads sits inside the dilated
        # footprint of a *different* split.
        for si in range(n_splits):
            rows = np.flatnonzero(codes == si)
            if rows.size == 0:
                continue
            others = dilated[np.arange(n_splits) != si].any(axis=0)
            bb = b[rows]
            conflict = np.zeros(rows.size, dtype=bool)
            for off in range(blocks_per_window):
                conflict |= others[bb + off]
            keep[pos[rows]] = ~conflict

    return pd.Series(keep, index=df.index, name="keep")


# ==========================================================================
# Audits - these raise
# ==========================================================================
def assert_group_disjoint(df: pd.DataFrame, group_key: str, split_col: str = "split") -> None:
    counts = df.groupby(group_key)[split_col].nunique()
    bad = counts[counts > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} group(s) of '{group_key}' appear in more than one split, "
            f"e.g. {list(bad.index[:5])}"
        )


def assert_no_sample_overlap(
    df: pd.DataFrame,
    split_col: str = "split",
    recording_col: str = "recording_id",
    block_col: str = "block",
) -> None:
    """Prove that no raw interrogator sample is read by two different splits."""
    blocks_per_window = C.WIN_LEN // C.WIN_HOP
    offenders = []
    for rec, sub in df.groupby(recording_col, sort=False):
        n_blocks = int(sub[block_col].max()) + blocks_per_window + 1
        seen = {}
        for split, s2 in sub.groupby(split_col, sort=False):
            cov = np.zeros(n_blocks, dtype=bool)
            b = s2[block_col].to_numpy()
            for off in range(blocks_per_window):
                cov[b + off] = True
            seen[split] = cov
        names = list(seen)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                shared = int((seen[names[i]] & seen[names[j]]).sum())
                if shared:
                    offenders.append((rec, names[i], names[j], shared))
    if offenders:
        raise AssertionError(
            f"raw-sample overlap between splits in {len(offenders)} recording/pair(s): "
            f"{offenders[:5]}"
        )


def assert_no_duplicate_ids(df: pd.DataFrame, id_col: str = "sample_id") -> None:
    dup = df[id_col].duplicated().sum()
    if dup:
        raise AssertionError(f"{dup} duplicated {id_col} values")


def split_report(df: pd.DataFrame, group_key: str, split_col: str = "split") -> pd.DataFrame:
    """Class x split window counts plus group counts, for the paper's tables."""
    pivot = pd.crosstab(df["label"], df[split_col])
    groups = df.groupby(["label", split_col])[group_key].nunique().unstack(fill_value=0)
    groups.columns = [f"{c}_groups" for c in groups.columns]
    out = pivot.join(groups)
    out.loc["TOTAL"] = out.sum()
    return out


def leakage_scorecard(df: pd.DataFrame, group_key: str, split_col: str = "split", **kw) -> pd.DataFrame:
    """Run every audit and return a pass/fail table instead of raising."""
    checks = []

    def run(name, fn):
        try:
            fn()
            checks.append({"check": name, "status": "PASS", "detail": ""})
        except AssertionError as exc:
            checks.append({"check": name, "status": "FAIL", "detail": str(exc)[:300]})

    run(f"groups of '{group_key}' disjoint across splits",
        lambda: assert_group_disjoint(df, group_key, split_col))
    run("no duplicate sample_id", lambda: assert_no_duplicate_ids(df))
    if "block" in df.columns and "recording_id" in df.columns:
        run("no raw-sample overlap between splits",
            lambda: assert_no_sample_overlap(df, split_col, **kw))
    # Every class must be present in every split, otherwise metrics are undefined.
    missing = []
    for split in df[split_col].unique():
        have = set(df.loc[df[split_col] == split, "label"])
        gap = set(df["label"].unique()) - have
        if gap:
            missing.append(f"{split}: {sorted(gap)}")
    checks.append(
        {
            "check": "all classes present in all splits",
            "status": "PASS" if not missing else "FAIL",
            "detail": "; ".join(missing),
        }
    )
    return pd.DataFrame(checks)
