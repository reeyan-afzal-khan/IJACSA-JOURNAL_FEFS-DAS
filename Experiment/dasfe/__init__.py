"""dasfe - multi-domain feature engineering for DAS event classification.

Shared library behind the notebook series in `Experiment/`.  Import it from a
notebook with:

    import sys; sys.path.insert(0, r"/mnt/b6bdcd1c-136a-435b-aeed-0e1b31c32749/Paper_Reeyan/Experiment")
    from dasfe import config as C, manifests, splits, extract

Design rule: every notebook stage reads its input from `Results/<dataset>/` and
writes its output there.  Nothing is passed between notebooks in memory, so any
stage can be re-run independently and the whole study is reproducible from disk.
"""
from __future__ import annotations

from . import config
from . import features_freq
from . import features_spatial
from . import features_tf
from . import features_time

__all__ = [
    "config", "manifests", "splits", "preprocess", "extract", "fusion",
    "selection", "balancing", "models", "evaluate",
    "features_time", "features_freq", "features_tf", "features_spatial",
]

DOMAINS = {
    "time": features_time,
    "freq": features_freq,
    "tf": features_tf,
    "spatial": features_spatial,
}

DOMAIN_PREFIX = {"time": "TIME", "freq": "FREQ", "tf": "TF", "spatial": "SPAT"}

FEATURE_COUNTS = {name: mod.N_FEATURES for name, mod in DOMAINS.items()}
TOTAL_FEATURES = sum(FEATURE_COUNTS.values())
assert TOTAL_FEATURES == 1002, f"multi-domain total must be 1002, got {TOTAL_FEATURES}"

__version__ = "1.0.0"
