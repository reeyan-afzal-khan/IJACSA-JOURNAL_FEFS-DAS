"""Execute a notebook's code cells in-process and report the first failure.

Used to validate the notebook series without depending on a working ZMQ kernel.
`display()` is stubbed so notebook-style calls work under a plain interpreter.

    python _run_notebook.py 01_dataset_inventory_and_leakage_audit.ipynb
    python _run_notebook.py 01 02          # prefix match
"""
from __future__ import annotations

import builtins
import re
import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).parent


def _display(*objs):
    import pandas as pd

    for o in objs:
        if isinstance(o, (pd.DataFrame, pd.Series)):
            print(o.to_string()[:4000])
        else:
            print(o)


def run(nb_path: Path, overrides: dict[str, str] | None = None) -> bool:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "display": _display}
    builtins.display = _display
    overrides = overrides or {}

    import matplotlib
    matplotlib.use("Agg")

    print(f"\n{'='*78}\nRUN {nb_path.name} {overrides or ''}\n{'='*78}")
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Rewrite the notebook's top-of-file configuration constants so one
        # notebook can be validated for both datasets without editing it.
        for key, val in overrides.items():
            src = re.sub(rf"^{key}(\s*)= .*$", rf"{key}\1= {val}", src, flags=re.M)
        try:
            exec(compile(src, f"{nb_path.name}[cell {i}]", "exec"), ns)
        except Exception:
            print(f"\n!!! FAILED in cell {i} of {nb_path.name}\n")
            print(src[:1500])
            print("-" * 60)
            traceback.print_exc()
            return False
    print(f"\nOK  {nb_path.name} - all code cells executed")
    return True


def main(argv: list[str]) -> int:
    overrides = {}
    names = []
    for a in argv:
        if "=" in a and a.split("=")[0].isupper():
            k, v = a.split("=", 1)
            overrides[k] = v
        else:
            names.append(a)

    targets = []
    for a in names:
        p = HERE / a
        targets += [p] if p.exists() else sorted(HERE.glob(f"{a}*.ipynb"))
    if not targets:
        targets = sorted(HERE.glob("[0-9][0-9]_*.ipynb"))
    ok = all(run(t, overrides) for t in targets)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
