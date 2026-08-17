"""Regenerate the notebook series from the specs in `notebook_specs/`.

Each spec is a plain .py file containing a module-level list called CELLS of
(kind, source) tuples, where kind is 'md' or 'code'.  Keeping notebook content
in .py form makes it reviewable in a diff; run this script to materialise the
.ipynb files.

    python _make_notebooks.py            # rebuild all
    python _make_notebooks.py 05 06      # rebuild selected
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
SPEC_DIR = HERE / "notebook_specs"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}


def build(cells: list[tuple[str, str]]) -> dict:
    out = []
    for kind, src in cells:
        lines = src.strip("\n").splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "execution_count": None,
                        "metadata": {}, "outputs": [], "source": lines})
    return {"cells": out, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}


def load_spec(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> None:
    wanted = set(argv)
    for spec_path in sorted(SPEC_DIR.glob("nb_*.py")):
        stem = spec_path.stem[3:]                      # nb_00_setup -> 00_setup
        if wanted and not any(stem.startswith(w) for w in wanted):
            continue
        mod = load_spec(spec_path)
        nb_path = HERE / f"{stem}.ipynb"
        nb_path.write_text(json.dumps(build(mod.CELLS), indent=1), encoding="utf-8")
        print(f"wrote {nb_path.name}  ({len(mod.CELLS)} cells)")


if __name__ == "__main__":
    main(sys.argv[1:])
