"""Create the 20 ten-seed case folders (acquisition-neutral names).

Prerequisite: the system is filled in — run once in _templates/eta:
    python problem.py        # generates _templates/eta/data/{evaluation,reference_law}.npz

Then:  python setup_cases.py
Group i (i=0..9): LHS seed = 101+i, member seed base = 1000*i+1.
Both USLW and US ten-seed campaigns share these folders; results are separated
by name prefix (uslw_seed*/us_seed*), driven by the ACQ variable of the
watchdog / job scripts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ETA_T = ROOT / "_templates" / "eta"
HARD_T = ROOT / "_templates" / "hard"

for req in (ETA_T / "data" / "evaluation.npz", ETA_T / "data" / "reference_law.npz"):
    if not req.exists():
        raise SystemExit(f"missing {req} — fill in problem.py TODOs and run "
                         f"'python problem.py' inside _templates/eta first")

for i in range(10):
    s = 101 + i
    base = 1000 * i + 1
    for template, name in ((ETA_T, f"eta_s{s}"), (HARD_T, f"hard_s{s}")):
        dest = ROOT / name
        if dest.exists():
            raise SystemExit(f"refusing to overwrite existing {dest}")
        shutil.copytree(template, dest)
    print(f"group {i+1}/10: LHS seed {s}, member seed base {base} -> eta_s{s}, hard_s{s}")
print("done")
