"""Create the 20 US-acquisition case folders under 10random_seeds_TOY_2D_US.

Group i (i = 0..9, ALL ten groups run this time - no pre-existing US results):
  LHS seed  = 101 + i
  seed base = 1000*i + 1   (round k members: base+2k, base+2k+1)
  -> eta_us_s{101+i}   (run 5 lambdas each: us / us_lam{...} result names)
  -> hard_us_s{101+i}

Usage:  python setup_cases.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ETA_T = ROOT / "_templates" / "eta"
HARD_T = ROOT / "_templates" / "hard"

for i in range(10):
    s = 101 + i
    base = 1000 * i + 1
    for template, name in ((ETA_T, f"eta_us_s{s}"), (HARD_T, f"hard_us_s{s}")):
        dest = ROOT / name
        if dest.exists():
            raise SystemExit(f"refusing to overwrite existing {dest}")
        shutil.copytree(template, dest)
    print(f"group {i+1}/10: LHS seed {s}, member seed base {base} "
          f"-> eta_us_s{s}, hard_us_s{s}")
print("done")
