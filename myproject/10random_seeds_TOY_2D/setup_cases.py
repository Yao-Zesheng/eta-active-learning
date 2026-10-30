"""Create the 18 remaining per-seed case folders under 10random_seeds_TOY_2D.

Seed plan (group 1 = the already-completed original runs, copied as
eta_uslw_s101 / hard_uslw_s101 and NOT touched here):

  group i (i = 1..9):  LHS seed  = 101 + i
                       seed base = 1000*i + 1   (round k members: base+2k, base+2k+1)
  -> eta_uslw_s{102..110}   from _templates/eta   (run 5 lambdas each on HPC)
  -> hard_uslw_s{102..110}  from _templates/hard

Usage:  python setup_cases.py [--only i,j,...]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ETA_T = ROOT / "_templates" / "eta"
HARD_T = ROOT / "_templates" / "hard"


def make_case(template: Path, dest: Path):
    if dest.exists():
        raise SystemExit(f"refusing to overwrite existing {dest}")
    shutil.copytree(template, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="comma-separated group indices 1..9")
    args = parser.parse_args()
    groups = [int(g) for g in args.only.split(",")] if args.only else list(range(1, 10))

    for i in groups:
        s = 101 + i
        base = 1000 * i + 1
        eta = ROOT / f"eta_uslw_s{s}"
        hard = ROOT / f"hard_uslw_s{s}"
        make_case(ETA_T, eta)
        make_case(HARD_T, hard)
        print(f"group {i+1}/10: LHS seed {s}, member seed base {base} "
              f"-> {eta.name}, {hard.name}")
    print("done")


if __name__ == "__main__":
    main()
