#!/usr/bin/env python3
"""Compare both policies on one workload pair.  python experiments/compare.py bitr_gcc --shortage 0.4 --seed 1"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import run_pair
from workloads import PAIR_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("pair", choices=PAIR_NAMES)
ap.add_argument("--shortage", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale", type=float, default=0.125)
a = ap.parse_args()
out, h = run_pair(a.pair, scale=a.scale, seed=a.seed, shortage=a.shortage, sample=False)
print(f"{a.pair}: capacity {h['capacity']} pages, dedicated ticks {h['dedicated']}")
for pol, rec in out.items():
    s = rec["sim"]
    rows = s.proc_rows()
    print(f"{pol:10s}", " | ".join(f"{r['name']} faults {r['major_faults']:6d} slowdown {r['elapsed']/d:5.2f}"
                                  for r, d in zip(rows, h["dedicated"])))
print("token summary:", out["token_lru"]["sim"].policy.summary())
