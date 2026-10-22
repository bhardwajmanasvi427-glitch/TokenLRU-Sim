#!/usr/bin/env python3
"""Single demonstration: two cyclic processes (300 + 340 pages) in 400 frames, both policies."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"), str(Path(__file__).resolve().parents[1])]
from tolru import SimConfig, run_simulation, dedicated_elapsed
from workloads import build_tiny_demo

ws, delays, cap, meta = build_tiny_demo()
cfg = SimConfig(capacity_pages=cap, seed=0, check_invariants=True)
ded = [dedicated_elapsed(w, cfg) for w in ws]
print(f"{meta['label']}  (synthetic; simulated ticks)")
for pol in ("global_lru", "token_lru"):
    sim = run_simulation(ws, cfg, pol, delays)
    print(f"\n== {pol}")
    for r, d in zip(sim.proc_rows(), ded):
        print(f"  {r['name']:4s} major faults {r['major_faults']:6d}  elapsed {r['elapsed']:8d}  slowdown {r['elapsed']/d:6.2f}")
    s = sim.policy.summary()
    keys = ("token_acquisitions", "protected_skips", "fallback_evictions") if pol == "token_lru" else ()
    if keys:
        print("  token:", {k: s[k] for k in keys}, "events:", sim.policy.events)
