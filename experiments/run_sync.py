#!/usr/bin/env python3
"""Evaluate the synchronisation-aware token priority (paper Sec. 5.3) on the BSP scenario.

    python experiments/run_sync.py [--seeds 6] [--out results]

Variants (identical workloads, capacity, I/O model):
  lru        global LRU baseline
  random     token-ordered LRU, random assignment                (paper's "initial" policy)
  sync       token-ordered LRU, sync-aware priority              (paper Sec. 5.3)
  sync+xfer  sync priority + hand-over of a held token to a waited-on process
             (OUR EXTENSION, not in the paper)
Token-handling regimes (the priority rule only acts when the token is being (re)assigned):
  retain         paper default: token kept until the holder exits (or a forced release; none configured)
  hold250k/100k  Case-2 forced release after 250k / 100k simulated ticks (+ equal unsuitable cooldown)
  per-fault      release_on_resolution=True: token released when the holder leaves the swapping state
                 (an option of this simulator; NOT what the paper describes)
Metrics: BSP application completion time relative to a run with unlimited memory ("app slowdown"; completion = the
later of the two ranks), total barrier wait of the ranks, hog completion, total major faults, token statistics.
"""
from __future__ import annotations

import argparse
import dataclasses
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT                                           # noqa: E402
from tolru import SimConfig, run_simulation                       # noqa: E402
from tolru.metrics import write_csv                                # noqa: E402
from workloads import capacity_for_shortage                        # noqa: E402
from workloads.bsp import build_bsp_scenario                       # noqa: E402

SHORTAGES = [0.3, 0.4, 0.5]
REGIMES = {
    "retain": dict(),
    "hold250k": dict(max_hold_ticks=250_000, unsuitable_cooldown_ticks=250_000),
    "hold100k": dict(max_hold_ticks=100_000, unsuitable_cooldown_ticks=100_000),
    "per-fault": dict(release_on_resolution=True),
}
VARIANTS = {
    "lru": None,
    "random": dict(sync_priority=False),
    "sync": dict(sync_priority=True),
    "sync+xfer": dict(sync_priority=True, sync_transfer=True),
}


def one(spec):
    seed, sh, regime = spec
    ws, delays, meta = build_bsp_scenario(seed)
    cap = capacity_for_shortage(ws, sh)
    base = SimConfig(capacity_pages=cap, seed=seed, **REGIMES[regime])
    floor_cfg = dataclasses.replace(base, capacity_pages=sum(w.peak_mad() for w in ws) + 1,
                                    max_hold_ticks=None, release_on_resolution=False)
    fr = run_simulation(ws, floor_cfg, "global_lru", delays).proc_rows()
    app_floor = max(fr[0]["finish"], fr[1]["finish"])
    rows = []
    for v, kw in VARIANTS.items():
        pol = "global_lru" if kw is None else "token_lru"
        sim = run_simulation(ws, dataclasses.replace(base, **(kw or {})), pol, delays)
        r = sim.proc_rows()
        s = sim.policy.summary()
        app = max(r[0]["finish"], r[1]["finish"])
        rows.append(dict(seed=seed, shortage=sh, regime=regime, variant=v, capacity_pages=cap,
                         app_finish=app, app_floor=app_floor, app_slowdown=app / app_floor,
                         rank_wait_ticks=r[0]["sync_wait_ticks"] + r[1]["sync_wait_ticks"],
                         hog_finish=r[2]["finish"], total_major_faults=sum(x["major_faults"] for x in r),
                         token_acquisitions=s.get("token_acquisitions"),
                         sync_priority_assignments=s.get("sync_priority_assignments"),
                         sync_transfers=s.get("sync_transfers"),
                         forced_release_timeout=s.get("forced_release_timeout")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()
    specs = [(s, sh, rg) for s in range(a.seeds) for sh in SHORTAGES for rg in REGIMES]
    with mp.Pool(a.workers) as pool:
        res = pool.map(one, specs, chunksize=1)
    rows = [r for rr in res for r in rr]
    write_csv(Path(a.out) / "sync_runs.csv", rows)
    import pandas as pd
    d = pd.DataFrame(rows)
    g = d.groupby(["regime", "shortage", "variant"]).agg(
        app_slowdown_mean=("app_slowdown", "mean"), app_slowdown_std=("app_slowdown", "std"),
        rank_wait_mean=("rank_wait_ticks", "mean"), faults_mean=("total_major_faults", "mean"),
        acquisitions_mean=("token_acquisitions", "mean"), prio_assignments_mean=("sync_priority_assignments", "mean"),
        transfers_mean=("sync_transfers", "mean"), n=("seed", "nunique")).reset_index()
    # per-seed paired comparison sync vs random
    p = d.pivot_table(index=["regime", "shortage", "seed"], columns="variant", values="app_slowdown").reset_index()
    pc = p.groupby(["regime", "shortage"]).apply(lambda x: pd.Series(dict(
        seeds_sync_better_than_random=int((x["sync"] < x["random"] - 1e-9).sum()),
        seeds_sync_worse_than_random=int((x["sync"] > x["random"] + 1e-9).sum()),
        seeds_identical=int(((x["sync"] - x["random"]).abs() <= 1e-9).sum()),
        seeds_xfer_better_than_sync=int((x["sync+xfer"] < x["sync"] - 1e-9).sum()),
        n=len(x)))).reset_index()
    g = g.merge(pc, on=["regime", "shortage"], how="left", suffixes=("", "_pairs")).drop(columns=["n_pairs"])
    write_csv(Path(a.out) / "sync_summary.csv", g.to_dict("records"))
    write_csv(Path(a.out) / "sync_paired.csv", pc.to_dict("records"))
    print(pc.to_string())
    print(g[g.variant != "lru"][["regime", "shortage", "variant", "app_slowdown_mean", "prio_assignments_mean"]].round(2).to_string())


if __name__ == "__main__":
    main()
