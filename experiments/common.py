"""Shared experiment machinery: run one workload pair under both policies."""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tolru import SimConfig, run_simulation, dedicated_elapsed            # noqa: E402
from tolru.metrics import pct_reduction, slowdown                        # noqa: E402
from workloads import build_pair, capacity_for_shortage                  # noqa: E402

POLICIES = ("global_lru", "token_lru")


def base_config(meta, capacity, seed, sample=True, **overrides) -> SimConfig:
    """Experiment configuration (every value is OUR choice; see docs/design.md).

    max_hold / cooldown = hold_factor x the pair's longest dedicated lifetime.  The
    paper says the threshold must be large to avoid needless token transfers.  NOTE:
    hold_factor=2 fires Case 2 in almost every run because CPU sharing alone already
    gives slowdown ~2, so the default is 4; results for 0(off)/2/4/8 are reported in
    the ablation (results/ablation_*.csv).
    """
    tps = meta["ticks_per_paper_second"]
    cfg = SimConfig(
        capacity_pages=capacity, seed=seed,
        sample_interval=int(2 * tps) if sample else 0,   # one sample per 2 paper-seconds
        nap_window=max(1, int(1 * tps)),                  # NAP window = 1 paper-second (paper: 1 s)
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg


def run_pair(pair, *, scale=0.125, seed=0, shortage=0.3, sample=True, policies=POLICIES,
             cfg_overrides=None, hold_factor=4.0):
    """Run `pair` under each policy with identical traces/capacity/I-O assumptions.

    Returns dict(policy -> run record) and a header with dedicated times.
    """
    ws, delays, meta = build_pair(pair, scale=scale, seed=seed)
    cap = capacity_for_shortage(ws, shortage)
    cfg0 = base_config(meta, cap, seed, sample=sample)
    ded = [dedicated_elapsed(w, cfg0) for w in ws]
    # Ideal CPU-sharing floor: both programs with unlimited memory (slowdown with ZERO page faults).
    floor_cfg = dataclasses.replace(cfg0, capacity_pages=sum(w.distinct_pages() for w in ws) + 1,
                                    sample_interval=0)
    floor = run_simulation(ws, floor_cfg, "global_lru", delays)
    floor_sd = [slowdown(r["elapsed"], d) for r, d in zip(floor.proc_rows(), ded)]

    overrides = dict(cfg_overrides or {})
    if "max_hold_ticks" not in overrides and hold_factor:
        overrides["max_hold_ticks"] = int(hold_factor * max(ded))
        overrides.setdefault("unsuitable_cooldown_ticks", int(hold_factor * max(ded)))
    cfg = dataclasses.replace(cfg0, **overrides)

    out = {}
    for pol in policies:
        t0 = time.perf_counter()
        sim = run_simulation(ws, cfg, pol, delays)
        wall = time.perf_counter() - t0
        out[pol] = dict(sim=sim, wall=wall)
    header = dict(pair=pair, seed=seed, shortage=shortage, scale=scale, capacity=cap, dedicated=ded,
                  floor_slowdown=floor_sd, workloads=ws, delays=delays, meta=meta, cfg=cfg)
    return out, header


def flatten(out, header):
    """Turn one run_pair() result into (per-process rows, per-system rows)."""
    proc_rows, sys_rows = [], []
    base_pf = {r["pid"]: r["major_faults"] for r in out["global_lru"]["sim"].proc_rows()} \
        if "global_lru" in out else {}
    for pol, rec in out.items():
        sim = rec["sim"]
        sysr = sim.system_row()
        summ = sim.policy.summary()
        sys_rows.append(dict(pair=header["pair"], seed=header["seed"], shortage=header["shortage"],
                             capacity_pages=header["capacity"], policy=pol, wall_seconds=round(rec["wall"], 3),
                             **{k: v for k, v in sysr.items() if k != "completion_order"},
                             completion_order=" > ".join(sim.procs[i].name for i in sysr["completion_order"]),
                             **summ,
                             token_hold_ticks_total=sum(h[2] - h[1] for h in getattr(sim.policy, "holds", [])),
                             false_lru_cross_total=sum(p.false_lru_cross for p in sim.procs),
                             false_lru_self_total=sum(p.false_lru_self for p in sim.procs)))
        for r, d, fl in zip(sim.proc_rows(), header["dedicated"], header["floor_slowdown"]):
            proc_rows.append(dict(pair=header["pair"], seed=header["seed"], shortage=header["shortage"],
                                  capacity_pages=header["capacity"], policy=pol, **r,
                                  dedicated=d, slowdown=slowdown(r["elapsed"], d),
                                  cpu_sharing_floor_slowdown=fl))
    return proc_rows, sys_rows
