"""Bulk-synchronous (BSP) scenario for the synchronisation-aware token priority (paper Sec. 5.3).

One node of a parallel machine runs
  * two ranks of a BSP application (sync group "bsp"): every superstep each rank makes `passes` passes over its
    working set (random page order) and then waits at a barrier for the other rank;
  * an independent memory hog (no barriers);
  * a short independent job that finishes early (so the token is released mid-run and has to be re-assigned).
ALL SYNTHETIC; the paper evaluates no such scenario (its experiments are on independent programs), so this is OUR
design for exercising the rule, not a reproduction of a paper experiment.  Page counts follow the 1/8-scale
workloads used elsewhere.
"""
from __future__ import annotations

import numpy as np

from tolru.process import Workload


def _rank(name, rng, ws, supersteps, passes, group="bsp"):
    parts, barriers, n = [], [], 0
    for s in range(supersteps):
        for _ in range(passes):
            parts.append(rng.permutation(ws).astype(np.int32))
            n += ws
        if s < supersteps - 1:
            barriers.append(n)
    return Workload(name, np.concatenate(parts), [(0, ws)], "bsp",
                    f"SYNTHETIC BSP rank: {supersteps} supersteps x {passes} passes over {ws} pages",
                    True, group, barriers)


def _cyclic(name, ws, passes, kind="custom", desc=""):
    return Workload(name, np.tile(np.arange(ws, dtype=np.int32), passes), [(0, ws)], kind, desc, True)


def build_bsp_scenario(seed: int = 0, ws0: int = 1500, ws1: int = 2000, supersteps: int = 10, passes: int = 3,
                       hog_ws: int = 2400, hog_passes: int = 30, short_ws: int = 800, short_passes: int = 8):
    """Returns (workloads, start_delays, meta).  Process order: rank0, rank1, hog, short job."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    ws = [_rank("rank0*", rng, ws0, supersteps, passes),
          _rank("rank1*", rng, ws1, supersteps, passes),
          _cyclic("hog*", hog_ws, hog_passes, "type1", "SYNTHETIC independent memory hog (cyclic)"),
          _cyclic("short*", short_ws, short_passes, "type1", "SYNTHETIC short independent job")]
    meta = dict(pair="bsp_node", label="BSP app (2 ranks) + hog + short job", seed=seed, synthetic=True,
                sum_peak_mad=sum(w.peak_mad() for w in ws))
    return ws, [0, 0, 0, 0], meta
