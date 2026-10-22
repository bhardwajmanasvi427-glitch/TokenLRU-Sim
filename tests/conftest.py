import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pytest

from tolru import SimConfig, Simulator, Workload, make_policy


def wl(name, trace, mad=None):
    return Workload(name, list(trace), [(0, mad if mad is not None else len(set(trace)))])


def make_sim(capacity, traces, policy="token_lru", delays=None, **cfgkw):
    """Build (but do not run) a simulator so tests can set up memory state by hand."""
    cfg = SimConfig(capacity_pages=capacity, **cfgkw)
    ws = [wl(f"p{i}", t) for i, t in enumerate(traces)]
    return Simulator(ws, cfg, make_policy(policy, cfg), delays)


def seed_resident(sim, pid, pages_ts):
    """Make pages resident for process pid with given {page: last-access-ts} (LRU order = given order)."""
    p = sim.procs[pid]
    for page, ts in pages_ts:
        sim.mem.install(p, page, ts)
        p.touched.add(page)
    return p
