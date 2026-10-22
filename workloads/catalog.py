"""Workload catalog: the paper's five interacting pairs and the Section 5.4 example.

ALL WORKLOADS ARE SYNTHETIC.  Their parameters are read off paper Table 1 and the
dedicated-environment graphs (Figs. 1-5) and then SCALED DOWN so Python can run
them.  They are not SPEC2000 executions and their results must not be presented
as reproductions of the original hardware measurements.

Scaling
-------
`scale` shrinks both page counts and reference counts uniformly (default 1/8).
One paper-second corresponds to 10_000 * scale references (= ticks, cpu_cost=1).
Page-count numbers below are "full scale" (4 KiB pages as in the paper); a
1 MB region is 256 pages.
"""
from __future__ import annotations

import numpy as np

from tolru.process import Workload
from . import generators as g

REFS_PER_SECOND_FULL = 10_000       # our choice: one page-touch = 100 microseconds of paper time

PAIR_NAMES = ["gzip_vortex3", "bitr_gcc", "vortex3_gcc", "vortex1_vortex3", "lu1_lu2"]
PAIR_LABELS = {
    "gzip_vortex3": "gzip / vortex3",
    "bitr_gcc": "bit-r / gcc",
    "vortex3_gcc": "vortex3 / gcc",
    "vortex1_vortex3": "vortex1 / vortex3",
    "lu1_lu2": "LU-1 / LU-2",
}
# paper Section 4.2 groups
PAIR_GROUPS = {"gzip_vortex3": 1, "bitr_gcc": 2, "vortex3_gcc": 3, "vortex1_vortex3": 6, "lu1_lu2": 7}


class _Scale:
    def __init__(self, scale):
        self.scale = scale
        self.tps = REFS_PER_SECOND_FULL * scale       # ticks per paper-second

    def P(self, pages):
        return max(1, int(round(pages * self.scale)))

    def R(self, seconds):
        return max(1, int(round(seconds * self.tps)))


# ----------------------------------------------------------------------------- programs
def make_gzip(rng, s: _Scale, name="gzip*"):
    # Table 1: 197.4 MB, 248.7 s.  Fig. 2: MAD ~49.5k pages from the start; hot set ~1k pages; sweep events
    # (NAP spikes) at ~42/50/90/97/148/150/198/200/245 s of sizes ~17k/31k/17.5k/20k/17k/31.5k/14k/31k/17.5k.
    ev = [(42, 17_000), (50, 31_000), (90, 17_500), (97, 20_000), (148, 17_000), (150, 31_500),
          (198, 14_000), (200, 31_000), (245, 17_500)]
    return g.quick_hot_with_sweeps(
        name, rng, alloc_pages=s.P(49_500), hot_pages=s.P(1_000),
        sweeps=[(s.R(t), s.P(n)) for t, n in ev], total_refs=s.R(248.7),
        description="SYNTHETIC gzip-like (type 1): up-front allocation, small hot set, sweep events read off Fig. 2")


def make_bitr(rng, s: _Scale, name="bit-r*"):
    # Table 1: 131.3 MB (~33.6k pages), 326.1 s.  Fig. 3: RSS ~33k pages for 0-168 s, then ~1.8k pages.
    # NOTE: Figs. 3/7 plot bit-r MAD at ~65.5k, twice Table 1's 131.3 MB; we follow Table 1 / the RSS.
    return g.quick_two_phase_cyclic(
        name, rng, ws_pages=s.P(33_600), refs_phase1=s.R(168), hot_pages=s.P(1_800),
        refs_phase2=s.R(158.1),
        description="SYNTHETIC bit-reversal-like (type 1): cyclic passes (0-168 s) then small hot set")


def make_vortex(rng, s: _Scale, which, name=None):
    # Table 1: vortex1 115 MB, 342.3 s; vortex3 131.2 MB, 398.0 s.  Fig. 3: stair-climbing MAD to ~29k (vortex1),
    # RSS/working set ~57-60% of MAD late in the run.
    peak, secs, stages = (29_400, 342.3, 13) if which == 1 else (33_600, 398.0, 14)
    return g.gradual(
        name or f"vortex{which}*", rng, peak_pages=s.P(peak), total_refs=s.R(secs),
        stages=stages, ws_frac=0.6,
        description=f"SYNTHETIC vortex{which}-like (type 2): staircase allocation, WS ~60% of peak MAD")


def make_gcc(rng, s: _Scale, name="gcc*"):
    # Table 1: 145 MB max, 218.7 s.  Fig. 1: ~3-4k pages to 75 s; spike ~33.5k at 75-85 s; ~12.5k 85-108 s;
    # ~5k 108-182 s; spike ~36.5k 182-200 s; ~13k 200-218.7 s.  Memory is released between phases.
    phases = [(3_500, 40), (4_000, 35), (33_500, 10), (12_500, 23), (5_000, 74), (36_500, 18), (13_000, 18.7)]
    return g.irregular_phases(
        name, rng, [(s.P(ws), s.R(sec)) for ws, sec in phases],
        description="SYNTHETIC gcc-like (type 3): two large spikes in demand, memory released between phases")


def make_lu(rng, s: _Scale, name="LU*"):
    # Table 1: 161.2 MB, 98.0 s.  Fig. 5: MAD ~40.3k pages; matrix released and re-allocated at ~0/21/41/61/81 s
    # (five ~20 s iterations); NAP ~2.2k pages/s between spikes => sequential traversal of the whole matrix.
    cycles = 5
    return g.lu_like(
        name, rng, alloc_pages=s.P(40_300), cycles=cycles,
        cycle_refs=s.R(98.0) // cycles,
        description="SYNTHETIC LU-like (type 3): matrix re-allocated every ~20 s, traversed sequentially in between")


# ----------------------------------------------------------------------------- pairs
def build_pair(pair: str, scale: float = 0.125, seed: int = 0):
    """Return (workloads, start_delays_ticks, meta) for one interacting pair.

    The trace RNG is seeded from (seed, pair) so both policies see identical traces.
    Start delays (gzip, and gcc in vortex3/gcc, start ~250 s late) are read from
    Figs. 6, 8, 12, 14 and are approximate.
    """
    if pair not in PAIR_NAMES:
        raise ValueError(f"unknown pair {pair!r}; choose from {PAIR_NAMES}")
    s = _Scale(scale)
    base = np.random.SeedSequence([seed, PAIR_NAMES.index(pair)])
    r1, r2 = [np.random.default_rng(x) for x in base.spawn(2)]
    if pair == "gzip_vortex3":
        ws = [make_gzip(r1, s, "gzip*"), make_vortex(r2, s, 3, "vortex3*")]
        delays = [s.R(250), 0]
    elif pair == "bitr_gcc":
        ws = [make_bitr(r1, s, "bit-r*"), make_gcc(r2, s, "gcc*")]
        delays = [0, 0]
    elif pair == "vortex3_gcc":
        ws = [make_vortex(r1, s, 3, "vortex3*"), make_gcc(r2, s, "gcc*")]
        delays = [0, s.R(250)]
    elif pair == "vortex1_vortex3":
        ws = [make_vortex(r1, s, 1, "vortex1*"), make_vortex(r2, s, 3, "vortex3*")]
        delays = [0, 0]
    else:  # lu1_lu2: same program/input, independent micro-level ordering
        ws = [make_lu(r1, s, "LU-1*"), make_lu(r2, s, "LU-2*")]
        delays = [0, 0]
    meta = dict(pair=pair, label=PAIR_LABELS[pair], group=PAIR_GROUPS[pair], scale=scale, seed=seed,
                ticks_per_paper_second=s.tps, synthetic=True,
                sum_peak_mad=sum(w.peak_mad() for w in ws))
    return ws, delays, meta


def capacity_for_shortage(workloads, shortage: float) -> int:
    """Memory = (1 - shortage) * sum of the programs' PEAK memory demands.

    This is the paper's definition of shortage ratio (Section 4.2 footnote 5):
    based on peak demands.  The paper's 20-50% range is a property of ITS machine
    settings; our `shortage` is an explicit experiment parameter.
    """
    if not (0.0 <= shortage < 1.0):
        raise ValueError("shortage must be in [0, 1)")
    return max(1, int(round((1.0 - shortage) * sum(w.peak_mad() for w in workloads))))


# ----------------------------------------------------------------------------- Section 5.4
def build_section54(scale: float = 0.25, loops: int = 1000):
    """Two instances of the cyclic program: 53 MB and 58 MB arrays, 60 MB of user memory.

    Page counts: MB * 256 (4 KiB pages).  Returns (workloads, delays, capacity_pages, meta).
    """
    small_pages = max(1, int(round(53 * 256 * scale)))
    large_pages = max(1, int(round(58 * 256 * scale)))
    cap = max(1, int(round(60 * 256 * scale)))
    w_small = g.cyclic_shrinking("small-53MB*", small_pages, loops=loops,
                                 description="SYNTHETIC Section 5.4 program, 53 MB array")
    w_large = g.cyclic_shrinking("large-58MB*", large_pages, loops=loops,
                                 description="SYNTHETIC Section 5.4 program, 58 MB array")
    meta = dict(pair="section54", label="Sec. 5.4: 53 MB + 58 MB, 60 MB memory", scale=scale,
                small_pages=small_pages, large_pages=large_pages, capacity_pages=cap,
                shortage_vs_sum=1 - cap / (small_pages + large_pages), synthetic=True)
    return [w_small, w_large], [0, 0], cap, meta


def build_tiny_demo(seed: int = 0):
    """Very small two-process scenario for the single demonstration (runs in well under a second)."""
    rng = np.random.default_rng(seed)
    a = g.cyclic_shrinking("A*", 300, loops=60, description="tiny cyclic demo process A")
    b = g.cyclic_shrinking("B*", 340, loops=60, description="tiny cyclic demo process B")
    return [a, b], [0, 0], 400, dict(pair="tiny_demo", label="tiny demo (300 + 340 pages, 400 frames)",
                                     synthetic=True)
