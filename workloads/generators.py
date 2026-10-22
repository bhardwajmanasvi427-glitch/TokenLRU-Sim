"""Synthetic page-reference trace generators.

IMPORTANT: every trace produced here is SYNTHETIC.  None of them is a real
SPEC2000 (or bit-reversal / LU / ...) execution trace.  They only approximate the
three memory-demand patterns of paper Section 3.3 and the shapes visible in the
dedicated-environment memory-time graphs (Figs. 1-5).

All generators are deterministic functions of (parameters, numpy Generator).
"""
from __future__ import annotations

import numpy as np

from tolru.process import Workload


class TraceBuilder:
    """Accumulates phases into one trace and records the MAD schedule."""

    def __init__(self, name, kind, description):
        self.name, self.kind, self.description = name, kind, description
        self.parts: list[np.ndarray] = []
        self.mad_steps: list[tuple[int, int]] = []
        self.length = 0
        self.hi = 0                      # highest page index touched so far + 1

    def _add(self, arr, mad=None):
        arr = np.asarray(arr, dtype=np.int32)
        if mad is not None:
            self.mad_steps.append((self.length, int(mad)))
        self.parts.append(arr)
        self.length += len(arr)

    # -- phase primitives ---------------------------------------------------------
    def alloc(self, n_pages, mad=None):
        """Sequential first touch of pages [hi, n_pages) (quick allocation burst)."""
        n_pages = int(n_pages)
        if n_pages > self.hi:
            self._add(np.arange(self.hi, n_pages), mad if mad is not None else n_pages)
            self.hi = n_pages
        elif mad is not None:
            self.mad_steps.append((self.length, int(mad)))

    def free(self, lo, hi, mad=None):
        """Release pages [lo, hi) (negative ids in the trace = unmap). Re-allocation re-touches fresh pages."""
        lo, hi = int(lo), int(hi)
        if hi > lo:
            self._add(-(np.arange(lo, hi) + 1), mad)
            if hi >= self.hi:
                self.hi = lo

    def sweep(self, lo, hi, passes=1, mad=None):
        lo, hi = int(lo), int(hi)
        one = np.arange(lo, hi)
        self._add(np.tile(one, passes), mad)
        self.hi = max(self.hi, hi)

    def cyclic(self, lo, hi, refs, mad=None):
        """Cycle sequentially through [lo, hi) for `refs` references (LRU-pathological pattern)."""
        lo, hi, refs = int(lo), int(hi), int(refs)
        span = max(1, hi - lo)
        self._add(lo + (np.arange(refs) % span), mad)
        self.hi = max(self.hi, hi if refs >= span else lo + refs)

    def random(self, rng, lo, hi, refs, mad=None, alloc_first=True):
        """Uniform random references over [lo, hi); optionally first-touch new pages sequentially."""
        lo, hi, refs = int(lo), int(hi), int(refs)
        if alloc_first and hi > self.hi:
            self.alloc(hi, mad)                 # first-touch burst; also records the MAD step
            mad = None
        self._add(rng.integers(lo, max(lo + 1, hi), size=max(0, refs)), mad)
        self.hi = max(self.hi, hi)

    def pick(self, rng, pages, refs, mad=None):
        """Random references drawn from an explicit page array."""
        pages = np.asarray(pages)
        self._add(rng.choice(pages, size=max(0, int(refs))), mad)

    def build(self) -> Workload:
        trace = np.concatenate(self.parts) if self.parts else np.zeros(0, dtype=np.int32)
        return Workload(self.name, trace, self.mad_steps or [(0, 0)], self.kind, self.description, True)


# ----------------------------------------------------------------------------------
# Section 5.4 program (cyclic array with shrinking access scope)
# ----------------------------------------------------------------------------------
def cyclic_shrinking(name, pages, loops=1000, shrink_every=10, factor=0.9, description=""):
    """The paper's Section 5.4 code: LOOP passes over an array; after every 10 loops
    the accessed scope shrinks by 10% (SIZE = 0.9*SIZE).  One reference per page."""
    size = int(pages)
    parts = []
    for i in range(loops):
        if size > 0:
            parts.append(np.arange(size, dtype=np.int32))
        if (i + 1) % shrink_every == 0:
            size = int(factor * size)
    trace = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)
    return Workload(name, trace, [(0, int(pages))], "type1",
                    description or "Section 5.4 cyclic array program (synthetic re-implementation)", True)


# ----------------------------------------------------------------------------------
# Type 2: gradually acquiring memory (staircase MAD), vortex-like
# ----------------------------------------------------------------------------------
def gradual(name, rng, peak_pages, total_refs, stages, ws_frac, description=""):
    """Stair-climbing allocation.  At each stage new pages are first-touched, then
    references are drawn uniformly from a working set that is `ws_frac` of the
    allocated pages (a fixed random subset, so the working set grows with MAD)."""
    b = TraceBuilder(name, "type2", description or "synthetic gradual-allocation program")
    rank = rng.permutation(peak_pages)
    per_stage = total_refs // stages
    for k in range(stages):
        pk = int(np.ceil(peak_pages * (k + 1) / stages))
        new = pk - b.hi
        b.alloc(pk, mad=pk)
        ws = np.nonzero(rank[:pk] < ws_frac * peak_pages)[0]
        if len(ws) == 0:
            ws = np.arange(pk)
        b.pick(rng, ws, max(0, per_stage - max(new, 0)))
    return b.build()


# ----------------------------------------------------------------------------------
# Type 1: quickly acquiring memory
# ----------------------------------------------------------------------------------
def quick_hot_with_sweeps(name, rng, alloc_pages, hot_pages, sweeps, total_refs, description=""):
    """gzip-like (Fig. 2): allocate everything up-front, small hot set (NAP ~1000 pages), and sweep
    events over part of the allocation at given times.  sweeps = [(start_ref, pages), ...]."""
    b = TraceBuilder(name, "type1", description or "synthetic quick-allocation program")
    b.alloc(alloc_pages, mad=alloc_pages)
    for start, size in sorted(sweeps):
        gap = start - b.length
        if gap > 0:
            b.random(rng, 0, hot_pages, gap, alloc_first=False)
        b.sweep(0, size)
    tail = total_refs - b.length
    if tail > 0:
        b.random(rng, 0, hot_pages, tail, alloc_first=False)
    return b.build()


def quick_two_phase_cyclic(name, rng, ws_pages, refs_phase1, hot_pages, refs_phase2, description=""):
    """bit-r-like: cyclic passes over a large resident array, then a much smaller hot set."""
    b = TraceBuilder(name, "type1", description or "synthetic bit-reversal-like program")
    b.cyclic(0, ws_pages, refs_phase1, mad=ws_pages)
    b.random(rng, 0, hot_pages, refs_phase2, alloc_first=False)
    return b.build()


# ----------------------------------------------------------------------------------
# Type 3: irregular demand (gcc-like spikes; LU-like periodic sweeps)
# ----------------------------------------------------------------------------------
def irregular_phases(name, rng, phases, description=""):
    """gcc-like (Fig. 1): list of (ws_pages, refs).  Entering a phase frees pages above the new
    working-set size (memory is released) or first-touches the extra pages (zero-fill); then uniform
    random references over the working set.  MAD follows ws_pages."""
    b = TraceBuilder(name, "type3", description or "synthetic irregular-demand program")
    for ws, refs in phases:
        ws = int(ws)
        if ws < b.hi:
            b.free(ws, b.hi, mad=ws)
        else:
            b.alloc(ws, mad=ws)
        b.random(rng, 0, ws, max(0, int(refs)), alloc_first=False)
    return b.build()


def lu_like(name, rng, alloc_pages, cycles, cycle_refs, description=""):
    """LU-like (Fig. 5): `cycles` iterations.  Each iteration releases the whole matrix, re-allocates it
    (sequential zero-fill touches = the RSS 'jump' / NAP spike), then traverses the whole matrix slowly and
    sequentially for the rest of the iteration (NAP ~2.2k pages/s over ~20 s covers ~the whole 40k-page matrix),
    i.e. its working set over an iteration is the whole matrix.  No randomness: the two LU instances differ only
    through scheduling, as the paper's LU-1/LU-2 (two runs of the same program)."""
    b = TraceBuilder(name, "type3", description or "synthetic LU-like program")
    per_page = max(1, (cycle_refs - alloc_pages) // max(1, alloc_pages))
    for i in range(cycles):
        if i:
            b.free(0, b.hi, mad=0)
        b.alloc(alloc_pages, mad=alloc_pages)
        b._add(np.repeat(np.arange(alloc_pages), per_page))
    return b.build()
