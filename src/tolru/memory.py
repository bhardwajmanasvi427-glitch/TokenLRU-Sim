"""Physical memory (page-frame) model.

A frame is "used" when it holds a resident page OR is reserved for a page whose
swap-in is still in flight on the disk.  Per-process resident pages live in an
OrderedDict ordered LRU -> MRU, so the global LRU victim is simply the process
whose oldest page has the smallest timestamp.  That is exact global LRU over the
union of all processes' pages, computed in O(#processes).
"""
from __future__ import annotations


class PhysicalMemory:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be >= 1 frame")
        self.capacity = capacity
        self.resident_total = 0
        self.inflight_total = 0

    # ---- frame accounting -----------------------------------------------
    def free(self) -> int:
        return self.capacity - self.resident_total - self.inflight_total

    def install(self, proc, page: int, ts: int) -> None:
        """Make `page` resident at the MRU end of proc's list."""
        proc.resident[page] = ts
        self.resident_total += 1

    def evict_lru_of(self, proc) -> int:
        """Remove and return proc's least-recently-used resident page."""
        page, _ = proc.resident.popitem(last=False)
        self.resident_total -= 1
        return page

    def free_process(self, proc) -> None:
        self.resident_total -= len(proc.resident)
        proc.resident.clear()
        self.inflight_total -= len(proc.inflight)
        proc.inflight.clear()

    # ---- victim search ---------------------------------------------------
    @staticmethod
    def lru_candidate(procs, exclude=None):
        """Return (proc, scan_steps): the process owning the globally oldest page.

        `exclude` removes one process from consideration (used by token-ordered
        LRU to skip the protected process).  Ties break on lower pid so the
        result is deterministic.
        """
        best = None
        best_ts = 0
        steps = 0
        for p in procs:
            if p is exclude or not p.resident:
                continue
            steps += 1
            ts = next(iter(p.resident.values()))
            if best is None or ts < best_ts:
                best, best_ts = p, ts
        return best, steps
