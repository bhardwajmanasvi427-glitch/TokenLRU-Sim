"""Conventional global LRU baseline.

Selects the least-recently-used page over the whole user memory, regardless of
which process owns it and regardless of whether the owner is currently
page-faulting.  There is intentionally NO token, NO swapping-status check and NO
protection here: this is the "original Linux"-style policy that the paper
compares against (with exact LRU instead of Linux's reference-bit approximation).
"""
from __future__ import annotations

from ..memory import PhysicalMemory
from .base import ReplacementPolicy


class GlobalLRU(ReplacementPolicy):
    name = "global_lru"

    def __init__(self):
        self.evictions = 0
        self.scan_steps = 0

    def select_victim(self, requester, now):
        procs = self.sim.procs
        victim, steps = PhysicalMemory.lru_candidate(procs)
        self.scan_steps += steps
        if victim is None:
            return None
        self.evictions += 1
        page = next(iter(victim.resident))        # its LRU page
        return victim, page

    def summary(self) -> dict:
        return {"policy_evictions": self.evictions, "victim_scan_steps": self.scan_steps}
