"""Process and workload model."""
from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence

# process states
NOT_STARTED, READY, BLOCKED_IO, WAIT_FRAME, DONE, BLOCKED_SYNC = range(6)
STATE_NAMES = ["not_started", "ready", "blocked_io", "wait_frame", "done", "blocked_sync"]


@dataclass
class Workload:
    """A per-process page-reference trace plus descriptive metadata.

    trace      : sequence of page ids (process-local virtual page numbers)
    mad_steps  : [(reference_index, MAD_pages), ...] sorted; memory allocation
                 demand in effect from that reference index onward
    kind       : "type1" quick | "type2" gradual | "type3" irregular | "custom"
    synthetic  : True for every workload in this repository (NOT SPEC2000 runs)
    sync_group : name of a bulk-synchronous group (None = independent process)
    barriers   : trace positions at which the process must wait (before executing the reference at that
                 position) until every other live member of its group has reached its own k-th barrier.
                 Models paper Sec. 5.3's "processes with synchronization / blocking messages waiting on them".
    """
    name: str
    trace: Sequence[int]
    mad_steps: list = field(default_factory=lambda: [(0, 0)])
    kind: str = "custom"
    description: str = ""
    synthetic: bool = True
    sync_group: str | None = None
    barriers: list = field(default_factory=list)

    def __post_init__(self):
        self.barriers = sorted(int(b) for b in self.barriers)
        if self.barriers and self.sync_group is None:
            raise ValueError("barriers require a sync_group")
        self.mad_steps = sorted(self.mad_steps) or [(0, 0)]
        self._mad_idx = [s[0] for s in self.mad_steps]

    def __len__(self) -> int:
        return len(self.trace)

    def mad_at(self, pos: int) -> int:
        i = bisect_right(self._mad_idx, pos) - 1
        return self.mad_steps[max(i, 0)][1]

    def peak_mad(self) -> int:
        return max((m for _, m in self.mad_steps), default=0)

    def distinct_pages(self) -> int:
        return len(set(int(x) for x in self.trace if int(x) >= 0))


class Process:
    """Runtime state of one simulated process."""

    def __init__(self, pid: int, workload: Workload, start_delay: int = 0):
        self.pid = pid
        self.name = workload.name
        self.workload = workload
        self.trace = [int(x) for x in workload.trace]   # python ints: fastest to index
        self.n_refs = len(self.trace)
        self.start_delay = start_delay
        self.pos = 0
        self.state = NOT_STARTED

        # page tables ---------------------------------------------------------
        # resident: page -> last-access timestamp, kept in LRU -> MRU order.
        self.resident: "OrderedDict[int, int]" = OrderedDict()
        self.inflight: dict[int, int] = {}      # page -> 1 while its swap-in is on the disk
        self.touched: set[int] = set()          # pages that have a backing copy (first touch done)

        # Swapping status (paper Section 5.2): True from the moment a major page
        # fault begins until the process is next dispatched ("returns to its
        # normal execution stage").  See docs/design.md for the discussion.
        self.swapping = False

        # wait-for-frame bookkeeping (rare: only when every frame is in flight)
        self.wait_page = -1
        self.wait_kind = ""

        # synchronisation (bulk-synchronous groups) -------------------------------
        self.sync_group = workload.sync_group
        self.barriers = list(workload.barriers)
        self.bi = 0                             # index of the next barrier this process must pass
        self.sync_wait_start: int | None = None
        self.sync_wait_ticks = 0
        self.barriers_passed = 0

        # statistics ----------------------------------------------------------
        self.start_time: int | None = None
        self.finish_time: int | None = None
        self.major_faults = 0
        self.minor_faults = 0
        self.evicted_pages = 0                  # pages of this process taken as victims
        self.false_lru_cross = 0                # evicted by ANOTHER process while this one was swapping
        self.false_lru_self = 0                 # evicted by its own fault while swapping

    @property
    def done(self) -> bool:
        return self.state == DONE

    def elapsed(self) -> int | None:
        if self.finish_time is None or self.start_time is None:
            return None
        return self.finish_time - self.start_time
