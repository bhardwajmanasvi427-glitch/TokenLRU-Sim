"""Simulation configuration.

Every numeric knob that the paper leaves unspecified lives here, so that the
experimental configuration is explicit and reproducible.  Time is *simulated*
time measured in integer ticks (never wall-clock time).

Values that come from the paper:
  * page_size = 4096 bytes (Section 3.2: "unit of 4 KB page").
Values that are OUR experimental choices (the paper does not specify them):
  * everything else (io_time, quantum, thresholds, windows, cooldowns ...).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class SimConfig:
    # ---- memory -----------------------------------------------------------
    capacity_pages: int = 1024          # user page frames available to the processes
    page_size: int = 4096               # bytes; paper Section 3.2 (4 KB pages)

    # ---- time model (simulated ticks) -------------------------------------
    cpu_cost: int = 1                   # ticks consumed by one memory reference (hit or zero-fill)
    io_time: int = 100                  # ticks the single disk needs to service one swap-in
    quantum: int = 50                   # round-robin time slice, in references
    start_time: int = 0

    # ---- randomness --------------------------------------------------------
    seed: int = 0                       # seeds randomized token assignment

    # ---- token-ordered LRU parameters (Section 5.3; values are OUR choices) --
    token_assignment: str = "random"    # "random" (paper's initial approach) | "first_fault"
    max_hold_ticks: int | None = None   # Case 2: forced release after holding this long (None = off)
    unsuitable_cooldown_ticks: int = 200_000   # how long a forced-out process stays token-unsuitable
    self_supply_window: int = 64        # Case 1: look at the holder's last N protected-mode faults
    self_supply_ratio: float = 0.9      # Case 1: force release if >= this fraction hit own pages
    release_on_resolution: bool = False # False = paper behaviour (token kept until exit / forced release)
    sync_priority: bool = True          # Sec. 5.3: highest token priority to a process others are blocked on
    sync_transfer: bool = False         # EXTENSION, NOT in the paper: hand the held token to a waited-on process

    # ---- instrumentation ----------------------------------------------------
    sample_interval: int = 0            # ticks between RSS/MAD/NAP samples (0 = off)
    nap_window: int = 1000              # NAP = pages accessed within this many ticks
    check_invariants: bool = False      # assert structural invariants after every fault (slow)
    log_events: bool = True             # keep the token event log

    def validate(self) -> None:
        if not isinstance(self.capacity_pages, int) or self.capacity_pages < 1:
            raise ValueError("capacity_pages must be an integer >= 1")
        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")
        if self.cpu_cost < 1:
            raise ValueError("cpu_cost must be >= 1")
        if self.io_time < 1:
            raise ValueError("io_time must be >= 1")
        if self.quantum < 1:
            raise ValueError("quantum must be >= 1")
        if self.token_assignment not in ("random", "first_fault"):
            raise ValueError("token_assignment must be 'random' or 'first_fault'")
        if self.max_hold_ticks is not None and self.max_hold_ticks < 1:
            raise ValueError("max_hold_ticks must be >= 1 or None")
        if self.unsuitable_cooldown_ticks < 0:
            raise ValueError("unsuitable_cooldown_ticks must be >= 0")
        if self.self_supply_window < 1:
            raise ValueError("self_supply_window must be >= 1")
        if not (0.0 < self.self_supply_ratio <= 1.0):
            raise ValueError("self_supply_ratio must be in (0, 1]")
        if self.sample_interval < 0 or self.nap_window < 1:
            raise ValueError("sample_interval must be >= 0 and nap_window >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
