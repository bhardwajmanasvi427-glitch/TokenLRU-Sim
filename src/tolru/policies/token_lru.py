"""Token-ordered LRU (Jiang & Zhang, Performance Evaluation 60 (2005), Section 5).

Paper mechanism -> code
-----------------------
* global, mutually exclusive token, states available/held  -> self.owner (None | pid)
* token requested only on a page fault caused by memory
  shortage, before the swap-in is serviced (do_swap_page)  -> on_major_fault()
* per-process "swapping status"                            -> Process.swapping (set/cleared by simulator)
* __get_free_pages() skips the program holding BOTH the
  token and the swapping status                            -> select_victim(): `protected`
* protection ends as soon as swapping status is cleared    -> protected() re-evaluates every call
* exception handler: take LRU pages from the privileged
  program itself if nobody else can supply them            -> fallback branch of select_victim()
* Section 5.3 case 1 (holder keeps eating its own pages)   -> self-supply window -> forced release
* Section 5.3 case 2 (holder keeps token too long)         -> max_hold_ticks   -> forced release
* token-unsuitable label removed after a fixed period      -> unsuitable_until / cooldown
* assignment: highest priority to a program that other programs
  are blocked on (barrier/synchronisation), otherwise random
  order among interacting programs                          -> _choose_acquirer()
  (single-node model: "waiting" = peers of a bulk-synchronous group blocked at a barrier;
   remote nodes / blocking messages are not modelled)
* cfg.sync_transfer (default OFF) is OUR EXTENSION, not part of the paper: it hands an already
  held token to a waited-on process when that process page-faults.

Nothing in here pins pages permanently: protection is evaluated per victim
search and disappears the moment the holder leaves the swapping state.
"""
from __future__ import annotations

import random
from collections import deque

from ..config import SimConfig
from ..memory import PhysicalMemory
from .base import ReplacementPolicy


class TokenOrderedLRU(ReplacementPolicy):
    name = "token_lru"

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # token state --------------------------------------------------------
        self.owner: int | None = None           # None => token available
        self.acquired_at: int | None = None
        self.unsuitable_until: dict[int, int] = {}
        self._self_hist: deque = deque(maxlen=cfg.self_supply_window)
        # statistics ----------------------------------------------------------
        self.acquisitions = 0
        self.transfers = 0                      # acquisitions by a different process than the previous holder
        self._last_owner: int | None = None
        self.releases = 0
        self.forced_release_self_supply = 0
        self.forced_release_timeout = 0
        self.releases_exit = 0
        self.releases_resolved = 0
        self.protected_skips = 0                # victim searches whose outcome the protection changed
        self.fallback_evictions = 0             # exception-handler evictions from the protected process
        self.holder_self_fallbacks = 0          # ... of which the holder's own fault supplied itself
        self.evictions = 0
        self.scan_steps = 0
        self.sync_priority_assignments = 0      # acquisitions decided by the sync-priority rule
        self.sync_transfers = 0                 # extension only
        self.holds: list[tuple] = []            # (pid, start, end, reason)
        self.events: list[tuple] = []           # (t, kind, pid, detail)

    # ---- helpers --------------------------------------------------------------
    @property
    def token_state(self) -> str:
        return "available" if self.owner is None else "held"

    def _log(self, t, kind, pid, detail=""):
        if self.cfg.log_events:
            self.events.append((t, kind, pid, detail))

    def _eligible(self, p, now) -> bool:
        return (not p.done) and self.unsuitable_until.get(p.pid, 0) <= now

    def protected(self):
        """The process whose pages are currently protected, or None.

        Protected <=> it holds the token AND its swapping status is on.
        """
        if self.owner is None:
            return None
        p = self.sim.procs[self.owner]
        return p if (p.swapping and not p.done) else None

    def _release(self, now, reason, cooldown=False):
        pid = self.owner
        if pid is None:
            return
        self.holds.append((pid, self.acquired_at, now, reason))
        self._log(now, "release_" + reason, pid)
        self.releases += 1
        if cooldown:
            self.unsuitable_until[pid] = now + self.cfg.unsuitable_cooldown_ticks
            self._log(now, "mark_unsuitable", pid, f"until={self.unsuitable_until[pid]}")
        self.owner = None
        self.acquired_at = None
        self._self_hist.clear()

    def _acquire(self, p, now):
        assert self.owner is None, "token already held"
        self.owner = p.pid
        self.acquired_at = now
        self.acquisitions += 1
        if self._last_owner is not None and self._last_owner != p.pid:
            self.transfers += 1
        self._last_owner = p.pid
        self._self_hist.clear()
        self._log(now, "acquire", p.pid)

    def _choose_acquirer(self, requester, now):
        """Randomised assignment among eligible programs that are page-faulting."""
        if self.cfg.token_assignment == "first_fault":
            return requester if self._eligible(requester, now) else None
        cands = [p for p in self.sim.procs if p.swapping and self._eligible(p, now)]
        if not cands:
            return None
        if self.cfg.sync_priority:
            # Sec. 5.3: a process that others are waiting on (synchronisation) has the highest priority;
            # among several, the one with most waiters; ties -> random.
            w = {p.pid: self.sim.waiters_on(p) for p in cands}
            best = max(w.values())
            if best > 0:
                self.sync_priority_assignments += 1
                return self.rng.choice([p for p in cands if w[p.pid] == best])
        return self.rng.choice(cands)

    # ---- hooks ---------------------------------------------------------------
    def on_tick(self, now):
        """Case 2: execution time while holding the token exceeds the threshold."""
        mh = self.cfg.max_hold_ticks
        if mh is not None and self.owner is not None and now - self.acquired_at > mh:
            self.forced_release_timeout += 1
            self._release(now, "timeout", cooldown=True)

    def on_major_fault(self, proc, shortage, now):
        self.on_tick(now)
        # The token is only ever requested when memory shortage forces a replacement.
        if not shortage:
            return
        if (self.owner is not None and self.cfg.sync_transfer and self.owner != proc.pid
                and self.sim.waiters_on(proc) > 0 and self.sim.waiters_on(self.sim.procs[self.owner]) == 0
                and self._eligible(proc, now)):
            self.sync_transfers += 1                # EXTENSION (not in the paper)
            self._release(now, "sync_transfer")
            self._acquire(proc, now)
            return
        if self.owner is not None:
            return
        chosen = self._choose_acquirer(proc, now)
        if chosen is not None:
            self._acquire(chosen, now)

    def on_swapping_end(self, proc, now):
        if self.cfg.release_on_resolution and self.owner == proc.pid:
            self.releases_resolved += 1
            self._release(now, "resolved")

    def on_exit(self, proc, now):
        if self.owner == proc.pid:
            self.releases_exit += 1
            self._release(now, "exit")

    def finalize(self, now):
        # Close any open hold interval so timelines are complete.
        if self.owner is not None:
            self.holds.append((self.owner, self.acquired_at, now, "end_of_run"))
            self.owner = None

    # ---- victim selection --------------------------------------------------------
    def select_victim(self, requester, now):
        procs = self.sim.procs
        prot = self.protected()

        # Unrestricted global-LRU choice (only used to detect whether protection mattered).
        unrestricted, steps0 = PhysicalMemory.lru_candidate(procs)
        # Normal scan: skip the protected process, keep looking at everybody else.
        victim, steps1 = PhysicalMemory.lru_candidate(procs, exclude=prot)
        self.scan_steps += steps0 + steps1
        if prot is not None and unrestricted is prot:
            self.protected_skips += 1

        holder_fault = prot is not None and requester is prot
        if victim is None:
            # Exception handler: nobody else can supply a page -> use the
            # protected process's own LRU pages instead of failing/deadlocking.
            if prot is not None and prot.resident:
                victim = prot
                self.fallback_evictions += 1
                if holder_fault:
                    self.holder_self_fallbacks += 1
            else:
                return None

        if holder_fault:
            self._record_holder_fault(requester, victim is requester, now)

        self.evictions += 1
        return victim, next(iter(victim.resident))

    def _record_holder_fault(self, holder, self_supplied: bool, now):
        """Case 1 monitor: does the holder keep taking pages from itself?"""
        others_alive = any((p is not holder) and (not p.done) for p in self.sim.procs)
        if not others_alive:
            return                              # nobody to protect against / harm
        self._self_hist.append(1 if self_supplied else 0)
        w = self.cfg.self_supply_window
        if len(self._self_hist) >= w and sum(self._self_hist) >= self.cfg.self_supply_ratio * w:
            self.forced_release_self_supply += 1
            self._release(now, "self_supply", cooldown=True)

    def summary(self) -> dict:
        return {
            "token_acquisitions": self.acquisitions,
            "token_transfers": self.transfers,
            "token_releases": self.releases,
            "forced_release_self_supply": self.forced_release_self_supply,
            "forced_release_timeout": self.forced_release_timeout,
            "release_on_exit": self.releases_exit,
            "release_on_resolution": self.releases_resolved,
            "sync_priority_assignments": self.sync_priority_assignments,
            "sync_transfers": self.sync_transfers,
            "protected_skips": self.protected_skips,
            "fallback_evictions": self.fallback_evictions,
            "holder_self_fallbacks": self.holder_self_fallbacks,
            "policy_evictions": self.evictions,
            "victim_scan_steps": self.scan_steps,
        }
