"""Policy interface used by the simulator core.

The simulator never looks at token state; it only calls these hooks.  That keeps
the baseline and the token-ordered policy fully independent.
"""
from __future__ import annotations


class ReplacementPolicy:
    name = "base"

    def attach(self, sim) -> None:
        self.sim = sim

    # Called right before a major (swap-in) fault is serviced -- the analogue of
    # the paper's "request the token before do_swap_page()".
    # `shortage` is True when no free frame exists (a replacement is needed).
    def on_major_fault(self, proc, shortage: bool, now: int) -> None:
        pass

    # Return (victim_process, page) or None if no page can be replaced right now.
    def select_victim(self, requester, now: int):
        raise NotImplementedError

    # The process has returned to normal execution (its swapping status was cleared).
    def on_swapping_end(self, proc, now: int) -> None:
        pass

    def on_exit(self, proc, now: int) -> None:
        pass

    def on_tick(self, now: int) -> None:
        pass

    def finalize(self, now: int) -> None:
        pass

    def summary(self) -> dict:
        return {}
