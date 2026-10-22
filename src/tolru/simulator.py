"""Discrete-event multiprogramming memory simulator.

Execution model (all times are SIMULATED ticks, not wall-clock)
---------------------------------------------------------------
* One CPU, round-robin with a quantum measured in references.
* One reference = one page touch; it costs `cpu_cost` ticks when it hits.
* First touch of a page is a *minor* (zero-fill) fault: no disk I/O, costs one
  reference time, but still needs a frame (so it can force a replacement).
* Touching a page that was evicted earlier is a *major* fault: the process
  blocks while ONE shared FIFO disk services the swap-in for `io_time` ticks.
  Because the disk is shared, heavy faulting by one process slows the other's
  faults too (the I/O-bandwidth effect discussed in paper Section 5.4).
* The frame is reserved when the swap-in starts (victim evicted then); the
  faulting reference completes when the I/O completes (guarantees progress).
* Evictions are free (no write-back is modelled -- documented limitation).
* A negative page id in a trace is an UNMAP/free of page (-id-1) (paper: programs such as LU
  and gcc release memory and re-allocate it; re-touching gives zero-filled pages, no disk read).
* "Swapping status": ON when a major fault starts, OFF when the process is next
  dispatched on the CPU, i.e. when it "returns to normal execution".  So the
  unresolved-page-fault window = waiting for the disk + waiting in the ready
  queue for the CPU -- exactly the window in which a process cannot touch its
  pages and they would become *false LRU* pages.
"""
from __future__ import annotations

import heapq
from collections import deque

from .config import SimConfig
from .memory import PhysicalMemory
from .process import (Process, Workload, NOT_STARTED, READY, BLOCKED_IO,
                      WAIT_FRAME, DONE, BLOCKED_SYNC)

ARRIVE, IO_DONE = 0, 1
_DONE_OK, _BLOCKED = 0, 1


class Simulator:
    def __init__(self, workloads, cfg: SimConfig, policy, start_delays=None):
        cfg.validate()
        self.cfg = cfg
        self.policy = policy
        delays = list(start_delays) if start_delays is not None else [0] * len(workloads)
        if len(delays) != len(workloads):
            raise ValueError("start_delays must match workloads")
        self.procs = [Process(i, w, delays[i]) for i, w in enumerate(workloads)]
        self.mem = PhysicalMemory(cfg.capacity_pages)
        self.groups: dict[str, list] = {}
        for p in self.procs:
            if p.sync_group is not None:
                self.groups.setdefault(p.sync_group, []).append(p)

        self.now = cfg.start_time
        self.ready: deque = deque()
        self.events: list = []
        self._seq = 0
        self.disk_free_at = 0
        self.disk_busy = 0
        self.idle_ticks = 0
        self.frame_waiters: deque = deque()
        self._retrying = False
        self.alive = 0
        self.completion_order: list[int] = []
        self.samples: list[tuple] = []
        self.next_sample = 0 if cfg.sample_interval > 0 else None
        self.disk_requests = 0
        policy.attach(self)

    # ------------------------------------------------------------------ events
    def _push(self, t, kind, pid, page=-1):
        self._seq += 1
        heapq.heappush(self.events, (t, self._seq, kind, pid, page))

    def _drain(self, upto):
        ev = self.events
        while ev and ev[0][0] <= upto:
            t, _, kind, pid, page = heapq.heappop(ev)
            p = self.procs[pid]
            if kind == ARRIVE:
                p.state = READY
                p.start_time = t
                self.ready.append(p)
            else:
                self._complete_io(p, page, t)

    def _complete_io(self, p, page, t):
        del p.inflight[page]
        self.mem.inflight_total -= 1
        self.mem.install(p, page, t)
        p.pos += 1                                  # the faulting reference is now done
        if p.pos >= p.n_refs:
            self._finish(p, t)
        else:
            p.state = READY
            self.ready.append(p)
        self._retry_waiters()

    # ---------------------------------------------------------------- lifecycle
    def _finish(self, p, t):
        p.state = DONE
        p.finish_time = t
        self.alive -= 1
        self.completion_order.append(p.pid)
        self.mem.free_process(p)
        self.policy.on_exit(p, t)
        if p.sync_group is not None:
            self._release_group(p.sync_group, t)    # peers may no longer have to wait for p
        self._retry_waiters()

    def run(self):
        for p in self.procs:
            if p.n_refs == 0:                       # empty trace: completes instantly
                p.state = DONE
                p.start_time = p.finish_time = self.now + p.start_delay
                self.completion_order.append(p.pid)
            else:
                self.alive += 1
                if p.start_delay == 0:
                    p.state = READY
                    p.start_time = self.now
                    self.ready.append(p)
                else:
                    self._push(self.now + p.start_delay, ARRIVE, p.pid)

        while self.alive > 0:
            self._drain(self.now)
            if self.alive == 0:
                break
            if not self.ready:
                if not self.events:
                    raise RuntimeError("deadlock: nothing runnable and no pending events")
                t = self.events[0][0]
                self._sample_until(t)
                if t > self.now:
                    self.idle_ticks += t - self.now
                    self.now = t
                continue
            p = self.ready.popleft()
            if p.done:
                continue
            self._dispatch(p)

        self.policy.finalize(self.now)
        if self.next_sample is not None:
            self._sample_until(self.now)
        return self

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, p):
        if p.swapping:                              # back to normal execution
            p.swapping = False
            self.policy.on_swapping_end(p, self.now)
        self.policy.on_tick(self.now)
        self._run_slice(p)

    def _run_slice(self, p):
        trace, n, res = p.trace, p.n_refs, p.resident
        cpu = self.cfg.cpu_cost
        ev = self.events
        sync = p.sync_group is not None and bool(p.barriers)
        for _ in range(self.cfg.quantum):
            if sync and p.bi < len(p.barriers) and p.pos == p.barriers[p.bi]:
                if self._arrive(p):                 # blocked at the barrier
                    return
            page = trace[p.pos]
            if page < 0:                            # free/unmap op: release page (-page-1), costs one tick
                pg = -page - 1
                if pg in res:
                    del res[pg]
                    self.mem.resident_total -= 1
                p.touched.discard(pg)               # a later touch is a fresh zero-fill (minor) fault
                if self.frame_waiters:
                    self._retry_waiters()
            elif page in res:                       # hit
                res.move_to_end(page)
                res[page] = self.now
            else:
                if self._fault(p, page) == _BLOCKED:
                    return
            p.pos += 1
            self.now += cpu
            if p.pos >= n:
                self._finish(p, self.now)
                return
            if ev and ev[0][0] <= self.now:
                self._drain(self.now)
            if self.next_sample is not None and self.now >= self.next_sample:
                self._sample_until(self.now)
        self.ready.append(p)                        # quantum expired

    # -------------------------------------------------------------- fault service
    def _fault(self, p, page):
        mem = self.mem
        if page not in p.touched:                   # minor / zero-fill fault
            p.minor_faults += 1
            if not self._try_minor(p, page):
                p.state = WAIT_FRAME
                p.wait_page, p.wait_kind = page, "minor"
                self.frame_waiters.append(p)
                return _BLOCKED
            return _DONE_OK

        # major fault: the page lives in swap
        p.major_faults += 1
        p.swapping = True
        # --- token acquisition point (paper: before do_swap_page) ---
        self.policy.on_major_fault(p, mem.free() == 0, self.now)
        if not self._begin_io(p, page):
            p.state = WAIT_FRAME
            p.wait_page, p.wait_kind = page, "major"
            self.frame_waiters.append(p)
        if self.cfg.check_invariants:
            self.check_invariants()
        return _BLOCKED

    def _try_minor(self, p, page) -> bool:
        if self.mem.free() == 0:
            v = self.policy.select_victim(p, self.now)
            if v is None:
                return False
            self._evict(v, p)
        self.mem.install(p, page, self.now)
        p.touched.add(page)
        if self.cfg.check_invariants:
            self.check_invariants()
        return True

    def _begin_io(self, p, page) -> bool:
        mem = self.mem
        if mem.free() == 0:
            v = self.policy.select_victim(p, self.now)
            if v is None:
                return False
            self._evict(v, p)
        p.inflight[page] = 1
        mem.inflight_total += 1
        start = max(self.now, self.disk_free_at)
        done = start + self.cfg.io_time
        self.disk_free_at = done
        self.disk_busy += self.cfg.io_time
        self.disk_requests += 1
        p.state = BLOCKED_IO
        self._push(done, IO_DONE, p.pid, page)
        return True

    def _evict(self, victim, requester):
        vproc, page = victim
        assert page in vproc.resident
        del vproc.resident[page]
        self.mem.resident_total -= 1
        vproc.evicted_pages += 1
        if vproc.swapping:                          # the page was LRU only because its owner was faulting
            if vproc is requester:
                vproc.false_lru_self += 1
            else:
                vproc.false_lru_cross += 1

    def _retry_waiters(self):
        if self._retrying or not self.frame_waiters:
            return
        self._retrying = True
        try:
            for _ in range(len(self.frame_waiters)):
                p = self.frame_waiters.popleft()
                if p.done:
                    continue
                if p.wait_kind == "major":
                    ok = self._begin_io(p, p.wait_page)
                else:
                    ok = self._try_minor(p, p.wait_page)
                    if ok:
                        p.pos += 1
                        if p.pos >= p.n_refs:
                            self._finish(p, self.now)
                        else:
                            p.state = READY
                            self.ready.append(p)
                if not ok:
                    self.frame_waiters.append(p)
        finally:
            self._retrying = False

    # --------------------------------------------------------------- synchronisation
    @staticmethod
    def _passed(q, k) -> bool:
        """Has member q already passed (or will never have) barrier k?"""
        return q.done or q.bi > k or q.bi >= len(q.barriers)

    def _arrive(self, p) -> bool:
        """p reached its barrier p.bi.  Returns True if p must block (peers have not arrived yet)."""
        k = p.bi
        peers = [q for q in self.groups[p.sync_group] if q is not p]
        if all(self._passed(q, k) or (q.state == BLOCKED_SYNC and q.bi == k) for q in peers):
            for q in peers:                         # last arriver: release everyone waiting at barrier k
                if q.state == BLOCKED_SYNC and q.bi == k:
                    self._unblock_sync(q, self.now)
            p.bi += 1
            p.barriers_passed += 1
            return False
        p.state = BLOCKED_SYNC
        p.sync_wait_start = self.now
        return True

    def _unblock_sync(self, q, t):
        q.sync_wait_ticks += t - q.sync_wait_start
        q.sync_wait_start = None
        q.bi += 1
        q.barriers_passed += 1
        q.state = READY
        self.ready.append(q)

    def _release_group(self, group, t):
        """A member finished: release peers whose barrier no longer waits on anyone."""
        for q in list(self.groups[group]):
            if q.state == BLOCKED_SYNC:
                k = q.bi
                if all(self._passed(m, k) or (m.state == BLOCKED_SYNC and m.bi == k)
                       for m in self.groups[group] if m is not q):
                    self._unblock_sync(q, t)

    def waiters_on(self, p) -> int:
        """Number of peers currently blocked at the barrier that p has not reached yet.

        p is then a *straggler* that other processes wait on (paper Sec. 5.3: highest token priority).
        """
        if p.sync_group is None or p.done or p.state == BLOCKED_SYNC or p.bi >= len(p.barriers):
            return 0
        k = p.bi
        return sum(1 for q in self.groups[p.sync_group]
                   if q is not p and q.state == BLOCKED_SYNC and q.bi == k)

    # ---------------------------------------------------------------- sampling
    def _sample_until(self, t):
        if self.next_sample is None:
            return
        while self.next_sample <= t:
            self._record(self.next_sample)
            self.next_sample += self.cfg.sample_interval

    def _record(self, ts):
        owner = getattr(self.policy, "owner", None)
        lo = ts - self.cfg.nap_window
        for p in self.procs:
            if p.state in (NOT_STARTED, DONE):
                continue
            nap = 0
            for _, last in reversed(p.resident.items()):
                if last < lo:
                    break
                nap += 1
            self.samples.append((ts, p.pid, len(p.resident), p.workload.mad_at(p.pos), nap,
                                 int(p.swapping), int(owner == p.pid), p.pos))

    # --------------------------------------------------------------- invariants
    def check_invariants(self):
        mem = self.mem
        assert mem.resident_total + mem.inflight_total <= mem.capacity, "more pages than frames"
        assert mem.resident_total == sum(len(p.resident) for p in self.procs), "resident count drift"
        assert mem.inflight_total == sum(len(p.inflight) for p in self.procs), "inflight count drift"
        for p in self.procs:
            assert not (set(p.resident) & set(p.inflight)), "page in two frames"
        owner = getattr(self.policy, "owner", None)
        assert owner is None or not self.procs[owner].done, "token held by finished process"

    # ------------------------------------------------------------------ results
    def proc_rows(self):
        rows = []
        for p in self.procs:
            rows.append(dict(pid=p.pid, name=p.name, refs=p.n_refs,
                             start=p.start_time, finish=p.finish_time, elapsed=p.elapsed(),
                             major_faults=p.major_faults, minor_faults=p.minor_faults,
                             evicted_pages=p.evicted_pages,
                             false_lru_cross=p.false_lru_cross, false_lru_self=p.false_lru_self,
                             sync_wait_ticks=p.sync_wait_ticks, barriers_passed=p.barriers_passed))
        return rows

    def system_row(self):
        makespan = max((p.finish_time or 0) for p in self.procs) if self.procs else 0
        busy = makespan - self.idle_ticks - self.cfg.start_time
        return dict(makespan=makespan, idle_ticks=self.idle_ticks,
                    cpu_utilisation=(busy / (makespan - self.cfg.start_time)) if makespan > self.cfg.start_time else None,
                    disk_busy_ticks=self.disk_busy, disk_requests=self.disk_requests,
                    total_major_faults=sum(p.major_faults for p in self.procs),
                    total_minor_faults=sum(p.minor_faults for p in self.procs),
                    completion_order=list(self.completion_order))


def run_simulation(workloads, cfg: SimConfig, policy_name: str, start_delays=None) -> Simulator:
    from .policies import make_policy
    sim = Simulator(workloads, cfg, make_policy(policy_name, cfg), start_delays)
    return sim.run()


def dedicated_elapsed(workload: Workload, cfg: SimConfig) -> int:
    """Execution time with the whole machine to itself and NO major faults.

    Memory is made large enough that nothing is ever evicted (the paper's
    "dedicated environment without major page faults").
    """
    import dataclasses
    big = dataclasses.replace(cfg, capacity_pages=max(1, workload.distinct_pages() + 1),
                              sample_interval=0, check_invariants=False)
    sim = run_simulation([workload], big, "global_lru")
    p = sim.procs[0]
    assert p.major_faults == 0, "dedicated run must have no major faults"
    return p.elapsed() or 0
