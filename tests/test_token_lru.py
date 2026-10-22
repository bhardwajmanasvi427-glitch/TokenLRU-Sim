"""Tests for the token mechanism, protection, exception handler, suitability monitoring (areas 2-11, 13)."""
import pytest

from tolru import SimConfig, run_simulation
from conftest import make_sim, seed_resident, wl


def _two_proc_state(cfg_kw=None):
    """P0 holds the token and owns the OLDEST pages; P1 owns newer pages."""
    sim = make_sim(8, [[0], [0]], policy="token_lru", **(cfg_kw or {}))
    seed_resident(sim, 0, [(1, 10), (2, 20)])
    seed_resident(sim, 1, [(7, 30), (8, 40)])
    pol = sim.policy
    pol._acquire(sim.procs[0], now=5)
    return sim, pol


# -------------------------------------------------------------------- token acquisition
def test_token_acquired_on_shortage_fault():
    sim = make_sim(4, [[0], [0]])
    p = sim.procs[0]
    p.swapping = True
    sim.policy.on_major_fault(p, shortage=True, now=1)
    assert sim.policy.owner == 0 and sim.policy.token_state == "held" and sim.policy.acquisitions == 1


def test_token_not_acquired_without_shortage():
    sim = make_sim(4, [[0], [0]])
    p = sim.procs[0]
    p.swapping = True
    sim.policy.on_major_fault(p, shortage=False, now=1)
    assert sim.policy.owner is None and sim.policy.token_state == "available"


def test_no_token_activity_when_memory_is_sufficient_end_to_end():
    traces = [[1, 2, 3, 1, 2, 3] * 20, [4, 5, 6, 4, 5, 6] * 20]
    sim = run_simulation([wl("a", traces[0]), wl("b", traces[1])], SimConfig(capacity_pages=100),
                         "token_lru")
    assert sim.policy.acquisitions == 0 and sim.system_row()["total_major_faults"] == 0


def test_token_acquired_end_to_end_only_after_memory_fills():
    seq0 = list(range(10)) * 8
    seq1 = list(range(10)) * 8
    sim = run_simulation([wl("a", seq0), wl("b", seq1)], SimConfig(capacity_pages=12), "token_lru")
    acquires = [e for e in sim.policy.events if e[1] == "acquire"]
    assert acquires, "memory shortage should have triggered a token acquisition"
    # frames: 12.  The first 12 first-touch references cannot cause a shortage-triggered major fault.
    assert acquires[0][0] > 12


def test_only_one_owner_at_a_time_and_double_acquire_rejected():
    sim, pol = _two_proc_state()
    with pytest.raises(AssertionError):
        pol._acquire(sim.procs[1], now=6)
    assert pol.owner == 0


def test_hold_intervals_never_overlap_in_a_real_run():
    seq = list(range(60)) * 15
    sim = run_simulation([wl("a", seq), wl("b", seq[::-1])],
                         SimConfig(capacity_pages=70, max_hold_ticks=3000, unsuitable_cooldown_ticks=2000,
                                   check_invariants=True), "token_lru")
    holds = sorted(sim.policy.holds, key=lambda h: h[1])
    for (_, s1, e1, _), (_, s2, e2, _) in zip(holds, holds[1:]):
        assert e1 <= s2, "token held by two processes at once"
    assert len(holds) >= 1


# -------------------------------------------------------------------- protection (area 4-6)
def test_holder_protected_while_swapping_other_process_supplies_victim():
    sim, pol = _two_proc_state()
    sim.procs[0].swapping = True
    victim, page = pol.select_victim(sim.procs[1], now=50)
    assert victim.pid == 1 and page == 7            # P0's older pages skipped
    assert pol.protected_skips == 1 and pol.fallback_evictions == 0


def test_protection_ends_when_swapping_status_clears():
    sim, pol = _two_proc_state()
    sim.procs[0].swapping = True
    assert pol.select_victim(sim.procs[1], 50)[0].pid == 1
    sim.procs[0].swapping = False                    # returned to normal execution
    victim, page = pol.select_victim(sim.procs[1], 60)
    assert victim.pid == 0 and page == 1             # true LRU pages are replaceable again


def test_token_without_swapping_status_gives_no_protection():
    sim, pol = _two_proc_state()
    assert sim.procs[0].swapping is False
    assert pol.protected() is None
    assert pol.select_victim(sim.procs[1], 50)[0].pid == 0


def test_skipping_picks_lru_among_remaining_processes():
    sim = make_sim(12, [[0], [0], [0]], policy="token_lru")
    seed_resident(sim, 0, [(1, 1)])                  # holder, oldest
    seed_resident(sim, 1, [(5, 25)])
    seed_resident(sim, 2, [(9, 15)])                 # oldest among non-protected
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    victim, page = sim.policy.select_victim(sim.procs[1], 50)
    assert (victim.pid, page) == (2, 9)


def test_protection_is_temporary_not_permanent_pinning():
    # holder keeps token but once its swapping ends its pages are evicted by other processes' faults
    sim, pol = _two_proc_state()
    sim.procs[0].swapping = True
    pol.select_victim(sim.procs[1], 50)
    sim.procs[0].swapping = False
    sim.mem.evict_lru_of(sim.procs[1])
    sim.mem.evict_lru_of(sim.procs[1])
    victim, _ = pol.select_victim(sim.procs[1], 70)
    assert victim.pid == 0


# -------------------------------------------------------------------- exception handler (area 7)
def test_fallback_to_protected_process_when_nobody_else_has_pages():
    sim = make_sim(4, [[0], [0]], policy="token_lru")
    seed_resident(sim, 0, [(1, 10), (2, 20)])
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    victim, page = sim.policy.select_victim(sim.procs[0], now=30)   # holder faults, P1 has nothing
    assert victim.pid == 0 and page == 1
    assert sim.policy.fallback_evictions == 1 and sim.policy.holder_self_fallbacks == 1


def test_fallback_when_non_holder_faults_and_only_protected_pages_exist():
    sim = make_sim(4, [[0], [0]], policy="token_lru")
    seed_resident(sim, 0, [(1, 10)])
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    victim, _ = sim.policy.select_victim(sim.procs[1], now=30)
    assert victim.pid == 0
    assert sim.policy.fallback_evictions == 1 and sim.policy.holder_self_fallbacks == 0


def test_no_victim_returns_none_instead_of_failing():
    sim = make_sim(4, [[0], [0]], policy="token_lru")
    assert sim.policy.select_victim(sim.procs[0], 0) is None


# -------------------------------------------------------------------- release rules (area 8-10)
def test_token_released_when_owner_exits():
    sim = run_simulation([wl("a", list(range(20)) * 3), wl("b", list(range(30)) * 20)],
                         SimConfig(capacity_pages=25), "token_lru")
    assert sim.policy.owner is None
    assert all(h[3] in ("exit", "self_supply", "timeout", "end_of_run") for h in sim.policy.holds)
    assert sim.policy.acquisitions >= 1


def test_release_on_resolution_option():
    sim = make_sim(8, [[0], [0]], release_on_resolution=True)
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    sim.policy.on_swapping_end(sim.procs[0], 9)
    assert sim.policy.owner is None and sim.policy.releases_resolved == 1


def test_default_keeps_token_after_resolution_as_in_the_paper():
    sim = make_sim(8, [[0], [0]])
    sim.policy._acquire(sim.procs[0], 0)
    sim.policy.on_swapping_end(sim.procs[0], 9)
    assert sim.policy.owner == 0                    # kept until exit / forced release


def test_forced_release_on_hold_timeout_marks_unsuitable():
    sim = make_sim(8, [[0], [0]], max_hold_ticks=10, unsuitable_cooldown_ticks=100)
    sim.policy._acquire(sim.procs[0], 0)
    sim.policy.on_tick(10)
    assert sim.policy.owner == 0                    # not exceeded yet (needs > 10)
    sim.policy.on_tick(11)
    assert sim.policy.owner is None
    assert sim.policy.forced_release_timeout == 1
    assert sim.policy.unsuitable_until[0] == 111
    assert sim.policy.holds[-1][3] == "timeout"


def test_unsuitable_cooldown_blocks_reacquisition_until_expiry():
    sim = make_sim(8, [[0], [0]], max_hold_ticks=10, unsuitable_cooldown_ticks=100,
                   token_assignment="first_fault")
    pol = sim.policy
    pol._acquire(sim.procs[0], 0)
    pol.on_tick(11)                                  # forced out, unsuitable until 111
    sim.procs[0].swapping = True
    pol.on_major_fault(sim.procs[0], True, now=50)
    assert pol.owner is None                         # still unsuitable
    pol.on_major_fault(sim.procs[0], True, now=111)
    assert pol.owner == 0                            # cooldown expired -> eligible again


def test_unsuitable_process_cannot_monopolise_other_process_may_take_token():
    sim = make_sim(8, [[0], [0]], max_hold_ticks=10, unsuitable_cooldown_ticks=1000)
    pol = sim.policy
    pol._acquire(sim.procs[0], 0)
    pol.on_tick(11)
    sim.procs[0].swapping = sim.procs[1].swapping = True
    for t in range(20, 25):
        pol.on_major_fault(sim.procs[0], True, now=t)
        if pol.owner is not None:
            break
    assert pol.owner == 1                            # random assignment only offers eligible processes


def test_case1_self_supply_monitor_forces_release():
    cfg = dict(self_supply_window=4, self_supply_ratio=0.75, unsuitable_cooldown_ticks=500)
    sim = make_sim(4, [[0], [0]], **cfg)
    seed_resident(sim, 0, [(i, i) for i in range(1, 12)])    # P1 has nothing -> holder must self-supply
    pol = sim.policy
    pol._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    for t in range(4):
        pol.select_victim(sim.procs[0], now=100 + t)   # victim is always the holder's own page
    assert pol.owner is None
    assert pol.forced_release_self_supply == 1
    assert pol.unsuitable_until[0] == 103 + 500


def test_case1_not_triggered_when_other_processes_supply_pages():
    sim = make_sim(40, [[0], [0]], self_supply_window=4, self_supply_ratio=0.75)
    seed_resident(sim, 0, [(i, i) for i in range(1, 6)])
    seed_resident(sim, 1, [(i, 100 + i) for i in range(1, 30)])
    pol = sim.policy
    pol._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    for t in range(8):
        victim, _ = pol.select_victim(sim.procs[0], now=200 + t)
        assert victim.pid == 1
    assert pol.owner == 0 and pol.forced_release_self_supply == 0


def test_case1_ignored_when_no_other_process_is_alive():
    sim = make_sim(4, [[0], [0]], self_supply_window=2, self_supply_ratio=1.0)
    seed_resident(sim, 0, [(i, i) for i in range(1, 9)])
    sim.procs[1].state = 4                           # DONE
    pol = sim.policy
    pol._acquire(sim.procs[0], 0)
    sim.procs[0].swapping = True
    for t in range(5):
        pol.select_victim(sim.procs[0], now=10 + t)
    assert pol.owner == 0


# -------------------------------------------------------------------- termination (area 13)
def test_process_terminating_while_holding_token_releases_it():
    short = list(range(8)) * 4
    long_ = list(range(30)) * 40
    sim = run_simulation([wl("short", short), wl("long", long_)],
                         SimConfig(capacity_pages=24, check_invariants=True), "token_lru")
    assert sim.policy.owner is None
    assert sim.procs[0].done and sim.procs[1].done


def test_exit_hook_clears_token_directly():
    sim = make_sim(8, [[0], [0]])
    sim.policy._acquire(sim.procs[1], 3)
    sim.procs[1].state = 4
    sim.policy.on_exit(sim.procs[1], 77)
    assert sim.policy.owner is None and sim.policy.holds[-1] == (1, 3, 77, "exit")


# -------------------------------------------------------------------- end-to-end invariants
def test_protected_process_never_evicted_except_by_documented_exception():
    seq0 = (list(range(40)) * 12)
    seq1 = (list(range(50)) * 12)
    cfg = SimConfig(capacity_pages=60, check_invariants=True)
    from tolru import Simulator, make_policy
    sim = Simulator([wl("a", seq0), wl("b", seq1)], cfg, make_policy("token_lru", cfg))
    pol = sim.policy
    orig = sim._evict
    state = {"fb": 0, "violations": 0}

    def checked(victim, requester):
        if pol.protected() is victim[0]:
            if pol.fallback_evictions <= state["fb"]:
                state["violations"] += 1             # evicted protected page without the exception
        state["fb"] = pol.fallback_evictions
        return orig(victim, requester)

    sim._evict = checked
    sim.run()
    assert state["violations"] == 0
    assert pol.protected_skips > 0


def test_resident_never_exceeds_capacity_and_pages_not_duplicated():
    seq = [i % 37 for i in range(3000)]
    for policy in ("global_lru", "token_lru"):
        sim = run_simulation([wl("a", seq), wl("b", seq[::-1]), wl("c", seq[5:])],
                             SimConfig(capacity_pages=40, check_invariants=True), policy)
        assert all(p.done for p in sim.procs)


def test_baseline_run_never_touches_token_state():
    seq = [i % 37 for i in range(2000)]
    sim = run_simulation([wl("a", seq), wl("b", seq)], SimConfig(capacity_pages=40), "global_lru")
    assert not hasattr(sim.policy, "owner")


# -------------------------------------------------------------------- determinism (area 14)
def _fingerprint(sim):
    return (sim.system_row(), sim.proc_rows(), sim.policy.summary(), list(sim.policy.events))


def test_deterministic_with_fixed_seed():
    seq = [i % 53 for i in range(4000)]
    runs = [run_simulation([wl("a", seq), wl("b", seq[7:])], SimConfig(capacity_pages=60, seed=11),
                           "token_lru") for _ in range(2)]
    assert _fingerprint(runs[0]) == _fingerprint(runs[1])


def test_random_assignment_depends_on_seed_stream_only():
    from tolru import TokenOrderedLRU
    picks = []
    for seed in (1, 1, 2):
        sim = make_sim(8, [[0], [0], [0]], seed=seed)
        for p in sim.procs:
            p.swapping = True
        sim.policy.on_major_fault(sim.procs[0], True, 0)
        picks.append(sim.policy.owner)
    assert picks[0] == picks[1]
