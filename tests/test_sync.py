"""Synchronisation groups (barriers) and sync-aware token priority (paper Sec. 5.3)."""
import pytest

from tolru import SimConfig, Simulator, Workload, make_policy, run_simulation
from tolru.process import BLOCKED_SYNC, DONE, READY
from conftest import make_sim, seed_resident


def swl(name, trace, group="g", barriers=()):
    return Workload(name, list(trace), [(0, len(set(trace)))], sync_group=group, barriers=list(barriers))


def sim_of(workloads, capacity=100, policy="token_lru", **kw):
    cfg = SimConfig(capacity_pages=capacity, **kw)
    return Simulator(workloads, cfg, make_policy(policy, cfg))


# ------------------------------------------------------------------ barriers
def test_barrier_makes_fast_process_wait_for_slow_one():
    fast = swl("fast", [1] * 10 + [2] * 10, barriers=[10])
    slow = swl("slow", [3] * 300 + [4] * 10, barriers=[300])
    sim = run_simulation([fast, slow], SimConfig(capacity_pages=50, quantum=20), "global_lru")
    f, s = sim.procs
    assert f.sync_wait_ticks > 0 and s.sync_wait_ticks == 0
    assert f.barriers_passed == 1 and s.barriers_passed == 1
    # the fast process cannot finish before the slow one reaches the barrier
    assert f.finish_time >= 300


def test_no_barrier_crossing_until_everyone_arrives():
    a = swl("a", list(range(5)) * 4, barriers=[10])
    b = swl("b", list(range(5)) * 40, barriers=[100])
    sim = sim_of([a, b], capacity=50, quantum=7, policy="global_lru")
    sim.run()
    # a was blocked until b had executed 100 references
    assert sim.procs[0].sync_wait_ticks >= 90


def test_peer_without_remaining_barriers_does_not_deadlock():
    a = swl("a", [1, 2, 3, 4], barriers=[2])
    b = swl("b", [7, 8], barriers=[])                # in the group but never synchronises
    sim = run_simulation([a, b], SimConfig(capacity_pages=10), "global_lru")
    assert all(p.done for p in sim.procs)


def test_finished_peer_releases_waiter():
    a = swl("a", [1, 2, 3, 4], barriers=[1])
    b = swl("b", [7, 8, 9], barriers=[3])            # b's barrier is at its end (position == length)
    b2 = Workload("b2", [7, 8, 9], [(0, 3)], sync_group="g", barriers=[])
    sim = run_simulation([a, b2], SimConfig(capacity_pages=10), "global_lru")
    assert all(p.done for p in sim.procs)


def test_multiple_barriers_and_groups_are_independent():
    g1 = [swl("a1", [1] * 40, "g1", [10, 20]), swl("a2", [2] * 40, "g1", [10, 20])]
    g2 = [swl("b1", [3] * 40, "g2", [30]), swl("b2", [4] * 40, "g2", [30])]
    sim = run_simulation(g1 + g2, SimConfig(capacity_pages=20, check_invariants=True), "token_lru")
    assert [p.barriers_passed for p in sim.procs] == [2, 2, 1, 1]


def test_barriers_require_a_group_and_workload_validation():
    with pytest.raises(ValueError):
        Workload("x", [1, 2], barriers=[1])


def test_independent_processes_unaffected_by_sync_machinery():
    seq = [i % 13 for i in range(500)]
    base = run_simulation([Workload("a", seq), Workload("b", seq)], SimConfig(capacity_pages=15), "token_lru")
    assert base.procs[0].sync_wait_ticks == 0 and base.policy.sync_priority_assignments == 0


# ------------------------------------------------------------------ waiters_on
def _straggler_setup(**kw):
    """P1 (straggler) and P2 are in a group; P2 is blocked at barrier 0 waiting for P1.  P0 is independent."""
    ws = [Workload("p0", [0]), swl("p1", [0, 1, 2], barriers=[2]), swl("p2", [0, 1, 2], barriers=[2])]
    sim = sim_of(ws, capacity=8, **kw)
    sim.procs[2].state = BLOCKED_SYNC
    sim.procs[2].sync_wait_start = 0
    return sim


def test_waiters_on_counts_blocked_peers_for_the_straggler_only():
    sim = _straggler_setup()
    assert sim.waiters_on(sim.procs[1]) == 1       # P2 waits for P1
    assert sim.waiters_on(sim.procs[2]) == 0       # a blocked process is not a straggler
    assert sim.waiters_on(sim.procs[0]) == 0       # independent process: nobody waits for it


# ------------------------------------------------------------------ token priority
@pytest.mark.parametrize("seed", range(25))
def test_token_goes_to_waited_on_process_regardless_of_seed(seed):
    sim = _straggler_setup(seed=seed)
    sim.procs[0].swapping = sim.procs[1].swapping = True
    sim.policy.on_major_fault(sim.procs[0], shortage=True, now=5)    # independent P0 faults first
    assert sim.policy.owner == 1                                      # ...but the straggler P1 gets the token
    assert sim.policy.sync_priority_assignments == 1


def test_without_waiters_assignment_is_random_among_swapping_processes():
    picks = set()
    for seed in range(30):
        sim = make_sim(8, [[0], [0], [0]], seed=seed)
        for p in sim.procs:
            p.swapping = True
        sim.policy.on_major_fault(sim.procs[0], True, 0)
        picks.add(sim.policy.owner)
    assert len(picks) > 1 and sim.policy.sync_priority_assignments == 0


def test_sync_priority_can_be_disabled():
    picks = set()
    for seed in range(40):
        sim = _straggler_setup(seed=seed, sync_priority=False)
        sim.procs[0].swapping = sim.procs[1].swapping = True
        sim.policy.on_major_fault(sim.procs[0], True, 5)
        picks.add(sim.policy.owner)
    assert picks == {0, 1}                                            # random again, P0 can win


def test_most_waited_on_process_wins():
    ws = [swl("s1", [0, 1, 2], "A", [2]), swl("w1", [0, 1, 2], "A", [2]),
          swl("s2", [0, 1, 2], "B", [2]), swl("w2", [0, 1, 2], "B", [2]), swl("w3", [0, 1, 2], "B", [2])]
    for seed in range(15):
        sim = sim_of(ws, capacity=8, seed=seed)
        for i in (1, 3, 4):
            sim.procs[i].state = BLOCKED_SYNC
            sim.procs[i].sync_wait_start = 0
        sim.procs[0].swapping = sim.procs[2].swapping = True
        sim.policy.on_major_fault(sim.procs[0], True, 1)
        assert sim.policy.owner == 2                                  # two waiters beat one


def test_unsuitable_straggler_is_not_given_the_token_despite_waiters():
    sim = _straggler_setup(max_hold_ticks=5, unsuitable_cooldown_ticks=1000)
    pol = sim.policy
    pol.unsuitable_until[1] = 500
    sim.procs[0].swapping = sim.procs[1].swapping = True
    pol.on_major_fault(sim.procs[0], True, 10)
    assert pol.owner == 0                                             # fairness rule still applies


def test_priority_does_not_trigger_without_shortage():
    sim = _straggler_setup()
    sim.procs[1].swapping = True
    sim.policy.on_major_fault(sim.procs[1], shortage=False, now=1)
    assert sim.policy.owner is None


# ------------------------------------------------------------------ extension (not in paper)
def test_sync_transfer_is_off_by_default_and_token_is_not_stolen():
    sim = _straggler_setup()
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[1].swapping = True
    sim.policy.on_major_fault(sim.procs[1], True, 3)
    assert sim.policy.owner == 0 and sim.policy.sync_transfers == 0


def test_sync_transfer_extension_hands_token_to_straggler():
    sim = _straggler_setup(sync_transfer=True)
    sim.policy._acquire(sim.procs[0], 0)
    sim.procs[1].swapping = True
    sim.policy.on_major_fault(sim.procs[1], True, 3)
    assert sim.policy.owner == 1 and sim.policy.sync_transfers == 1
    assert sim.policy.holds[-1][3] == "sync_transfer"


# ------------------------------------------------------------------ end to end
def test_bsp_run_completes_with_invariants_and_is_deterministic():
    def run():
        ws = [swl(f"r{i}", [(j * 7 + i) % 40 for j in range(400)], "bsp", [100, 200, 300]) for i in range(2)]
        ws.append(Workload("hog", [j % 60 for j in range(800)]))
        return run_simulation(ws, SimConfig(capacity_pages=70, seed=3, check_invariants=True), "token_lru")
    a, b = run(), run()
    assert all(p.done for p in a.procs)
    assert a.proc_rows() == b.proc_rows() and a.policy.summary() == b.policy.summary()
    assert [p.barriers_passed for p in a.procs[:2]] == [3, 3]


def test_live_peer_with_no_barriers_never_blocks_the_other_member():
    a = swl("a", [1, 2, 3, 4], barriers=[2])
    b = swl("b", [7] * 500, barriers=[])               # still running when a reaches its barrier
    sim = run_simulation([a, b], SimConfig(capacity_pages=10, quantum=10), "global_lru")
    assert sim.procs[0].sync_wait_ticks == 0 and sim.procs[0].finish_time < sim.procs[1].finish_time
