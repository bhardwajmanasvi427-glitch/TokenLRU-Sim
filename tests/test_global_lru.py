"""Tests for the conventional global-LRU baseline (test areas 1, 12 and baseline invariants)."""
import pytest

from tolru import SimConfig, run_simulation, dedicated_elapsed
from conftest import make_sim, seed_resident, wl


def test_lru_known_reference_sequence():
    # Classic textbook string; LRU with 3 frames has exactly 10 faults (5 cold + 5 replacement).
    seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    sim = run_simulation([wl("p", seq)], SimConfig(capacity_pages=3), "global_lru")
    p = sim.procs[0]
    assert p.minor_faults == 5                  # 5 distinct pages, each first-touched once
    assert p.major_faults == 5                  # refaults of evicted pages
    assert p.minor_faults + p.major_faults == 10


def test_global_lru_picks_oldest_page_across_processes():
    sim = make_sim(6, [[0], [0]], policy="global_lru")
    seed_resident(sim, 0, [(1, 10), (2, 40)])
    seed_resident(sim, 1, [(7, 20), (8, 30)])
    victim, page = sim.policy.select_victim(sim.procs[1], now=50)
    assert victim.pid == 0 and page == 1        # ts=10 is the global LRU page


def test_baseline_ignores_swapping_status_and_has_no_token():
    sim = make_sim(6, [[0], [0]], policy="global_lru")
    seed_resident(sim, 0, [(1, 10)])
    seed_resident(sim, 1, [(7, 20)])
    sim.procs[0].swapping = True                # would be "protected" under the token policy
    victim, page = sim.policy.select_victim(sim.procs[1], now=50)
    assert victim.pid == 0                      # baseline does NOT skip it
    assert not hasattr(sim.policy, "owner")
    assert "protected_skips" not in sim.policy.summary()


def test_ties_break_deterministically_on_lowest_pid():
    sim = make_sim(6, [[0], [0]], policy="global_lru")
    seed_resident(sim, 0, [(1, 10)])
    seed_resident(sim, 1, [(7, 10)])
    assert sim.policy.select_victim(sim.procs[1], 50)[0].pid == 0


def test_no_victim_when_nothing_resident():
    sim = make_sim(2, [[0]], policy="global_lru")
    assert sim.policy.select_victim(sim.procs[0], 0) is None


# ------------------------------------------------------------------ edge cases (area 12)
def test_empty_trace_completes_immediately():
    sim = run_simulation([wl("e", [])], SimConfig(capacity_pages=2), "global_lru")
    assert sim.procs[0].done and sim.procs[0].elapsed() == 0
    assert sim.system_row()["total_major_faults"] == 0


def test_empty_trace_alongside_real_process():
    sim = run_simulation([wl("e", []), wl("r", [1, 2, 1, 2])], SimConfig(capacity_pages=2), "token_lru")
    assert all(p.done for p in sim.procs)


def test_repeated_references_hit_after_first_touch():
    sim = run_simulation([wl("r", [5] * 100)], SimConfig(capacity_pages=1), "global_lru")
    p = sim.procs[0]
    assert (p.minor_faults, p.major_faults) == (1, 0)
    assert p.elapsed() == 100                   # 100 refs * cpu_cost 1


@pytest.mark.parametrize("bad", [dict(capacity_pages=0), dict(capacity_pages=-3), dict(io_time=0),
                                 dict(quantum=0), dict(token_assignment="nope"),
                                 dict(self_supply_ratio=0.0), dict(self_supply_window=0),
                                 dict(max_hold_ticks=0), dict(unsuitable_cooldown_ticks=-1)])
def test_invalid_configuration_rejected(bad):
    kw = dict(capacity_pages=4)
    kw.update(bad)
    with pytest.raises(ValueError):
        SimConfig(**kw).validate()


@pytest.mark.parametrize("policy", ["global_lru", "token_lru"])
def test_single_frame_two_processes_makes_progress(policy):
    traces = [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]]
    sim = run_simulation([wl("a", traces[0]), wl("b", traces[1])],
                         SimConfig(capacity_pages=1, check_invariants=True), policy)
    assert all(p.done for p in sim.procs)       # no deadlock with a single frame


def test_dedicated_run_has_no_major_faults_and_elapsed_equals_refs():
    w = wl("d", [1, 2, 3, 1, 2, 3, 4, 5])
    assert dedicated_elapsed(w, SimConfig(capacity_pages=2)) == len(w)


def test_start_delay_respected():
    sim = run_simulation([wl("a", [1, 2]), wl("b", [1, 2])], SimConfig(capacity_pages=8), "global_lru",
                         start_delays=[0, 500])
    assert sim.procs[1].start_time == 500 and sim.procs[1].finish_time >= 502


# ------------------------------------------------------------------ free / unmap op (negative page id)
def test_unmap_releases_frame_and_retouch_is_zero_fill_not_major():
    # touch 1,2 ; free page 1 (-2) ; touch 3 (fits in the freed frame) ; re-touch 1 -> minor, never major
    sim = run_simulation([wl("f", [1, 2, -2, 3, 1])], SimConfig(capacity_pages=2, check_invariants=True),
                         "global_lru")
    p = sim.procs[0]
    assert p.major_faults == 0 and p.minor_faults == 4


def test_unmap_of_swapped_out_page_clears_backing_copy():
    # capacity 1: page 1 is evicted by 2; freeing 1 then touching it again must be zero-fill, not a swap-in
    sim = run_simulation([wl("f", [1, 2, -2, 1])], SimConfig(capacity_pages=1), "global_lru")
    assert sim.procs[0].major_faults == 0


def test_distinct_pages_ignores_free_ops():
    assert wl("d", [1, 2, -2, 1]).distinct_pages() == 2
