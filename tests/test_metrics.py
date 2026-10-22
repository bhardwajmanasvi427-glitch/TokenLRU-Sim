"""Metric-definition tests (area 15) plus workload/catalog sanity checks."""
import math

import pytest

from tolru.metrics import pct_reduction, slowdown, mean_std, write_csv
from workloads import build_pair, build_section54, capacity_for_shortage, PAIR_NAMES


def test_page_fault_reduction_formula():
    assert pct_reduction(328_551, 179_908) == pytest.approx(45.24, abs=0.01)   # paper Table 2 value
    assert pct_reduction(77_680, 168_215) == pytest.approx(-116.55, abs=0.01)  # increase stays negative


def test_slowdown_and_slowdown_reduction():
    assert slowdown(523, 100) == pytest.approx(5.23)
    assert pct_reduction(5.23, 2.63) == pytest.approx(49.71, abs=0.01)


def test_undefined_metrics_are_explicit_none_not_misleading_numbers():
    assert pct_reduction(0, 0) is None and pct_reduction(0, 5) is None and pct_reduction(None, 1) is None
    assert slowdown(10, 0) is None and slowdown(None, 5) is None and slowdown(5, None) is None


def test_mean_std():
    assert mean_std([1, 2, 3]) == (2.0, 1.0)
    assert mean_std([None, 4]) == (4.0, 0.0)
    assert mean_std([]) == (None, None)


def test_write_csv_roundtrip(tmp_path):
    p = write_csv(tmp_path / "x.csv", [{"a": 1, "b": None}, {"a": 2, "c": 3}])
    txt = p.read_text().splitlines()
    assert txt[0] == "a,b,c" and txt[1] == "1,," and txt[2] == "2,,3"


@pytest.mark.parametrize("pair", PAIR_NAMES)
def test_pair_workloads_are_reproducible_and_labelled_synthetic(pair):
    a, da, ma = build_pair(pair, scale=0.02, seed=3)
    b, db, mb = build_pair(pair, scale=0.02, seed=3)
    assert da == db and ma == mb
    for x, y in zip(a, b):
        assert list(x.trace) == list(y.trace)
        assert x.synthetic and x.name.endswith("*") and len(x) > 0
        assert x.peak_mad() >= 1


def test_lu_pair_is_two_runs_of_the_same_program_and_frees_its_matrix():
    ws, _, _ = build_pair("lu1_lu2", scale=0.02, seed=0)
    assert list(ws[0].trace) == list(ws[1].trace)             # paper footnote 4: same program, same input
    assert (ws[0].trace < 0).sum() > 0                         # matrix released and re-allocated each iteration
    assert ws[0].mad_steps[0][1] > 0 and any(m == 0 for _, m in ws[0].mad_steps)


def test_capacity_for_shortage_and_validation():
    ws, _, _ = build_pair("bitr_gcc", scale=0.02)
    assert capacity_for_shortage(ws, 0.0) == sum(w.peak_mad() for w in ws)
    assert capacity_for_shortage(ws, 0.5) < capacity_for_shortage(ws, 0.2)
    with pytest.raises(ValueError):
        capacity_for_shortage(ws, 1.0)
    with pytest.raises(ValueError):
        build_pair("nope")


def test_section54_sizes():
    ws, delays, cap, meta = build_section54(scale=1.0, loops=10)
    assert meta["small_pages"] == 53 * 256 and meta["large_pages"] == 58 * 256 and cap == 60 * 256
    assert len(ws[0]) == 10 * 53 * 256              # 10 loops at full size, no shrink yet
    # shrink rule: after 10 loops SIZE becomes int(0.9*SIZE)
    ws2, *_ = build_section54(scale=1.0, loops=11)
    assert len(ws2[0]) == 10 * 53 * 256 + int(0.9 * 53 * 256)
