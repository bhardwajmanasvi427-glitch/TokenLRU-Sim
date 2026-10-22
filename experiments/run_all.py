#!/usr/bin/env python3
"""Run every experiment and write raw CSVs to results/.

    python experiments/run_all.py                 # defaults: 5 seeds, scale 1/8
    python experiments/run_all.py --seeds 3 --scale 0.0625     # quicker
    python experiments/run_all.py --shortage 0.3  # skip calibration, fixed shortage for all pairs

Stages
  1. calibration : BASELINE ONLY, seed 0, shortages 20/30/40/50 %; per pair pick the smallest shortage
                   at which the baseline shows "considerable performance degradation"
                   (paper Sec. 4.2 methodology).  The token policy is never consulted for this choice.
  2. main        : both policies, calibrated shortage, N seeds, with RSS/MAD/NAP samples.
  3. sweep       : both policies at 20/30/40/50 % shortage, N seeds (no samples).
  4. ablation    : token policy only: Case-2 hold threshold {off,2,4,8}x and assignment random/first_fault.
  5. section54   : the paper's two-process 53 MB / 58 MB example.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, POLICIES, run_pair, flatten, base_config                    # noqa: E402
from tolru import SimConfig, run_simulation, dedicated_elapsed                      # noqa: E402
from tolru.metrics import write_csv, slowdown                                        # noqa: E402
from workloads import PAIR_NAMES, build_section54                                    # noqa: E402

SHORTAGES = [0.2, 0.3, 0.4, 0.5]
DEGRADATION_THRESHOLD = 1.5       # baseline slowdown must exceed 1.5x its CPU-sharing floor (our definition)


def job(spec):
    out, h = run_pair(spec["pair"], scale=spec["scale"], seed=spec["seed"], shortage=spec["shortage"],
                      sample=spec.get("sample", False), policies=spec.get("policies", POLICIES),
                      cfg_overrides=spec.get("cfg"), hold_factor=spec.get("hold_factor", 4.0))
    proc_rows, sys_rows = flatten(out, h)
    tag = dict(variant=spec.get("variant", ""))
    for r in proc_rows + sys_rows:
        r.update(tag)
    samples, events, holds = [], [], []
    for pol, rec in out.items():
        sim = rec["sim"]
        if spec.get("sample"):
            for (t, pid, rss, mad, nap, swp, tok, pos) in sim.samples:
                samples.append(dict(pair=spec["pair"], seed=spec["seed"], policy=pol, t=t, pid=pid,
                                    name=sim.procs[pid].name, rss=rss, mad=mad, nap=nap, swapping=swp,
                                    holds_token=tok, progress_refs=pos))
        if pol == "token_lru":
            for (t, kind, pid, detail) in sim.policy.events:
                events.append(dict(pair=spec["pair"], seed=spec["seed"], t=t, event=kind, pid=pid,
                                   name=sim.procs[pid].name, detail=detail))
            for (pid, a, b, why) in sim.policy.holds:
                holds.append(dict(pair=spec["pair"], seed=spec["seed"], pid=pid, name=sim.procs[pid].name,
                                  start=a, end=b, reason=why))
    meta = dict(pair=spec["pair"], seed=spec["seed"], shortage=spec["shortage"], capacity=h["capacity"],
                dedicated=h["dedicated"], floor=h["floor_slowdown"], delays=h["delays"],
                config=h["cfg"].to_dict())
    return dict(proc=proc_rows, sys=sys_rows, samples=samples, events=events, holds=holds, meta=meta)


def collect(results, prefix, outdir, keep=("proc", "sys", "samples", "events", "holds")):
    names = dict(proc="process", sys="system", samples="samples", events="token_events", holds="token_holds")
    for k in keep:
        rows = [r for res in results for r in res[k]]
        if rows:
            write_csv(outdir / f"{prefix}_{names[k]}.csv", rows)


def calibrate(pairs, scale, pool, outdir, fixed=None):
    """Baseline-only: choose per-pair memory shortage (decided WITHOUT looking at the token policy)."""
    specs = [dict(pair=p, scale=scale, seed=0, shortage=s, policies=("global_lru",), variant="calib")
             for p in pairs for s in SHORTAGES]
    res = pool.map(job, specs)
    rows, chosen = [], {}
    for r in res:
        m = r["meta"]
        sd = [x["slowdown"] for x in r["proc"]]
        ratios = [a / b if (a and b) else None for a, b in zip(sd, m["floor"])]
        rows.append(dict(pair=m["pair"], shortage=m["shortage"], capacity_pages=m["capacity"],
                         baseline_slowdown_0=sd[0], baseline_slowdown_1=sd[1],
                         floor_slowdown_0=m["floor"][0], floor_slowdown_1=m["floor"][1],
                         degradation_ratio_max=max(x for x in ratios if x is not None),
                         baseline_major_faults=r["sys"][0]["total_major_faults"]))
    for p in pairs:
        cand = sorted((x for x in rows if x["pair"] == p), key=lambda x: x["shortage"])
        pick = next((x["shortage"] for x in cand if x["degradation_ratio_max"] >= DEGRADATION_THRESHOLD),
                    cand[-1]["shortage"])
        chosen[p] = fixed if fixed is not None else pick
    for x in rows:
        x["chosen"] = int(chosen[x["pair"]] == x["shortage"])
        x["criterion"] = f"max(baseline slowdown / CPU-sharing floor) >= {DEGRADATION_THRESHOLD}"
    write_csv(outdir / "calibration.csv", rows)
    return chosen


def section54(scale, outdir, seed=0):
    ws, delays, cap, meta = build_section54(scale=scale)
    rows_p, rows_s, samples, events = [], [], [], []
    cfg0 = SimConfig(capacity_pages=cap, seed=seed,
                     sample_interval=max(1, int(2 * 10_000 * scale)), nap_window=max(1, int(10_000 * scale)))
    ded = [dedicated_elapsed(w, cfg0) for w in ws]
    import dataclasses
    cfg = dataclasses.replace(cfg0, max_hold_ticks=int(4 * max(ded)), unsuitable_cooldown_ticks=int(4 * max(ded)))
    for pol in POLICIES:
        t0 = time.perf_counter()
        sim = run_simulation(ws, cfg, pol, delays)
        wall = time.perf_counter() - t0
        for r, d in zip(sim.proc_rows(), ded):
            rows_p.append(dict(pair="section54", policy=pol, seed=seed, **r, dedicated=d,
                               slowdown=slowdown(r["elapsed"], d)))
        sr = sim.system_row()
        rows_s.append(dict(pair="section54", policy=pol, seed=seed, capacity_pages=cap,
                           wall_seconds=round(wall, 3), **{k: v for k, v in sr.items() if k != "completion_order"},
                           completion_order=" > ".join(sim.procs[i].name for i in sr["completion_order"]),
                           **sim.policy.summary()))
        for (t, pid, rss, mad, nap, swp, tok, pos) in sim.samples:
            samples.append(dict(pair="section54", seed=seed, policy=pol, t=t, pid=pid, name=sim.procs[pid].name,
                                rss=rss, mad=mad, nap=nap, swapping=swp, holds_token=tok, progress_refs=pos))
        if pol == "token_lru":
            for (t, kind, pid, detail) in sim.policy.events:
                events.append(dict(pair="section54", seed=seed, t=t, event=kind, pid=pid,
                                   name=sim.procs[pid].name, detail=detail))
    write_csv(outdir / "section54_process.csv", rows_p)
    write_csv(outdir / "section54_system.csv", rows_s)
    write_csv(outdir / "section54_samples.csv", samples)
    write_csv(outdir / "section54_token_events.csv", events)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=0.125, help="page/reference scale vs paper (default 1/8)")
    ap.add_argument("--seeds", type=int, default=5, help="number of independent repetitions")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--shortage", type=float, default=None,
                    help="fixed memory shortage for every pair (skips calibration choice)")
    ap.add_argument("--pairs", nargs="*", default=PAIR_NAMES)
    ap.add_argument("--s54-scale", type=float, default=0.25, help="scale of the Section 5.4 example")
    ap.add_argument("--io-time", type=int, default=None, help="override simulated disk service time (ticks)")
    ap.add_argument("--quantum", type=int, default=None)
    ap.add_argument("--self-window", type=int, default=None)
    ap.add_argument("--self-ratio", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["sweep", "ablation", "section54"])
    a = ap.parse_args()

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg_over = {k: v for k, v in dict(io_time=a.io_time, quantum=a.quantum, self_supply_window=a.self_window,
                                      self_supply_ratio=a.self_ratio).items() if v is not None}
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    t_start = time.time()
    pool = mp.Pool(a.workers)

    print("[1/5] calibration (baseline only) ...", flush=True)
    chosen = calibrate(a.pairs, a.scale, pool, outdir, fixed=a.shortage)
    print("      chosen shortage:", chosen, flush=True)

    print("[2/5] main runs ...", flush=True)
    specs = [dict(pair=p, scale=a.scale, seed=s, shortage=chosen[p], sample=True, variant="main",
                  cfg=cfg_over or None) for p in a.pairs for s in seeds]
    res = pool.map(job, specs, chunksize=1)
    collect(res, "main", outdir)

    if "sweep" not in a.skip:
        print("[3/5] memory-pressure sweep ...", flush=True)
        specs = [dict(pair=p, scale=a.scale, seed=s, shortage=sh, sample=False, variant="sweep",
                      cfg=cfg_over or None) for p in a.pairs for sh in SHORTAGES for s in seeds]
        res = pool.map(job, specs, chunksize=1)
        collect(res, "sweep", outdir, keep=("proc", "sys"))

    if "ablation" not in a.skip:
        print("[4/5] ablations (token policy only) ...", flush=True)
        specs = []
        for p in a.pairs:
            for s in seeds:
                for hf in (0, 2.0, 4.0, 8.0):
                    specs.append(dict(pair=p, scale=a.scale, seed=s, shortage=chosen[p], policies=("token_lru",),
                                      hold_factor=hf, variant=f"hold_factor={hf if hf else 'off'}",
                                      cfg=cfg_over or None))
                specs.append(dict(pair=p, scale=a.scale, seed=s, shortage=chosen[p], policies=("token_lru",),
                                  variant="assignment=first_fault",
                                  cfg=dict(cfg_over, token_assignment="first_fault")))
        res = pool.map(job, specs, chunksize=1)
        collect(res, "ablation", outdir, keep=("proc", "sys"))
    pool.close()
    pool.join()

    if "section54" not in a.skip:
        print("[5/5] Section 5.4 example ...", flush=True)
        meta54 = section54(a.s54_scale, outdir)
    else:
        meta54 = None

    manifest = dict(scale=a.scale, seeds=seeds, chosen_shortage=chosen, section54=meta54,
                    degradation_threshold=DEGRADATION_THRESHOLD, cfg_overrides=cfg_over,
                    default_config=SimConfig().to_dict(), elapsed_wall_seconds=round(time.time() - t_start, 1),
                    note="All workloads are SYNTHETIC; times are simulated ticks.")
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"done in {time.time() - t_start:.0f}s -> {outdir}", flush=True)


if __name__ == "__main__":
    main()
