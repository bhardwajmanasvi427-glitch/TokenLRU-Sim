#!/usr/bin/env python3
"""Consolidate raw CSVs into summary tables and the paper comparison.

Reads  results/main_process.csv, main_system.csv, sweep_*.csv, ablation_*.csv
Writes results/consolidated_results.csv, paper_comparison.csv, paper_agreement.csv,
       sweep_summary.csv, ablation_summary.csv, results_table.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT                                       # noqa: E402
from paper_reference import PAPER_TABLE2                       # noqa: E402
from workloads import PAIR_NAMES, PAIR_LABELS                  # noqa: E402

RES = ROOT / "results"


def pct_red(base, new):
    base = np.asarray(base, dtype=float)
    new = np.asarray(new, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 * (base - new) / base
    out[base == 0] = np.nan                     # undefined, never a misleading number
    return out


def per_seed_table(proc: pd.DataFrame) -> pd.DataFrame:
    """One row per (pair, seed, pid) with baseline/token columns side by side + per-seed reductions."""
    keys = ["pair", "seed", "pid", "shortage"]
    cols = ["name", "major_faults", "minor_faults", "slowdown", "elapsed", "false_lru_cross", "false_lru_self"]
    b = proc[proc.policy == "global_lru"].set_index(keys)[cols].add_prefix("base_")
    t = proc[proc.policy == "token_lru"].set_index(keys)[cols].add_prefix("tok_")
    m = b.join(t).reset_index()
    m["pf_reduction_pct"] = pct_red(m.base_major_faults, m.tok_major_faults)
    m["sd_reduction_pct"] = pct_red(m.base_slowdown, m.tok_slowdown)
    return m


def consolidate(proc: pd.DataFrame) -> pd.DataFrame:
    m = per_seed_table(proc)
    rows = []
    for pair in PAIR_NAMES:
        for pid in (0, 1):
            g = m[(m.pair == pair) & (m.pid == pid)]
            if g.empty:
                continue
            r = dict(pair=pair, label=PAIR_LABELS[pair], pid=pid, program=g.base_name.iloc[0],
                     shortage=g.shortage.iloc[0], n_seeds=len(g))
            for k in ("base_major_faults", "tok_major_faults", "base_slowdown", "tok_slowdown",
                      "pf_reduction_pct", "sd_reduction_pct", "base_false_lru_cross", "tok_false_lru_cross"):
                r[k + "_mean"] = g[k].mean()
                r[k + "_std"] = g[k].std(ddof=1) if len(g) > 1 else 0.0
            r["pf_reduction_pct_of_means"] = float(pct_red([r["base_major_faults_mean"]],
                                                           [r["tok_major_faults_mean"]])[0])
            r["sd_reduction_pct_of_means"] = float(pct_red([r["base_slowdown_mean"]], [r["tok_slowdown_mean"]])[0])
            r["seeds_token_better_faults"] = int((g.tok_major_faults < g.base_major_faults).sum())
            r["seeds_token_better_slowdown"] = int((g.tok_slowdown < g.base_slowdown).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def paper_comparison(cons: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in cons.iterrows():
        P = PAPER_TABLE2[r.pair]
        i = int(r.pid)
        paper_pf_red_calc = float(pct_red([P["lru_pf"][i]], [P["tok_pf"][i]])[0])
        paper_sd_red_calc = float(pct_red([P["lru_sd"][i]], [P["tok_sd"][i]])[0])
        for metric, ours_b, ours_t, pb, pt in (
            ("major page faults", r.base_major_faults_mean, r.tok_major_faults_mean, P["lru_pf"][i], P["tok_pf"][i]),
            ("slowdown", r.base_slowdown_mean, r.tok_slowdown_mean, P["lru_sd"][i], P["tok_sd"][i]),
        ):
            ours_red = r.pf_reduction_pct_of_means if metric.startswith("major") else r.sd_reduction_pct_of_means
            paper_red = paper_pf_red_calc if metric.startswith("major") else paper_sd_red_calc
            rows.append(dict(
                pair=r.pair, program=P["names"][i], our_program=r.program, metric=metric,
                paper_baseline=pb, paper_token=pt, paper_reduction_pct=round(paper_red, 1),
                our_baseline=ours_b, our_token=ours_t, our_reduction_pct=ours_red,
                abs_diff_baseline=ours_b - pb, abs_diff_token=ours_t - pt,
                reduction_diff_points=ours_red - paper_red,
                # relative diffs only meaningful for slowdown (dimensionless); counts depend on our 1/8 scale
                rel_diff_baseline_pct=(100 * (ours_b - pb) / pb) if metric == "slowdown" else None,
                rel_diff_token_pct=(100 * (ours_t - pt) / pt) if metric == "slowdown" else None,
                direction_agrees=bool(np.sign(ours_red) == np.sign(paper_red)) if pd.notna(ours_red) else None,
                ours_seeds_better=int(r.seeds_token_better_faults if metric.startswith("major")
                                      else r.seeds_token_better_slowdown),
                n_seeds=int(r.n_seeds)))
    return pd.DataFrame(rows)


def sweep_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    m = per_seed_table(sweep)
    out = []
    for (pair, sh), g in m.groupby(["pair", "shortage"]):
        per_seed = g.groupby("seed").agg(base_pf=("base_major_faults", "sum"), tok_pf=("tok_major_faults", "sum"),
                                         base_sd=("base_slowdown", "mean"), tok_sd=("tok_slowdown", "mean"))
        out.append(dict(pair=pair, shortage=sh, n_seeds=len(per_seed),
                        base_total_faults=per_seed.base_pf.mean(), tok_total_faults=per_seed.tok_pf.mean(),
                        total_fault_reduction_pct=float(pct_red([per_seed.base_pf.mean()], [per_seed.tok_pf.mean()])[0]),
                        base_mean_slowdown=per_seed.base_sd.mean(), tok_mean_slowdown=per_seed.tok_sd.mean(),
                        mean_slowdown_reduction_pct=float(pct_red([per_seed.base_sd.mean()], [per_seed.tok_sd.mean()])[0]),
                        seeds_token_fewer_faults=int((per_seed.tok_pf < per_seed.base_pf).sum()),
                        seeds_token_lower_slowdown=int((per_seed.tok_sd < per_seed.base_sd).sum())))
    return pd.DataFrame(out)


def ablation_summary(abl: pd.DataFrame, main_proc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    allv = pd.concat([abl, main_proc[main_proc.policy == "token_lru"].assign(variant="default (hold=4x, random)")])
    for (pair, variant), g in allv.groupby(["pair", "variant"]):
        per_seed = g.groupby("seed").agg(pf=("major_faults", "sum"), sd=("slowdown", "mean"))
        rows.append(dict(pair=pair, variant=variant, n_seeds=len(per_seed),
                         mean_total_faults=per_seed.pf.mean(), mean_slowdown=per_seed.sd.mean()))
    return pd.DataFrame(rows)


def md_table(df, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(fmt.get(c, lambda v: str(v))(r[c]) for c in cols) + " |\n"
    return head + body


def main():
    proc = pd.read_csv(RES / "main_process.csv")
    cons = consolidate(proc)
    cons.to_csv(RES / "consolidated_results.csv", index=False)
    cmp_ = paper_comparison(cons)
    cmp_.to_csv(RES / "paper_comparison.csv", index=False)

    sweep = pd.read_csv(RES / "sweep_process.csv")
    sw = sweep_summary(sweep)
    sw.to_csv(RES / "sweep_summary.csv", index=False)
    abl = pd.read_csv(RES / "ablation_process.csv")
    ab = ablation_summary(abl, proc)
    ab.to_csv(RES / "ablation_summary.csv", index=False)

    # agreement summary
    ag = cmp_.groupby("metric").agg(n=("direction_agrees", "size"), directional_agreements=("direction_agrees", "sum"),
                                    mean_abs_reduction_diff_points=("reduction_diff_points", lambda s: s.abs().mean()))
    ag.to_csv(RES / "paper_agreement.csv")

    f1 = lambda v: f"{v:,.0f}" if pd.notna(v) else "n/a"
    f2 = lambda v: f"{v:.2f}" if pd.notna(v) else "n/a"
    f0p = lambda v: f"{v:+.0f}%" if pd.notna(v) else "n/a"
    t = cons.assign(prog=cons.program, sh=(cons.shortage * 100).round(0).astype(int).astype(str) + "%")
    txt = "## Measured results (this simulator; synthetic workloads; mean of seeds)\n\n"
    txt += md_table(t, ["label", "prog", "sh", "base_major_faults_mean", "tok_major_faults_mean", "pf_reduction_pct_of_means",
                        "base_slowdown_mean", "tok_slowdown_mean", "sd_reduction_pct_of_means", "seeds_token_better_slowdown"],
                    dict(base_major_faults_mean=f1, tok_major_faults_mean=f1, pf_reduction_pct_of_means=f0p,
                         base_slowdown_mean=f2, tok_slowdown_mean=f2, sd_reduction_pct_of_means=f0p))
    (RES / "results_table.md").write_text(txt)
    print(txt)
    print(cmp_[cmp_.metric == "slowdown"][["pair", "program", "paper_baseline", "paper_token", "our_baseline", "our_token",
                                           "paper_reduction_pct", "our_reduction_pct", "direction_agrees"]].round(2).to_string())
    print(cmp_[cmp_.metric != "slowdown"][["pair", "program", "paper_reduction_pct", "our_reduction_pct", "direction_agrees",
                                           "ours_seeds_better"]].round(1).to_string())
    print(ag)
    print(sw.round(2).to_string())
    print(ab.round(2).to_string())


if __name__ == "__main__":
    main()
