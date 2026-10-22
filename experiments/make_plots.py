#!/usr/bin/env python3
"""Generate all measured-data figures from results/*.csv into plots/.

Every figure has a provenance caption: MEASURED (this simulator, synthetic workloads),
PAPER-DERIVED (Table 2 reference values) or both.  The data behind each figure is saved
to plots/data/figNN_*.csv so figures can be regenerated/inspected without re-running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT                                  # noqa: E402
from paper_reference import PAPER_TABLE2                  # noqa: E402
from summarize import consolidate, pct_red                # noqa: E402
from workloads import PAIR_NAMES, PAIR_LABELS             # noqa: E402

RES, PLOTS = ROOT / "results", ROOT / "plots"
DATA = PLOTS / "data"
PLOTS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# validated with the dataviz palette validator (blue/orange pass CVD + contrast checks)
C_BASE, C_TOK = "#2a78d6", "#eb6834"
C_P0, C_P1 = "#6250d6", "#008300"            # process identity (not policy) in RSS/timeline plots
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POL_LABEL = {"global_lru": "Global LRU (baseline)", "token_lru": "Token-ordered LRU"}

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 160, "font.size": 10, "axes.titlesize": 11,
    "axes.labelsize": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def caption(fig, text, y=0.005):
    fig.text(0.01, y, text, fontsize=7.8, color=INK2, ha="left", va="bottom", wrap=True)


def save(fig, name):
    fig.savefig(PLOTS / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def prog_labels(cons):
    return [f"{PAIR_LABELS[r.pair].split(' / ')[r.pid]}" for r in cons.itertuples()]


# ----------------------------------------------------------------------------- fig 1-3
def fig_totals(main_proc, cons):
    tot = (main_proc.groupby(["pair", "seed", "policy"]).major_faults.sum().reset_index()
           .groupby(["pair", "policy"]).major_faults.agg(["mean", "std"]).reset_index())
    tot.to_csv(DATA / "fig01_total_faults.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(PAIR_NAMES))
    w = 0.36
    for k, pol in enumerate(("global_lru", "token_lru")):
        d = tot[tot.policy == pol].set_index("pair").loc[PAIR_NAMES]
        b = ax.bar(x + (k - 0.5) * w, d["mean"], w - 0.04, yerr=d["std"], color=[C_BASE, C_TOK][k],
                   label=POL_LABEL[pol], error_kw=dict(lw=1, capsize=3, ecolor=INK2))
        for xi, v in zip(x + (k - 0.5) * w, d["mean"]):
            ax.text(xi, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [PAIR_LABELS[p] for p in PAIR_NAMES])
    ax.set_ylabel("major page faults (both programs, sum)")
    ax.set_title("Total page faults: global LRU vs token-ordered LRU")
    ax.legend()
    caption(fig, "MEASURED: synthetic workloads, 1/8 scale, memory shortage calibrated per pair on baseline only. "
                 "Bars = mean over 5 seeds, whiskers = std. Counts are NOT comparable to the paper's absolute counts.")
    save(fig, "fig01_total_page_faults.png")


def fig_per_process(cons):
    cons[["pair", "program", "base_major_faults_mean", "base_major_faults_std", "tok_major_faults_mean",
          "tok_major_faults_std"]].to_csv(DATA / "fig02_per_process_faults.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(cons))
    w = 0.38
    ax.bar(x - w / 2, cons.base_major_faults_mean, w - 0.04, yerr=cons.base_major_faults_std, color=C_BASE,
           label=POL_LABEL["global_lru"], error_kw=dict(lw=1, capsize=2.5, ecolor=INK2))
    ax.bar(x + w / 2, cons.tok_major_faults_mean, w - 0.04, yerr=cons.tok_major_faults_std, color=C_TOK,
           label=POL_LABEL["token_lru"], error_kw=dict(lw=1, capsize=2.5, ecolor=INK2))
    ax.set_xticks(x, [f"{PAIR_LABELS[r.pair]}\n{r.program}" for r in cons.itertuples()], fontsize=7.5)
    ax.set_ylabel("major page faults")
    ax.set_title("Per-process page faults for each workload pair")
    ax.legend()
    caption(fig, "MEASURED: mean +/- std over 5 seeds (synthetic workloads). A process whose faults rise under the token "
                 "policy is shown as measured, not hidden.")
    save(fig, "fig02_per_process_page_faults.png")


def fig_slowdown(cons):
    cons[["pair", "program", "base_slowdown_mean", "base_slowdown_std", "tok_slowdown_mean",
          "tok_slowdown_std"]].to_csv(DATA / "fig03_slowdown.csv", index=False)
    floor = pd.read_csv(RES / "main_process.csv").query("policy=='global_lru'").groupby(["pair", "pid"]) \
        .cpu_sharing_floor_slowdown.mean()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(cons))
    w = 0.38
    ax.bar(x - w / 2, cons.base_slowdown_mean, w - 0.04, yerr=cons.base_slowdown_std, color=C_BASE,
           label=POL_LABEL["global_lru"], error_kw=dict(lw=1, capsize=2.5, ecolor=INK2))
    ax.bar(x + w / 2, cons.tok_slowdown_mean, w - 0.04, yerr=cons.tok_slowdown_std, color=C_TOK,
           label=POL_LABEL["token_lru"], error_kw=dict(lw=1, capsize=2.5, ecolor=INK2))
    for i, r in enumerate(cons.itertuples()):
        f = floor[(r.pair, r.pid)]
        ax.plot([i - 0.45, i + 0.45], [f, f], color=INK, lw=1.2, ls=(0, (3, 2)),
                label="CPU-sharing floor (no page faults)" if i == 0 else None)
    ax.set_xticks(x, [f"{PAIR_LABELS[r.pair]}\n{r.program}" for r in cons.itertuples()], fontsize=7.5)
    ax.set_ylabel("slowdown = interacting time / dedicated time")
    ax.set_title("Execution slowdown")
    ax.legend()
    caption(fig, "MEASURED (synthetic workloads, 5 seeds). Dashed line: slowdown the pair would have with unlimited memory "
                 "(CPU sharing only) -- slowdown above it is attributable to paging.")
    save(fig, "fig03_slowdown.png")


# ----------------------------------------------------------------------------- fig 4
def fig_rss(samples):
    s0 = samples[samples.seed == 0]
    fig, axes = plt.subplots(len(PAIR_NAMES), 2, figsize=(11, 12), sharey="row")
    for i, pair in enumerate(PAIR_NAMES):
        for j, pol in enumerate(("global_lru", "token_lru")):
            ax = axes[i, j]
            d = s0[(s0.pair == pair) & (s0.policy == pol)]
            for pid, col in ((0, C_P0), (1, C_P1)):
                q = d[d.pid == pid]
                if q.empty:
                    continue
                nm = q.name.iloc[0]
                ax.plot(q.t / 1000, q.rss, color=col, lw=1.4)
                ax.plot(q.t / 1000, q.mad, color=col, lw=0.8, ls=(0, (2, 2)), alpha=0.7)
                ax.annotate(nm, (q.t.iloc[-1] / 1000, q.rss.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                            fontsize=7, color=INK2, va="center", annotation_clip=False)
            ax.set_title(f"{PAIR_LABELS[pair]} - {POL_LABEL[pol]}", fontsize=9)
            ax.set_xlabel("simulated time (10^3 ticks)", fontsize=8)
            if j == 0:
                ax.set_ylabel("pages")
            ax.margins(x=0.12)
    d = s0[["pair", "policy", "t", "pid", "name", "rss", "mad", "nap"]]
    d.to_csv(DATA / "fig04_rss_mad_timeseries.csv", index=False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=INK2, lw=1.4), Line2D([], [], color=INK2, lw=0.8, ls=(0, (2, 2)))],
               ["RSS (resident pages)", "MAD (allocation demand)"], loc="upper right", ncol=2, fontsize=8)
    fig.suptitle("Resident set size and memory allocation demand over simulated time", x=0.35, y=0.995)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    caption(fig, "MEASURED, seed 0, synthetic workloads. Compare the RSS of each process between the two columns (the "
                 "paper's Figs. 6-10 vs 12-16 analogue); time axes differ because completion times differ.", y=0.0)
    save(fig, "fig04_rss_vs_time.png")


def fig_mad_rss_nap(samples, pair="bitr_gcc"):
    s = samples[(samples.seed == 0) & (samples.pair == pair)]
    names = s.groupby("pid").name.first()
    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex="col")
    for c, pid in enumerate((0, 1)):
        for r, (col, lab) in enumerate((("mad", "MAD (allocation demand)"), ("rss", "RSS (resident pages)"),
                                        ("nap", "NAP (pages touched in last window)"))):
            ax = axes[r, c]
            for pol, colr in (("global_lru", C_BASE), ("token_lru", C_TOK)):
                q = s[(s.policy == pol) & (s.pid == pid)]
                ax.plot(q.t / 1000, q[col], color=colr, lw=1.2, label=POL_LABEL[pol])
            ax.set_ylabel(lab.split(" (")[0] + " (pages)")
            if r == 0:
                ax.set_title(f"{names[pid]}  -  {lab.split(' (')[0]}/RSS/NAP shown separately", fontsize=9)
            ax.text(0.99, 0.92, lab, transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)
        axes[2, c].set_xlabel("simulated time (10^3 ticks)")
    axes[0, 0].legend(fontsize=7.5, loc="center right")
    fig.suptitle(f"MAD, RSS and NAP as separate quantities - {PAIR_LABELS[pair]}", y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    caption(fig, "MEASURED, seed 0, synthetic workloads. NAP window = 1 scaled paper-second. MAD is defined by the "
                 "workload generator; RSS and NAP are measured by the simulator.", y=0.0)
    save(fig, "fig08_mad_rss_nap.png")


# ----------------------------------------------------------------------------- fig 5
def fig_token_timeline(holds, samples, proc):
    fig, axes = plt.subplots(len(PAIR_NAMES), 1, figsize=(10, 8.5), sharex=False)
    reason_style = {"exit": ("o", "token released: owner exited"), "timeout": ("s", "forced release: hold timeout"),
                    "self_supply": ("D", "forced release: self-supply (Case 1)"), "end_of_run": ("x", "end of run")}
    seen = set()
    rows = []
    for ax, pair in zip(axes, PAIR_NAMES):
        pr = proc[(proc.pair == pair) & (proc.seed == 0) & (proc.policy == "token_lru")]
        for r in pr.itertuples():
            ax.barh(r.pid, (r.finish - r.start) / 1000, left=r.start / 1000, height=0.5, color="#e4e3df",
                    label="process lifetime" if "life" not in seen else None)
            seen.add("life")
            ax.text(r.start / 1000, r.pid + 0.32, r.name, fontsize=7, color=INK2, va="bottom")
        for h in holds[(holds.pair == pair) & (holds.seed == 0)].itertuples():
            ax.barh(h.pid, (h.end - h.start) / 1000, left=h.start / 1000, height=0.5, color=C_TOK,
                    label="token held" if "held" not in seen else None)
            seen.add("held")
            m, lab = reason_style.get(h.reason, ("o", h.reason))
            ax.plot(h.end / 1000, h.pid, m, color=INK, ms=5, label=lab if lab not in seen else None)
            seen.add(lab)
            rows.append(dict(pair=pair, pid=h.pid, start=h.start, end=h.end, reason=h.reason))
        sw = samples[(samples.pair == pair) & (samples.seed == 0) & (samples.policy == "token_lru")]
        ax.set_yticks([0, 1], [PAIR_LABELS[pair].split(" / ")[0], PAIR_LABELS[pair].split(" / ")[1]], fontsize=8)
        ax.set_ylim(-0.6, 1.9)
        ax.set_title(PAIR_LABELS[pair], fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
    axes[-1].set_xlabel("simulated time (10^3 ticks)")
    h_, l_ = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h_.append(hh)
                l_.append(ll)
    fig.legend(h_, l_, loc="upper right", fontsize=7.5, ncol=1)
    fig.suptitle("Token ownership and release timeline (token-ordered LRU)", x=0.35, y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    pd.DataFrame(rows).to_csv(DATA / "fig05_token_timeline.csv", index=False)
    caption(fig, "MEASURED, seed 0, synthetic workloads: orange = interval during which the process owns the global token.",
            y=0.0)
    save(fig, "fig05_token_timeline.png")


# ----------------------------------------------------------------------------- fig 6
def fig_pressure(sw):
    sw.to_csv(DATA / "fig06_pressure_sweep.csv", index=False)
    fig, axes = plt.subplots(2, len(PAIR_NAMES), figsize=(13, 5.2), sharex=True)
    for j, pair in enumerate(PAIR_NAMES):
        d = sw[sw.pair == pair].sort_values("shortage")
        x = d.shortage * 100
        axes[0, j].plot(x, d.base_total_faults, "-o", color=C_BASE, lw=1.6, ms=4, label=POL_LABEL["global_lru"])
        axes[0, j].plot(x, d.tok_total_faults, "-o", color=C_TOK, lw=1.6, ms=4, label=POL_LABEL["token_lru"])
        axes[1, j].plot(x, d.base_mean_slowdown, "-o", color=C_BASE, lw=1.6, ms=4)
        axes[1, j].plot(x, d.tok_mean_slowdown, "-o", color=C_TOK, lw=1.6, ms=4)
        axes[0, j].set_title(PAIR_LABELS[pair], fontsize=9)
        axes[1, j].set_xticks([20, 30, 40, 50])
    fig.supxlabel("memory shortage (% of summed peak MAD)", fontsize=10, y=0.06)
    axes[0, 0].set_ylabel("total major faults")
    axes[1, 0].set_ylabel("mean slowdown")
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Baseline vs token-ordered LRU under increasing memory pressure", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    caption(fig, "MEASURED, mean over 5 seeds, synthetic workloads. Shortage = 1 - capacity / sum of the programs' peak MAD. "
                 "The paper's 20-50% range refers to ITS machine settings; here it is an explicit sweep parameter.", y=0.0)
    save(fig, "fig06_memory_pressure.png")


# ----------------------------------------------------------------------------- fig 7
def fig_aggregate(cons):
    rows = []
    for r in cons.itertuples():
        P = PAPER_TABLE2[r.pair]
        i = r.pid
        rows.append(dict(pair=r.pair, program=r.program,
                         ours_pf_red=r.pf_reduction_pct_of_means, paper_pf_red=float(pct_red([P["lru_pf"][i]], [P["tok_pf"][i]])[0]),
                         ours_sd_red=r.sd_reduction_pct_of_means, paper_sd_red=float(pct_red([P["lru_sd"][i]], [P["tok_sd"][i]])[0])))
    d = pd.DataFrame(rows)
    d.to_csv(DATA / "fig07_aggregate_vs_paper.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(d))
    for ax, (o, p, ttl) in zip(axes, (("ours_pf_red", "paper_pf_red", "Page-fault reduction (%)"),
                                      ("ours_sd_red", "paper_sd_red", "Slowdown reduction (%)"))):
        ax.bar(x - 0.2, d[o], 0.38, color=C_TOK, label="Measured (this simulator, synthetic)")
        ax.bar(x + 0.2, d[p], 0.38, color="#bdbbb3", hatch="///", edgecolor="white", label="Paper Table 2 (published)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(x, [f"{PAIR_LABELS[a]}: {b.rstrip('*')}" for a, b in zip(d.pair, d.program)],
                      fontsize=7.5, rotation=40, ha="right")
        ax.set_title(ttl)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Aggregate comparison across workload pairs: measured vs published reductions", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    caption(fig, "MEASURED (orange, mean of 5 seeds, synthetic 1/8-scale workloads) vs PAPER-DERIVED (hatched, Table 2, real "
                 "Linux 2.2 kernel runs). Directional agreement only -- not numerical reproduction.", y=0.0)
    save(fig, "fig07_aggregate_vs_paper.png")


def fig_section54(s54, p54):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, pol in zip(axes, ("global_lru", "token_lru")):
        d = s54[s54.policy == pol]
        for pid, col in ((0, C_P0), (1, C_P1)):
            q = d[d.pid == pid]
            ax.plot(q.t / 1000, q.rss, color=col, lw=1.4, label=f"{q.name.iloc[0]} RSS")
        cap = 3840
        ax.axhline(cap, color=INK2, lw=0.9, ls=(0, (4, 3)))
        ax.text(0.99, cap * 1.01, "user memory (60 MB at 1/4 scale)", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7, color=INK2)
        ev = pd.read_csv(RES / "section54_token_events.csv")
        if pol == "token_lru":
            for e in ev.itertuples():
                ax.axvline(e.t / 1000, color=C_TOK, lw=1, ls=":")
                ax.text(e.t / 1000, 40, e.event.replace("_", " "), rotation=90, fontsize=7, color=C_TOK, va="bottom")
        ax.set_title(POL_LABEL[pol])
        ax.set_xlabel("simulated time (10^3 ticks)")
        ax.set_ylabel("resident pages")
        ax.legend(fontsize=8, loc="lower right")
    s54.to_csv(DATA / "fig09_section54_rss.csv", index=False)
    fig.suptitle("Section 5.4 example: 53 MB and 58 MB cyclic processes in 60 MB (scaled 1/4)", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    caption(fig, "MEASURED, synthetic re-implementation of the paper's program. Baseline = exact LRU on a cyclic scan (its "
                 "worst case), so the baseline slowdown here is far larger than the paper's Linux measurement.", y=0.0)
    save(fig, "fig09_section54.png")


def fig_sync():
    d = pd.read_csv(RES / "sync_runs.csv")
    regs = ["retain", "hold250k", "hold100k", "per-fault"]
    reg_lab = {"retain": "retain until\nexit (paper\ndefault)", "hold250k": "Case 2:\nrelease after\n250k ticks",
               "hold100k": "Case 2:\nrelease after\n100k ticks", "per-fault": "release when\nswapping ends\n(not paper\ndefault)"}
    var = ["lru", "random", "sync", "sync+xfer"]
    col = {"lru": C_BASE, "random": C_TOK, "sync": "#6250d6", "sync+xfer": "#008300"}
    lab = {"lru": "Global LRU", "random": "Token, random assignment", "sync": "Token, sync-aware priority (paper Sec. 5.3)",
           "sync+xfer": "Sync priority + hand-over (our extension, not in paper)"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, sh in zip(axes, sorted(d.shortage.unique())):
        for j, v in enumerate(var):
            m = [d[(d.shortage == sh) & (d.regime == r) & (d.variant == v)].app_slowdown for r in regs]
            ax.bar(np.arange(len(regs)) + (j - 1.5) * 0.2, [x.mean() for x in m], 0.18, color=col[v],
                   hatch="///" if v == "sync+xfer" else None, edgecolor="white" if v == "sync+xfer" else None,
                   yerr=[x.std() for x in m], error_kw=dict(lw=0.8, capsize=2, ecolor=INK2), label=lab[v])
        ax.set_yscale("log")
        ax.set_xticks(range(len(regs)), [reg_lab[r] for r in regs], fontsize=7)
        ax.set_title(f"{int(sh * 100)}% memory shortage")
    axes[0].set_ylabel("BSP application completion time\n/ time with unlimited memory (log scale)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Synchronisation-aware token priority: BSP application (2 ranks) + hog + short job", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    d.to_csv(DATA / "fig10_sync_priority.csv", index=False)
    caption(fig, "MEASURED, 6 seeds, synthetic scenario of our own design (the paper evaluates no BSP scenario). Under the paper-default "
                 "regime the priority never acts (identical to random); it matters when the token is re-assigned while ranks wait.", y=0.0)
    save(fig, "fig10_sync_priority.png")


def main():
    proc = pd.read_csv(RES / "main_process.csv")
    cons = consolidate(proc)
    samples = pd.read_csv(RES / "main_samples.csv")
    holds = pd.read_csv(RES / "main_token_holds.csv")
    sw = pd.read_csv(RES / "sweep_summary.csv") if (RES / "sweep_summary.csv").exists() else None
    fig_totals(proc, cons)
    fig_per_process(cons)
    fig_slowdown(cons)
    fig_rss(samples)
    fig_token_timeline(holds, samples, proc)
    if sw is not None:
        fig_pressure(sw)
    fig_aggregate(cons)
    fig_mad_rss_nap(samples)
    if (RES / "sync_runs.csv").exists():
        fig_sync()
    fig_section54(pd.read_csv(RES / "section54_samples.csv"), pd.read_csv(RES / "section54_process.csv"))


if __name__ == "__main__":
    main()
