#!/usr/bin/env python3
"""Build the technical report (HTML + PDF) from results/*.csv, plots/*.png and captured outputs.

    python experiments/build_report.py          -> docs/technical_report.html and docs/technical_report.pdf

All numbers in tables come from the CSVs; the narrative quotes values that were read from those same CSVs
(re-check them if you change workloads/parameters and rerun).
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT                                   # noqa: E402
from paper_reference import PAPER_TABLE2                   # noqa: E402

RES, PLOTS, DOCS = ROOT / "results", ROOT / "plots", ROOT / "docs"


def img(path, caption, width="100%"):
    b = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b}" style="width:{width}"/>'
            f'<figcaption>{caption}</figcaption></figure>')


def table(df, fmt=None, cls=""):
    fmt = fmt or {}
    h = "<table class='%s'><thead><tr>%s</tr></thead><tbody>" % (cls, "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns))
    for _, r in df.iterrows():
        h += "<tr>" + "".join(f"<td>{(fmt.get(c, str)(r[c]) if pd.notna(r[c]) else 'n/a')}</td>" for c in df.columns) + "</tr>"
    return h + "</tbody></table>"


n0 = lambda v: f"{v:,.0f}"
n2 = lambda v: f"{v:.2f}"
sg = lambda v: f"{v:+.0f}%"

cons = pd.read_csv(RES / "consolidated_results.csv")
cmp_ = pd.read_csv(RES / "paper_comparison.csv")
ag = pd.read_csv(RES / "paper_agreement.csv")
sw = pd.read_csv(RES / "sweep_summary.csv")
abl = pd.read_csv(RES / "ablation_summary.csv")
calib = pd.read_csv(RES / "calibration.csv")
s54p = pd.read_csv(RES / "section54_process.csv")
s54s = pd.read_csv(RES / "section54_system.csv")
msys = pd.read_csv(RES / "main_system.csv")
mproc = pd.read_csv(RES / "main_process.csv")

# ---------------------------------------------------------------- tables
t_main = cons.assign(shortage=(cons.shortage * 100).round().astype(int).astype(str) + "%")[
    ["label", "program", "shortage", "base_major_faults_mean", "tok_major_faults_mean", "pf_reduction_pct_of_means",
     "base_slowdown_mean", "tok_slowdown_mean", "sd_reduction_pct_of_means", "seeds_token_better_slowdown"]]
t_main.columns = ["Pair", "Program", "Shortage", "LRU faults", "Token faults", "Fault red.", "LRU slowdown",
                  "Token slowdown", "Slowdown red.", "Seeds better (of 5)"]
T_MAIN = table(t_main, {"LRU faults": n0, "Token faults": n0, "Fault red.": sg, "LRU slowdown": n2,
                        "Token slowdown": n2, "Slowdown red.": sg})

rows = []
for _, r in cmp_[cmp_.metric == "major page faults"].iterrows():
    s = cmp_[(cmp_.pair == r.pair) & (cmp_.program == r.program) & (cmp_.metric == "slowdown")].iloc[0]
    rows.append({"Pair": r.pair, "Program": r.program,
                 "Paper faults LRU→token": f"{r.paper_baseline:,.0f} → {r.paper_token:,.0f} ({r.paper_reduction_pct:+.0f}%)",
                 "Ours faults LRU→token": f"{r.our_baseline:,.0f} → {r.our_token:,.0f} ({r.our_reduction_pct:+.0f}%)",
                 "Paper slowdown": f"{s.paper_baseline:.2f} → {s.paper_token:.2f} ({s.paper_reduction_pct:+.0f}%)",
                 "Ours slowdown": f"{s.our_baseline:.2f} → {s.our_token:.2f} ({s.our_reduction_pct:+.0f}%)",
                 "Direction (faults / slowdown)": f"{'yes' if r.direction_agrees else 'NO'} / {'yes' if s.direction_agrees else 'NO'}"})
T_CMP = table(pd.DataFrame(rows))
T_AG = table(ag.rename(columns={"metric": "Metric", "n": "Programs", "directional_agreements": "Same sign as paper",
                                "mean_abs_reduction_diff_points": "Mean |difference| (pct. points)"}),
             {"Mean |difference| (pct. points)": n0})

s = sw.assign(shortage=(sw.shortage * 100).round().astype(int).astype(str) + "%")[
    ["pair", "shortage", "base_total_faults", "tok_total_faults", "total_fault_reduction_pct", "base_mean_slowdown",
     "tok_mean_slowdown", "mean_slowdown_reduction_pct", "seeds_token_lower_slowdown"]]
s.columns = ["Pair", "Shortage", "LRU faults", "Token faults", "Fault red.", "LRU mean slowdown", "Token mean slowdown",
             "Slowdown red.", "Seeds better (of 5)"]
T_SW = table(s, {"LRU faults": n0, "Token faults": n0, "Fault red.": sg, "LRU mean slowdown": n2,
                 "Token mean slowdown": n2, "Slowdown red.": sg})

ab = abl.pivot(index="pair", columns="variant", values="mean_slowdown").reset_index()
T_AB = table(ab, {c: n2 for c in ab.columns if c != "pair"})

fl = msys.groupby("policy")[["false_lru_cross_total", "false_lru_self_total", "token_acquisitions", "protected_skips",
                             "fallback_evictions", "forced_release_timeout", "forced_release_self_supply"]].sum().reset_index()
fl.columns = ["Policy", "False-LRU evictions by OTHER processes", "False-LRU evictions by own fault",
              "Token acquisitions", "Protected skips", "Fallback evictions", "Forced releases (timeout)",
              "Forced releases (self-supply)"]
T_FL = table(fl, {c: (lambda v: f"{v:,.0f}") for c in fl.columns if c != "Policy"})

a54 = s54p[["policy", "name", "major_faults", "elapsed", "dedicated", "slowdown"]].copy()
a54.columns = ["Policy", "Process", "Major faults", "Elapsed (ticks)", "Dedicated (ticks)", "Slowdown"]
T_54 = table(a54, {"Major faults": n0, "Elapsed (ticks)": n0, "Dedicated (ticks)": n0, "Slowdown": n2})

cal = calib.assign(shortage=(calib.shortage * 100).round().astype(int).astype(str) + "%")[
    ["pair", "shortage", "degradation_ratio_max", "chosen"]]
cal.columns = ["Pair", "Shortage", "Baseline slowdown / CPU-sharing floor (max over programs)", "Chosen"]
T_CAL = table(cal, {"Baseline slowdown / CPU-sharing floor (max over programs)": n2, "Chosen": lambda v: "✔" if v else ""})

# workload audit table (what the paper says vs what we used)
AUDIT = pd.DataFrame([
    ["gzip", "Table 1: 197.4 MB, 248.7 s; Fig. 2: MAD ≈49.5k pages from start, NAP ≈1k with sweep spikes at ≈42/50/90/97/148/150/198/200/245 s",
     "49,500 pages; hot set 1,000; 9 sweep events at those times/sizes; starts ≈250 s into the pair (Fig. 6)", "Corrected (was 6 sweeps)"],
    ["vortex1 / vortex3", "Table 1: 115.0 / 131.2 MB, 342.3 / 398.0 s; Fig. 3: stair-climbing MAD, RSS ≈ 57–80% of MAD",
     "29,400 / 33,600 pages, 13 / 14 stairs, working set 60% of peak MAD", "Corrected (WS was 75%)"],
    ["bit-r", "Table 1: 131.3 MB, 326.1 s; Fig. 3: RSS ≈33k for 0–168 s then ≈1.8k (plotted MAD ≈65.5k, inconsistent with Table 1)",
     "33,600 pages cyclic for 168 s, then 1,800-page hot set for 158 s", "Corrected (switch was 165 s); MAD discrepancy noted"],
    ["gcc", "Table 1: 145.0 MB max, 218.7 s; Fig. 1: ≈3–4k to 75 s, spike ≈33.5k 75–85 s, ≈12.5k to 108 s, ≈5k to 182 s, spike ≈36.5k 182–200 s, ≈13k to 218.7 s; Fig. 8: starts ≈250 s in vortex3/gcc",
     "Phases 3.5k/4k/33.5k/12.5k/5k/36.5k/13k with those durations; memory is freed between phases", "Corrected (durations and frees)"],
    ["LU", "Table 1: 161.2 MB, 98.0 s; Fig. 5: MAD ≈40.3k, matrix released and re-allocated at ≈0/21/41/61/81 s, NAP ≈2.2k pages/s between",
     "40,300 pages; 5 iterations of free + re-allocate + slow sequential traversal of the whole matrix; LU-1 = LU-2 (same program, footnote 4)",
     "Corrected twice: (1) added free/re-allocate; (2) hot set replaced by whole-matrix traversal"],
    ["Section 5.4 program", "Sec. 5.4 code: 1000 loops, SIZE ×0.9 every 10 loops; 53 MB and 58 MB arrays, 60 MB memory",
     "Same loop structure, scaled 1/4 (3,392 and 3,712 pages in 3,840 frames)", "Verified"],
], columns=["Program", "What the paper states", "What the synthetic workload uses", "Status"])
T_AUDIT = table(AUDIT)

MAP = pd.DataFrame([
    ["Global, mutually exclusive token (available / held)", "TokenOrderedLRU.owner, _acquire, _release", "Implemented"],
    ["Token requested only on a page fault caused by memory shortage, before the swap-in (do_swap_page)", "Simulator._fault → policy.on_major_fault(shortage = no free frame)", "Implemented"],
    ["Per-process swapping status; on while swapping, off on return to normal execution", "Process.swapping; set on major fault, cleared on next dispatch", "Implemented (simulator interpretation)"],
    ["__get_free_pages() skips the program holding both token and swapping status", "TokenOrderedLRU.protected() and select_victim()", "Implemented"],
    ["Protection ends as soon as swapping ends", "protected() re-evaluated on every victim search", "Implemented"],
    ["Exception handler: take the holder's own LRU pages if no one else can supply", "fallback branch of select_victim()", "Implemented"],
    ["Case 1: holder repeatedly selects pages from itself → release + token-unsuitable", "_record_holder_fault (window / ratio)", "Implemented (thresholds ours)"],
    ["Case 2: holding time over threshold → release + token-unsuitable for a period", "on_tick, unsuitable_until", "Implemented (threshold ours)"],
    ["Random token assignment among interacting programs", "_choose_acquirer (seeded RNG)", "Implemented"],
    ["Priority for processes others are waiting on (synchronisation / blocking messages)", "Workload.sync_group/barriers, Simulator.waiters_on(), TokenOrderedLRU._choose_acquirer", "Implemented for barrier-synchronised processes on one node; remote nodes and message passing are not modelled"],
    ["(Extension, not in the paper) hand a held token to a waited-on process", "cfg.sync_transfer (default off)", "Our addition, evaluated separately"],
    ["Linux 2.2.14 kernel implementation", "—", "NOT implemented (simulator only)"],
], columns=["Paper element", "Code", "Status"])
T_MAP = table(MAP)

syn = pd.read_csv(RES / "sync_summary.csv")
syp = pd.read_csv(RES / "sync_paired.csv")
REG_NAME = {"retain": "retain until exit (paper default)", "hold250k": "Case 2 forced release after 250k ticks",
            "hold100k": "Case 2 forced release after 100k ticks", "per-fault": "release when swapping ends (not paper default)"}
sp = syn.pivot_table(index=["regime", "shortage"], columns="variant", values="app_slowdown_mean").reset_index()
prio = syn[syn.variant == "sync"][["regime", "shortage", "prio_assignments_mean"]]
sp = sp.merge(prio, on=["regime", "shortage"]).merge(syp[["regime", "shortage", "seeds_sync_better_than_random",
        "seeds_sync_worse_than_random", "seeds_identical"]], on=["regime", "shortage"])
sp["regime"] = sp["regime"].map(REG_NAME)
sp["shortage"] = (sp.shortage * 100).round().astype(int).astype(str) + "%"
sp = sp[["regime", "shortage", "lru", "random", "sync", "sync+xfer", "prio_assignments_mean",
         "seeds_sync_better_than_random", "seeds_sync_worse_than_random", "seeds_identical"]]
sp.columns = ["Token handling", "Shortage", "Global LRU", "Token, random", "Token, sync priority", "Sync priority + hand-over (extension)",
              "Priority assignments (mean)", "Seeds sync better", "Seeds sync worse", "Seeds identical"]
T_SYNC = table(sp, {c: n2 for c in ["Global LRU", "Token, random", "Token, sync priority",
                                     "Sync priority + hand-over (extension)", "Priority assignments (mean)"]})
tot = syp[["seeds_sync_better_than_random", "seeds_sync_worse_than_random", "seeds_identical"]].sum()
pf = syp[syp.regime == "per-fault"]
rt = syp[syp.regime == "retain"]

pytest_txt = html.escape((DOCS / "pytest_output.txt").read_text())
demo_txt = html.escape((DOCS / "demo_output.txt").read_text())

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin: 0 0 4px; } h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #bbb; padding-bottom: 3px; }
h3 { font-size: 11pt; margin-top: 14px; } .sub { color: #555; margin-bottom: 14px; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 8px 0 12px; page-break-inside: auto; }
th, td { border: 1px solid #ccc; padding: 3px 5px; vertical-align: top; text-align: left; } th { background: #eee; }
tr { page-break-inside: avoid; }
figure { margin: 10px 0 14px; page-break-inside: avoid; } figcaption { font-size: 8.3pt; color: #444; margin-top: 3px; }
pre { background: #f4f3ef; border: 1px solid #ccc; padding: 8px; font-size: 8pt; white-space: pre-wrap; }
.note { background: #fff6e5; border-left: 4px solid #eb6834; padding: 6px 10px; margin: 10px 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; }
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8"><title>Token-ordered LRU: simulator-based case study</title><style>{CSS}</style></head><body>
<h1>Token-ordered LRU page replacement: a simulator-based case study</h1>
<div class="sub">Technical report on the implementation and evaluation of the algorithm of S. Jiang and X. Zhang,
<i>Token-ordered LRU: an effective page replacement policy and its implementation in Linux systems</i>, Performance Evaluation 60 (2005) 5–29.</div>

<div class="note"><b>Scope statement.</b> This is a <b>discrete-event simulator</b> reproduction. It is not a Linux kernel modification, and all workloads
are <b>synthetic</b> approximations of the paper's programs (names carry a * in all figures), parameterised from the paper's Table 1 and Figs. 1–5 and scaled to 1/8.
Time is simulated ticks, never wall-clock. Published numbers are shown only as labelled reference data.</div>

<h2>1. Summary</h2>
<p>We implemented a conventional global-LRU baseline and the token-ordered LRU policy (paper Sections 5.2–5.3, including the synchronisation-aware priority) in one simulator, ran both on identical
workload pairs mirroring the paper's five interacting groups plus the Section 5.4 example, and measured page faults, slowdown, RSS/MAD/NAP, token activity
and false-LRU evictions over 5 seeds, a memory-pressure sweep and ablations.</p>
<ul>
<li><b>Mechanism confirmed.</b> Counted directly, evictions of pages belonging to a process that was itself faulting ("false LRU" evictions, paper Sec. 1) fell from
{fl.iloc[0, 1]:,.0f} (cross-process) + {fl.iloc[0, 2]:,.0f} (self) under global LRU to {fl.iloc[1, 1]:,.0f} + {fl.iloc[1, 2]:,.0f} under token-ordered LRU, summed over the five pairs and 5 seeds.</li>
<li><b>Clear wins</b> in bit-r (faults −98%, slowdown −51%), LU-1/LU-2 (slowdown 42.4 → 2.8 / 5.4), vortex3 in vortex3/gcc, and the Section 5.4 example (total faults 316,400 → 8,145).</li>
<li><b>Unfavourable or neutral cases are reported, not hidden:</b> gzip is unchanged; gcc is much worse in bit-r/gcc (faults 210 → 5,021) and the <i>pair-average</i>
slowdown there rises 2.01 → 2.17; vortex1 gets worse on average in vortex1/vortex3; at low shortage the policy is neutral or slightly worse.</li>
<li><b>Synchronisation-aware priority (Sec. 5.3) is implemented</b> and evaluated on a bulk-synchronous scenario of our own design (Section 6.7): under the paper's default
token handling it never acts; where the token is re-assigned while processes wait at barriers it helps in most runs ({int(tot.iloc[0])} of {int(tot.sum())} paired seed runs better than random assignment, {int(tot.iloc[1])} worse, {int(tot.iloc[2])} identical), with clear counter-examples.</li>
<li><b>Agreement with the paper is partial:</b> same sign of improvement for {int(ag.iloc[0, 2])}/10 programs (faults) and {int(ag.iloc[1, 2])}/10 (slowdown). Absolute magnitudes are not comparable.</li>
</ul>

<h2>2. Background: false LRU pages and the token</h2>
<p>Global LRU treats a page as replaceable when it has not been touched recently. A process blocked on a page fault cannot touch its pages, so they age and
are taken by other processes ("false LRU" pages), which then fault in turn; the effect cascades into thrashing (paper Sec. 1, 4.3). Token-ordered LRU
gives one process at a time a temporary privilege: while it holds the token <i>and</i> is swapping, its pages are skipped by victim selection so it can build its working
set; when its swapping status turns off the privilege lapses, so its true-LRU pages can be replaced again. Two monitors (Sec. 5.3) retire an abusive holder.</p>
<figure>{img(DOCS / 'diagrams' / 'flowchart.png', 'Figure A (conceptual illustration, not measured data): fault handling and victim selection in token-ordered LRU.', '62%')}</figure>

<h2>3. Implementation and paper mapping</h2>
{T_MAP}
<p><b>Execution model.</b> One CPU with round-robin scheduling (quantum 50 references); a hit costs 1 tick; one shared FIFO disk services each swap-in in 100 ticks;
frames are reserved when a swap-in starts; evictions are free (no write-back). <b>Swapping status</b> is turned on when a major fault starts and off when the process is next
dispatched, i.e. across the disk wait <i>and</i> the ready-queue wait. The token is kept until the owner exits or a monitor forces a release, as the paper's description implies.</p>
<p><b>Fault terminology.</b> The paper counts only <i>major</i> faults (page fetched from disk). In this simulator "minor" means first-touch zero-fill
(no I/O), which is a different use of the word than the paper's (page-table relink of a resident page); all fault counts and reductions here are major faults only.</p>
<figure>{img(DOCS / 'diagrams' / 'architecture.png', 'Figure B (conceptual illustration): simulator architecture. The baseline policy contains no token code.', '100%')}</figure>

<h2>4. Workloads: audit against the paper</h2>
<p>An earlier version of this project took its workload parameters from memory. They were re-audited against Table 1, Figs. 1–8 and the Section 5.4 text and corrected
where they disagreed; all results in this report come from the corrected workloads.</p>
{T_AUDIT}

<h2>5. Experimental design</h2>
<ul>
<li>Both policies run on identical traces, capacity, I/O model and start offsets; token assignment uses a seeded RNG; 5 seeds (0–4).</li>
<li>Slowdown = interacting elapsed time / elapsed time of the program alone with no major faults (paper Sec. 4.1). Reductions use the formulas of Section 6 of the brief.</li>
<li>Memory shortage = 1 − capacity / Σ peak MAD (paper footnote 5). For each pair the shortage in the 20–50% range was chosen <b>from baseline runs only</b>:
the smallest value at which the baseline's slowdown exceeds 1.5× its CPU-sharing floor. The token policy never influenced this choice.</li>
<li>Thresholds not given by the paper are our choices: Case 2 hold limit = 4× the longest dedicated run, same cooldown; Case 1 window 64 faults with ≥90% self-supplied;
I/O time 100 ticks; quantum 50.</li>
</ul>
{T_CAL}

<h2>6. Results</h2>
<h3>6.1 Per-program results (mean of 5 seeds)</h3>
{T_MAIN}
{img(PLOTS / 'fig01_total_page_faults.png', 'Figure 1 (measured): total major faults per pair. Bars = mean of 5 seeds; whiskers = std.')}
{img(PLOTS / 'fig02_per_process_page_faults.png', 'Figure 2 (measured): major faults per process. Processes that get worse are shown as measured.')}
{img(PLOTS / 'fig03_slowdown.png', 'Figure 3 (measured): slowdown; dashed line = slowdown from CPU sharing alone (no paging).')}

<h3>6.2 Memory behaviour: RSS, MAD and NAP</h3>
{img(PLOTS / 'fig04_rss_vs_time.png', 'Figure 4 (measured, seed 0): RSS (solid) and MAD (dashed) over simulated time, baseline (left) vs token-ordered LRU (right).')}
{img(PLOTS / 'fig08_mad_rss_nap.png', 'Figure 5 (measured, seed 0): MAD, RSS and NAP shown as separate quantities for bit-r/gcc.')}

<h3>6.3 Token behaviour</h3>
{img(PLOTS / 'fig05_token_timeline.png', 'Figure 6 (measured, seed 0): token ownership intervals. The token is held from the first shortage-triggered fault until the owner exits.')}
{T_FL}
<p>In the main runs no forced releases occurred (the 4× hold threshold was never exceeded and the Case 1 ratio was never reached), and the fallback exception handler was never needed.
Protection did act: {fl.iloc[1, 4]:,.0f} victim searches would have chosen the protected process under plain LRU.
The ablation below shows the Case 2 threshold <i>does</i> matter when set tightly.</p>

<h3>6.4 Section 5.4 example (53 MB and 58 MB arrays, 60 MB memory, scaled 1/4)</h3>
{img(PLOTS / 'fig09_section54.png', 'Figure 7 (measured): RSS of the two processes. The token goes to the small process; the large one shrinks, then grows once the small one reduces its working set, as the paper describes for its Fig. 11.')}
{T_54}
<p>Baseline slowdown is extreme because exact LRU on a cyclic scan larger than memory faults on every reference; the paper's Linux baseline was less pathological, so the magnitude should not be compared with the paper.</p>

<h3>6.5 Memory-pressure sweep (20–50% shortage)</h3>
{img(PLOTS / 'fig06_memory_pressure.png', 'Figure 8 (measured): effect of memory shortage. The paper’s 20–50% is a property of its machine settings; here it is an explicit sweep parameter.')}
{T_SW}
<p>The benefit is not uniform: bit-r/gcc is neutral or slightly worse below 50% shortage; LU benefits strongly at 20–40% but not at 50%, where both processes thrash regardless; vortex1/vortex3 has almost no baseline faults below 50%.</p>

<h3>6.6 Ablations (token policy only; mean slowdown)</h3>
{T_AB}
<p>Only LU is sensitive: a Case 2 threshold of 2× the dedicated run forces repeated releases (mean slowdown 32.2 versus 4.1), confirming the paper's warning that the threshold must be large to avoid excessive token transfers. Random versus first-fault assignment made no difference in these runs.</p>

<h3>6.7 Synchronisation-aware token priority (paper Sec. 5.3)</h3>
<p><b>What the paper says.</b> In a parallel system the highest priority for the token goes to a process that others are waiting on (synchronisation or blocking messages),
because relieving it benefits the processes coordinating with it; otherwise the token goes to interacting programs in random order.</p>
<p><b>How it is modelled.</b> Processes may belong to a bulk-synchronous group with barrier positions in their traces. A process reaching its barrier blocks until every other live member of its group
has reached its own matching barrier. A process is "waited on" when it has not reached the barrier at which at least one peer is blocked; among several candidates the one with most waiters wins, ties are random.
The fairness rule still applies (a token-unsuitable process is not eligible). Remote nodes and message passing are not modelled; this is a single-node approximation of the rule.
We also built an <i>extension that is not in the paper</i> (<code>sync_transfer</code>, off by default) that hands an already-held token to a waited-on process.</p>
<p><b>Scenario (our design; the paper evaluates none).</b> Two ranks of a BSP application (10 supersteps, 3 passes over 1,500 / 2,000 pages, barrier after every superstep), an independent memory hog (2,400 pages, cyclic)
and a short independent job, with 30/40/50% memory shortage, 6 seeds. Metric: completion time of the BSP application relative to a run with unlimited memory.</p>
{img(PLOTS / 'fig10_sync_priority.png', 'Figure 10 (measured): BSP application completion time (log scale) for four token-handling regimes. The priority rule only acts when the token is re-assigned while ranks wait.')}
{T_SYNC}
<ul>
<li><b>Paper-default handling (token kept until its holder exits): the rule is inert.</b> The token is assigned once, at the first shortage, before any rank is waiting, so sync priority and random assignment were identical in {int(rt.seeds_identical.sum())} of {int(rt.seeds_identical.sum() + rt.seeds_sync_better_than_random.sum() + rt.seeds_sync_worse_than_random.sum())} paired runs.
The rule is therefore a refinement for systems where the token is re-assigned, not a general improvement.</li>
<li><b>Token released whenever the holder stops swapping (frequent re-assignment): large gains.</b> The sync rule beat random assignment in {int(pf.seeds_sync_better_than_random.sum())} of {int(len(pf) * 6)} paired runs. Random assignment is itself harmful there
(many token transfers, the effect the paper warns about); sync priority brings the application back near the global-LRU level at 30% shortage, and well below it at 40% (see table).</li>
<li><b>Case-2 forced releases: mixed.</b> It helps in most runs but is clearly worse than random in some (for example 40% shortage with the 250k threshold), so no uniform improvement is claimed.</li>
<li><b>Extension (not in the paper):</b> handing the token to the waited-on process helps in many settings (e.g. 40% shortage under paper-default handling) but not all (e.g. 50% shortage with Case-2 release).</li>
</ul>
<p><i>Caveats:</i> one scenario, six seeds, synthetic traces, and a single-node stand-in for inter-node synchronisation; the benefit of the rule in a real cluster depends on factors not modelled here.</p>

<h2>7. Comparison with the published results</h2>
{img(PLOTS / 'fig07_aggregate_vs_paper.png', 'Figure 9 (measured vs paper-derived): reductions for each program. Orange = this simulator; hatched = paper Table 2.')}
{T_CMP}
{T_AG}
<p><b>Reading this honestly.</b> Directional agreement is partial and numerical agreement is not claimed. The simulator differs from the original experiment in: synthetic traces at 1/8 scale;
exact timestamp LRU instead of Linux’s reference-bit approximation; a single fixed-time disk; no write-back, swap cache or read-ahead; our own choice of shortage per pair and of the monitor thresholds.
Several results depend on which process receives the token first: in bit-r/gcc the token went to bit-r in all five seeds, which helps bit-r and hurts gcc, whereas the paper reports gcc benefiting strongly.
We did not adjust parameters to force a match.</p>

<h2>8. Validation</h2>
<p>109 automated tests cover the 15 areas of the brief (known LRU sequence, token acquisition only under shortage, protection, restoration of eligibility, fallback, release on completion/timeout/exit,
unsuitable marking and cooldown, single owner, empty traces, tiny memory, termination while holding the token, determinism, metric formulas) plus the free/unmap operation and the synchronisation groups, barriers and token priority (mutation checks confirm that disabling the priority or inverting it fails 26 tests).
Deliberately breaking the protection, the fallback or the shortage check made tests fail (mutation checks). Structural invariants (frames never over-committed, no duplicate resident pages,
token never held by a finished process) were asserted on all five pairs at two pressure levels under both policies. A clean rerun of the experiments reproduced the CSV outputs exactly.</p>
<pre>$ python -m pytest -q
{pytest_txt}
$ python experiments/demo.py
{demo_txt}</pre>
{img(DOCS / 'diagrams' / 'sequence.png', 'Figure C (conceptual illustration): the life of one token episode.', '45%')}

<h2>9. Limitations and threats to validity</h2>
<ul>
<li>Simulator, not a kernel patch; none of the paper's Linux 2.2.14 code was built or run.</li>
<li>Workloads are synthetic and scaled; they follow the paper's tables and figures but are not SPEC2000/bit-r/LU executions. The paper's plotted bit-r MAD (≈65.5k pages) disagrees with its Table 1 (131.3 MB ≈ 33.6k pages); we followed Table 1 and the RSS curve.</li>
<li>Synchronisation-aware priority is modelled only for barrier-synchronised processes on one node. Real parallel systems wait on remote nodes and messages, which are not modelled; the evaluation scenario is our own design.</li>
<li>Baseline pathologies (exact LRU on cyclic scans: LU and Section 5.4) inflate baseline slowdowns relative to Linux; reduction percentages there overstate what a Linux baseline would show.</li>
<li>Five seeds; random assignment among symmetric processes makes some pairs seed-dependent (e.g. vortex1/vortex3: token-better in 3 of 5 seeds for vortex1).</li>
</ul>

<h2>10. Conclusion</h2>
<p>The simulator implements every element of the paper's token-ordered LRU algorithm, including a single-node model of the synchronisation-aware priority, and demonstrates the paper’s central mechanism:
protecting a swapping token holder sharply reduces false-LRU evictions and, in several workload pairs and in the Section 5.4 example, large reductions in faults and slowdown follow.
The effect is workload- and pressure-dependent, with neutral and negative cases that the paper’s five-pair table does not show; verifying those against a real kernel is outside what a simulator can establish.</p>

<h2>Appendix: reproduce</h2>
<pre>pip install -r requirements.txt
python -m pytest -q
python experiments/run_all.py --seeds 5
python experiments/summarize.py
python experiments/make_plots.py
python experiments/run_sync.py --seeds 6    # synchronisation-aware priority experiment
python experiments/write_report.py
python experiments/build_report.py</pre>
</body></html>"""

out_html = DOCS / "technical_report.html"
out_html.write_text(HTML, encoding="utf-8")
print("wrote", out_html)

try:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(out_html.resolve().as_uri())
        pg.pdf(path=str(DOCS / "technical_report.pdf"), format="A4", print_background=True,
               margin=dict(top="14mm", bottom="14mm", left="12mm", right="12mm"))
        b.close()
    print("wrote", DOCS / "technical_report.pdf")
except Exception as e:                                       # PDF is optional; HTML is always produced
    print("PDF step skipped:", e)
