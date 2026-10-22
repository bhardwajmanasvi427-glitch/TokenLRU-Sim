# Token-ordered LRU page replacement - simulator-based case study

Implements and evaluates the algorithm of Jiang & Zhang, *Token-ordered LRU: an effective page replacement policy and its
implementation in Linux systems*, Performance Evaluation 60 (2005) 5-29, as a **discrete-event Python simulator**
(global LRU baseline vs token-ordered LRU on identical workloads).

**What this is not:** it is not a Linux kernel modification, and the workloads are *synthetic* look-alikes of the paper's
programs (marked `*`), not SPEC2000 runs. Time is simulated ticks, never wall-clock. See `docs/design.md`.

## Setup and commands (Python 3.10+; numpy, pandas, matplotlib, pytest)

```bash
pip install -r requirements.txt
python -m pytest -q                              # 109 unit/invariant tests
python experiments/demo.py                       # single demonstration (< 2 s)
python experiments/compare.py gzip_vortex3 --shortage 0.3 --seed 0   # one pair, both policies
python experiments/run_all.py --seeds 5          # all experiments -> results/*.csv  (~2 min, 2 workers)
python experiments/summarize.py                  # consolidated tables + paper comparison
python experiments/run_sync.py --seeds 6         # synchronisation-aware priority experiment -> results/sync_*.csv
python experiments/make_plots.py                 # figures -> plots/ (+ plots/data/*.csv)
python experiments/write_report.py               # docs/results.md
python experiments/build_report.py               # docs/technical_report.pdf/.html (with figures)
```
Configurable: `--scale`, `--seeds/--seed0`, `--shortage`, `--io-time`, `--quantum`, `--self-window`, `--self-ratio`,
`--pairs`; all other parameters are fields of `SimConfig` (`src/tolru/config.py`).

## Layout

`src/tolru/` simulator (`simulator.py`, `memory.py`, `process.py`, `config.py`, `metrics.py`, `policies/{global_lru,token_lru}.py`) ·
`workloads/` generators + catalog · `tests/` · `experiments/` · `results/` raw CSV · `plots/` figures ·
`docs/` technical report (`technical_report.pdf/.html`), design (`design.md`), generated results (`results.md`), diagrams (`diagrams/`), presentation notes (`presentation.md`).

## Headline findings (details and every unfavourable case in `docs/technical_report.pdf` and `docs/results.md`)

* The mechanism behaves as the paper describes: the token is taken only under memory shortage, other processes absorb the
  evictions while the holder is swapping (42,278 protected victim searches in the main runs), and protection lapses when it stops.
  False-LRU evictions (a process's pages evicted while it was itself faulting) fell from 197,434 (by others) + 124,746 (own fault)
  under global LRU to 21,095 + 35,187 under token-ordered LRU.
* Clear wins: bit-r (faults -98%), LU-1/LU-2 (slowdown 42.4 -> 2.8 / 5.4), vortex3 in vortex3/gcc, the Section 5.4 example
  (total faults 316,400 -> 8,145).
* Unfavourable or neutral: gzip unchanged; gcc much worse in bit-r/gcc (faults 210 -> 5,021; the token went to bit-r in all
  5 seeds) and that pair's average slowdown rises 2.01 -> 2.17; vortex1 worse on average in vortex1/vortex3 (token split 3/2 between the
  processes across seeds); at low memory shortage the policy is neutral or slightly worse.
* Directional agreement with the paper's Table 2 is partial: same sign of improvement for 6/10 programs (faults) and 7/10
  (slowdown). Magnitudes differ a lot and were not tuned.
* A tight Case-2 threshold (2x the dedicated run) hurts LU badly (mean slowdown 32.2 vs 4.1), matching the paper's warning that the
  threshold must be large.

## Workload fidelity

Workload parameters were audited against paper Table 1, Figs. 1-8 and Sec. 5.4 and corrected where they differed (see section 4 of the
technical report). Notable modelling points: LU and gcc free and re-allocate memory (negative page ids in a trace = unmap); LU traverses
its whole matrix each iteration. All are still synthetic, scaled to 1/8.

## Figures (provenance in each caption)

fig01 total faults · fig02 per-process faults · fig03 slowdown · fig04 RSS+MAD vs time · fig05 token timeline ·
fig06 memory-pressure sweep · fig07 aggregate vs paper (measured + paper-derived) · fig08 MAD/RSS/NAP separately ·
fig09 Sec. 5.4 example. Diagrams in `docs/diagrams/` are *conceptual* (flowchart, architecture, sequence).

## Checklist status

Done: baseline; token + swapping status; protected victim selection; exception handler; Case 1 and Case 2 monitors with
cooldown; random assignment; 3 memory-demand patterns; 5 pairs; Sec. 5.4 example; multi-seed runs; pressure sweep;
ablations; metrics incl. MAD/RSS/NAP, token counts/holds/forced releases/skips/fallbacks; tests incl. invariants; plots;
paper comparison; synchronisation-aware priority (single-node barrier-group model). Not done: real kernel patch; real SPEC2000
traces; remote-node/message blocking; no write-back traffic. Minor faults are first-touch only.

## Synchronisation-aware token priority (paper Sec. 5.3)
Processes can belong to a bulk-synchronous group (`Workload.sync_group`, `barriers`); a process reaching a barrier blocks until
its peers arrive. When the token is free, a process that others are waiting on gets the highest priority (ties random);
otherwise random order. Scenario: `workloads/bsp.py` (our own design), experiment `experiments/run_sync.py`, figure `plots/fig10_sync_priority.png`.
Honest findings: under the paper-default token handling (kept until exit) the rule never acts (identical to random in all runs);
when the token is re-assigned each time the holder stops swapping it helped in every paired seed; with Case-2 forced releases
results are mixed, with some clear losses. `cfg.sync_transfer` (hand-over of a held token) is **our extension, not in the paper**, off by default.
Remote nodes and blocking messages are not modelled.
