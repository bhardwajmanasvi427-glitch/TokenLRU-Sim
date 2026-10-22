# Design, paper mapping and assumptions

Reference: S. Jiang and X. Zhang, "Token-ordered LRU: an effective page replacement policy and its implementation in
Linux systems", *Performance Evaluation* 60 (2005) 5-29. Sections quoted below were read in the supplied PDF.

## 1. Paper requirement -> code -> test

| Paper requirement | Where | Test |
|---|---|---|
| Sec. 5.1 baseline: global LRU over all processes' pages | `policies/global_lru.py`, `memory.lru_candidate` | `tests/test_global_lru.py` |
| Sec. 5.2 global mutually exclusive token, available/held | `TokenOrderedLRU.owner`, `_acquire`, `_release` | `test_only_one_owner...`, `test_hold_intervals_never_overlap...` |
| Token requested only on a fault caused by memory shortage, before `do_swap_page()` | `Simulator._fault` calls `policy.on_major_fault(p, shortage=free()==0)` before the swap-in starts | `test_token_acquired_on_shortage_fault`, `test_token_not_acquired_without_shortage`, `test_no_token_activity_when_memory_is_sufficient_end_to_end` |
| Per-process swapping status: on during swap-in, off on return to normal execution | `Process.swapping`; set in `_fault`, cleared in `Simulator._dispatch` | `test_token_without_swapping_status_gives_no_protection` |
| `__get_free_pages()` skips the program holding BOTH token and swapping status | `TokenOrderedLRU.protected()` + `select_victim` | `test_holder_protected_while_swapping...`, `test_skipping_picks_lru_among_remaining_processes` |
| Protection removed as soon as swapping status is off (not permanent pinning) | `protected()` is re-evaluated on every victim search | `test_protection_ends_when_swapping_status_clears`, `test_protection_is_temporary_not_permanent_pinning` |
| Exception handler: take pages from the privileged process if nobody else can supply | fallback branch in `select_victim` | `test_fallback_*`, `test_no_victim_returns_none...` |
| Sec. 5.3 case 1: holder keeps selecting pages from itself -> release + "token-unsuitable" | `_record_holder_fault` (window/ratio config) | `test_case1_*` |
| Sec. 5.3 case 2: hold time over a threshold -> release + unsuitable for a fixed period | `on_tick`, `unsuitable_until` | `test_forced_release_on_hold_timeout...`, `test_unsuitable_cooldown...` |
| Random token assignment among interacting programs | `_choose_acquirer` (seeded `random.Random`) | `test_random_assignment_depends_on_seed_stream_only`, `test_deterministic_with_fixed_seed` |
| Token goes away when the owner terminates | `on_exit` | `test_process_terminating_while_holding_token_releases_it` |
| Sec. 4.2 five workload pairs; Sec. 5.4 two-process example | `workloads/catalog.py` | `tests/test_metrics.py` |
| Sec. 6 / Table 2 metrics | `metrics.py`, `experiments/summarize.py`, `paper_reference.py` | `tests/test_metrics.py` |

The baseline never imports token code; a test asserts it has no `owner` attribute and ignores `swapping`.
Mutation checks run during development (protection disabled / fallback disabled / shortage check removed) each made
the suite fail, i.e. the tests do exercise these rules.

## 2. Execution model (simulated time only)

One CPU, round-robin (quantum 50 references). A hit costs 1 tick. First touch of a page is a *minor* fault (zero-fill,
no I/O, but still needs a frame). Touching an evicted page is a *major* fault: the process blocks while ONE shared FIFO
disk performs the swap-in (100 ticks), so heavy faulting by one process delays the other's I/O (the I/O-bandwidth
effect the paper discusses in Sec. 5.4 / 6). The frame is reserved when the swap-in starts. Evictions are free (no
write-back).

**Swapping status in a simulator.** The paper turns the status on while a program is swapping and off when it returns
to normal execution. Here it turns on when a major fault starts and off when the process is next dispatched on the CPU.
So the "unresolved fault" window = waiting for the disk + waiting in the ready queue. That is precisely when its pages
cannot be touched and would turn into false-LRU pages. Merely owning pages never sets the status.

**Token retention.** The paper hopes the holder "finishes as soon as possible to release the space and the token"; the
token is therefore kept until the owner exits or a Sec. 5.3 monitor forces a release (default
`release_on_resolution=False`). Protection itself applies only while swapping status is on.

## 3. Parameters the paper does not give (OUR choices, all in `SimConfig`)

io_time 100, quantum 50, cpu_cost 1; Case-1 window 64 faults with >=90% self-supplied; Case-2 hold threshold and
cooldown = 4x the pair's longest dedicated run (`hold_factor`; the paper says the threshold must be large; the ablation
`results/ablation_summary.csv` varies it over off/2/4/8x and the random-vs-first-fault assignment); NAP window = one
scaled paper-second; per-pair memory shortage chosen by a **baseline-only** criterion (smallest of 20/30/40/50% where
baseline slowdown >= 1.5x its CPU-sharing floor), never by looking at the token policy.

## 4. Differences from the original and limitations

* **Simulator, not a kernel patch.** Nothing here runs in or modifies the Linux kernel. The paper's Linux 2.2.14 code
  base is not available and 2.2 interfaces do not exist in modern kernels; no kernel build was attempted.
* **Synthetic workloads** (labelled with `*`): parameters audited against paper Table 1 and Figs. 1-8, scaled to 1/8. They are not SPEC2000,
  bit-reversal or LU executions and no claim of reproducing the hardware numbers is made.
* Exact LRU (timestamps) instead of Linux's reference-bit/clock approximation; no write-back, swap cache, read-ahead,
  or multi-page swap clusters; one disk with fixed service time; no minor-fault page-cache hits.
* Synchronisation-aware token priority (Sec. 5.3) is implemented as a **single-node model**: processes in a bulk-synchronous
  group block at barrier positions until all live peers arrive (`BLOCKED_SYNC`); `Simulator.waiters_on(p)` counts peers blocked
  waiting for `p`; `_choose_acquirer` gives the free token to the candidate with most waiters (ties random), else random.
  Remote nodes and blocking messages are not modelled. The rule is inert under the paper-default token retention (token is
  assigned once, before anyone waits) and only applies with `token_assignment="random"` (not `first_fault`).
* `sync_transfer` (hand a held token to a waited-on faulting process) is **our extension, not in the paper**, default off.
* The BSP evaluation scenario (`workloads/bsp.py`) is our own synthetic design; the paper evaluates no such workload.
* Case-2 "execution time while holding the token" is measured as simulated elapsed time since acquisition.
* The paper's page size is 4 KiB; sizes here are *page IDs*, no real RAM is allocated.
* LU-1 and LU-2 are the same deterministic trace (paper footnote 4: two executions of LU with the same input); they differ only through scheduling and token assignment.
* Terminology: the paper's "minor fault" is a page-table relink of a resident page; here "minor" means first-touch zero-fill. Only major faults are compared with the paper.
* A negative page id in a trace is an unmap/free of page (-id-1) (used by LU and gcc, which release and re-allocate memory).
* The paper's plotted bit-r MAD (~65.5k pages) disagrees with its Table 1 (131.3 MB ~ 33.6k pages); Table 1 and the RSS curve were followed.
* Randomised assignment leaves outcomes order-dependent in symmetric pairs (e.g. vortex1/vortex3: which process wins the
  token varies by seed, so one of the two gets worse in some seeds). This is reported, not hidden.
