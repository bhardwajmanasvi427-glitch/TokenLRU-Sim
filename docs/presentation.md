# Presentation notes (about 5 minutes)

**Problem.** Under memory shortage several memory-hungry programs run together. Global LRU picks victims by recency, but a
program that is blocked on a page fault cannot touch its pages, so they look old ("false LRU pages") and are taken by
others. Every program then fails to build its working set and the system thrashes.

**Idea (paper Sec. 5).** One global token. When a page fault happens under memory shortage, one program takes it. While
that program is swapping (token AND swapping status) its pages are skipped in victim selection, so it can build its
working set. As soon as it is back to normal execution its pages are ordinary LRU candidates again (not pinned). If nobody
else can supply pages, the exception handler takes the holder's own. Two monitors stop abuse: holder eating its own pages
(Case 1) or holding too long (Case 2) -> token released, program marked token-unsuitable for a while.

**What we built.** A discrete-event simulator (CPU round-robin, shared disk, frames) with two independent policies,
five synthetic workload pairs mirroring the paper's, the Sec. 5.4 example, 5 seeds, a memory-pressure sweep, ablations,
109 tests (incl. invariants and sync groups; mutation-checked), plus a single-node model of the sync-aware token priority.

**What we found.** See `docs/technical_report.pdf`. False-LRU evictions fell by roughly 80%. Big wins for bit-r, LU and the Sec. 5.4 example;
no effect for gzip; gcc worse in bit-r/gcc (the token went to bit-r every time); neutral/worse at low shortage. Same sign as the paper for 6/10 (faults)
and 7/10 (slowdown) programs; magnitudes not comparable.

**Limits (say them out loud).** Simulator not kernel; synthetic workloads not SPEC2000; exact LRU not Linux clock;
no write-back; sync-aware priority only as a local barrier model (inert under paper-default token retention, mixed under Case-2 releases); thresholds are our choices.
