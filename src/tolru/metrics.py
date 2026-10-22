"""Metric definitions (exactly as specified in the project brief) and CSV export."""
from __future__ import annotations

import csv
import math
from pathlib import Path


def pct_reduction(baseline, new):
    """100 * (baseline - new) / baseline.  None when baseline is 0/None (undefined)."""
    if baseline is None or new is None or baseline == 0:
        return None
    return 100.0 * (baseline - new) / baseline


def slowdown(elapsed, dedicated):
    """Interacting execution time / dedicated execution time.  None if undefined."""
    if elapsed is None or dedicated is None or dedicated <= 0:
        return None
    return elapsed / dedicated


def mean_std(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})
    return path
