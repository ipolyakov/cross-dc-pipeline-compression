#!/usr/bin/env python3
"""Full Table 5/6/7 style from sweep stdout."""
import re, statistics
from pathlib import Path

def parse(path):
    times = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"elapsed time per iteration \(ms\):\s*([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
    return times[5:] if len(times) > 5 else times

def med(vals):
    return statistics.median(vals) if vals else None

base = Path("results/sweep")
sets = {
    "lat0_4070s": base / "lat0_4070s",
    "lat30_4070s": base / "lat30_4070s",
    "early_lat0": base / "early_lat0",
    "early_lat30": base / "early_lat30",
}
print("=== Medians (ms, last 15 windows) ===")
for name, d in sets.items():
    if not d.exists():
        continue
    print(f"\n{name}:")
    for p in sorted(d.glob("*.txt")):
        m = med(parse(p))
        if m:
            print(f"  {p.name}: {m:.1f}")
