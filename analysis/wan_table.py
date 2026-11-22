#!/usr/bin/env python3
"""Table 8 style from AWS logs in results/aws/."""
import re, statistics
from pathlib import Path

def medians(path):
    times = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"elapsed time per iteration \(ms\):\s*([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
    return times[5:] if len(times) > 5 else times

aws = Path("results/aws")
print("=== Table 8 style (median of last 15 windows) ===")
for p in sorted(aws.glob("*.txt")):
    t = medians(p)
    if t:
        print(f"{p.name}: median {statistics.median(t):.1f} ms  (n={len(t)})")
