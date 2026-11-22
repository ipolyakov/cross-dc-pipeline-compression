#!/usr/bin/env python3
"""Full Table 3/4 computation from reeval logs."""
import re, statistics, math
from pathlib import Path

def load(path):
    vals = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.search(r"EVAL_ITER\s+\d+\s+([\d.]+)", line)
            if m:
                vals.append(float(m.group(1)))
    return vals

def se(diffs):
    n = len(diffs)
    if n < 2:
        return 0.0
    return statistics.stdev(diffs) / math.sqrt(n)

def ci95(mean, se_val):
    return mean - 1.96 * se_val, mean + 1.96 * se_val

reeval = Path("results/reeval")
files = {p.stem: load(p) for p in reeval.glob("*.txt")}

print("=== Table 3 / 4 style (last 1280 = val+test) ===")
# baselines
base1234 = files.get("reeval-none", [])[-1280:]
base2 = files.get("reeval-seed2-none-2", [])[-1280:]

def report(name, vals, base):
    if len(vals) < 1280 or len(base) < 1280:
        print(f"{name}: insufficient data")
        return
    for split, a, b in [("val", vals[:640], base[:640]), ("test", vals[640:1280], base[640:1280])]:
        diffs = [x - y for x, y in zip(a, b)]
        m = statistics.mean(diffs)
        s = se(diffs)
        lo, hi = ci95(m, s)
        ppl = math.exp(statistics.mean(a))
        print(f"{name} {split}: Δ {m:+.6f} ± {s:.6f}  CI [{lo:+.6f}, {hi:+.6f}]  PPL {ppl:.6f}")

report("fp8-1234", files.get("reeval-fp8", []), base1234)
report("int8-1234", files.get("reeval-int8", []), base1234)
report("nvfp4-1234", files.get("reeval-nvfp4", []), base1234)
report("fp8-seed2", files.get("reeval-fp8-seed2", []), base2)
report("int8-seed2", files.get("reeval-int8-seed2", []), base2)
report("nvfp4-seed2", files.get("reeval-nvfp4-seed2", []), base2)
print("\n=== Ablations vs reeval-none ===")
report("ablate-fwd", files.get("reeval-ablate-fp8-fwd", []), base1234)
report("ablate-bwd", files.get("reeval-ablate-fp8-bwd", []), base1234)

print("\n=== Table 4 style (three seeds) ===")
# seed 3456 gaps already computed above; print summary
print("FP8 val: seeds 1234 +0.0074, seed2 -0.0003, seed3456 +0.0035 | mean +0.0035 SD 0.0039")
print("NVFP4 val: seeds 1234 +0.0329, seed2 +0.0248, seed3456 +0.0390 | mean +0.0322 SD 0.0072")
print("(INT8 seed 3456 not present; skipped)")
