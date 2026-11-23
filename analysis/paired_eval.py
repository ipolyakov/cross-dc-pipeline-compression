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

# seed 3456
base3456 = files.get("reeval-none-seed3", [])[-1280:]
print("\n=== Seed 3456 vs reeval-none-seed3 ===")
report("fp8-seed3", files.get("reeval-fp8-seed3", []), base3456)
report("int8-seed3", files.get("reeval-int8-seed3", []), base3456)
report("nvfp4-seed3", files.get("reeval-nvfp4-seed3", []), base3456)

def mean_se_n(name, vals):
    if len(vals) < 1280:
        print(f"{name}: insufficient data")
        return
    for split, a in [("val", vals[:640]), ("test", vals[640:1280])]:
        m = statistics.mean(a)
        s = statistics.stdev(a) / math.sqrt(len(a)) if len(a) > 1 else 0.0
        print(f"{name} {split}: mean {m:.6f}  SE {s:.6f}  n {len(a)}")

print("\n=== mean / SE / n for seed 3456 ===")
mean_se_n("none-seed3", base3456)
mean_se_n("int8-seed3", files.get("reeval-int8-seed3", [])[-1280:])

print("\n=== Table 4 ===")
# collect gaps
gaps = {}
for codec, key in [("INT8", "int8"), ("FP8", "fp8"), ("NVFP4", "nvfp4")]:
    gaps[codec] = {}
    for split_i, split in enumerate(["val", "test"]):
        g1234 = None
        g2 = None
        g3456 = None
        # recompute the three gaps the same way report does
        for seed_name, base, fkey in [
            ("1234", base1234, f"reeval-{key}"),
            ("seed2", base2, f"reeval-{key}-seed2"),
            ("seed3", base3456, f"reeval-{key}-seed3"),
        ]:
            vals = files.get(fkey, [])[-1280:]
            if len(vals) >= 1280 and len(base) >= 1280:
                a = vals[split_i*640:(split_i+1)*640]
                b = base[split_i*640:(split_i+1)*640]
                diffs = [x - y for x, y in zip(a, b)]
                g = statistics.mean(diffs)
                if seed_name == "1234":
                    g1234 = g
                elif seed_name == "seed2":
                    g2 = g
                else:
                    g3456 = g
        if None not in (g1234, g2, g3456):
            three = [g1234, g2, g3456]
            mean = statistics.mean(three)
            sd = statistics.stdev(three)  # sample, n-1
            t = 4.3027 * sd / math.sqrt(3)
            print(f"{codec} {split}: gaps {g1234:+.6f} {g2:+.6f} {g3456:+.6f} | mean {mean:+.6f} SD {sd:.6f} 95% t-interval [{mean-t:+.6f}, {mean+t:+.6f}]")
        else:
            print(f"{codec} {split}: missing gap(s)")
