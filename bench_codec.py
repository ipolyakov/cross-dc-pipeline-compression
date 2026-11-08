#!/usr/bin/env python3
"""Encode/decode time, packed size and break-even bandwidth of the p2p codecs.

Run on one GPU inside the training image. The patch must already be applied
(the AWS image has it; otherwise git apply docker/methods/cross-dc-pipeline-compression/pp_p2p_compression.patch).

    sudo docker run --rm --gpus '"device=0"' --ipc host \
        -v "$PWD":/bench:ro \
        -e PYTHONPATH=/SDP4Bit \
        "$IMAGE" \
        python3 /bench/bench_codec.py
"""

import math
import statistics
import sys

sys.path.insert(0, "/SDP4Bit")

import torch

from megatron.core.pipeline_parallel.p2p_compression import CODECS

WARMUP = 20
ITERS = 100

# What actually crosses a stage boundary.
# AWS 6.7B: sequence parallel shards the activation across the 8 TP ranks of a
# stage, so each rank sends (s/TP, micro_batch, h). The full tensor is what a
# TP=1 stage would send, and what the 539M dev runs send.
SHAPES = {
    "aws_shard_6.7B": (256, 1, 4096),   # 2048/8, b=1, h=4096
    "aws_full_6.7B": (2048, 1, 4096),
    "dev_539M": (2048, 1, 1280),
}


def cuda_ms(fn, iters):
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(iters):
        starter.record()
        fn()
        ender.record()
        torch.cuda.synchronize()
        samples.append(starter.elapsed_time(ender))
    samples.sort()
    return statistics.median(samples), samples[len(samples) // 10], samples[-1 - len(samples) // 10]


def bench(name, shape, codec):
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(WARMUP):
        wire = codec.encode(x)
        codec.decode(wire, shape, torch.bfloat16)
    torch.cuda.synchronize()

    enc_med, enc_p10, enc_p90 = cuda_ms(lambda: codec.encode(x), ITERS)
    wire = codec.encode(x)
    torch.cuda.synchronize()
    dec_med, dec_p10, dec_p90 = cuda_ms(lambda: codec.decode(wire, shape, torch.bfloat16), ITERS)

    bf16 = x.numel() * 2
    packed = wire.numel()  # uint8
    saved = bf16 - packed
    total_s = (enc_med + dec_med) / 1e3
    # bytes/s -> Mbit/s. Below this bandwidth the codec pays for itself.
    break_even_mbit = (saved / total_s) * 8 / 1e6 if total_s else float("inf")

    decoded = codec.decode(wire, shape, torch.bfloat16)
    err = (decoded.float() - x.float()).abs()
    rmse = err.square().mean().sqrt().item()

    return {
        "codec": name,
        "shape": shape,
        "numel": x.numel(),
        "bf16": bf16,
        "packed": packed,
        "ratio": bf16 / packed,
        "enc": enc_med,
        "dec": dec_med,
        "enc_p10": enc_p10,
        "enc_p90": enc_p90,
        "dec_p10": dec_p10,
        "dec_p90": dec_p90,
        "break_mbit": break_even_mbit,
        "max_abs": err.max().item(),
        "rmse": rmse,
    }


def main():
    assert torch.cuda.is_available(), "no CUDA"
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"warmup {WARMUP}, timed {ITERS}, CUDA events, median")
    print()
    header = (
        f"{'shape':<16} {'codec':<6} {'numel':>10} {'bf16':>10} {'packed':>10} "
        f"{'ratio':>6} {'enc ms':>8} {'dec ms':>8} {'B* Mbit/s':>10} "
        f"{'max|e|':>9} {'rmse':>9}"
    )
    print(header)
    rows = []
    for shape_name, shape in SHAPES.items():
        for codec_name in ("fp8", "int8", "nvfp4"):
            row = bench(codec_name, shape, CODECS[codec_name])
            row["shape_name"] = shape_name
            rows.append(row)
            print(
                f"{shape_name:<16} {codec_name:<6} {row['numel']:>10} "
                f"{row['bf16']:>10} {row['packed']:>10} {row['ratio']:>6.2f} "
                f"{row['enc']:>8.3f} {row['dec']:>8.3f} {row['break_mbit']:>10.0f} "
                f"{row['max_abs']:>9.4f} {row['rmse']:>9.5f}"
            )
    print()
    print("B* is the link bandwidth at which encode+decode costs as much as")
    print("sending the saved bytes. Slower than B*: compression wins.")
    print("enc/dec are medians; the step pays both, on the two ends of the link.")


if __name__ == "__main__":
    main()
