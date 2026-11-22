# Low-bit activation compression for pipeline-parallel LLM training across datacenters

Code, launch scripts and run logs for the paper "Low-Bit Activation Compression for
Pipeline-Parallel LLM Training over Bandwidth-Constrained Cross-Datacenter Links"
(I. Polyakov, A. Dukhanov).

The patch adds three codecs (FP8, INT8, NVFP4) to the pipeline point-to-point
communication of Megatron-LM. They quantize the activations and activation gradients
sent between pipeline stages. Everything inside a stage runs in the original precision.

## Contents

| Path | Purpose |
|------|---------|
| `pp_p2p_compression.patch` | codecs and their integration into Megatron-LM |
| `Dockerfile` | training image |
| `run_pp.sh` | launch on the two-host development cluster |
| `run_pp_aws.sh` | launch on the cross-region AWS hosts |
| `<LIMITER SCRIPT>` | bandwidth and delay limiter (`tc`) |
| `bench_codec.py` | codec microbenchmark |
| `analysis/` | scripts that recompute the tables from the logs |
| `results/` | run logs and saved script output |

## Setup

1. Check out the Megatron-LM fork (SDP4Bit code base): https://github.com/ByteDance-Seed/SDP4Bit , commit 25e219d.
2. Apply the patch: `git apply pp_p2p_compression.patch`.
3. Build the image: `docker build -t pp-compress .`

The runs used Ubuntu 22.04, Python 3.10, CUDA 12.8.0 and PyTorch 2.9.1 (cu128).

## Usage

Two environment variables select the codec. All ranks must use the same values.

| Variable | Values | Applies to |
|----------|--------|------------|
| `PP_P2P_COMPRESS` | `none`, `fp8`, `int8`, `nvfp4` | forward activations |
| `PP_P2P_COMPRESS_GRAD` | same; default: the forward setting | backward activation gradients |

The patch uses blocking point-to-point communication. With virtual pipeline stages,
add `--no-overlap-p2p-communication`.

Launch:

- development cluster: `./run_pp.sh NAME ITERS CODEC`
- cross-region: `./run_pp_aws.sh <ARGUMENTS>`

## Recomputing the tables

The scripts read the logs in `results/` and need no GPU. Run them from the repository
root, for example `python3 analysis/paired_eval.py`. Their saved output is in
`results/tables/`.

| Table | Script | Logs |
|-------|--------|------|
| 3, 4 | `analysis/paired_eval.py` | `results/reeval/` |
| 5, 6 | `analysis/sweep_tables.py` | `results/sweep/` |
| 8 | `analysis/wan_table.py` | `results/aws/` |
| 10 | `bench_codec.py` | `results/microbench/` |

Notes:

- Table 7 follows from Table 5: the step time minus the unthrottled step time of the
  same codec.
- `wan_table.py` prints the medians of the tuned-transport runs. The logs of the
  default-transport runs are in `results/aws/` and use the same protocol (median of
  the last 15 logging windows).
- `results/sweep/early_lat0` and `early_lat30` are an earlier session of the sweeps,
  without INT8.

## Not included

- The preprocessed dataset (the source is the public OpenWebText corpus).
- Checkpoints.
- Training-loss curves. Figures 2 and 3 and Table 9 cannot be recomputed from this
  repository.

## Reproducibility note

A seed reproduces a run only on the same GPU model: the GPU random generator gives
different initial weights on different models.

## License and citation

Apache-2.0, see `LICENSE` and `NOTICE`. To cite, see `CITATION.cff`.
