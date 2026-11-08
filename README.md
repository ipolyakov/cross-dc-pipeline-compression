# Cross-datacenter pipeline compression

Code for the paper on block-scaled compression of the activations and activation gradients that pipeline parallelism sends between stages.

The codecs sit in Megatron's pipeline point-to-point path. Computation inside a stage stays in its original precision. Compression is off unless the environment variables below are set, so an unset run is ordinary training.

This repository is the paper-specific piece. It is not a fork of the training codebase. The image is built from the Dockerfile here: public CUDA 12.8, PyTorch 2.9.1, then [SDP4Bit](https://github.com/ByteDance-Seed/SDP4Bit) pinned at `25e219de81ccb4e4a46d11f361b9cc44ffc86c9a` (repository HEAD, last commit 2024-12-11) with the two patches in `docker/methods/cross-dc-pipeline-compression/` applied. Dataset and credentials are not included.

## Build the image

From the repository root, on a machine with a GPU and the NVIDIA container toolkit:

```bash
docker build -t sdp-4bit:latest .
```

The build clones Apex unpinned. SDP4Bit is pinned. No AWS keys are copied into the image. The dev launcher mounts the dataset from the host. The AWS launcher reads keys from the local `aws` CLI and passes `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` into the container; `cluster.env` holds bucket names only.

Point `DEV_REGISTRY` (dev) or `IMAGE_REPO` (AWS) in `cluster.env` at the image you just built. `run_pp.sh` and `run_pp_aws.sh` then `docker run` it and apply a newer copy of the patch at start if the image does not already contain it.

## Codecs

| Variable | Meaning | Values | Default |
|---|---|---|---|
| `PP_P2P_COMPRESS` | codec of the forward activations | `none`, `fp8`, `int8`, `nvfp4` | `none` |
| `PP_P2P_COMPRESS_GRAD` | codec of the backward activation gradients | same | `PP_P2P_COMPRESS` |

| Codec | Layout | Bytes per element (bf16 is 2) |
|---|---|---|
| `fp8` | E4M3, block 128, float32 scale per block | 1.03 |
| `int8` | symmetric int8, block 128, float32 scale per block | 1.03 |
| `nvfp4` | E2M1, block 16, E4M3 scale per block, float32 scale per tensor | 0.56 |

The code is plain PyTorch, in `megatron/core/pipeline_parallel/p2p_compression.py` after the patch, hooked into `_communicate`. It needs blocking point-to-point, which non-interleaved pipeline parallelism uses. The first send on each rank prints `pipeline p2p compression: activations=..., gradients=...`.

## Layout

```
Dockerfile
cluster.env.example
bench_codec.py
docker/methods/cross-dc-pipeline-compression/
    pp_p2p_compression.patch
    sdp4bit_int64_offset_fix.patch
    download_dataset.sh
    run_pp.sh              # 539M, two dev hosts, one GPU each
    run_pp_aws.sh          # 6.7B, two AWS regions, 8xH100 each
    run_pp_container.sh    # what the launchers run inside the image
    throttle.sh            # token-bucket limit on one interface
    step_time.sh           # median step time from a Megatron log
```

`run_pp.sh` and `run_pp_aws.sh` source `../../../cluster.env`, so they must stay in `docker/methods/cross-dc-pipeline-compression/` with `cluster.env` at the repository root. Copy `cluster.env.example` to `cluster.env` and fill it in. That file holds registry and bucket names; AWS keys are read from the local `aws` CLI at launch and are not stored here.

## Apply the patch

Inside a checkout of SDP4Bit:

```bash
git apply /path/to/pp_p2p_compression.patch
```

The image build applies it the same way. `run_pp_container.sh` applies it at start when the image does not already contain it. Check, on a host with two GPUs, from the SDP4Bit directory:

```bash
torchrun --nproc_per_node=2 tools/test_p2p_compression.py
```

It ends with `ALL CHECKS PASSED`.

## Dev cluster (539M, 30 000 steps and the bandwidth sweep)

Two hosts, one GPU each, pipeline degree 2, untied embeddings. Run every command on both hosts, with `MACHINE_RANK=0` on the master. `PP_IF` is the interface that carries the traffic; `throttle.sh` is applied to that interface on both hosts.

```bash
export MASTER_ADDR=<master-ip> PP_GPU=0 PP_IF=<iface>    # MACHINE_RANK=0 or 1
./run_pp.sh --test                                        # both hosts, no training
./throttle.sh $PP_IF off
for c in none fp8 nvfp4; do ./run_pp.sh loss30k-$c 30000 $c; done
for r in off 1gbit 500mbit 200mbit 100mbit; do
    ./throttle.sh $PP_IF $r
    for c in none fp8 nvfp4; do
        EVAL_INTERVAL=100000 SAVE_INTERVAL=0 ./run_pp.sh step-$r-$c 200 $c
    done
done
./throttle.sh $PP_IF off
./step_time.sh "$DEV_SDP4BIT_EXPERIMENTS/pp539M/step-500mbit-fp8/output"
```

`throttle.sh` installs `tc qdisc replace dev IFACE root tbf rate RATE burst 256kb latency 50ms`. Traffic control does not shape RDMA, so the launcher sets `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=$PP_IF`, which keeps NCCL on TCP over the throttled interface. Check the resulting rate with `iperf3` between the hosts. The sweep runs are 200 iterations, logged every 10; `step_time.sh` drops the first 5 log lines and prints the median of the rest.

A training run is detached from the script. Starting the same command again waits for the running container instead of launching a second one. A run whose checkpoint is already at the requested iteration is skipped, so the loop can be restarted.

## AWS (6.7B, two regions, 200 steps)

One stage per host, tensor parallelism 8, sequence parallel. The public IPs are of instances that already exist.

```bash
./run_pp_aws.sh wan-200-none 200 none <key.pem> \
    us-east-1 <east-public-ip> us-west-2 <west-public-ip>
./run_pp_aws.sh wan-200-fp8  200 fp8  <key.pem> \
    us-east-1 <east-public-ip> us-west-2 <west-public-ip>
./run_pp_aws.sh wan-200-nvfp4 200 nvfp4 <key.pem> \
    us-east-1 <east-public-ip> us-west-2 <west-public-ip>
```

The script opens UDP 51820 between the public IPs, brings up a WireGuard tunnel and runs both stages. Follow a run with `ssh -i <key.pem> ubuntu@<public-ip> sudo docker logs -f ppaws-<name>`.

## Codec microbenchmark

`bench_codec.py` times encode and decode with CUDA events and reports the break-even bandwidth. Run it on one GPU inside the training image, with the patch applied:

```bash
sudo docker run --rm --gpus '"device=0"' --ipc host \
    -v "$PWD":/bench:ro -e PYTHONPATH=/SDP4Bit \
    "$IMAGE" python3 /bench/bench_codec.py
```

## License

New files added by the patch are Apache-2.0. Files modified from Megatron-LM keep their NVIDIA license headers. The launch scripts in this repository are released under Apache-2.0.
