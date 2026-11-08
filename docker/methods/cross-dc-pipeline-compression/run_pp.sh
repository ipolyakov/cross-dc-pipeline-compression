#!/usr/bin/env bash
# Megatron PP=2 runs of the 539M model across the two dev hosts, one GPU per host, for the paper on
# compressing pipeline activations. Run the same command on both hosts, MACHINE_RANK=0 on the master:
#
#   MACHINE_RANK=0|1 MASTER_ADDR=IP [PP_GPU=N] ./run_pp.sh NAME ITERS [CODEC [CODEC_GRAD]]
#   MACHINE_RANK=0|1 MASTER_ADDR=IP [PP_GPU=N] ./run_pp.sh --test
#   ./run_pp.sh --test-local
#
#   NAME        run name: outputs go to $DEV_SDP4BIT_EXPERIMENTS/pp539M/NAME (output, tensorboard,
#               checkpoints); a name that already has checkpoints resumes from them, and a name whose
#               checkpoint is at ITERS is skipped, so a loop over runs can simply be started again
#   ITERS       --train-iters, also --lr-decay-iters so that the cosine schedule ends with the run
#   CODEC       activations: none (default) | fp8 | int8 | nvfp4   (PP_P2P_COMPRESS)
#   CODEC_GRAD  their gradients, defaults to CODEC                 (PP_P2P_COMPRESS_GRAD)
#   --test      tools/test_p2p_compression.py over the network (two hosts, one GPU each); every rank
#               prints the step it is in, and a rank that cannot reach the other fails after 3 minutes
#   --test-local the same test on this host alone with its two GPUs: checks the code without the network
#
# Environment:
#   PP_GPU         index of the GPU to use on this host (nvidia-smi -L), default 0
#   EVAL_INTERVAL  validation every N iterations, default 500; 100000 for step-time runs
#   SAVE_INTERVAL  checkpoint every N iterations, default 1000; 0 for no checkpoints
#   PP_EXTRA       more Megatron arguments
#   PP_RECOMPUTE   activation recomputation inside a stage: selective (default), full, or off.
#                  selective recomputes only the core attention; full recomputes each transformer
#                  layer (uniform, 1 layer per checkpoint); off stores every activation.
#                  This does not change the tensor sent across the pipeline boundary.
#   PP_SEED        Megatron --seed. Unset keeps the script default (1234). A convergence
#                  seed sweep must set this; a new NAME per seed, or checkpoints collide.
#   PP_IF          network interface for NCCL (NCCL_SOCKET_IFNAME); set it to the interface that
#                  throttle.sh limits, so that the limit applies to the training traffic
#   PP_DEBUG=1     verbose torchrun, c10d and NCCL logs (rendezvous, network interfaces, connections)
#   PP_PORT        rendezvous port, default 6012
#   PP_RETRIES     a training run that fails (network drop, NCCL error) is restarted from its last
#                  checkpoint after 60 s, up to this many times in total, default 5; both hosts
#                  restart on their own and meet again at the rendezvous
#
# Embeddings and output weights are untied (--untie-embeddings-and-output-weights): tied, their
# gradient all-reduce would cross the slow link every step. NCCL_IB_DISABLE=1 keeps NCCL on TCP, so
# that the tc limit of throttle.sh applies to its traffic. The p2p compression patch is applied at
# start unless the image already has it.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../cluster.env"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PP_GPU=${PP_GPU:-0}
GPUS="\"device=$PP_GPU\""
if [[ "${1:-}" == "--test-local" ]]; then
    MACHINE_RANK=0
    MASTER_ADDR=127.0.0.1
    GPUS=all
fi
: "${MACHINE_RANK:?set MACHINE_RANK=0 on the master host, 1 on the other}"
: "${MASTER_ADDR:?set MASTER_ADDR to the IP of the master host}"

if [[ "${1:-}" == "--test" || "${1:-}" == "--test-local" ]]; then
    MODE=${1#--}
    NAME=test-$(date +%Y%m%d-%H%M%S)
    ITERS=0
    CODEC=none
    CODEC_GRAD=none
else
    if [[ $# -lt 2 ]]; then
        sed -n '2,/^$/p' "${BASH_SOURCE[0]}"
        exit 1
    fi
    MODE=train
    NAME=$1
    ITERS=$2
    CODEC=${3:-none}
    CODEC_GRAD=${4:-$CODEC}
    if [[ ! "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ITERS must be a positive integer, got '$ITERS'" >&2
        exit 1
    fi
    for c in "$CODEC" "$CODEC_GRAD"; do
        case "$c" in none|fp8|int8|nvfp4) ;; *) echo "unknown codec '$c'" >&2; exit 1 ;; esac
    done
fi

IF_ENV=()
if [[ -n "${PP_IF:-}" && "$MODE" != "test-local" ]]; then
    IF_ENV=(-e NCCL_SOCKET_IFNAME="$PP_IF" -e GLOO_SOCKET_IFNAME="$PP_IF")
fi
if [[ "${PP_DEBUG:-0}" == "1" ]]; then
    IF_ENV+=(-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET -e TORCH_DISTRIBUTED_DEBUG=DETAIL
             -e TORCH_CPP_LOG_LEVEL=INFO -e LOGLEVEL=INFO)
fi

EXPERIMENT_NAME=pp539M/$NAME
sudo mkdir -p "$DEV_SDP4BIT_EXPERIMENTS/$EXPERIMENT_NAME"
# The dataset is often on NFS with root_squash, so the container cannot create a
# new seed's index cache beside it (chmod returns Operation not permitted).
# The experiments directory is writable; keep the cache there.
INDEX_CACHE="$DEV_SDP4BIT_EXPERIMENTS/index-cache"
sudo mkdir -p "$INDEX_CACHE"
TRACKER="$DEV_SDP4BIT_EXPERIMENTS/$EXPERIMENT_NAME/checkpoints/latest_checkpointed_iteration.txt"
checkpoint() { sudo cat "$TRACKER" 2>/dev/null | tr -d '[:space:]' || true; }
if [[ "$MODE" == "train" && "$(checkpoint)" == "$ITERS" ]]; then
    echo "run $EXPERIMENT_NAME: already done (checkpoint at $ITERS), skipped"
    exit 0
fi
echo "run $EXPERIMENT_NAME: mode=$MODE iters=$ITERS activations=$CODEC gradients=$CODEC_GRAD" \
     "recompute=${PP_RECOMPUTE:-selective} seed=${PP_SEED:-default} gpus=$GPUS rank=$MACHINE_RANK master=$MASTER_ADDR:${PP_PORT:-6012} nccl_if=${PP_IF:-auto}"

IMAGE="$DEV_REGISTRY/sdp-4bit:latest"

# The container of a training run is detached from this script: Ctrl+C, a closed SSH session or
# the end of this script leave it running, and running the same command again waits for it instead
# of starting a second one. It is found by its EXPERIMENT_NAME, also when an older version of this
# script started it.
running_container() {
    local id
    for id in $(sudo docker ps -q --filter "ancestor=$IMAGE"); do
        if sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$id" |
            grep -qx "EXPERIMENT_NAME=$EXPERIMENT_NAME"; then
            echo "$id"
            return
        fi
    done
}

wait_container() {
    local id=$1 rc logs
    sudo docker logs -f --tail 20 "$id" &
    logs=$!
    rc=$(sudo docker wait "$id" 2>/dev/null) || rc=1
    kill "$logs" 2>/dev/null || true
    wait "$logs" 2>/dev/null || true
    sudo docker rm "$id" >/dev/null 2>&1 || true
    return "${rc:-1}"
}

run_container() {  # attached and removed at exit; detached (prints the ID) with PP_DETACH=1
local how=(--rm)
if [[ -n "${PP_DETACH:-}" ]]; then how=(-d); fi
# TODO: remove the pretrain_gpt.py sed after the image is rebuilt from the Dockerfile,
# which already sets is_distributed_storage=True.
# TODO: remove the helpers build with that same rebuild. The Dockerfile compiles it.
sudo docker run "${how[@]}" \
    --gpus "$GPUS" \
    --network host \
    --ipc host \
    -e SLURM_GPUS_PER_NODE=1 \
    -e SLURM_NNODES=2 \
    -e SLURM_NODEID="$MACHINE_RANK" \
    -e MASTER_ADDR="$MASTER_ADDR" \
    -e PP_PORT="${PP_PORT:-6012}" \
    -e PYTHONUNBUFFERED=1 \
    -e EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    -e PP_MODE="$MODE" \
    -e PP_ITERS="$ITERS" \
    -e PP_P2P_COMPRESS="$CODEC" \
    -e PP_P2P_COMPRESS_GRAD="$CODEC_GRAD" \
    -e PP_EVAL_INTERVAL="${EVAL_INTERVAL:-500}" \
    -e PP_SAVE_INTERVAL="${SAVE_INTERVAL:-1000}" \
    -e PP_RECOMPUTE="${PP_RECOMPUTE:-selective}" \
    -e PP_SEED="${PP_SEED:-}" \
    -e PP_DATA_CACHE="${PP_DATA_CACHE:-/experiments/sdp4bit/index-cache}" \
    -e PP_EXTRA="${PP_EXTRA:-}" \
    -e NCCL_IB_DISABLE=1 \
    "${IF_ENV[@]}" \
    -v "$DEV_SDP4BIT_DATASET":/datasets/sdp4bit-openwebtext \
    -v "$DEV_SDP4BIT_EXPERIMENTS":/experiments/sdp4bit \
    -v "$HERE":/pp:ro \
    "$IMAGE" bash -c "cd /SDP4Bit && sed -i 's/is_distributed_storage=args.distributed_storage/is_distributed_storage=True/' pretrain_gpt.py && if ! python3 -c 'from megatron.core.datasets import helpers' >/dev/null 2>&1; then make -C megatron/core/datasets -j\$(nproc); fi && bash /pp/run_pp_container.sh"
}

start_or_attach() {
    local id
    id=$(running_container)
    if [[ -n "$id" ]]; then
        echo "run $EXPERIMENT_NAME: already running in container ${id:0:12}, waiting for it"
    else
        id=$(PP_DETACH=1 run_container)
        echo "run $EXPERIMENT_NAME: started container ${id:0:12}" \
             "(stop it with: sudo docker stop ${id:0:12}, on both hosts)"
    fi
    wait_container "$id"
}

if [[ "$MODE" != "train" ]]; then
    run_container
    exit
fi
RETRIES=${PP_RETRIES:-5}
for ((attempt = 1; attempt <= RETRIES; attempt++)); do
    if start_or_attach; then rc=0; else rc=$?; fi
    done_at=$(checkpoint)
    if [[ "$done_at" == "$ITERS" || $rc -eq 0 ]]; then
        echo "run $EXPERIMENT_NAME: finished (exit $rc, checkpoint ${done_at:-none})"
        exit 0
    fi
    echo "run $EXPERIMENT_NAME: exit $rc, last checkpoint ${done_at:-none}; attempt $attempt of $RETRIES"
    if ((attempt < RETRIES)); then
        sleep 60
    fi
done
exit 1
