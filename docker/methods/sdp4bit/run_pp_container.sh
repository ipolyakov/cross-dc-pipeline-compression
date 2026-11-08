#!/usr/bin/env bash
# Inside the sdp-4bit container, started by run_pp.sh or run_pp_aws.sh: prepares the baseline
# script for PP=2 and runs it (or the p2p compression test). Settings come from the PP_* variables.
#
# PP_MODEL   539M (default, one GPU per stage) or 6.7B (tensor parallel across the GPUs of the stage)
# PP_TP      tensor-parallel size; default 1 for 539M and 8 for 6.7B

set -euo pipefail

# training.py calls /push_progress.sh after every log line; nothing has to be pushed
printf '#!/bin/sh\nexit 0\n' > /push_progress.sh

cd /SDP4Bit
# The image already contains one revision of this patch. A newer checkout of the patch file
# then neither applies nor reverses, and a plain git apply aborts the run.
if git apply --reverse --check /pp/pp_p2p_compression.patch 2>/dev/null; then
    echo "p2p compression patch: already in the image"
elif git apply --check /pp/pp_p2p_compression.patch 2>/dev/null; then
    git apply /pp/pp_p2p_compression.patch
    echo "p2p compression patch: applied"
elif [[ -f megatron/core/pipeline_parallel/p2p_compression.py ]]; then
    echo "p2p compression patch: not applicable, keeping the copy already in the image"
else
    echo "p2p compression patch does not apply and is not in the image" >&2
    git apply /pp/pp_p2p_compression.patch
fi

# TODO: remove this sed after the image is rebuilt from the Dockerfile, which already
# sets is_distributed_storage=True. Until then each host builds its own index cache:
# Megatron otherwise writes the npy files only on global rank 0.
sed -i 's/is_distributed_storage=args.distributed_storage/is_distributed_storage=True/' pretrain_gpt.py

# TODO: remove this build after the image is rebuilt from the Dockerfile, which compiles
# megatron/core/datasets/helpers. Without it a host that has no index cache cannot build one.
if ! python3 -c 'from megatron.core.datasets import helpers' >/dev/null 2>&1; then
    make -C megatron/core/datasets -j"$(nproc)"
fi

OUT=/experiments/sdp4bit/$EXPERIMENT_NAME
mkdir -p "$OUT"

if [[ "$PP_MODE" == "test-local" ]]; then
    echo "test on this host: two local GPUs, no network"
    torchrun --standalone --nproc_per_node 2 tools/test_p2p_compression.py 2>&1 | tee -a "$OUT/output"
    exit "${PIPESTATUS[0]}"
fi
if [[ "$PP_MODE" == "test" ]]; then
    # torchrun of rank 0 listens on MASTER_ADDR:PP_PORT and waits for rank 1; nothing is printed
    # until both have joined, so a silent stop here means the hosts do not see each other on this port
    echo "rank $SLURM_NODEID: torchrun rendezvous at $MASTER_ADDR:$PP_PORT, waiting for the other host"
    torchrun --nnodes 2 --node_rank "$SLURM_NODEID" --nproc_per_node 1 \
        --master_addr "$MASTER_ADDR" --master_port "$PP_PORT" \
        tools/test_p2p_compression.py 2>&1 | tee -a "$OUT/output"
    exit "${PIPESTATUS[0]}"
fi

PP_MODEL=${PP_MODEL:-539M}
case "$PP_MODEL" in
    539M|539m)
        T=sample_scripts/accuracy/539M-experimental/baseline/train.sh
        TP_SIZE=${PP_TP:-1}
        ;;
    6.7B|6_7B|6.7b)
        T=sample_scripts/accuracy/6_7B/baseline/train.sh
        TP_SIZE=${PP_TP:-8}
        ;;
    *)
        echo "unknown PP_MODEL '$PP_MODEL' (use 539M or 6.7B)" >&2
        exit 1
        ;;
esac

# Sequence parallel shards the activation across the tensor-parallel ranks, so a stage
# boundary carries one shard per rank (one full tensor in total) instead of a full copy
# per rank. It requires TP > 1. Megatron rejects it unless CUDA_DEVICE_MAX_CONNECTIONS=1,
# and the default async tensor-parallel all-reduce has the same requirement.
export CUDA_DEVICE_MAX_CONNECTIONS=1
EXTRA="--untie-embeddings-and-output-weights"
if [[ "$TP_SIZE" != 1 ]]; then
    EXTRA="$EXTRA --sequence-parallel"
fi
EXTRA="$EXTRA ${PP_EXTRA:-}"
# Dataset tree is not writable (NFS root_squash). Index cache goes to the
# experiments mount instead; is_distributed_storage makes each host build its own.
if [[ -n "${PP_DATA_CACHE:-}" ]] && [[ "${PP_EXTRA:-}" != *--data-cache-path* ]]; then
    EXTRA="$EXTRA --data-cache-path $PP_DATA_CACHE"
fi

# Recomputation trades compute for memory inside a stage. It does not change the
# pipeline-boundary tensor or how many times that tensor is sent. Default selective
# is what the SDP4Bit scripts already request via --recompute-activations.
# full needs a method: Megatron rejects full without one. uniform/1 checkpoints every layer.
PP_RECOMPUTE=${PP_RECOMPUTE:-selective}
case "$PP_RECOMPUTE" in
    selective)
        ;;
    full)
        sed -i \
            -e '/--recompute-activations/d' \
            -e 's/--recompute-granularity[[:space:]]\+selective/--recompute-granularity full/' \
            "$T"
        if ! grep -q -- '--recompute-granularity full' "$T"; then
            EXTRA="$EXTRA --recompute-granularity full"
        fi
        EXTRA="$EXTRA --recompute-method uniform --recompute-num-layers 1"
        ;;
    off|none)
        sed -i \
            -e '/--recompute-activations/d' \
            -e '/--recompute-granularity/d' \
            "$T"
        ;;
    *)
        echo "unknown PP_RECOMPUTE '$PP_RECOMPUTE' (use off, selective or full)" >&2
        exit 1
        ;;
esac

# A second --seed later in the script would win, so replace the existing one.
# Megatron's default is 1234 when the flag is absent.
PP_SEED=${PP_SEED:-}
if [[ -n "$PP_SEED" ]]; then
    if ! [[ "$PP_SEED" =~ ^[0-9]+$ ]]; then
        echo "PP_SEED must be an integer, got '$PP_SEED'" >&2
        exit 1
    fi
    if grep -q -- '--seed' "$T"; then
        sed -i "s/--seed[[:space:]]\+[0-9][0-9]*/--seed $PP_SEED/" "$T"
    else
        EXTRA="$EXTRA --seed $PP_SEED"
    fi
fi

sed -i \
    -e "s/^MASTER_PORT=.*/MASTER_PORT=$PP_PORT/" \
    -e "s/--tensor-model-parallel-size [0-9][0-9]*/--tensor-model-parallel-size $TP_SIZE/" \
    -e "s/--pipeline-model-parallel-size [0-9][0-9]*/--pipeline-model-parallel-size 2/" \
    -e "s/--train-iters [0-9][0-9]*/--train-iters $PP_ITERS/" \
    -e "s/--lr-decay-iters [0-9][0-9]*/--lr-decay-iters $PP_ITERS/" \
    -e "s/--log-interval [0-9][0-9]*/--log-interval 10/" \
    -e "s/--eval-interval [0-9][0-9]*/--eval-interval $PP_EVAL_INTERVAL/" \
    -e "s/--eval-iters [0-9][0-9]*/--eval-iters 20/" \
    -e "s|--max-position-embeddings 2048|--max-position-embeddings 2048 $EXTRA|" \
    "$T"
if [[ "$PP_SAVE_INTERVAL" == "0" ]]; then
    sed -i '/--save /d' "$T"
else
    sed -i "s/--save-interval [0-9][0-9]*/--save-interval $PP_SAVE_INTERVAL/" "$T"
fi
grep -q -- "--pipeline-model-parallel-size 2" "$T" || { echo "failed to set PP=2 in $T" >&2; exit 1; }
grep -q -- "--tensor-model-parallel-size $TP_SIZE" "$T" || { echo "failed to set TP=$TP_SIZE in $T" >&2; exit 1; }
if [[ "$TP_SIZE" != 1 ]]; then
    grep -q -- "--sequence-parallel" "$T" || { echo "failed to add --sequence-parallel to $T" >&2; exit 1; }
fi
echo "---- $T (model=$PP_MODEL tp=$TP_SIZE recompute=$PP_RECOMPUTE seed=${PP_SEED:-default} CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS)"
cat "$T"
echo "----"

bash "$T" 2>&1 | tee -a "$OUT/output"
