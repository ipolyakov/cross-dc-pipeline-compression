# Training image for the paper. Starts from the public CUDA image and inlines the two
# private base images (ml-base, ml-common-torch29) that the lab Dockerfile used to build on.
#
#   docker build -t sdp-4bit:latest .
#
# SDP4Bit is pinned to the commit this work trained on (repository HEAD, unchanged since
# 2024-12-11). Apex is not pinned: the image that produced the tables was built from whatever
# master was current that day, and that commit was not recorded.
#
# No credentials are baked in. Pass AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY at run time
# if the container itself has to read S3. The dev launcher mounts the dataset instead.

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3-pip \
        python3-dev \
        git \
        build-essential \
        curl \
        unzip \
        pybind11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.9.1+cu128 torchvision==0.24.1+cu128 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install \
        tensorboard==2.20.0 \
        numpy==1.26.4 \
        scipy==1.13.1 \
        datasets==2.21.0 \
        matplotlib==3.10.3 \
        "transformer_engine[core_cu12]==2.11.0" \
        packaging==26.2 \
        psutil==5.9.5 \
        huggingface_hub==1.26.1 \
        scikit-learn==1.6.1 \
        ninja \
        regex \
        pybind11 && \
    pip install --no-build-isolation "transformer_engine[pytorch]==2.11.0" \
        --config-settings="--build-option=-j4" && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
        pip install -v --no-build-isolation .

RUN git clone https://github.com/ByteDance-Seed/SDP4Bit.git && \
    cd SDP4Bit && git checkout 25e219de81ccb4e4a46d11f361b9cc44ffc86c9a

COPY docker/methods/cross-dc-pipeline-compression/sdp4bit_int64_offset_fix.patch /tmp/sdp4bit_int64_offset_fix.patch
COPY docker/methods/cross-dc-pipeline-compression/pp_p2p_compression.patch /tmp/pp_p2p_compression.patch
RUN cd SDP4Bit && \
    git apply /tmp/sdp4bit_int64_offset_fix.patch && \
    git apply /tmp/pp_p2p_compression.patch

# Without this, only global rank 0 writes the index cache and every other host
# dies opening the files. There is no shared filesystem between the two hosts.
RUN sed -i 's/is_distributed_storage=args.distributed_storage/is_distributed_storage=True/' \
        SDP4Bit/pretrain_gpt.py

# gpt_dataset imports this extension while building the index cache.
RUN make -C SDP4Bit/megatron/core/datasets -j"$(nproc)"

# The 4070 is sm_89 and the H100 is sm_90. Upstream jet_quant lists 86 and 90 only.
RUN sed -i "s/'-gencode', 'arch=compute_86,code=sm_86',/'-gencode', 'arch=compute_86,code=sm_86', '-gencode', 'arch=compute_89,code=sm_89',/g" \
        SDP4Bit/tools/jet_quant_cuda/setup.py && \
    sed -i "s/'-gencode', 'arch=compute_90,code=sm_90',/'-gencode', 'arch=compute_90,code=sm_90', '-gencode', 'arch=compute_120,code=sm_120,'/g" \
        SDP4Bit/tools/jet_quant_cuda/setup.py && \
    sed -i "s/'build_ext': BuildExtensionForDebug/'build_ext': BuildExtensionForDebug.with_options(use_ninja=True)/g" \
        SDP4Bit/tools/jet_quant_cuda/setup.py && \
    cd SDP4Bit/tools/jet_quant_cuda && MAX_JOBS=16 python3 ./setup.py build_ext

COPY docker/methods/cross-dc-pipeline-compression/download_dataset.sh /download_dataset.sh
RUN chmod +x /download_dataset.sh

# Point the two scripts this paper runs at the mounted dataset and experiment
# directories. The weight-diff and gradient-quant variants of SDP4Bit are not built.
RUN find SDP4Bit/sample_scripts/accuracy -type f -exec sed -i \
        -e 's|VOCAB_FILE=|VOCAB_FILE=/datasets/sdp4bit-openwebtext/vocab.json|g' \
        -e 's|MERGE_FILE=|MERGE_FILE=/datasets/sdp4bit-openwebtext/merges.txt|g' \
        -e 's|DATA_PATH=|DATA_PATH=/datasets/sdp4bit-openwebtext/openwebtext_text_document|g' \
        -e 's|TENSORBOARD_DIR=|TENSORBOARD_DIR=/experiments/sdp4bit/${EXPERIMENT_NAME}/tensorboard|g' \
        -e 's|\$CHECKPOINT_PATH|/experiments/sdp4bit/${EXPERIMENT_NAME}/checkpoints|g' \
        -e 's|--wandb-project NeurIPS||g' \
        -e 's|--wandb-save-dir ${WANDB_DIR}||g' \
        -e 's|--wandb-exp-name 1_3B-Grad-Quant||g' \
        -e 's|--overlap-grad-reduce||g' \
        -e 's|torchrun|/download_dataset.sh\ntorchrun|g' \
        {} +

# 539M: the 1.3B gradient-quant script with this paper's width and batch, then
# the quantization flags removed. run_pp.sh calls the baseline copy.
RUN mkdir -p SDP4Bit/sample_scripts/accuracy/539M-experimental && \
    cp -r SDP4Bit/sample_scripts/accuracy/1_3B/grad_quant \
          SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant && \
    sed -i \
        -e 's/--hidden-size 2048/--hidden-size 1280/g' \
        -e 's/--micro-batch-size 4/--micro-batch-size 1/g' \
        -e 's/--global-batch-size 256/--global-batch-size 16/g' \
        SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant/train.sh && \
    cp -r SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant \
          SDP4Bit/sample_scripts/accuracy/539M-experimental/baseline && \
    sed -i \
        -e 's/--quantized-gradients//g' \
        -e 's/--gq-group-size-inter 128//g' \
        -e 's/--gradient-quantization-bits-inter 4//g' \
        -e 's/--gq-group-size-intra 128//g' \
        -e 's/--gradient-quantization-bits-intra 8//g' \
        -e 's/--hadamard-transform//g' \
        SDP4Bit/sample_scripts/accuracy/539M-experimental/baseline/train.sh

# 6.7B: upstream baseline with this paper's batch. run_pp_aws.sh calls it.
RUN sed -i \
        -e 's/--micro-batch-size 2/--micro-batch-size 1/g' \
        -e 's/--global-batch-size 256/--global-batch-size 16/g' \
        SDP4Bit/sample_scripts/accuracy/6_7B/baseline/train.sh

WORKDIR /SDP4Bit
