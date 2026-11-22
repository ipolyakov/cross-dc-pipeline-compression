FROM ml-common-torch29:latest

RUN apt-get update && \
    apt-get install -y pybind11-dev

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex && APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .

RUN git clone https://github.com/ByteDance-Seed/SDP4Bit.git

COPY ./sdp4bit_int64_offset_fix.patch .
RUN cd SDP4Bit && git apply ../sdp4bit_int64_offset_fix.patch

# Optional compression of the activations and gradients sent between pipeline stages, off unless
# PP_P2P_COMPRESS is set (see README.md)
COPY ./pp_p2p_compression.patch .
RUN cd SDP4Bit && git apply ../pp_p2p_compression.patch

# The index cache is built on local disk. Without this, only global rank 0 writes the npy
# files and every other host dies opening them (there is no shared filesystem).
RUN sed -i 's/is_distributed_storage=args.distributed_storage/is_distributed_storage=True/' SDP4Bit/pretrain_gpt.py

ARG QUANT_SETUP_PY=SDP4Bit/tools/jet_quant_cuda/setup.py
RUN sed -i "s/'-gencode', 'arch=compute_86,code=sm_86',/'-gencode', 'arch=compute_86,code=sm_86', '-gencode', 'arch=compute_89,code=sm_89',/g" ${QUANT_SETUP_PY}
RUN sed -i "s/'-gencode', 'arch=compute_90,code=sm_90',/'-gencode', 'arch=compute_90,code=sm_90', '-gencode', 'arch=compute_120,code=sm_120,'/g" ${QUANT_SETUP_PY}
RUN sed -i "s/'build_ext': BuildExtensionForDebug/'build_ext': BuildExtensionForDebug.with_options(use_ninja=True)/g" ${QUANT_SETUP_PY}

RUN cd SDP4Bit/tools/jet_quant_cuda && \
    MAX_JOBS=16 python3 ./setup.py build_ext

RUN apt update && apt install python3-dev
#RUN DEBIAN_FRONTEND=noninteractive apt install -y awscli
RUN apt update && apt install -y curl unzip && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -r awscliv2.zip ./aws
RUN pip install regex # Used by megatron gpt2 tokenizer
RUN pip install pybind11

# gpt_dataset imports this extension while building the index cache. The old sed did not
# match the Makefile, so the module was never compiled into the image.
RUN make -C SDP4Bit/megatron/core/datasets -j"$(nproc)"

RUN sed -i 's/print_rank_last(string)/print_rank_last(string)\n    import subprocess\n    subprocess.run(["\/push_progress.sh"])/g' SDP4Bit/megatron/training.py

# Make sh configuration
RUN mkdir -p SDP4Bit/sample_scripts/accuracy/539M-experimental

RUN cp -r SDP4Bit/sample_scripts/accuracy/1_3B/grad_quant SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant
RUN cp -r SDP4Bit/sample_scripts/accuracy/6_7B/quantWeightDiff_Grad SDP4Bit/sample_scripts/accuracy/6_7B/weight_diff4b
RUN cp -r SDP4Bit/sample_scripts/accuracy/6_7B/quantWeightDiff_Grad SDP4Bit/sample_scripts/accuracy/6_7B/weight_diff8b
RUN cp -r SDP4Bit/sample_scripts/accuracy/6_7B/quantWeightDiff_Grad SDP4Bit/sample_scripts/accuracy/6_7B/wd4b_gq
RUN cp -r SDP4Bit/sample_scripts/accuracy/6_7B/quantWeightDiff_Grad SDP4Bit/sample_scripts/accuracy/6_7B/wd8b_gq
RUN cp -r SDP4Bit/sample_scripts/accuracy/6_7B/quantWeightDiff_Grad SDP4Bit/sample_scripts/accuracy/6_7B/grad_quant

RUN cp -r SDP4Bit/sample_scripts/accuracy/1_3B/grad_quant SDP4Bit/sample_scripts/accuracy/1_3B/weight_diff4b

COPY download_dataset.sh .
COPY push_progress.sh .

COPY .aws /root/.aws

# NB pass SLURM_NODEID env to docker container
# NB pass /datasets/sdp4bit-openwebtext path from the host
# NB pass /experiments/sdp4bit from the host
# NB pass EXPERIMENT_NAME from the host (like 20260701-1000)
#    sed -i 's/--save-interval 5002/--save-interval 100/g' ${TARGET_PATH}/train.sh && \

ARG SCRIPTS_PATH=SDP4Bit/sample_scripts/accuracy
RUN find ${SCRIPTS_PATH} -type f -exec sed -i 's/VOCAB_FILE=/VOCAB_FILE=\/datasets\/sdp4bit-openwebtext\/vocab.json/g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/MERGE_FILE=/MERGE_FILE=\/datasets\/sdp4bit-openwebtext\/merges.txt/g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/DATA_PATH=/DATA_PATH=\/datasets\/sdp4bit-openwebtext\/openwebtext_text_document/g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/TENSORBOARD_DIR=/TENSORBOARD_DIR=\/experiments\/sdp4bit\/${EXPERIMENT_NAME}\/tensorboard/g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/\$CHECKPOINT_PATH/\/experiments\/sdp4bit\/${EXPERIMENT_NAME}\/checkpoints/g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/--wandb-project NeurIPS//g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/--wandb-save-dir ${WANDB_DIR}//g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/--wandb-exp-name 1_3B-Grad-Quant//g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/--overlap-grad-reduce//g' {} + && \
    find ${SCRIPTS_PATH} -type f -exec sed -i 's/torchrun/\/download_dataset.sh\ntorchrun/g' {} +

#    sed -i 's/--save-interval 5002/--save-interval 150/g' ${TARGET_PATH}/train.sh && \
ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant/
RUN sed -i 's/--hidden-size 2048/--hidden-size 1280/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 4/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh
# For qW (quantize weights). Quantize weights difference does not work as the code is not available and commented out
# && \
#    sed -i 's/--quantized-gradients/--quantized-gradients --quantized-weights --weight-quantization-bits 4 --wq-group-size 2048/g' ${TARGET_PATH}/train.sh

RUN cp -r SDP4Bit/sample_scripts/accuracy/539M-experimental/grad_quant SDP4Bit/sample_scripts/accuracy/539M-experimental/baseline

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/539M-experimental/baseline/
#RUN sed -i 's/--quantized-weights//g' ${TARGET_PATH}/train.sh && \
#    sed -i 's/--weight-quantization-bits 4//g' ${TARGET_PATH}/train.sh && \
#    sed -i 's/--wq-group-size 2048//g' ${TARGET_PATH}/train.sh && \
RUN sed -i 's/--quantized-gradients//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-inter 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-inter 4//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-intra 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-intra 8//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--hadamard-transform//g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/weight_diff4b/
RUN sed -i 's/--quantized-gradients//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-inter 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-inter 4//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-intra 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-intra 8//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--hadamard-transform//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/weight_diff8b/
RUN sed -i 's/--quantized-gradients//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-inter 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-inter 4//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-intra 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-intra 8//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--hadamard-transform//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--weight-quantization-bits 4/--weight-quantization-bits 8/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/wd4b_gq/
RUN sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/wd8b_gq/
RUN sed -i 's/--weight-quantization-bits 4/--weight-quantization-bits 8/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/grad_quant/
RUN sed -i 's/--quantized-weights//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--weight-quantization-bits 4//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--wq-group-size 2048//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/6_7B/baseline/
RUN sed -i 's/--micro-batch-size 2/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

ARG TARGET_PATH=SDP4Bit/sample_scripts/accuracy/1_3B/weight_diff4b/
RUN sed -i 's/--quantized-gradients/--quantized-weights/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-inter 128/--wq-group-size 2048/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-inter 4/--weight-quantization-bits 4/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gq-group-size-intra 128//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--gradient-quantization-bits-intra 8//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--hadamard-transform//g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--micro-batch-size 4/--micro-batch-size 1/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--global-batch-size 256/--global-batch-size 16/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--train-iters 80000/--train-iters 640000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--lr-decay-iters 70000/--lr-decay-iters 560000/g' ${TARGET_PATH}/train.sh && \
    sed -i 's/--eval-interval 100/--eval-interval 100/g' ${TARGET_PATH}/train.sh

