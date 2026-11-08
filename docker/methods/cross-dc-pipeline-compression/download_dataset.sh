#!/usr/bin/env bash
# Fetches the preprocessed OpenWebText tree if the mount does not already have it.
# Credentials are the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY of the container,
# which run_pp_aws.sh passes in. The dev launcher mounts the dataset, so this is a no-op there.

echo "Checking s3:"
aws s3 ls

DATASET_DIR=/datasets/sdp4bit-openwebtext
if [[ ! -d "$DATASET_DIR" ]]; then
    mkdir -p $DATASET_DIR
    aws s3 sync ${S3_DATASETS:?}/openwebtext-sdp4bit $DATASET_DIR
fi
