#!/usr/bin/env bash
# Cross-region pipeline runs on two AWS hosts (one stage per host).
#
# Private IPs of the two regions do not route to each other (the VPCs share a CIDR, so peering
# cannot be used). This script brings up a WireGuard tunnel over the public IPs, which is the
# address NCCL and torchrun then use. One tunnel is one flow: expect about the single-stream
# iperf number, not the multi-stream one.
#
# Run from a machine that has the AWS CLI, the SSH key and this repo:
#
#   ./run_pp_aws.sh --test KEY us-east-2 EAST_PUBLIC us-west-2 WEST_PUBLIC
#   ./run_pp_aws.sh wan-200-none 200 none KEY us-east-2 EAST_PUBLIC us-west-2 WEST_PUBLIC
#   PP_MODEL=539M ./run_pp_aws.sh wan-200-fp8 200 fp8 KEY us-east-2 EAST_PUBLIC us-west-2 WEST_PUBLIC
#
#   --test      tools/test_p2p_compression.py, one GPU per host. Look for ALL CHECKS PASSED.
#               Does not download the dataset.
#   NAME ITERS CODEC KEY REGION0 PUB0 REGION1 PUB1 [CODEC_GRAD]
#               CODEC: none | fp8 | int8 | nvfp4. CODEC_GRAD defaults to CODEC.
#   The first region is rank 0 (pipeline stage 0).
#
#   PP_MODEL=6.7B   default. TP=8, PP=2, one stage per host, all 8 GPUs, sequence parallel.
#                   PP_RECOMPUTE=selective (default), full, or off. Selective is the SDP4Bit default.
#   PP_MODEL=539M   the dev-cluster model, 1 GPU per host.
#   EVAL_INTERVAL   default 100000 (no eval during a short run)
#   SAVE_INTERVAL   default 0
#   PP_PORT         default 6012
#   PP_PULL=1       docker pull even when the image is already on the host
#   PP_DEBUG=1      NCCL and c10d logs
#
# The first real run also downloads OpenWebText onto the instance store of each host
# (the regional replica from S3_DATASETS_REPLICAS when that region has one). Containers are
# detached: a dropped SSH session does not stop them. Logs:
#   ssh -i KEY ubuntu@PUBLIC sudo docker logs -f ppaws-NAME

set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/../../../cluster.env"
SSH_USER=${SSH_USER:-ubuntu}
IMAGE="$IMAGE_REPO:latest"
WG_PORT=51820
TUNNEL_MASTER=10.255.0.1
TUNNEL_OTHER=10.255.0.2

usage() { sed -n '2,/^$/p' "${BASH_SOURCE[0]}"; exit 1; }

if [[ "${1:-}" == "--test" ]]; then
    if [[ $# -ne 6 ]]; then
        usage
    fi
    MODE=test
    NAME=test-$(date +%Y%m%d-%H%M%S)
    ITERS=0
    CODEC=none
    CODEC_GRAD=none
    KEY=$2
    REGION0=$3
    PUB0=$4
    REGION1=$5
    PUB1=$6
    MODEL=${PP_MODEL:-539M}
    GPUS_PER_NODE=1
    GPU_ARG=device=0
else
    if [[ $# -lt 8 || $# -gt 9 ]]; then
        usage
    fi
    MODE=train
    NAME=$1
    ITERS=$2
    CODEC=$3
    KEY=$4
    REGION0=$5
    PUB0=$6
    REGION1=$7
    PUB1=$8
    CODEC_GRAD=${9:-${PP_CODEC_GRAD:-$CODEC}}
    MODEL=${PP_MODEL:-6.7B}
    if [[ ! "$ITERS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ITERS must be a positive integer, got '$ITERS'" >&2
        exit 1
    fi
    for c in "$CODEC" "$CODEC_GRAD"; do
        case "$c" in none|fp8|int8|nvfp4) ;; *) echo "unknown codec '$c'" >&2; exit 1 ;; esac
    done
    case "$MODEL" in
        6.7B|6_7B|6.7b) GPUS_PER_NODE=${PP_TP:-8} ;;
        539M|539m) GPUS_PER_NODE=${PP_TP:-1} ;;
        *) echo "unknown PP_MODEL '$MODEL'" >&2; exit 1 ;;
    esac
    if [[ "$GPUS_PER_NODE" == 1 ]]; then
        GPU_ARG=device=0
    else
        GPU_ARG=all
    fi
fi
if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "NAME must match [A-Za-z0-9._-]+, got '$NAME'" >&2
    exit 1
fi
if [[ ! -f "$KEY" ]]; then
    echo "SSH key not found: $KEY" >&2
    exit 1
fi

AWS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_KEY=$(aws configure get aws_secret_access_key)
if [[ -z "$AWS_KEY_ID" || -z "$AWS_SECRET_KEY" ]]; then
    echo "No AWS credentials on this machine: run aws configure" >&2
    exit 1
fi

# "id private_ip vpc subnet sg"
describe_host() {
    local region=$1 pub=$2
    aws ec2 describe-instances --region "$region" \
        --filters "Name=ip-address,Values=$pub" "Name=instance-state-name,Values=running" \
        --query 'Reservations[0].Instances[0].[InstanceId,PrivateIpAddress,VpcId,SubnetId,SecurityGroups[0].GroupId]' \
        --output text
}

echo "Resolving instances"
read -r ID0 PRIV0 VPC0 SUB0 SG0 <<<"$(describe_host "$REGION0" "$PUB0")"
read -r ID1 PRIV1 VPC1 SUB1 SG1 <<<"$(describe_host "$REGION1" "$PUB1")"
if [[ "$ID0" != i-* || "$ID1" != i-* ]]; then
    echo "Could not find running instances for $PUB0 ($REGION0) and $PUB1 ($REGION1)" >&2
    echo "Got: '$ID0' and '$ID1'" >&2
    exit 1
fi
echo "rank 0: $ID0 $REGION0 public $PUB0 private $PRIV0"
echo "rank 1: $ID1 $REGION1 public $PUB1 private $PRIV1"

open_udp() {
    local region=$1 sg=$2 cidr=$3
    local out
    if ! out=$(aws ec2 authorize-security-group-ingress --region "$region" --group-id "$sg" \
        --ip-permissions "IpProtocol=udp,FromPort=$WG_PORT,ToPort=$WG_PORT,IpRanges=[{CidrIp=$cidr,Description=pp-wireguard}]" 2>&1); then
        if [[ "$out" != *InvalidPermission.Duplicate* ]]; then
            echo "$out" >&2
            exit 1
        fi
    fi
}
echo "Opening UDP $WG_PORT between the public IPs"
open_udp "$REGION0" "$SG0" "$PUB1/32"
open_udp "$REGION1" "$SG1" "$PUB0/32"

SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 -i "$KEY")
SCP=(scp -o StrictHostKeyChecking=accept-new -i "$KEY")

# Packages, GPU runtime, the image, and a WireGuard key pair. Last line is "WGKEY priv pub".
bootstrap() {
    local pub=$1
    "${SSH[@]}" "$SSH_USER@$pub" "AWS_ACCESS_KEY_ID='$AWS_KEY_ID' AWS_SECRET_ACCESS_KEY='$AWS_SECRET_KEY' ECR_REGION='$ECR_REGION' ECR_REGISTRY='$ECR_REGISTRY' IMAGE='$IMAGE' PP_PULL='${PP_PULL:-0}' GPU_ARG='$GPU_ARG' GPUS_PER_NODE='$GPUS_PER_NODE' bash -s" <<'EOF'
set -euo pipefail
echo "STEP gpu"
if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi is missing on this AMI" >&2
    exit 1
fi
nvidia-smi -L
sudo mkdir -p /mnt/localssd
if ! mountpoint -q /mnt/localssd; then
    dev=$(lsblk -dnb -o NAME,SIZE,TYPE | awk '$3=="disk" && $2+0>=200*1024*1024*1024 {print $1}' | while read -r n; do
        if ! lsblk -nro MOUNTPOINT "/dev/$n" | grep -q .; then echo "$n"; break; fi
    done)
    if [[ -n "${dev:-}" ]]; then
        echo "Formatting /dev/$dev for /mnt/localssd"
        sudo mkfs.ext4 -F "/dev/$dev"
        sudo mount "/dev/$dev" /mnt/localssd
    else
        echo "WARNING: no free disk >= 200G, dataset and docker stay on the root volume"
    fi
fi
sudo mkdir -p /mnt/localssd/datasets /mnt/localssd/experiments
export DEBIAN_FRONTEND=noninteractive
echo "STEP packages"
pkgs=()
command -v docker >/dev/null || pkgs+=(docker.io)
command -v wg >/dev/null || pkgs+=(wireguard)
command -v aws >/dev/null || pkgs+=(awscli)
command -v ping >/dev/null || pkgs+=(iputils-ping)
if [[ ${#pkgs[@]} -gt 0 ]]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${pkgs[@]}" >/dev/null
fi

# p5 GPUs sit behind NVSwitch. Until nvidia-fabricmanager reports the fabric ready, CUDA
# init fails inside the container. Same start and wait as docker/common/run_host.sh:
# the AMI already has the package, do not apt-install a versioned fabricmanager over it.
echo "STEP fabric"
sudo systemctl enable --now nvidia-fabricmanager
for _ in $(seq 30); do
    FABRIC=$(nvidia-smi -q | grep -A2 -E '^[[:space:]]*Fabric[[:space:]]*$' || true)
    [[ "$FABRIC" != *"In Progress"* ]] && break
    sleep 5
done
echo "GPU fabric:"
echo "$FABRIC" | grep -E 'State|Status' | sed 's/  */ /g' | sort | uniq -c || true

# Image layers live in containerd, which data-root does not move. A pull onto the root
# volume already failed here with "no space left on device". Bind the existing store
# onto the instance store instead of deleting it.
if mountpoint -q /mnt/localssd && ! mountpoint -q /var/lib/containerd && [[ -d /var/lib/containerd ]]; then
    src=$(df -P /var/lib/containerd | awk 'NR==2{print $1}')
    dst=$(df -P /mnt/localssd | awk 'NR==2{print $1}')
    if [[ "$src" != "$dst" ]]; then
        echo "STEP containerd-on-localssd"
        sudo systemctl stop docker.service docker.socket containerd || true
        if [[ -d /mnt/localssd/containerd ]] && [[ -n $(sudo ls -A /mnt/localssd/containerd 2>/dev/null || true) ]]; then
            echo "WARNING: /var/lib/containerd is on $src and /mnt/localssd/containerd already has data; not moving it" >&2
        else
            sudo rm -rf /mnt/localssd/containerd
            sudo mv /var/lib/containerd /mnt/localssd/containerd
            sudo mkdir -p /var/lib/containerd
            sudo mount --bind /mnt/localssd/containerd /var/lib/containerd
            echo "containerd data -> /mnt/localssd/containerd"
        fi
    fi
fi
# data-root first, nvidia-ctk after it. run_host.sh does this so a rewritten daemon.json
# cannot drop the nvidia runtime; without that runtime --gpus injects no devices.
if mountpoint -q /mnt/localssd; then
    echo "STEP docker-data-root"
    sudo mkdir -p /mnt/localssd/docker /etc/docker
    if ! command -v python3 >/dev/null; then
        echo "WARNING: python3 is missing, docker data stays on the root volume"
    else
        sudo python3 -c 'import json,os
p="/etc/docker/daemon.json"
cfg={}
if os.path.isfile(p) and os.path.getsize(p)>0:
    try:
        cfg=json.load(open(p))
    except Exception as e:
        print("WARNING: leaving daemon.json unchanged:", e)
        cfg=None
if cfg is not None and cfg.get("data-root")!="/mnt/localssd/docker":
    cfg["data-root"]="/mnt/localssd/docker"
    with open(p,"w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print("docker data-root -> /mnt/localssd/docker")
'
    fi
fi
echo "STEP nvidia-container-toolkit"
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor --yes --batch -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl daemon-reload
sudo systemctl enable docker
if systemctl list-unit-files | grep -q '^containerd.service'; then
    sudo systemctl enable containerd
    sudo systemctl restart containerd
fi
sudo systemctl restart docker
mkdir -p "$HOME/.aws"
cat > "$HOME/.aws/credentials" <<CREDS
[default]
aws_access_key_id=$AWS_ACCESS_KEY_ID
aws_secret_access_key=$AWS_SECRET_ACCESS_KEY
CREDS
chmod 600 "$HOME/.aws/credentials"
echo "STEP image"
if [[ "$PP_PULL" != 1 ]] && sudo docker image inspect "$IMAGE" >/dev/null 2>&1 </dev/null; then
    echo "image already present: $IMAGE"
else
    aws ecr get-login-password --region "$ECR_REGION" | sudo docker login --username AWS --password-stdin "$ECR_REGISTRY"
    sudo docker pull "$IMAGE" </dev/null
fi
echo "STEP cuda"
# bash -s feeds this script on stdin. docker run must not read it: the entrypoint would
# swallow the WireGuard commands below and the host would exit 0 without a key.
sudo docker run --rm --gpus "$GPU_ARG" "$IMAGE" \
    python3 -c "import torch; n=torch.cuda.device_count(); print('cuda devices', n, flush=True); assert torch.cuda.is_available() and n >= $GPUS_PER_NODE, ('CUDA is not visible in the container', n)" \
    </dev/null
echo "STEP wireguard-key"
umask 077
priv=$(wg genkey) || { echo "wg genkey failed" >&2; exit 1; }
pub=$(printf '%s\n' "$priv" | wg pubkey) || { echo "wg pubkey failed" >&2; exit 1; }
if [[ -z "$priv" || -z "$pub" ]]; then
    echo "wg did not produce a key pair" >&2
    exit 1
fi
printf 'WGKEY %s %s\n' "$priv" "$pub"
echo "wireguard key ready"
EOF
}

echo "Bootstrapping both hosts (packages, image pull, wireguard keys). The image pull can take a long time."
log0=$(mktemp)
log1=$(mktemp)
bootstrap "$PUB0" >"$log0" 2>&1 &
p0=$!
bootstrap "$PUB1" >"$log1" 2>&1 &
p1=$!
while true; do
    alive=0
    kill -0 "$p0" 2>/dev/null && alive=1
    kill -0 "$p1" 2>/dev/null && alive=1
    echo "rank 0: $(grep -v '^WGKEY ' "$log0" | tail -n 1 || true)"
    echo "rank 1: $(grep -v '^WGKEY ' "$log1" | tail -n 1 || true)"
    [[ $alive -eq 1 ]] || break
    sleep 20
done
fail=0
wait "$p0" || fail=1
wait "$p1" || fail=1
if [[ $fail -ne 0 ]]; then
    echo "----- rank 0 ($PUB0) -----" >&2
    grep -v '^WGKEY ' "$log0" >&2 || true
    echo "----- rank 1 ($PUB1) -----" >&2
    grep -v '^WGKEY ' "$log1" >&2 || true
    exit 1
fi
read -r _ PRIVKEY0 PUBKEY0 <<<"$(grep -a --color=never '^WGKEY ' "$log0" | tail -n 1 || true)"
read -r _ PRIVKEY1 PUBKEY1 <<<"$(grep -a --color=never '^WGKEY ' "$log1" | tail -n 1 || true)"
if [[ -z "${PUBKEY0:-}" || -z "${PUBKEY1:-}" ]]; then
    echo "WireGuard keys were not printed. Last lines from each host:" >&2
    echo "----- rank 0 ($PUB0) -----" >&2
    grep -a --color=never -v '^WGKEY ' "$log0" | tail -n 40 >&2 || true
    echo "----- rank 1 ($PUB1) -----" >&2
    grep -a --color=never -v '^WGKEY ' "$log1" | tail -n 40 >&2 || true
    exit 1
fi

# AllowedIPs is the tunnel address of the peer, not its public endpoint.
wgconf0=$(mktemp)
wgconf1=$(mktemp)
cat > "$wgconf0" <<EOF
[Interface]
Address = $TUNNEL_MASTER/24
ListenPort = $WG_PORT
PrivateKey = $PRIVKEY0
MTU = 1380

[Peer]
PublicKey = $PUBKEY1
AllowedIPs = $TUNNEL_OTHER/32
Endpoint = $PUB1:$WG_PORT
PersistentKeepalive = 25
EOF
cat > "$wgconf1" <<EOF
[Interface]
Address = $TUNNEL_OTHER/24
ListenPort = $WG_PORT
PrivateKey = $PRIVKEY1
MTU = 1380

[Peer]
PublicKey = $PUBKEY0
AllowedIPs = $TUNNEL_MASTER/32
Endpoint = $PUB0:$WG_PORT
PersistentKeepalive = 25
EOF

install_wg() {
    local pub=$1 conf=$2
    "${SCP[@]}" "$conf" "$SSH_USER@$pub:/tmp/wg0.conf"
    "${SSH[@]}" "$SSH_USER@$pub" "sudo mv /tmp/wg0.conf /etc/wireguard/wg0.conf && sudo chmod 600 /etc/wireguard/wg0.conf && sudo systemctl enable wg-quick@wg0 && sudo systemctl restart wg-quick@wg0 && ip -br addr show wg0"
}
echo "Starting the tunnel"
install_wg "$PUB0" "$wgconf0"
install_wg "$PUB1" "$wgconf1"
echo "Checking the tunnel from rank 1"
"${SSH[@]}" "$SSH_USER@$PUB1" "ping -c 3 -W 3 $TUNNEL_MASTER"

case "$MODEL" in
    539M|539m) EXP_MODEL=539M ;;
    *) EXP_MODEL=6.7B ;;
esac
EXPERIMENT_NAME=pp${EXP_MODEL}/$NAME
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-0}
PP_PORT=${PP_PORT:-6012}

# Prefer the dataset replica in the host's own region. The container still receives the URI,
# but the sync below finishes before torchrun so the two ranks do not time out on each other.
replica_uri() {
    local region=$1 uri=$S3_DATASETS part
    if [[ -n "${S3_DATASETS_REPLICAS:-}" ]]; then
        local parts
        IFS=',' read -ra parts <<< "${S3_DATASETS_REPLICAS}"
        for part in "${parts[@]}"; do
            part=${part// /}
            if [[ "$part" == "$region="* ]]; then
                uri=${part#*=}
            fi
        done
    fi
    printf '%s\n' "$uri"
}

sync_dataset() {
    local pub=$1 uri=$2 uri_region=$3
    local primary_region=${S3_DATASETS_REGION:-${ECR_REGION:-us-east-1}}
    "${SSH[@]}" "$SSH_USER@$pub" "AWS_ACCESS_KEY_ID='$AWS_KEY_ID' AWS_SECRET_ACCESS_KEY='$AWS_SECRET_KEY' DATA_URI='$uri' DATA_REGION='$uri_region' PRIMARY='$S3_DATASETS' PRIMARY_REGION='$primary_region' bash -s" <<'EOF'
set -euo pipefail
dest=/mnt/localssd/datasets/sdp4bit-openwebtext
sudo mkdir -p "$dest"
sudo chown -R "$(id -u):$(id -g)" "$dest"
ready() {
    [[ -f $dest/vocab.json && -f $dest/merges.txt ]] && compgen -G "$dest/openwebtext_text_document*" >/dev/null
}
if ready; then
    echo "dataset already present in $dest"
    exit 0
fi
# The instance CLI defaults to its own region. HeadObject against a bucket in
# another region returns 301 Moved Permanently unless --region is set.
sync_from() {
    echo "aws s3 sync --region $2 $1/openwebtext-sdp4bit -> $dest"
    aws s3 sync --region "$2" "$1/openwebtext-sdp4bit" "$dest"
}
if [[ "$DATA_URI" != "$PRIMARY" ]]; then
    sync_from "$DATA_URI" "$DATA_REGION" || echo "replica sync failed, trying $PRIMARY"
fi
if ! ready; then
    sync_from "$PRIMARY" "$PRIMARY_REGION"
fi
ready || { echo "dataset is missing vocab.json, merges.txt or openwebtext_text_document*" >&2; ls -la "$dest" >&2; exit 1; }
echo "dataset ready"
EOF
}

if [[ "$MODE" == "train" ]]; then
    URI0=$(replica_uri "$REGION0")
    URI1=$(replica_uri "$REGION1")
    echo "Downloading OpenWebText on both hosts (rank 0: $URI0, rank 1: $URI1)"
    sync_dataset "$PUB0" "$URI0" "$REGION0" >"$log0" 2>&1 &
    p0=$!
    sync_dataset "$PUB1" "$URI1" "$REGION1" >"$log1" 2>&1 &
    p1=$!
    while true; do
        alive=0
        kill -0 "$p0" 2>/dev/null && alive=1
        kill -0 "$p1" 2>/dev/null && alive=1
        echo "rank 0: $(tail -n 1 "$log0" || true)"
        echo "rank 1: $(tail -n 1 "$log1" || true)"
        [[ $alive -eq 1 ]] || break
        sleep 20
    done
    fail=0
    wait "$p0" || fail=1
    wait "$p1" || fail=1
    if [[ $fail -ne 0 ]]; then
        echo "----- dataset rank 0 -----" >&2
        cat "$log0" >&2
        echo "----- dataset rank 1 -----" >&2
        cat "$log1" >&2
        exit 1
    fi
else
    URI0=$S3_DATASETS
    URI1=$S3_DATASETS
fi

echo "run $EXPERIMENT_NAME: mode=$MODE model=$MODEL gpus/node=$GPUS_PER_NODE iters=$ITERS codec=$CODEC/$CODEC_GRAD"
echo "master=$TUNNEL_MASTER:$PP_PORT (tunnel). Stop with: sudo docker stop ppaws-$NAME   on both hosts"

DEBUG_ENV=
if [[ "${PP_DEBUG:-0}" == 1 ]]; then
    DEBUG_ENV="-e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET -e TORCH_DISTRIBUTED_DEBUG=DETAIL"
fi

start_rank() {
    local pub=$1 rank=$2 data_uri=$3
    "${SCP[@]}" "$HERE/run_pp_container.sh" "$HERE/pp_p2p_compression.patch" "$SSH_USER@$pub:/tmp/"
    "${SSH[@]}" "$SSH_USER@$pub" "bash -s" <<EOF
set -euo pipefail
mkdir -p "\$HOME/pp-aws"
mv /tmp/run_pp_container.sh /tmp/pp_p2p_compression.patch "\$HOME/pp-aws/"
chmod +x "\$HOME/pp-aws/run_pp_container.sh"
sudo docker rm -f "ppaws-$NAME" >/dev/null 2>&1 || true
sudo docker run -d --name "ppaws-$NAME" \
    --gpus $GPU_ARG \
    --network host --ipc host \
    --ulimit memlock=-1:-1 \
    --ulimit stack=67108864:67108864 \
    --cap-add IPC_LOCK \
    -e SLURM_GPUS_PER_NODE=$GPUS_PER_NODE \
    -e SLURM_NNODES=2 \
    -e SLURM_NODEID=$rank \
    -e MASTER_ADDR=$TUNNEL_MASTER \
    -e PP_PORT=$PP_PORT \
    -e PYTHONUNBUFFERED=1 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
    -e PP_MODE=$MODE \
    -e PP_MODEL=$MODEL \
    -e PP_TP=$GPUS_PER_NODE \
    -e PP_ITERS=$ITERS \
    -e PP_P2P_COMPRESS=$CODEC \
    -e PP_P2P_COMPRESS_GRAD=$CODEC_GRAD \
    -e PP_EVAL_INTERVAL=$EVAL_INTERVAL \
    -e PP_SAVE_INTERVAL=$SAVE_INTERVAL \
    -e PP_RECOMPUTE=${PP_RECOMPUTE:-selective} \
    -e PP_SEED=${PP_SEED:-} \
    -e PP_EXTRA="${PP_EXTRA:-}" \
    -e NCCL_IB_DISABLE=1 \
    -e NCCL_NET=Socket \
    -e NCCL_NET_PLUGIN=none \
    -e NCCL_SOCKET_IFNAME=wg0 \
    -e GLOO_SOCKET_IFNAME=wg0 \
    -e NCCL_SOCKET_FAMILY=AF_INET \
    $DEBUG_ENV \
    -e AWS_ACCESS_KEY_ID=$AWS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_KEY \
    -e S3_DATASETS=$data_uri \
    -e S3_EXPERIMENTS=${S3_EXPERIMENTS:-} \
    -v /mnt/localssd/datasets:/datasets \
    -v /mnt/localssd/experiments:/experiments/sdp4bit \
    -v "\$HOME/pp-aws":/pp:ro \
    $IMAGE bash /pp/run_pp_container.sh </dev/null
echo "started \$(sudo docker ps -q --filter name=ppaws-$NAME)"
EOF
}

start_rank "$PUB0" 0 "$URI0"
start_rank "$PUB1" 1 "$URI1"

echo
echo "Containers are detached. Follow rank 0 with:"
echo "  ssh -i $KEY $SSH_USER@$PUB0 sudo docker logs -f ppaws-$NAME"
echo "Output file on each host:"
echo "  /mnt/localssd/experiments/$EXPERIMENT_NAME/output"
echo
echo "Stop the containers (instances keep billing until you terminate them):"
echo "  ssh -i $KEY $SSH_USER@$PUB0 sudo docker rm -f ppaws-$NAME"
echo "  ssh -i $KEY $SSH_USER@$PUB1 sudo docker rm -f ppaws-$NAME"
echo "Do not use run_host.sh for this experiment."
