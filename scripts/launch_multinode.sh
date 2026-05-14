#!/usr/bin/env bash
# =============================================================================
# launch_multinode.sh — Launch preprocess_embeddings.py across 4 × 8×H100 nodes
#
# Usage (run on EACH worker node):
#   NODE_RANK=0 bash scripts/launch_multinode.sh   # on worker-0
#   NODE_RANK=1 bash scripts/launch_multinode.sh   # on worker-1
#   NODE_RANK=2 bash scripts/launch_multinode.sh   # on worker-2
#   NODE_RANK=3 bash scripts/launch_multinode.sh   # on worker-3
#
# Or pass NODE_RANK as the first positional argument:
#   bash scripts/launch_multinode.sh 0
#
# Cluster topology
# ----------------
#   worker-0  172.24.59.214  (MASTER — rendezvous point)
#   worker-1  172.27.38.249
#   worker-2  172.27.63.68
#   worker-3  172.26.8.85
#   GPUs per node : 8 × H100
#   Total GPUs    : 32  (world_size = 32)
# =============================================================================

set -euo pipefail

# ---- Resolve node rank ------------------------------------------------------
NODE_RANK="${NODE_RANK:-${1:-}}"
if [[ -z "$NODE_RANK" ]]; then
    echo "ERROR: set NODE_RANK env var or pass it as first argument (0-3)." >&2
    exit 1
fi

# ---- Proxy & HuggingFace mirror ---------------------------------------------
export https_proxy=http://10.8.36.21:80
export http_proxy=http://10.8.36.21:80
export HF_HOME=/share/project/congsheng/model
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ---- Git identity (for any git ops in subprocesses) -------------------------
git config --global user.name  "ACondaway"
git config --global user.email "2542426789@qq.com"

# ---- Conda environment -------------------------------------------------------
source /share/project/ac_code/miniconda3/bin/activate
conda activate qwen_decoder

# ---- Cluster / rendezvous config --------------------------------------------
MASTER_ADDR="172.24.59.214"
MASTER_PORT="${MASTER_PORT:-29500}"
NNODES=2
NPROC_PER_NODE=8

# ---- Job paths --------------------------------------------------------------
WMDEC_ROOT="/share/project/ac_code/qwen_decoder/WMDEC"
ENCODER_CKPT="/share/project/congsheng/qwen_visual/qwen3_5_visual_encoder_4b.pt"
CONFIG="${WMDEC_ROOT}/scripts/preprocess_config.yaml"

# ---- Tuning knobs -----------------------------------------------------------
BATCH_SIZE=4096
NUM_IO_WORKERS=192
PREFETCH_BATCHES=32
MAX_FRAMES_PER_CYCLE=0   # 0 = process everything; e.g. 200000 for incremental

# ---- InfiniBand / NCCL env --------------------------------------------------
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export NCCL_DEBUG=INFO

# ---- Launch -----------------------------------------------------------------
echo "======================================================================"
echo " WMDEC multi-node preprocessing"
echo " node_rank=${NODE_RANK}  master=${MASTER_ADDR}:${MASTER_PORT}"
echo " world_size=$((NNODES * NPROC_PER_NODE))  batch=${BATCH_SIZE}"
echo " io_workers=${NUM_IO_WORKERS}  prefetch=${PREFETCH_BATCHES}"
echo "======================================================================"

cd "${WMDEC_ROOT}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    scripts/preprocess_embeddings.py \
        --config           "${CONFIG}" \
        --encoder_ckpt     "${ENCODER_CKPT}" \
        --batch_size       "${BATCH_SIZE}" \
        --num_io_workers   "${NUM_IO_WORKERS}" \
        --prefetch_batches "${PREFETCH_BATCHES}" \
        --max_frames_per_cycle "${MAX_FRAMES_PER_CYCLE}" \

