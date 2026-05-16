# WMDEC Preprocessing Runbook

Multi-node Qwen visual embedding extraction across 4 × 8×H100 workers.  
All commands are run **from within a worker node** (SSH in first).

---

## Cluster

| Node | Hostname | IP |
|---|---|---|
| worker-0 (master) | `job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-0` | `172.24.59.214` |
| worker-1 | `job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-1` | `172.27.38.249` |
| worker-2 | `job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-2` | `172.27.63.68` |
| worker-3 | `job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-3` | `172.26.8.85` |

SSH in:
```bash
ssh -CAXY job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-0.caomingyu.baai-emllm_galbot.cn-neimongol-helingeer.job@ssh.platform-cuihu.jingneng-inner.ac.cn -p 2222
```

---

## Key Paths

| Item | Path |
|---|---|
| Project root | `/share/project/ac_code/qwen_decoder/WMDEC` |
| Encoder checkpoint | `/share/project/congsheng/qwen_visual/qwen3_5_visual_encoder_4b.pt` |
| Dataset source | `/share/project/hotel/lerobot_multiimage_data_1fps/event/` |
| Embedding output | `/share/project/congsheng/all-qwen-embeddings-event-all/` |
| Config | `scripts/preprocess_config.yaml` |
| Logs | `logs/worker{0-3}.log` |

---

## Environment Setup

Already baked into `launch_multinode.sh`. If running manually:

```bash
export https_proxy=http://10.8.36.21:80
export http_proxy=http://10.8.36.21:80
export HF_HOME=/share/project/congsheng/model
export HF_ENDPOINT=https://hf-mirror.com
git config --global user.name  "ACondaway"
git config --global user.email "2542426789@qq.com"
source /share/project/ac_code/miniconda3/bin/activate
conda activate qwen_decoder
```

---

## Tuning Knobs

Edit these at the top of `scripts/launch_multinode.sh` before launching:

| Variable | Current | Notes |
|---|---|---|
| `BATCH_SIZE` | `1024` | Images per GPU per step. H100 80 GB holds up to ~2048. |
| `NUM_IO_WORKERS` | `128` | JPEG loader threads per GPU rank. Node has 192 CPUs. |
| `PREFETCH_BATCHES` | `12` | RAM batches buffered ahead of GPU. Each ~600 MB; node has 1.97 TB RAM. |
| `MAX_FRAMES_PER_CYCLE` | `0` | `0` = process everything. Set e.g. `500000` for incremental runs. |

```bash
vi /share/project/ac_code/qwen_decoder/WMDEC/scripts/launch_multinode.sh
```

---

## Launch

### Option A — from worker-0, launch all nodes via SSH into tmux

```bash
WMDEC=/share/project/ac_code/qwen_decoder/WMDEC
W0=job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-0
W1=job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-1
W2=job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-2
W3=job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-3

# Rotate old logs
for i in 0 1 2 3; do mv ${WMDEC}/logs/worker${i}.log ${WMDEC}/logs/worker${i}_prev.log 2>/dev/null; done

# Launch workers 1 2 3 (they wait at rendezvous for master)
for i in 1 2 3; do
  eval w=\$W${i}
  ssh $w "tmux new-session -d -s wmdec_run 'NODE_RANK=${i} bash ${WMDEC}/scripts/launch_multinode.sh > ${WMDEC}/logs/worker${i}.log 2>&1'"
  echo "  worker-${i} launched"
done

# Launch master last (worker-0)
tmux new-session -d -s wmdec_run \
  "NODE_RANK=0 bash ${WMDEC}/scripts/launch_multinode.sh > ${WMDEC}/logs/worker0.log 2>&1"
echo "  worker-0 (master) launched"
```

### Option B — run manually on each node

```bash
# On worker-0:
cd /share/project/ac_code/qwen_decoder/WMDEC
tmux new-session -d -s wmdec_run 'NODE_RANK=0 bash scripts/launch_multinode.sh > logs/worker0.log 2>&1'

# On worker-1:
cd /share/project/ac_code/qwen_decoder/WMDEC
tmux new-session -d -s wmdec_run 'NODE_RANK=1 bash scripts/launch_multinode.sh > logs/worker1.log 2>&1'

# On worker-2:
cd /share/project/ac_code/qwen_decoder/WMDEC
tmux new-session -d -s wmdec_run 'NODE_RANK=2 bash scripts/launch_multinode.sh > logs/worker2.log 2>&1'

# On worker-3:
cd /share/project/ac_code/qwen_decoder/WMDEC
tmux new-session -d -s wmdec_run 'NODE_RANK=3 bash scripts/launch_multinode.sh > logs/worker3.log 2>&1'
```

> Always use `tmux` so the job survives SSH disconnects.  
> Attach anytime with: `tmux attach -t wmdec_run`

---

## Stop All Nodes

```bash
# From worker-0, stops all 4 nodes:
bash /share/project/ac_code/qwen_decoder/WMDEC/scripts/stop_all.sh
```

Or manually on each node:
```bash
tmux kill-session -t wmdec_run
pkill -9 -f preprocess_embeddings
pkill -9 -f torchrun
```

---

## Monitor

### Live status table — refreshes every 3 s (no tmux needed)
```bash
bash /share/project/ac_code/qwen_decoder/WMDEC/scripts/monitor_nodes.sh
```

### tmux — 4 panes with live `nvidia-smi` per node + process bar
```bash
bash /share/project/ac_code/qwen_decoder/WMDEC/scripts/monitor_nodes.sh --tmux
```

### tmux — 4 panes tailing log files per node
```bash
bash /share/project/ac_code/qwen_decoder/WMDEC/scripts/monitor_nodes.sh --log
```

### GPU util on all 4 nodes at once
```bash
for i in 0 1 2 3; do
  echo "=== worker-$i ==="
  ssh job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-$i \
    'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits' \
    | awk -F, '{printf "  GPU%s  util=%3s%%  mem=%5s/%sMiB\n",$1,$2,$3,$4}' 2>/dev/null
done
```

### Check torchrun alive on all nodes
```bash
for i in 0 1 2 3; do
  echo -n "worker-$i: "
  ssh job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-$i \
    'pgrep -c torchrun 2>/dev/null && echo torchrun_alive || echo DEAD'
done
```

### Clean log tail (rank 0 progress)
```bash
grep -v 'evaluate_expr\|recording.py\|run.py\|\*\*\*\|NCCL\|INFO' \
  /share/project/ac_code/qwen_decoder/WMDEC/logs/worker0.log | tail -20
```

---

## Check Progress

### Frames written per dataset (sorted by count)
```bash
find /share/project/congsheng/all-qwen-embeddings-event-all -mindepth 1 -maxdepth 1 -type d \
  | while read d; do
      cnt=$(find "$d" -name '*.pt' 2>/dev/null | wc -l)
      printf '%8d  %s\n' $cnt $(basename $d)
    done | sort -rn
```

### Total .pt files written
```bash
find /share/project/congsheng/all-qwen-embeddings-event-all -name '*.pt' | wc -l
```

---

## Resume After Crash

Just relaunch — already-written `.pt` files are detected via a single `os.walk`
and skipped automatically. No flags or changes needed.

---

## Add / Remove Datasets

Edit `scripts/preprocess_config.yaml`. All entries use `type: lerobot_without_text`:

```yaml
- name: my-new-dataset
  image_dir: /share/project/hotel/lerobot_multiimage_data_1fps/event/my-new-dataset
  type: lerobot_without_text
  # camera_key: left_head   # only set when camera dir is not the default "image"
```

Multi-camera datasets with non-default `camera_key` in current config:

| Dataset | `camera_key` | Other available cameras |
|---|---|---|
| `galbot_real` | `left_head` | `left_wrist`, `right_head`, `right_wrist` |
| `agibot_world_multiview` | `hand_left_color` | `hand_right_color`, `head_color` |

---

## IB / NCCL Config

Already set inside `launch_multinode.sh` — no manual action needed:

```bash
export NCCL_IB_DISABLE=0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=mlx5_100,mlx5_101,mlx5_102,mlx5_103,mlx5_104,mlx5_105,mlx5_106,mlx5_107
export NCCL_IB_GID_INDEX=7
export NCCL_DEBUG=INFO
```

---

## Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/launch_multinode.sh` | Main launch script — run with `NODE_RANK=N` |
| `scripts/stop_all.sh` | Kill all 4 nodes from worker-0 |
| `scripts/monitor_nodes.sh` | Live status / tmux monitor / log tailing |
| `scripts/preprocess_config.yaml` | Dataset list and output root |
| `scripts/preprocess_embeddings.py` | Main Python entry point |


python scripts/preprocess_single_node.py --config scripts/preprocess_config.yaml  --encoder_ckpt /share/project/congsheng/qwen_visual/qwen3_5_visual_encoder_4b.pt --num_gpus 8 --num_io_workers 64 --batch_size 128