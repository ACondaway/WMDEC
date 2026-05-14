#!/usr/bin/env bash
# =============================================================================
# monitor_nodes.sh — Live status monitor for all 4 embedding-worker nodes
#
# Modes:
#   bash scripts/monitor_nodes.sh          # compact table, refreshes every 3s
#   bash scripts/monitor_nodes.sh --tmux   # 5-pane tmux (4 nodes + summary)
#   bash scripts/monitor_nodes.sh --log    # tail preprocess log on each node
#
# Requirements:
#   • Passwordless SSH from this node to worker-0..3 (already verified)
#   • tmux (optional, for --tmux mode)
# =============================================================================

set -euo pipefail

WORKERS=(
  "job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-0"
  "job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-1"
  "job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-2"
  "job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-3"
)
WMDEC_ROOT="/share/project/ac_code/qwen_decoder/WMDEC"
SESSION="wmdec_monitor"
INTERVAL=3

MODE="${1:-}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_gpu_line() {
  # returns: idx | util% | mem_used/mem_total MB | temp°C
  ssh -o ConnectTimeout=4 -o BatchMode=yes "$1"     'nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu        --format=csv,noheader,nounits 2>/dev/null' 2>/dev/null   | awk -F, '{printf "  GPU%s  util=%3s%%  mem=%5s/%sMB  temp=%s°C\n",$1,$2,$3,$4,$5}'
}

_proc_line() {
  # show torchrun process count + CPU% of top python proc
  ssh -o ConnectTimeout=4 -o BatchMode=yes "$1"     'pgrep -c torchrun 2>/dev/null || echo 0; top -bn1 -p $(pgrep -d, python3 2>/dev/null || echo 1) 2>/dev/null | awk "NR>7{s+=$9} END{printf \"%.0f\", s}"'     2>/dev/null | paste - - |     awk '{printf "  torchrun_procs=%s  python_cpu=%.0f%%\n",$1,$2}'
}

# ---------------------------------------------------------------------------
# Mode: compact table (default, no tmux needed)
# ---------------------------------------------------------------------------
if [[ -z "$MODE" ]]; then
  echo "Monitoring ${#WORKERS[@]} nodes — Ctrl-C to quit (refresh every ${INTERVAL}s)"
  while true; do
    clear
    echo "=================================================================="
    echo " WMDEC Node Status   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    for i in "${!WORKERS[@]}"; do
      w="${WORKERS[$i]}"
      short="worker-$i"
      echo ""
      echo "[$short]  $w"
      _gpu_line  "$w" &
      _proc_line "$w" &
      wait
    done
    echo ""
    echo "------------------------------------------------------------------"
    echo " nvidia-smi summary (all GPUs, worker-0):"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total       --format=csv,noheader,nounits 2>/dev/null       | awk -F, '{printf "  GPU%s util=%3s%% mem=%5s/%sMB\n",$1,$2,$3,$4}' || true
    sleep "$INTERVAL"
  done
  exit 0
fi

# ---------------------------------------------------------------------------
# Mode: tmux — 5 panes (one per node + bottom summary bar)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--tmux" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux not found"; exit 1; }

  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -x 220 -y 60

  for i in 0 1 2 3; do
    w="${WORKERS[$i]}"
    if [[ $i -eq 0 ]]; then
      tmux rename-window -t "${SESSION}:0" "nodes"
      tmux send-keys -t "${SESSION}:0"         "ssh $w 'watch -n2 \"nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | column -t\"'" Enter
    else
      tmux split-window -t "$SESSION" -v
      tmux send-keys -t "$SESSION"         "ssh $w 'watch -n2 \"nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | column -t\"'" Enter
    fi
    if [[ $i -eq 1 ]]; then
      tmux select-layout -t "$SESSION" tiled
    fi
  done
  tmux select-layout -t "$SESSION" tiled

  # Bottom pane: aggregate CPU+proc summary across all nodes
  tmux split-window -t "$SESSION" -v -l 6
  POLL="while true; do clear; for i in 0 1 2 3; do echo -n \"worker-\$i: \"; ssh ${WORKERS[$i]} 'pgrep -c torchrun 2>/dev/null | tr -d \\\\n && echo \" torchrun procs, cpu_py=\$(top -bn1 2>/dev/null | grep python | head -1 | awk \"{print \\\\\\\$9}\")%\"' 2>/dev/null || echo unreachable; done; sleep 3; done"
  tmux send-keys -t "$SESSION" "$POLL" Enter

  tmux attach -t "$SESSION"
  exit 0
fi

# ---------------------------------------------------------------------------
# Mode: --log — tail the most recent preprocess log on each worker
# ---------------------------------------------------------------------------
if [[ "$MODE" == "--log" ]]; then
  command -v tmux >/dev/null 2>&1 || { echo "tmux not found (needed for --log)"; exit 1; }

  LOG_GLOB="${WMDEC_ROOT}/logs/preprocess_*.log"
  tmux kill-session -t "${SESSION}_log" 2>/dev/null || true
  tmux new-session -d -s "${SESSION}_log" -x 220 -y 60

  for i in 0 1 2 3; do
    w="${WORKERS[$i]}"
    CMD="ssh $w 'f=\$(ls -t ${LOG_GLOB} 2>/dev/null | head -1); [ -n \"\$f\" ] && tail -f \"\$f\" || echo \"No log found at ${LOG_GLOB}\"'"
    if [[ $i -eq 0 ]]; then
      tmux send-keys -t "${SESSION}_log" "$CMD" Enter
    else
      tmux split-window -t "${SESSION}_log" -v
      tmux send-keys -t "${SESSION}_log" "$CMD" Enter
    fi
  done
  tmux select-layout -t "${SESSION}_log" tiled
  tmux attach -t "${SESSION}_log"
  exit 0
fi

echo "Unknown mode '$MODE'. Use --tmux or --log."
exit 1
