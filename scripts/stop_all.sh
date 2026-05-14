#!/usr/bin/env bash
# Run from ANY worker node to stop all 4 nodes.
WORKERS=(
  job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-0
  job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-1
  job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-2
  job-27cd4d00-5f3e-46ae-923f-fd24c4aea7ab-worker-3
)
for i in 0 1 2 3; do
  echo -n "Stopping worker-${i}... "
  ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no ${WORKERS[$i]}     'tmux kill-session -t wmdec_run 2>/dev/null; pkill -9 -f preprocess_embeddings 2>/dev/null; pkill -9 -f torchrun 2>/dev/null; echo done' || echo 'unreachable'
done
