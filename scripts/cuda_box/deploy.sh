#!/usr/bin/env bash
# Runs on the LOCAL machine. Syncs the repo to a remote Linux NVIDIA GPU host
# (Ubuntu 24.04, driver installed, reachable over SSH) and (re)runs setup.sh.
# rsync --delete mirrors this checkout into ~/slangvision on the host: point
# it only at a disposable host.
# Usage: scripts/cuda_box/deploy.sh [host]
#        SLANGVISION_CUDA_HOST=gpu-host scripts/cuda_box/deploy.sh
set -euo pipefail

HOST="${1:-${SLANGVISION_CUDA_HOST:-}}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGVISION_CUDA_HOST or pass an SSH host as the first argument." >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

# Never sync anything git ignores (local notes, outputs, environments).
FILTERS=(--filter=':- .gitignore')
if [[ -f "$REPO_ROOT/.git/info/exclude" ]]; then
    FILTERS+=(--exclude-from="$REPO_ROOT/.git/info/exclude")
fi

echo "== rsync -> $HOST:~/slangvision =="
rsync -az --delete "${FILTERS[@]}" \
    --exclude '.git' \
    --exclude '.env' \
    --exclude '__pycache__' \
    --exclude '*.egg-info' \
    --exclude '.pytest_cache' \
    --exclude 'dist' \
    --exclude '.venv' \
    --exclude '.claude' \
    --exclude 'CLAUDE.local.md' \
    --exclude '.DS_Store' \
    "$REPO_ROOT/" "$HOST:slangvision/"

echo "== remote setup =="
ssh "$HOST" 'bash slangvision/scripts/cuda_box/setup.sh'

echo "== deploy done. run tests with: scripts/cuda_box/run_tests.sh =="
