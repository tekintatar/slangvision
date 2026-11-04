#!/usr/bin/env bash
# Runs on the LOCAL machine. Runs the pipeline comparison on the remote GPU
# host (prepared by deploy.sh) and copies the JSON result back.
# Usage: SLANGVISION_CUDA_HOST=gpu-host scripts/cuda_box/run_bench.sh [out.json]
set -euo pipefail

HOST="${SLANGVISION_CUDA_HOST:-}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGVISION_CUDA_HOST to the SSH host for the GPU box." >&2
    exit 2
fi
OUT="${1:-bench_$(date +%Y%m%d_%H%M%S).json}"
# a fresh file per run on the host, printed on the last line of output
# shellcheck disable=SC2016
REMOTE=$(ssh "$HOST" 'f=$(mktemp --suffix=.json) && cd ~/slangvision && CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 ~/venvs/slangvision/bin/python benchmarks/compare_pipelines.py --json "$f" >&2 && echo "$f"')
scp "$HOST:$REMOTE" "$OUT"
# the remote path is expanded here on purpose
# shellcheck disable=SC2029
ssh "$HOST" "rm -f $(printf %q "$REMOTE")"
echo "== result: $OUT =="
