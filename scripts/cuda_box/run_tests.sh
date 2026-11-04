#!/usr/bin/env bash
# Runs on the LOCAL machine. Executes the test suite on the remote GPU host
# (prepared by deploy.sh) against the installed wheel. Extra arguments go to
# pytest.
# Usage: SLANGVISION_CUDA_HOST=gpu-host scripts/cuda_box/run_tests.sh [pytest args...]
set -euo pipefail

HOST="${SLANGVISION_CUDA_HOST:-}"
if [[ -z "$HOST" ]]; then
    echo "Set SLANGVISION_CUDA_HOST to the SSH host for the GPU box." >&2
    exit 2
fi
ARGS=""
if [[ $# -gt 0 ]]; then
    ARGS=$(printf ' %q' "$@")
fi
# The arguments are quoted above and intentionally expanded on the client.
# shellcheck disable=SC2029
# Run from an empty directory (removed afterwards) so pytest collects only the
# test folder and imports the installed wheel, not the checkout.
ssh "$HOST" "d=\$(mktemp -d) && cd \$d && CUDA_PATH=~/cuda-shim LD_LIBRARY_PATH=~/cuda-shim/lib64 ~/venvs/slangvision/bin/pytest ~/slangvision/tests -q -ra -p no:cacheprovider$ARGS; s=\$?; rm -rf \$d; exit \$s"
