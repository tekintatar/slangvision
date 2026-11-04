#!/usr/bin/env bash
# Runs ON the remote GPU host (invoked by deploy.sh). Idempotent — safe to
# re-run after every deploy. Assumes Ubuntu 24.04 with the NVIDIA driver
# installed and the repo synced to ~/slangvision.
#
# Installs the built wheel (not the checkout) on the validated stack: torch
# 2.13 (cu130), torchvision 0.28, SlangPy 0.42.0. NVRTC and the CUDA runtime
# headers come from torch's bundled cu13 libraries through a small shim.
set -euo pipefail

REPO=~/slangvision
VENV=~/venvs/slangvision

echo "== apt packages =="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip

echo "== build release artifacts =="
BUILD_ROOT=$(mktemp -d)
trap 'rm -rf "$BUILD_ROOT"' EXIT
python3 -m venv "$BUILD_ROOT/venv"
"$BUILD_ROOT/venv/bin/pip" install -q --upgrade pip build==1.4.0 twine==7.0.0
rm -rf "$REPO/dist"
"$BUILD_ROOT/venv/bin/python" -m build "$REPO"
"$BUILD_ROOT/venv/bin/twine" check "$REPO"/dist/*
WHEEL=$(realpath "$REPO"/dist/*.whl)

echo "== clean environment: torch 2.13 (cu130), torchvision 0.28, the wheel =="
rm -rf "$VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q torch==2.13.0 torchvision==0.28.0
"$VENV/bin/pip" install -q slangpy==0.42.0 "${WHEEL}[dev,bench]"
"$VENV/bin/pip" check

echo "== CUDA shim (NVRTC 13 + headers from torch's bundled cu13 libs) =="
# Env for CUDA runs (exported by run_tests.sh / run_bench.sh):
#   CUDA_PATH=~/cuda-shim  LD_LIBRARY_PATH=~/cuda-shim/lib64
SP=$(ls -d "$VENV"/lib/python3*/site-packages)
rm -rf ~/cuda-shim && mkdir -p ~/cuda-shim/lib64
ln -sfn "$SP/nvidia/cu13/include" ~/cuda-shim/include
ln -sf "$SP"/nvidia/cu13/lib/libnvrtc* ~/cuda-shim/lib64/
ln -sf "$SP"/nvidia/cu13/lib/libcudart* ~/cuda-shim/lib64/
ln -sf "$SP/nvidia/cu13/lib/libnvrtc.so.13" ~/cuda-shim/lib64/libnvrtc.so

echo "== environment report =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
"$VENV/bin/python" -c "import slangpy, torch, torchvision, slangvision; print('slangvision', slangvision.__version__, '| slangpy', slangpy.__version__, '| torch', torch.__version__, '| torchvision', torchvision.__version__)"
ls ~/cuda-shim/lib64/libnvrtc.so ~/cuda-shim/include/cuda_runtime.h > /dev/null && echo "cuda shim: OK (cu13)"

echo "== setup done =="
