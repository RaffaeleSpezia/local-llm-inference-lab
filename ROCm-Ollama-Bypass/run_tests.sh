#!/usr/bin/env bash
set -euo pipefail

VENV_PATH=${VENV_PATH:-$HOME/venv-rocm311}
if [ -d "$VENV_PATH" ]; then
  source "$VENV_PATH/bin/activate"
fi

export HF_HOME=${HF_HOME:-/mnt/raid0/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-/mnt/raid0/hf_cache/transformers}
export TORCH_HOME=${TORCH_HOME:-/mnt/raid0/torch_cache}
export PYTHONPATH=${PYTHONPATH:-$(pwd)}

if [ "$#" -eq 0 ]; then
  set -- tests/test_app.py
fi

pytest "$@" -q
