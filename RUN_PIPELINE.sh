#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -lt 4 ]]; then
  echo "Usage: $0 --images-dir PATH --manifest FILE [--gpu] [--skip-enhanced]" >&2
  exit 2
fi
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
GPU=""
for arg in "$@"; do [[ "$arg" == "--gpu" ]] && GPU="--gpu"; done
python download_models.py $GPU
python run_pipeline.py "$@"
