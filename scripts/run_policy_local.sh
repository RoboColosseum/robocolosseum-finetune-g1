#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${POLICY_PYTHON:-$PWD/.venv-policy/bin/python}" -m g1_deploy.policy_server \
  --model models/groot-n17-g1-carry-box-40k \
  --host 127.0.0.1 --port 8008 --device cuda:0 "$@"
