#!/usr/bin/env bash
# Populate the HF cache through the exact deployed loader, then verify offline.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
policy_python="${POLICY_PYTHON:-$PWD/.venv-policy/bin/python}"
model_path="${MODEL_PATH:-$PWD/models/groot-n17-g1-carry-box-40k}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-$HOME/hf}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo 'Loading policy online to download required resources (CUDA required)...'
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$policy_python" - "$model_path" <<'PY'
import sys
from g1_deploy.policy_server import GrootBackend
backend = GrootBackend(sys.argv[1], "cuda:0")
print("Online load successful:", backend.schema()["action_shape"])
PY

# Separate process releases the GPU allocation before loading again.
echo 'Verifying policy and processors with Hugging Face offline mode enabled...'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$policy_python" - "$model_path" <<'PY'
import sys
from g1_deploy.policy_server import GrootBackend
backend = GrootBackend(sys.argv[1], "cuda:0")
print("OFFLINE LOAD OK:", backend.schema()["action_shape"])
print("Model and processors loaded; inference and robot execution were not tested.")
PY
