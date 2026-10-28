#!/usr/bin/env bash
# Stage an isolated inference environment; do not start a policy or robot client.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=89f2322b4ad016e192437adc1a2c253b05bab246
source_dir="$PWD/third_party/GalaxeaVLA"
uv_bin="${UV_BIN:-$HOME/.local/bin/uv}"
if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p third_party
  git clone https://github.com/OpenGalaxea/GalaxeaVLA.git "$source_dir"
  git -C "$source_dir" checkout "$revision"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || {
  echo 'Unexpected Galaxea source revision; review before changing it.' >&2
  exit 1
}
if [[ ! -x .venv-galaxea/bin/python ]]; then
  "$uv_bin" venv --python 3.10 .venv-galaxea
fi
# Inference dependencies only; upstream's training/simulator stack is not needed.
"$uv_bin" pip install --python .venv-galaxea/bin/python -r requirements/galaxea.txt
"$uv_bin" pip install --python .venv-galaxea/bin/python --no-deps -e "$source_dir"
.venv-galaxea/bin/python - <<'PY'
import torch
from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
from g05.models.g05.inferencer import PolicyInferencer
from g05.data_processor.processor.galaxea_cot_processor import GalaxeaCoTProcessor
from g05.tokenizer.interface.vq_base import VQActionTokenizer
print('Galaxea imports ready; CUDA:', torch.cuda.is_available())
print('Checkpoint loading and inference still require training G1 metadata and processor assets.')
PY
