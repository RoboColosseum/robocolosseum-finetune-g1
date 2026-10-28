#!/usr/bin/env bash
# Reuse the deployment machine's CUDA PyTorch, isolate Transformers 4.x.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
revision=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
source_dir="$PWD/third_party/lingbot-vla-v2"
if [[ ! -d "$source_dir/.git" ]]; then
  mkdir -p third_party
  git clone https://github.com/robbyant/lingbot-vla-v2.git "$source_dir"
  git -C "$source_dir" checkout "$revision"
fi
[[ "$(git -C "$source_dir" rev-parse HEAD)" == "$revision" ]] || {
  echo 'Unexpected LingBot source revision; review before changing it.' >&2
  exit 1
}
if [[ ! -x .venv-lingbot/bin/python ]]; then
  "${POLICY_PYTHON:-$PWD/.venv-policy/bin/python}" -m venv --system-site-packages .venv-lingbot
fi
.venv-lingbot/bin/pip install -r requirements/lingbot.txt
.venv-lingbot/bin/python scripts/patch_lingbot_inference.py "$source_dir"
.venv-lingbot/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-4B-Instruct',
                  revision='ebb281ec70b05090aa6165b016eac8ec08e71b17',
                  allow_patterns=['*.json', '*.txt', '*.jinja'],
                  local_dir='models/base/Qwen3-VL-4B-Instruct')
PY
