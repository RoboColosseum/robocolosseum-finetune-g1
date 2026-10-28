#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${UI_PYTHON:-python3}" -m g1_deploy.web_ui "$@"
