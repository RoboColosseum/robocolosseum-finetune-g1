#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec bash scripts/run_model.sh groot-plate-into-rack "$@"
