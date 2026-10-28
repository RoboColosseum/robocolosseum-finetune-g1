#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${WBC_PYTHON:-python}" -m g1_deploy.wbc_client "$@"
