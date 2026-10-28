#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${POLICY_PYTHON:-$PWD/.venv-policy/bin/python}" -m g1_deploy.run_model "$@"
