#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec "${POLICY_PYTHON:-python}" -m g1_deploy.policy_server "$@"
