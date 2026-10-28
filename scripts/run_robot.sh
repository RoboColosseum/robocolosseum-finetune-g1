#!/usr/bin/env bash
# Default is dry-run; execute must be explicitly selected.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/run_robot.sh [observe|dry-run|execute] [G1_INTERFACE] [DURATION_SECONDS]'
  echo 'Defaults: dry-run eth0 30. WBC_PYTHON and ROBOT_CONFIG may override paths.'
  exit 0
fi
if (( $# > 3 )); then echo 'Too many arguments; use --help' >&2; exit 2; fi
mode="${1:-dry-run}"
case "$mode" in observe|dry-run|execute) ;; *) echo 'Mode must be observe, dry-run or execute' >&2; exit 2;; esac
wbc_python="${WBC_PYTHON:-$PWD/.venv-wbc/bin/python}"
robot_config="${ROBOT_CONFIG:-config/robot.example.json}"
exec "$wbc_python" -m g1_deploy.wbc_client --config "$robot_config" \
  --mode "$mode" --interface "${2:-${G1_INTERFACE:-eth0}}" --duration "${3:-30}"
