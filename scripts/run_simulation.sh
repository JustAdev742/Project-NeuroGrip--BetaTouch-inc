#!/usr/bin/env bash
# Run the whole system against simulated hardware, with the live text UI.
#
#   ./scripts/run_simulation.sh            # run until interrupted
#   ./scripts/run_simulation.sh 30         # run for 30 seconds
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
DURATION="${1:-0}"
exec env PYTHONPATH=src python3 -m neurogrip run \
    --profile simulation --duration "$DURATION"
