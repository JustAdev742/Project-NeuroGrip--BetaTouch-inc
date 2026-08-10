#!/usr/bin/env bash
# Run against real hardware.
#
# Refuses to start if the motor controller is not present, because a hand that
# silently falls back to simulation would be worse than one that says why it
# will not start.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

PORT="${NEUROGRIP__SERVO__PORT:-/dev/ttyUSB0}"
if [ ! -e "$PORT" ]; then
    echo "error: motor controller not found at $PORT" >&2
    echo "hint:  ls /dev/serial/by-id/   or run with --profile simulation" >&2
    exit 1
fi
exec python3 -m neurogrip run --profile hardware "$@"
