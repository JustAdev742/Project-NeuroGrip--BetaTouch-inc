#!/usr/bin/env bash
# Run in mock mode (no hardware needed) — development / demo
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PROJ_DIR/src"
exec python3 -m app.main --config configs/default.yaml --mock
