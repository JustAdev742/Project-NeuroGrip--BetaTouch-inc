#!/usr/bin/env bash
# Everything CI runs, locally.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "── ruff ──────────────────────────────────────────"
python3 -m ruff check src tests

echo "── tests ─────────────────────────────────────────"
python3 -m pytest -q

echo "── scenarios ─────────────────────────────────────"
PYTHONPATH=src python3 -m neurogrip simulate all --log-level ERROR

echo
echo "all checks passed"
