#!/usr/bin/env bash
# Self-tests and a health report. Safe to run at any time: motion tests are
# skipped unless explicitly requested.
set -euo pipefail
cd "$(dirname "$0")/.."
exec env PYTHONPATH=src python3 -m neurogrip diagnose "$@"
