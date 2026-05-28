#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python -m app.main --config configs/default.yaml
