#!/usr/bin/env bash
# Run in hardware mode — real GPIO, I2C, camera
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Verify hardware prerequisites
echo "  Checking hardware..."
if ! command -v i2cdetect &>/dev/null; then
    echo "  ⚠️  i2c-tools not installed. Run: sudo apt install i2c-tools"
fi
if [ ! -e /dev/i2c-1 ]; then
    echo "  ⚠️  I2C bus not found. Enable with: sudo raspi-config"
fi
if ! ls /dev/video* &>/dev/null; then
    echo "  ⚠️  No camera found at /dev/video*"
fi

export PYTHONPATH="$PROJ_DIR/src"
export PROSTHETIC_SERVO_DRIVER=lgpio
exec python3 -m app.main --config configs/default.yaml
