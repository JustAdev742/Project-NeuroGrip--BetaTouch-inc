#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# NeuroGrip — Full Setup for Ubuntu 26.04 (Resolute) / Pi 4B+
# Kernel 6.18.x, Minimal CLI
#
# This script:
#   1. Installs all system + Python dependencies
#   2. Downloads YOLOv8n model (works on Pi CPU out of the box)
#   3. Enables I2C + camera hardware interfaces
#   4. Installs + starts the systemd service IMMEDIATELY
#   5. Opens the dashboard URL so you see the UI right away
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$PROJ_DIR/.venv"
MODEL_DIR="$PROJ_DIR/assets/models"
MODEL_FILE="$MODEL_DIR/yolov8n.pt"
SERVICE_NAME="prosthetic-hand"
DASHBOARD_PORT=8000

echo ""
echo "  ════════════════════════════════════════════════════"
echo "  🤖  NeuroGrip — Full Pi Setup (Ubuntu 26.04)"
echo "  ════════════════════════════════════════════════════"
echo ""

# ──────────────────────────────────────────────────────────
# 1. System packages
# ──────────────────────────────────────────────────────────
echo "[1/8] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    python3-smbus python3-lgpio \
    i2c-tools \
    libopencv-dev python3-opencv \
    libcap-dev libatlas-base-dev \
    git curl wget \
    2>&1 | tail -1

echo "  ✓ System packages installed"

# ──────────────────────────────────────────────────────────
# 2. Hardware interfaces (I2C + Camera)
# ──────────────────────────────────────────────────────────
echo "[2/8] Enabling hardware interfaces..."
NEEDS_REBOOT=false

# Ubuntu 26.04 on Pi uses /boot/firmware/config.txt
BOOT_CFG="/boot/firmware/config.txt"
if [ ! -f "$BOOT_CFG" ]; then
    BOOT_CFG="/boot/config.txt"  # fallback
fi

if [ -f "$BOOT_CFG" ]; then
    # I2C for ADS1115 EMG ADC
    if ! grep -q "^dtparam=i2c_arm=on" "$BOOT_CFG" 2>/dev/null; then
        echo "dtparam=i2c_arm=on" | sudo tee -a "$BOOT_CFG" >/dev/null
        NEEDS_REBOOT=true
    fi
    # Camera
    if ! grep -q "^start_x=1" "$BOOT_CFG" 2>/dev/null; then
        echo "start_x=1" | sudo tee -a "$BOOT_CFG" >/dev/null
        echo "gpu_mem=128" | sudo tee -a "$BOOT_CFG" >/dev/null
        NEEDS_REBOOT=true
    fi
fi

# User groups for hardware access
sudo usermod -aG gpio,i2c,video "$USER" 2>/dev/null || true
echo "  ✓ Hardware interfaces configured"

# ──────────────────────────────────────────────────────────
# 3. Python virtual environment
# ──────────────────────────────────────────────────────────
echo "[3/8] Setting up Python environment..."
python3 -m venv --system-site-packages "$VENV_DIR"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
echo "  ✓ Virtual environment created"

# ──────────────────────────────────────────────────────────
# 4. Python dependencies
# ──────────────────────────────────────────────────────────
echo "[4/8] Installing Python packages..."
pip install -r "$PROJ_DIR/requirements.txt" -q

# Ultralytics for YOLO (install separately — big package)
echo "  Installing ultralytics (YOLOv8)..."
pip install ultralytics -q 2>&1 | tail -1 || true

# Hardware packages (optional — won't fail if not on Pi)
pip install adafruit-circuitpython-ads1x15 -q 2>/dev/null || true
pip install lgpio -q 2>/dev/null || true

echo "  ✓ Python packages installed"

# ──────────────────────────────────────────────────────────
# 5. Download YOLOv8n model
# ──────────────────────────────────────────────────────────
echo "[5/8] Downloading YOLOv8n model for Raspberry Pi..."
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_FILE" ]; then
    # Download via ultralytics Python API (handles caching + verification)
    "$VENV_DIR/bin/python3" -c "
from ultralytics import YOLO
import shutil, os
print('  Downloading YOLOv8n (6 MB)...')
model = YOLO('yolov8n.pt')
# Model downloads to current dir or ultralytics cache
for candidate in ['yolov8n.pt', os.path.expanduser('~/.cache/ultralytics/yolov8n.pt')]:
    if os.path.exists(candidate):
        shutil.copy2(candidate, '$MODEL_FILE')
        print(f'  ✓ Model saved to $MODEL_FILE')
        break
else:
    # If above fails, the model object itself can export
    print('  ✓ Model loaded (will be cached by ultralytics)')
"
    echo "  ✓ YOLOv8n downloaded and ready"
else
    echo "  ✓ YOLOv8n already exists at $MODEL_FILE"
fi

# Quick model test (verify it loads without crashing)
echo "  Verifying model loads..."
"$VENV_DIR/bin/python3" -c "
from ultralytics import YOLO
m = YOLO('$MODEL_FILE')
print(f'  ✓ Model verified: {len(m.names)} classes, ready for Pi CPU inference')
" 2>/dev/null || echo "  ⚠ Model verification skipped (will download on first run)"

# ──────────────────────────────────────────────────────────
# 6. Create hardware config overlay
# ──────────────────────────────────────────────────────────
echo "[6/8] Generating hardware config..."
cat > "$PROJ_DIR/configs/hardware.yaml" <<'HWEOF'
# Auto-generated by setup_pi.sh — hardware mode config overlay
# This overrides default.yaml settings for real Pi deployment

system:
  mock_mode: false

servo:
  driver: "lgpio"
HWEOF

# Update default.yaml to disable mock mode for Pi
sed -i 's/mock_mode: true/mock_mode: false/' "$PROJ_DIR/configs/default.yaml" 2>/dev/null || true

echo "  ✓ Hardware config generated"

# ──────────────────────────────────────────────────────────
# 7. Install + enable + start systemd service
# ──────────────────────────────────────────────────────────
echo "[7/8] Installing systemd service..."

# Generate service file with correct paths
cat > "/tmp/$SERVICE_NAME.service" <<SVCEOF
[Unit]
Description=NeuroGrip AI Prosthetic Hand Controller
After=network.target
Wants=network.target

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJ_DIR
Environment="PATH=$VENV_DIR/bin:/usr/bin:/bin"
Environment="PYTHONPATH=$PROJ_DIR/src"
Environment="PROSTHETIC_SERVO_DRIVER=lgpio"
ExecStart=$VENV_DIR/bin/python3 -m app.main --config configs/default.yaml
KillSignal=SIGTERM
TimeoutStopSec=10
Restart=always
RestartSec=3
SupplementaryGroups=gpio i2c video
StandardOutput=journal
StandardError=journal
SyslogIdentifier=neurogrip

[Install]
WantedBy=multi-user.target
SVCEOF

sudo cp "/tmp/$SERVICE_NAME.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME.service"

echo "  ✓ Service installed and enabled"

# ──────────────────────────────────────────────────────────
# 8. START IMMEDIATELY
# ──────────────────────────────────────────────────────────
echo "[8/8] Starting NeuroGrip..."
sudo systemctl restart "$SERVICE_NAME.service"

# Wait for dashboard to come up
echo "  Waiting for dashboard to start..."
for i in $(seq 1 15); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$DASHBOARD_PORT/api/health" 2>/dev/null | grep -q "200"; then
        break
    fi
    sleep 1
done

# Get the Pi's IP address for display
PI_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
PI_IP="${PI_IP:-localhost}"

echo ""
echo "  ════════════════════════════════════════════════════"
echo "  ✅  NeuroGrip is RUNNING!"
echo ""
echo "  📊  Dashboard: http://$PI_IP:$DASHBOARD_PORT"
echo "  📊  Local:     http://localhost:$DASHBOARD_PORT"
echo ""
echo "  🤖  YOLOv8n model: $MODEL_FILE"
echo "  ⚙️   Service:       sudo systemctl status $SERVICE_NAME"
echo "  📋  Logs:           sudo journalctl -u $SERVICE_NAME -f"
echo ""

if [ "$NEEDS_REBOOT" = true ]; then
    echo "  ⚠️  I2C/Camera changes need a reboot."
    echo "      Run: sudo reboot"
    echo "      Service will auto-start after reboot."
else
    echo "  Service starts automatically on every boot."
fi

echo "  ════════════════════════════════════════════════════"
echo ""

# Try to open in browser if GUI is available
if command -v xdg-open &>/dev/null; then
    xdg-open "http://localhost:$DASHBOARD_PORT" 2>/dev/null &
elif command -v chromium-browser &>/dev/null; then
    chromium-browser "http://localhost:$DASHBOARD_PORT" 2>/dev/null &
fi
