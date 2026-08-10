# Installation

Three ways to run this, in increasing order of commitment. Start at the top; you
do not need hardware to get a working system.

---

## 1. Simulation only (no hardware, no dependencies)

The runtime core is standard library only. If you have Python 3.11 or newer, you
have everything you need.

```bash
git clone https://github.com/JustAdev742/Project-NeuroGrip--BetaTouch-inc.git
cd Project-NeuroGrip--BetaTouch-inc

# Nothing to install. Run it:
PYTHONPATH=src python3 -m neurogrip simulate all
```

Expected output:

```
grasp-bottle: PASS (8.0 s)
  ✓ hand is holding the object (contacts: 5)
  ...
5/5 scenarios passed
```

If that works, the entire stack — EMG filtering, vision, fusion, motion control,
the ESP32 wire protocol, the safety rules — is running.

### Watching it work

```bash
# A live text dashboard, 30 seconds of simulated operation:
PYTHONPATH=src python3 -m neurogrip run --profile simulation --duration 30

# One scenario, verbosely:
PYTHONPATH=src python3 -m neurogrip simulate grasp-bottle --log-level DEBUG

# What the system thinks of itself:
PYTHONPATH=src python3 -m neurogrip info --profile simulation
```

---

## 2. Development install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

That gives you the `neurogrip` command directly (no `PYTHONPATH`), plus pytest,
ruff and mypy.

```bash
pytest                             # 392 tests, ~2 minutes
ruff check src tests
neurogrip config --check           # validate configuration
```

See [testing.md](testing.md) for what the test suite covers and how to run
subsets of it.

### Optional extras

Each is genuinely optional — the code guards every import and degrades to a
working fallback.

| Extra | Installs | Needed for |
|---|---|---|
| `vision` | numpy, opencv-python, onnxruntime | Real camera input and neural inference. Without it, vision runs a classical fallback. |
| `hardware` | pyserial | Talking to a real ESP32 or EMG front end. |
| `ui` | pillow | Camera preview in the Tk touchscreen UI. |
| `all` | everything above plus `dev` | |

```bash
pip install -e ".[all]"
```

---

## 3. Deploying to the device

Target is a Linux SBC (Raspberry Pi 4/5 or equivalent) driving a touchscreen,
with the ESP32 on USB.

### System packages

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-tk libatlas-base-dev
```

`python3-tk` is needed only for the touchscreen UI; the text and null renderers
work without it.

### Install

```bash
sudo mkdir -p /opt/neurogrip && sudo chown "$USER" /opt/neurogrip
git clone https://github.com/JustAdev742/Project-NeuroGrip--BetaTouch-inc.git /opt/neurogrip
cd /opt/neurogrip
python3 -m venv .venv
.venv/bin/pip install -e ".[vision,hardware,ui]"
```

### Serial permissions

```bash
sudo usermod -aG dialout "$USER"    # log out and back in
```

Give the controller a stable name, so a reboot cannot renumber it into a
different device:

```bash
# /etc/udev/rules.d/99-neurogrip.rules
SUBSYSTEM=="tty", ATTRS{idVendor}=="303a", SYMLINK+="neurogrip-motor"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Then set `servo.port = "/dev/neurogrip-motor"` in your configuration.

### Flash the firmware

```bash
cd firmware/esp32_motor_controller
pio run --target upload
```

See [hardware.md](hardware.md) for wiring and the bring-up sequence.

### Configure

```bash
cp config/hardware.toml config/site.toml   # edit for your build
neurogrip config --check --config config/site.toml
```

`--check` is not optional in practice. It catches the class of mistake that is
otherwise invisible: a misspelled key silently falls back to its default, so
`max_forse = 0.4` does not fail — it is ignored, and the hand runs at 0.85. See
[configuration.md](configuration.md).

### Run as a service

```ini
# /etc/systemd/system/neurogrip.service
[Unit]
Description=NeuroGrip prosthetic hand
After=multi-user.target

[Service]
Type=simple
User=neurogrip
WorkingDirectory=/opt/neurogrip
ExecStart=/opt/neurogrip/.venv/bin/neurogrip run --config config/site.toml
Restart=on-failure
RestartSec=2

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now neurogrip
journalctl -u neurogrip -f
```

`Restart=on-failure` is safe: an unclean exit is detected on the next start, and
the system comes back in Manual mode with the AI disabled until the user
re-enables it. A restart loop therefore cannot repeatedly re-enter the state that
crashed it. See [safety.md](safety.md).

---

## First run on real hardware

Do these in order. Each one can fail in a way that makes the next meaningless.

```bash
neurogrip config --check          # 1. configuration is coherent
neurogrip test link               # 2. the link is good, not merely present
neurogrip diagnose                # 3. every device reports healthy
neurogrip test range              # 4. every finger reaches its travel  ⚠ moves
neurogrip test estop              # 5. the stop actually stops it       ⚠ moves
neurogrip calibrate servo         # 6. measure tendon slack             ⚠ moves
neurogrip calibrate emg           # 7. calibrate the user's muscles
```

Steps 4–6 move the hand. Do not run them while the socket is worn, and keep the
hand clear of anything it could close on.

Step 5 is the one people skip. An emergency stop that has never been tested is
not a safety system, it is an assumption.

---

## Troubleshooting

**`neurogrip: command not found`** — the virtualenv is not active, or you
installed without `-e`. Use `PYTHONPATH=src python3 -m neurogrip` as a fallback.

**`could not open serial port`** — check group membership (`groups | grep
dialout`), that the device exists (`ls -l /dev/serial/by-id/`), and that nothing
else has it open (`fuser /dev/ttyUSB0`).

**Power-on self-test fails on the motor controller** — the link test tells you
whether it is the cable or the firmware: `neurogrip test link`. Zero replies
means power or wiring; CRC errors mean noise.

**The hand starts in Manual and will not switch to AI Assist** — something has
degraded. Check the Diagnostics screen or run `neurogrip diagnose`. The most
common causes are a stale EMG calibration and no camera.

**Tk errors on a headless machine** — set `ui.renderer = "text"` or `"null"`.
