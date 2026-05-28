# 🤖 NeuroGrip — AI-Assisted Prosthetic Hand

An intelligent prosthetic hand prototype that uses **EMG signals** for user intent and **computer vision** for automatic grasp classification. Built for a school showcase, inspired by [DeGol et al. (IEEE EMBC 2016)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5325038/).

> ⚠️ **Educational prototype only** — not a medical device.

---

## Features

| Feature | Description |
|---|---|
| **EMG Control** | User controls the hand via muscle signals — AI never overrides the user |
| **Vision Grasp** | Palm camera + YOLO classifies objects → recommends optimal grip |
| **8 Hand Modes** | Normal (AI+EMG), Typing, Gaming, Mouse, Basketball, Tennis, Baseball, Cricket |
| **Auto-Training** | Capture & label photos of unknown objects for model fine-tuning |
| **Smartwatch Dashboard** | Live telemetry, mode switching, mini games, GitHub updates |
| **21 Grip Profiles** | Each tuned for 180° / 1.5 kg-cm servos with sine-curve motion |
| **Safe Fallback** | Any failure → SAFE grip + emergency stop |

## Hardware

- **Brain**: Raspberry Pi 4 Model B+ (Ubuntu 26.04 Resolute, kernel 6.18)
- **Servos**: 5× SG90 — 180° range, 1.5 kg-cm torque, 50 Hz PWM
- **EMG**: MyoWare 2.0 → ADS1115 16-bit ADC (I2C)
- **Camera**: Pi Camera Module 3 or USB webcam (V4L2)
- **Display**: Mini touchscreen for smartwatch dashboard

## Quick Start

### Development (no hardware)
```bash
git clone https://github.com/your-username/prosthetic-hand-prototype.git
cd prosthetic-hand-prototype
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/run_mock.sh
# Dashboard: http://localhost:8000
```

### Raspberry Pi (Ubuntu 26.04)
```bash
git clone https://github.com/your-username/prosthetic-hand-prototype.git
cd prosthetic-hand-prototype
./scripts/setup_pi.sh    # installs everything + systemd service
sudo reboot              # enable I2C + camera
sudo systemctl start prosthetic-hand
# Dashboard: http://<pi-ip>:8000
```

### Environment Overrides
```bash
export PROSTHETIC_SERVO_DRIVER=lgpio     # use modern GPIO backend
export PROSTHETIC_DASHBOARD_PORT=8080    # change dashboard port
export PROSTHETIC_SYSTEM_MOCK_MODE=true  # force mock mode
```

## Architecture

```
EMG Sensor → ADS1115 → EMG Reader → Intent (REST/OPEN/CLOSE/HOLD/CANCEL)
                                          ↓
Camera → YOLO Detector → Grip Selector ← Intent
                              ↓
                     Grip Library (21 profiles)
                              ↓
                   Servo Driver (lgpio/RPi.GPIO)
                              ↓
                      5× 180° Servos
                              ↓
                  Dashboard ← WebSocket ← Status
```

**Design Principle**: The user is always in control. EMG intent is the primary signal. Vision only *recommends* a grip — the user must confirm with EMG CLOSE. Manual overrides from the dashboard always take priority over AI.

## Project Structure

```
├── assets/dashboard/     # Smartwatch UI (HTML/CSS/JS)
├── configs/              # YAML configuration
│   ├── default.yaml      # Main config (includes others)
│   ├── servo.yaml        # Servo pins, limits, PWM specs
│   ├── emg.yaml          # EMG thresholds, smoothing
│   ├── vision.yaml       # YOLO model config
│   └── grip_mappings.yaml
├── scripts/
│   ├── setup_pi.sh       # Ubuntu 26.04 setup
│   ├── run_mock.sh       # Development mode
│   ├── run_hardware.sh   # Hardware mode
│   └── prosthetic-hand.service  # systemd unit
├── src/
│   ├── app/
│   │   ├── main.py              # Entry point
│   │   ├── controller/
│   │   │   ├── main_controller.py  # Orchestrator
│   │   │   └── state_machine.py    # IDLE→DETECTING→GRIPPING→HOLDING
│   │   ├── models/
│   │   │   ├── grip_types.py    # GripType + HandMode enums
│   │   │   └── status.py        # SystemStatus dataclass
│   │   └── utils/               # Config, logging, timing
│   └── modules/
│       ├── camera/              # OpenCV + V4L2
│       ├── emg/                 # ADS1115 reader + mock
│       ├── grip/                # Grip library + selector
│       ├── servo/               # lgpio/RPi.GPIO driver + mock
│       ├── vision/              # YOLO detector + mock
│       └── dashboard/           # FastAPI server + WebSocket
├── tests/                       # pytest suite (21 tests)
├── training_data/               # Auto-captured training images
└── pyproject.toml               # Modern Python packaging
```

## Hand Modes

| Mode | EMG CLOSE | EMG HOLD | Use Case |
|---|---|---|---|
| Normal | AI-selected grip | — | Daily object handling |
| Typing | Key tap | Home position | Keyboard input |
| Gaming | Action press | WASD position | PC gaming |
| Mouse | Click | Rest on mouse | Computer use |
| Basketball 🏀 | Shooting release | Ball palm | Court play |
| Tennis 🎾 | Backhand grip | Forehand grip | Racket sports |
| Baseball ⚾ | Power swing | Bat grip | Batting |
| Cricket 🏏 | Power swing | V-grip bat | Batting |

## Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

## Research Reference

> DeGol, J., Akhtar, A., Manja, B., & Bretl, T. (2016). *Automatic Grasp Selection using a Camera in a Hand Prosthesis.* IEEE EMBC. [PMC5325038](https://pmc.ncbi.nlm.nih.gov/articles/PMC5325038/)

Key finding: Palm camera + CNN achieves 93.2% grasp classification accuracy without external sensors.

## License

MIT
