"""
Application entry point — starts controller + dashboard.

Handles SIGTERM/SIGINT gracefully for systemd service management
on Ubuntu 26.04 (Raspberry Pi).

Usage:
  python -m app.main --mock              # development (no hardware)
  python -m app.main --config myconf.yaml  # custom config
  systemctl start prosthetic-hand        # via systemd service
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from app.controller.main_controller import MainController
from app.utils.config import load_config
from modules.dashboard.server import DashboardServer


def main() -> None:
	parser = argparse.ArgumentParser(
		description="AI-assisted prosthetic hand controller",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=(
			"Environment overrides:\n"
			"  PROSTHETIC_SYSTEM_MOCK_MODE=true\n"
			"  PROSTHETIC_SERVO_DRIVER=lgpio\n"
			"  PROSTHETIC_DASHBOARD_PORT=8080"
		),
	)
	parser.add_argument("--config", default="configs/default.yaml",
	                     help="Path to YAML config file")
	parser.add_argument("--mock", action="store_true",
	                     help="Run with mock hardware (no GPIO/I2C required)")
	args = parser.parse_args()

	config = load_config(args.config)
	dashboard_cfg = config.get("system", {}).get("dashboard", {})
	host = dashboard_cfg.get("host", "0.0.0.0")
	port = int(dashboard_cfg.get("port", 8000))
	mock_mode = args.mock or config.get("system", {}).get("mock_mode", False)

	root_dir = Path(__file__).resolve().parents[2]
	assets_dir = root_dir / "assets" / "dashboard"

	# Create controller first so we can wire its handlers into the dashboard
	controller = MainController(args.config)

	dashboard = DashboardServer(
		host=host,
		port=port,
		assets_dir=assets_dir,
		emg_override_handler=controller.handle_emg_override,
		mode_change_handler=controller.handle_mode_change,
		manual_grip_handler=controller.handle_manual_grip,
		training_capture_handler=controller.handle_training_capture,
		training_stats_handler=controller.get_training_stats,
	)
	dashboard.start_in_thread()

	# Wire dashboard into controller
	controller.dashboard = dashboard

	# ── Graceful shutdown on SIGTERM/SIGINT (systemd sends SIGTERM) ──
	def shutdown_handler(signum, frame):
		print("\n  ⏹  Received signal — shutting down gracefully...")
		controller.shutdown()
		sys.exit(0)

	signal.signal(signal.SIGTERM, shutdown_handler)
	signal.signal(signal.SIGINT, shutdown_handler)

	mode_text = "🧪  MOCK MODE" if mock_mode else "⚡  HARDWARE MODE"
	print(f"\n  🤖  NeuroGrip — AI Prosthetic Hand")
	print(f"  {mode_text}")
	print(f"  📊  Dashboard: http://localhost:{port}")
	print(f"  Press Ctrl+C to stop.\n")

	try:
		controller.run(mock_mode=mock_mode)
	except KeyboardInterrupt:
		pass
	finally:
		controller.shutdown()


if __name__ == "__main__":
	main()
