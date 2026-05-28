# Deployment

1) Install dependencies on the Pi:
	- scripts/setup_pi.sh
2) Run mock mode first:
	- scripts/run_mock.sh
3) Switch servo driver to gpio in configs/servo.yaml
4) Run real mode:
	- scripts/run_real.sh

Open the dashboard at http://pi-ip-address:8000
