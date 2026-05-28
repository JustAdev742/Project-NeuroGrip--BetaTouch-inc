# EMG Calibration

1) Record resting signal for 10 seconds.
2) Record a strong close signal for 10 seconds.
3) Adjust thresholds in configs/emg.yaml:
	- rest_threshold just above resting baseline
	- close_threshold in the upper half of strong contraction
	- open_threshold slightly above rest
4) Verify intent states in the dashboard.
