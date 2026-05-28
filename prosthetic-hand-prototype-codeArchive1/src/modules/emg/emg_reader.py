"""
EMG Reader — ADS1115 ADC over I2C on Raspberry Pi / Ubuntu.

Hardware: MyoWare 2.0 or equivalent EMG sensor → ADS1115 16-bit ADC.
I2C bus: /dev/i2c-1 (default on Pi).

On Ubuntu 26.04, ensure I2C is enabled:
  sudo raspi-config  → Interface Options → I2C → Enable
  sudo usermod -aG i2c $USER
  sudo apt install python3-smbus i2c-tools
  i2cdetect -y 1   # should show 0x48 (ADS1115 default address)
"""

from __future__ import annotations

import time
from typing import Optional

from app.models.status import EMGIntent
from app.utils.logging import get_logger
from .interface import EMGInterface
from .emg_processing import EMGConfig, EMGProcessor


class EMGReader(EMGInterface):
	"""
	Reads real EMG signal from an ADS1115 ADC via I2C.

	Normalises the raw voltage to 0.0–1.0 and feeds it through
	the EMGProcessor for smoothing, intent classification, and
	quality estimation.
	"""

	def __init__(self, config: dict) -> None:
		self.config = config
		self.logger = get_logger("emg")
		self.processor = EMGProcessor(
			EMGConfig(
				smoothing_alpha=config.get("smoothing_alpha", 0.2),
				rest_threshold=config.get("rest_threshold", 0.2),
				open_threshold=config.get("open_threshold", 0.25),
				hold_threshold=config.get("hold_threshold", 0.4),
				close_threshold=config.get("close_threshold", 0.6),
				cancel_threshold=config.get("cancel_threshold", 0.9),
				min_quality=config.get("min_quality", 0.4),
			)
		)
		self.intent = EMGIntent.REST
		self.quality = 1.0
		self.signal = 0.0
		self._adc = None
		self._channel = None
		self._override_intent: Optional[EMGIntent] = None
		self._override_expires: float = 0.0
		self._setup_adc()

	def _setup_adc(self) -> None:
		"""Initialise the ADS1115 on I2C bus 1 (Pi default)."""
		try:
			import board
			import busio
			import adafruit_ads1x15.ads1115 as ADS
			from adafruit_ads1x15.analog_in import AnalogIn
		except ImportError as exc:
			raise RuntimeError(
				"ADS1115 dependencies not installed. On Ubuntu 26.04:\n"
				"  pip install adafruit-circuitpython-ads1x15\n"
				"  sudo apt install python3-smbus i2c-tools\n"
				"  sudo raspi-config  # enable I2C"
			) from exc

		i2c_address = int(self.config.get("i2c_address", 0x48))
		channel_index = int(self.config.get("channel", 0))
		gain = float(self.config.get("gain", 1))  # ±4.096V at gain=1

		try:
			i2c = busio.I2C(board.SCL, board.SDA)
			self._adc = ADS.ADS1115(i2c, address=i2c_address, gain=gain)
			channels = [ADS.P0, ADS.P1, ADS.P2, ADS.P3]
			self._channel = AnalogIn(self._adc, channels[channel_index])
			self.logger.info(
				f"ADS1115 ready: address=0x{i2c_address:02x}, "
				f"channel=A{channel_index}, gain={gain}"
			)
		except Exception as exc:
			raise RuntimeError(
				f"Failed to open ADS1115 on I2C bus. "
				f"Check wiring and run: i2cdetect -y 1\n"
				f"Error: {exc}"
			) from exc

	def _read_raw(self) -> float:
		"""Read a single sample and normalise to 0.0–1.0."""
		if not self._channel:
			return 0.0
		try:
			voltage = float(self._channel.voltage)
		except Exception as exc:
			self.logger.warning(f"ADC read error: {exc}")
			return self.signal  # Return last known value

		max_voltage = float(self.config.get("max_voltage", 3.3))
		return max(0.0, min(1.0, voltage / max_voltage))

	def set_override(self, intent: EMGIntent) -> None:
		"""Allow dashboard to temporarily override EMG intent."""
		self._override_intent = intent
		self._override_expires = time.monotonic() + 2.5

	def read_signal(self) -> float:
		# Handle override
		if self._override_intent and time.monotonic() < self._override_expires:
			# Map intent to a synthetic signal level
			synth = {"REST": 0.1, "OPEN": 0.22, "HOLD": 0.5, "CLOSE": 0.75, "CANCEL": 0.95}
			self.signal = synth.get(self._override_intent.value, 0.1)
			self.intent = self._override_intent
			self.quality = 0.95
			return self.signal
		elif self._override_intent and time.monotonic() >= self._override_expires:
			self._override_intent = None

		self.signal = self._read_raw()
		self.intent, self.quality = self.processor.update(self.signal)
		return self.signal

	def get_intent(self) -> EMGIntent:
		return self.intent

	def signal_quality(self) -> float:
		return self.quality
