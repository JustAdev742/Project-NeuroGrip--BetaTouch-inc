import time


class RateController:
	def __init__(self, hz: float):
		self.period = 1.0 / max(hz, 0.001)
		self.next_time = time.perf_counter()

	def sleep(self) -> None:
		self.next_time += self.period
		delay = self.next_time - time.perf_counter()
		if delay > 0:
			time.sleep(delay)
		else:
			self.next_time = time.perf_counter()


class IntervalTimer:
	def __init__(self, hz: float):
		self.period = 1.0 / max(hz, 0.001)
		self.last_time = 0.0

	def ready(self) -> bool:
		now = time.perf_counter()
		if now - self.last_time >= self.period:
			self.last_time = now
			return True
		return False


class FpsTracker:
	def __init__(self):
		self.last_time = time.perf_counter()
		self.frames = 0
		self.value = 0.0

	def tick(self) -> float:
		self.frames += 1
		now = time.perf_counter()
		delta = now - self.last_time
		if delta >= 1.0:
			self.value = self.frames / delta
			self.frames = 0
			self.last_time = now
		return self.value
