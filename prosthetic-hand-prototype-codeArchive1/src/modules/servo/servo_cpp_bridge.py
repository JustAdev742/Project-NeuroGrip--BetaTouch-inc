class ServoCppBridge:
	def __init__(self) -> None:
		self.available = False

	def is_available(self) -> bool:
		return self.available
