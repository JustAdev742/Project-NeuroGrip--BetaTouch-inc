import json
import logging
import os
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
	def format(self, record: logging.LogRecord) -> str:
		payload: Dict[str, Any] = {
			"level": record.levelname,
			"name": record.name,
			"message": record.getMessage(),
		}
		if record.exc_info:
			payload["exc_info"] = self.formatException(record.exc_info)
		return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str) -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	handler = logging.StreamHandler()
	handler.setFormatter(JsonFormatter())
	logger.addHandler(handler)
	logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
	logger.propagate = False
	return logger
