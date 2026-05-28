from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Callable, Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.models.status import EventRecord, SystemStatus
from .handlers import ConnectionManager, register_routes

EVENT_BUFFER_SIZE = 200


class DashboardServer:
	def __init__(
		self,
		host: str,
		port: int,
		assets_dir: Path,
		emg_override_handler: Optional[Callable[[str], bool]] = None,
		mode_change_handler: Optional[Callable[[str], bool]] = None,
		manual_grip_handler: Optional[Callable[[str], bool]] = None,
		training_capture_handler: Optional[Callable[[str], dict]] = None,
		training_stats_handler: Optional[Callable[[], dict]] = None,
	) -> None:
		self.host = host
		self.port = port
		self.assets_dir = assets_dir
		self._last_status: Dict[str, object] = {}
		self._events: deque[Dict[str, object]] = deque(maxlen=EVENT_BUFFER_SIZE)
		self._ws_manager = ConnectionManager()
		self.app = FastAPI(title="Prosthetic Hand Dashboard", docs_url=None, redoc_url=None)

		register_routes(
			self.app,
			status_provider=self._get_status,
			events_provider=self._get_events,
			emg_override_handler=emg_override_handler,
			mode_change_handler=mode_change_handler,
			manual_grip_handler=manual_grip_handler,
			training_capture_handler=training_capture_handler,
			training_stats_handler=training_stats_handler,
			manager=self._ws_manager,
		)

		# Serve static dashboard files (must be mounted LAST so API routes take priority)
		self.app.mount("/", StaticFiles(directory=str(self.assets_dir), html=True), name="static")

	# ------------------------------------------------------------------
	# Status & Events
	# ------------------------------------------------------------------

	def _get_status(self) -> Dict[str, object]:
		return self._last_status

	def _get_events(self) -> List[Dict[str, object]]:
		return list(self._events)

	def update_status(self, status: SystemStatus) -> None:
		self._last_status = status.to_dict()

	def add_event(self, event: EventRecord) -> None:
		self._events.append(event.to_dict())

	# ------------------------------------------------------------------
	# Server lifecycle
	# ------------------------------------------------------------------

	def serve(self) -> None:
		uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

	def start_in_thread(self) -> threading.Thread:
		thread = threading.Thread(target=self.serve, daemon=True)
		thread.start()
		return thread
