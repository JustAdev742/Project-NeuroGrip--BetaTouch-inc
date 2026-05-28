from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse


class ConnectionManager:
	"""Manages active WebSocket connections and broadcasts status updates."""

	def __init__(self) -> None:
		self._connections: Set[WebSocket] = set()

	async def connect(self, websocket: WebSocket) -> None:
		await websocket.accept()
		self._connections.add(websocket)

	def disconnect(self, websocket: WebSocket) -> None:
		self._connections.discard(websocket)

	async def broadcast(self, payload: Dict[str, Any]) -> None:
		message = json.dumps(payload)
		dead: List[WebSocket] = []
		for ws in self._connections:
			try:
				await ws.send_text(message)
			except Exception:
				dead.append(ws)
		for ws in dead:
			self._connections.discard(ws)

	@property
	def client_count(self) -> int:
		return len(self._connections)


def register_routes(
	app: FastAPI,
	status_provider: Callable[[], Dict[str, object]],
	events_provider: Callable[[], List[Dict[str, object]]],
	emg_override_handler: Optional[Callable[[str], bool]] = None,
	mode_change_handler: Optional[Callable[[str], bool]] = None,
	manual_grip_handler: Optional[Callable[[str], bool]] = None,
	training_capture_handler: Optional[Callable[[str], dict]] = None,
	training_stats_handler: Optional[Callable[[], dict]] = None,
	manager: Optional[ConnectionManager] = None,
) -> None:
	"""Register all API and WebSocket routes on the FastAPI app."""

	ws_manager = manager or ConnectionManager()

	# ------------------------------------------------------------------
	# REST endpoints
	# ------------------------------------------------------------------

	@app.get("/api/status")
	def get_status() -> Dict[str, object]:
		return status_provider() or {}

	@app.get("/api/events")
	def get_events() -> List[Dict[str, object]]:
		return events_provider()

	@app.get("/api/health")
	def health_check() -> Dict[str, object]:
		status = status_provider() or {}
		return {
			"ok": True,
			"mock_mode": status.get("mock_mode_active", False),
			"system_state": status.get("system_state", "UNKNOWN"),
			"hand_mode": status.get("hand_mode", "NORMAL"),
			"ws_clients": ws_manager.client_count,
			"uptime": status.get("uptime_seconds", 0),
		}

	# ------------------------------------------------------------------
	# EMG Override
	# ------------------------------------------------------------------

	@app.post("/api/emg-override")
	async def emg_override(body: Dict[str, str]) -> JSONResponse:
		intent = body.get("intent", "").upper()
		valid = {"REST", "CLOSE", "OPEN", "HOLD", "CANCEL"}
		if intent not in valid:
			return JSONResponse(
				{"ok": False, "error": f"Invalid intent '{intent}'. Must be one of {valid}"},
				status_code=400,
			)
		if emg_override_handler:
			success = emg_override_handler(intent)
			return JSONResponse({"ok": success, "intent": intent})
		return JSONResponse({"ok": False, "error": "No handler registered"}, status_code=501)

	# ------------------------------------------------------------------
	# Hand Mode Switch
	# ------------------------------------------------------------------

	@app.post("/api/mode")
	async def change_mode(body: Dict[str, str]) -> JSONResponse:
		mode = body.get("mode", "").upper()
		if mode_change_handler:
			success = mode_change_handler(mode)
			if success:
				return JSONResponse({"ok": True, "mode": mode})
			return JSONResponse({"ok": False, "error": f"Invalid mode '{mode}'"}, status_code=400)
		return JSONResponse({"ok": False, "error": "No handler registered"}, status_code=501)

	# ------------------------------------------------------------------
	# Manual Grip Selection
	# ------------------------------------------------------------------

	@app.post("/api/grip")
	async def select_manual_grip(body: Dict[str, str]) -> JSONResponse:
		grip = body.get("grip", "").upper()
		if manual_grip_handler:
			success = manual_grip_handler(grip)
			if success:
				return JSONResponse({"ok": True, "grip": grip})
			return JSONResponse(
				{"ok": False, "error": f"Grip '{grip}' not available in current mode"},
				status_code=400,
			)
		return JSONResponse({"ok": False, "error": "No handler registered"}, status_code=501)

	# ------------------------------------------------------------------
	# Training Capture (auto-train for unknown objects)
	# ------------------------------------------------------------------

	@app.post("/api/training/capture")
	async def training_capture(body: Dict[str, str]) -> JSONResponse:
		label = body.get("label", "").strip().lower()
		if not label:
			return JSONResponse({"ok": False, "error": "Label required"}, status_code=400)
		if training_capture_handler:
			result = training_capture_handler(label)
			return JSONResponse(result, status_code=200 if result.get("ok") else 400)
		return JSONResponse({"ok": False, "error": "No handler registered"}, status_code=501)

	@app.get("/api/training/stats")
	def training_stats() -> Dict[str, object]:
		if training_stats_handler:
			return training_stats_handler()
		return {"total": 0, "labels": {}}

	# ------------------------------------------------------------------
	# GitHub Integration (proxy to avoid CORS)
	# ------------------------------------------------------------------

	@app.get("/api/github/issues")
	async def github_issues() -> JSONResponse:
		"""Fetch latest issues from the configured GitHub repo."""
		try:
			import urllib.request
			import json as json_mod

			# Default to this project's hypothetical repo; override via config
			repo = "your-username/prosthetic-hand-prototype"
			url = f"https://api.github.com/repos/{repo}/issues?state=all&per_page=10&sort=updated"
			req = urllib.request.Request(url, headers={"User-Agent": "prosthetic-hand-dashboard"})
			with urllib.request.urlopen(req, timeout=5) as resp:
				data = json_mod.loads(resp.read().decode())
			issues = [
				{
					"number": i.get("number"),
					"title": i.get("title"),
					"state": i.get("state"),
					"updated_at": i.get("updated_at"),
					"labels": [l.get("name") for l in i.get("labels", [])],
					"url": i.get("html_url"),
				}
				for i in data[:10]
			]
			return JSONResponse({"ok": True, "issues": issues})
		except Exception as exc:
			return JSONResponse({"ok": False, "issues": [], "error": str(exc)})

	@app.get("/api/github/releases")
	async def github_releases() -> JSONResponse:
		"""Fetch latest releases from the configured GitHub repo."""
		try:
			import urllib.request
			import json as json_mod

			repo = "your-username/prosthetic-hand-prototype"
			url = f"https://api.github.com/repos/{repo}/releases?per_page=5"
			req = urllib.request.Request(url, headers={"User-Agent": "prosthetic-hand-dashboard"})
			with urllib.request.urlopen(req, timeout=5) as resp:
				data = json_mod.loads(resp.read().decode())
			releases = [
				{
					"tag": r.get("tag_name"),
					"name": r.get("name"),
					"published": r.get("published_at"),
					"url": r.get("html_url"),
				}
				for r in data[:5]
			]
			return JSONResponse({"ok": True, "releases": releases})
		except Exception as exc:
			return JSONResponse({"ok": False, "releases": [], "error": str(exc)})

	# ------------------------------------------------------------------
	# WebSocket
	# ------------------------------------------------------------------

	@app.websocket("/ws")
	async def websocket_feed(websocket: WebSocket) -> None:
		await ws_manager.connect(websocket)
		try:
			while True:
				try:
					await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
				except asyncio.TimeoutError:
					pass
				payload = status_provider() or {}
				await websocket.send_json(payload)
				await asyncio.sleep(0.08)  # ~12 updates/sec
		except (WebSocketDisconnect, Exception):
			ws_manager.disconnect(websocket)
