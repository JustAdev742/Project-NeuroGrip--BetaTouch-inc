"""
Config Loader — YAML config with environment variable overrides.

Environment override format: PROSTHETIC_<SECTION>_<KEY>=value
  Example: PROSTHETIC_SYSTEM_MOCK_MODE=true
           PROSTHETIC_SERVO_DRIVER=lgpio
           PROSTHETIC_DASHBOARD_PORT=8080
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
	with path.open("r", encoding="utf-8") as handle:
		data = yaml.safe_load(handle) or {}
	if not isinstance(data, dict):
		raise ValueError(f"Config must be a mapping: {path}")
	return data


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
	for key, value in update.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			base[key] = deep_merge(base[key], value)
		else:
			base[key] = value
	return base


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Scan for PROSTHETIC_* environment variables and inject them.

	PROSTHETIC_SECTION_KEY=value  →  config["section"]["key"] = value
	Type inference: true/false → bool, integers → int, floats → float.
	"""
	prefix = "PROSTHETIC_"
	for key, raw in os.environ.items():
		if not key.startswith(prefix):
			continue
		parts = key[len(prefix):].lower().split("_", 1)
		if len(parts) != 2:
			continue
		section, param = parts

		# Coerce type
		value: Any = raw
		if raw.lower() in ("true", "1", "yes"):
			value = True
		elif raw.lower() in ("false", "0", "no"):
			value = False
		else:
			try:
				value = int(raw)
			except ValueError:
				try:
					value = float(raw)
				except ValueError:
					pass  # keep as string

		config.setdefault(section, {})[param] = value

	return config


def load_config(config_path: str) -> Dict[str, Any]:
	path = Path(config_path).resolve()
	config = load_yaml(path)

	# Process includes (layered configs)
	includes = config.get("includes", [])
	if includes:
		base_dir = path.parent
		for include in includes:
			include_path = (base_dir / include).resolve()
			if include_path.exists():
				include_data = load_yaml(include_path)
				config = deep_merge(config, include_data)

	# Environment variable overrides take highest priority
	config = _apply_env_overrides(config)

	return config
