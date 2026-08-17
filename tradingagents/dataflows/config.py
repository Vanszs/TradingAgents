import threading
from copy import deepcopy
from typing import Dict, Optional

import tradingagents.default_config as default_config

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
_lock = threading.RLock()


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    with _lock:
        if _config is None:
            _config = deepcopy(default_config.DEFAULT_CONFIG)


def set_config(config: Dict):
    """Update the configuration with custom values.

    Dict-valued keys (e.g. ``data_vendors``) are merged one level deep so a
    partial update like ``{"data_vendors": {"core_stock_apis": "alpha_vantage"}}``
    keeps the other nested keys from the default; scalar keys are replaced.
    """
    global _config
    with _lock:
        initialize_config()
        incoming = deepcopy(config)
        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(_config.get(key), dict):
                _config[key].update(value)
            else:
                _config[key] = value


def get_config() -> Dict:
    """Get the current configuration."""
    with _lock:
        if _config is None:
            initialize_config()
        return deepcopy(_config)


def is_point_in_time_mode(config: Optional[Dict] = None) -> bool:
    """Return whether live vendor data must be rejected."""
    active = get_config() if config is None else config
    return bool(active.get("point_in_time_mode") or active.get("backtest_mode"))


# Initialize with default config
initialize_config()
