"""
HIMARI OPUS V2 - Hot-Reload Configuration Manager
==================================================

File watcher-based configuration manager with validation and callbacks.

Features:
- Watch layer3_config.yaml for changes
- Reload without restart (validation before applying)
- Callback notification on config change
- Thread-safe access
- Graceful error handling

Version: 3.1 Enhanced
"""

import yaml
import time
import os
from pathlib import Path
from typing import Dict, Callable, Optional, Any
from threading import Thread, Lock
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Hot-reload configuration manager for Layer 3.
    
    Features:
    - Watch layer3_config.yaml for changes
    - Reload without restart (validation before applying)
    - Callback notification on config change
    - Thread-safe access
    """
    
    def __init__(self, config_path: str = "config/layer3_config.yaml", poll_interval: int = 5):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
            poll_interval: Polling interval in seconds
        """
        self.config_path = Path(config_path)
        self.poll_interval = poll_interval
        
        self._config: Dict = {}
        self._lock = Lock()
        self._last_modified = 0.0
        self._callbacks = []
        self._watcher_thread: Optional[Thread] = None
        self._running = False
        
        # Initial load
        self.reload_config()
    
    def load_config(self) -> Dict:
        """Load config from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def validate_config(self, config: Dict) -> bool:
        """
        Validate config structure before applying.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_sections = ['position_sizing', 'risk_management', 'validation_criteria']
        
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        # Validate kelly_fraction range
        kelly_fraction = config.get('position_sizing', {}).get('bayesian_kelly', {}).get('kelly_fraction')
        if kelly_fraction is not None and not (0 < kelly_fraction <= 0.5):
            logger.error(f"Invalid kelly_fraction: {kelly_fraction}. Must be in (0, 0.5]")
            return False
        
        # Validate max_leverage
        max_leverage = config.get('risk_management', {}).get('max_leverage')
        if max_leverage is not None and not (1.0 <= max_leverage <= 3.0):
            logger.error(f"Invalid max_leverage: {max_leverage}. Must be in [1.0, 3.0]")
            return False
        
        # Validate conformal coverage
        coverage = config.get('position_sizing', {}).get('conformal_prediction', {}).get('coverage')
        if coverage is not None and not (0.5 <= coverage <= 0.99):
            logger.error(f"Invalid coverage: {coverage}. Must be in [0.5, 0.99]")
            return False
        
        return True
    
    def reload_config(self) -> bool:
        """
        Reload config if file changed.
        
        Returns:
            True if reloaded, False if no change or error
        """
        try:
            if not self.config_path.exists():
                logger.warning(f"Config file not found: {self.config_path}")
                return False
            
            current_mtime = os.path.getmtime(self.config_path)
            
            if current_mtime <= self._last_modified:
                return False  # No change
            
            # Load new config
            new_config = self.load_config()
            
            # Validate before applying
            if not self.validate_config(new_config):
                logger.error("Config validation failed. Keeping previous config.")
                return False
            
            # Apply new config
            with self._lock:
                old_config = self._config
                self._config = new_config
                self._last_modified = current_mtime
            
            logger.info(f"Config reloaded successfully at {time.time()}")
            
            # Notify callbacks
            self._notify_callbacks(old_config, new_config)
            
            return True
        
        except Exception as e:
            logger.error(f"Error reloading config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value by dot-separated path.
        
        Args:
            key_path: Dot-separated path (e.g., 'position_sizing.bayesian_kelly.kelly_fraction')
            default: Default value if not found
            
        Returns:
            Config value or default
        
        Example:
            >>> config.get('position_sizing.bayesian_kelly.kelly_fraction')
            0.25
        """
        with self._lock:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    return default
            
            return value if value is not None else default
    
    def register_callback(self, callback: Callable[[Dict, Dict], None]):
        """
        Register callback for config changes.
        
        Args:
            callback: Function(old_config, new_config) -> None
        """
        self._callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")
    
    def _notify_callbacks(self, old_config: Dict, new_config: Dict):
        """Notify registered callbacks of config change."""
        for callback in self._callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                logger.error(f"Callback error in {callback.__name__}: {e}")
    
    def start_watcher(self):
        """Start background thread to watch for config changes."""
        if self._watcher_thread is not None:
            logger.warning("Config watcher already running")
            return
        
        self._running = True
        self._watcher_thread = Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()
        logger.info(f"Config watcher started (poll interval: {self.poll_interval}s)")
    
    def stop_watcher(self):
        """Stop config watcher thread."""
        self._running = False
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=self.poll_interval + 1)
            self._watcher_thread = None
        logger.info("Config watcher stopped")
    
    def _watch_loop(self):
        """Background loop to check for config changes."""
        while self._running:
            self.reload_config()
            time.sleep(self.poll_interval)
    
    def get_all(self) -> Dict:
        """Get complete configuration (thread-safe)."""
        with self._lock:
            return self._config.copy()
    
    def get_state(self) -> Dict:
        """Get current state for monitoring."""
        return {
            'config_path': str(self.config_path),
            'config_exists': self.config_path.exists(),
            'last_modified': self._last_modified,
            'watcher_running': self._running,
            'callback_count': len(self._callbacks),
            'poll_interval_sec': self.poll_interval
        }
