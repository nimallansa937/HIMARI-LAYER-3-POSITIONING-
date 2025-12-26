"""
Unit tests for Config Manager
"""

import pytest
import sys
import os
import tempfile
import yaml
sys.path.insert(0, 'src')

from core.layer3_config_manager import ConfigManager


class TestConfigManager:
    """Test suite for Hot-Reload Config Manager."""
    
    def test_initialization_with_missing_file(self):
        """Test initialization with non-existent file."""
        config = ConfigManager(config_path="nonexistent.yaml")
        
        assert config._config == {}
    
    def test_load_valid_config(self):
        """Test loading valid configuration."""
        # Create temp config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'position_sizing': {
                    'bayesian_kelly': {'kelly_fraction': 0.25}
                },
                'risk_management': {'max_leverage': 1.0},
                'validation_criteria': {'max_confidence': 1.0}
            }, f)
            temp_path = f.name
        
        try:
            config = ConfigManager(config_path=temp_path)
            
            assert config.get('position_sizing.bayesian_kelly.kelly_fraction') == 0.25
            assert config.get('risk_management.max_leverage') == 1.0
        finally:
            os.unlink(temp_path)
    
    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'position_sizing': {
                    'bayesian_kelly': {'kelly_fraction': 0.3}
                },
                'risk_management': {},
                'validation_criteria': {}
            }, f)
            temp_path = f.name
        
        try:
            config = ConfigManager(config_path=temp_path)
            
            value = config.get('position_sizing.bayesian_kelly.kelly_fraction')
            assert value == 0.3
            
            # Test default for missing key
            default = config.get('nonexistent.key', 'default')
            assert default == 'default'
        finally:
            os.unlink(temp_path)
    
    def test_validation_invalid_kelly(self):
        """Test validation rejects invalid Kelly fraction."""
        config = ConfigManager(config_path="nonexistent.yaml")
        
        invalid_config = {
            'position_sizing': {
                'bayesian_kelly': {'kelly_fraction': 0.8}  # > 0.5
            },
            'risk_management': {},
            'validation_criteria': {}
        }
        
        assert config.validate_config(invalid_config) == False
    
    def test_validation_invalid_leverage(self):
        """Test validation rejects invalid leverage."""
        config = ConfigManager(config_path="nonexistent.yaml")
        
        invalid_config = {
            'position_sizing': {},
            'risk_management': {'max_leverage': 5.0},  # > 3.0
            'validation_criteria': {}
        }
        
        assert config.validate_config(invalid_config) == False
    
    def test_callback_registration(self):
        """Test callback registration."""
        config = ConfigManager(config_path="nonexistent.yaml")
        
        callback_called = []
        
        def my_callback(old, new):
            callback_called.append((old, new))
        
        config.register_callback(my_callback)
        
        assert len(config._callbacks) == 1
    
    def test_get_state(self):
        """Test state retrieval."""
        config = ConfigManager(config_path="nonexistent.yaml", poll_interval=10)
        
        state = config.get_state()
        
        assert 'config_path' in state
        assert state['poll_interval_sec'] == 10
        assert state['watcher_running'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
