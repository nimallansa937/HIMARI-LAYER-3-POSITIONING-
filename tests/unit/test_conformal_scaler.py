"""
Unit tests for Conformal Position Scaler
"""

import pytest
import sys
sys.path.insert(0, 'src')

from engines.conformal_scaler import ConformalPositionScaler


class TestConformalPositionScaler:
    """Test suite for Conformal Position Scaler."""
    
    def test_initialization(self):
        """Test scaler initialization."""
        scaler = ConformalPositionScaler(coverage=0.90, window_size=200)
        
        assert scaler.coverage == 0.90
        assert scaler.alpha == pytest.approx(0.10)
        assert scaler.window_size == 200
        assert len(scaler.residuals) == 0
    
    def test_insufficient_samples(self):
        """Test that scaler returns 1.0 scale with insufficient samples."""
        scaler = ConformalPositionScaler(min_samples=20)
        
        position, diagnostics = scaler.scale_position(base_position_usd=10000)
        
        assert position == 10000
        assert diagnostics['scale_factor'] == 1.0
        assert diagnostics['status'] == 'insufficient_samples'
    
    def test_null_safety_predicted(self):
        """Test NULL safety for predicted_return."""
        scaler = ConformalPositionScaler()
        
        scaler.update(predicted_return=None, actual_return=0.05)
        
        assert scaler.null_rejections == 1
        assert len(scaler.residuals) == 0
    
    def test_null_safety_actual(self):
        """Test NULL safety for actual_return."""
        scaler = ConformalPositionScaler()
        
        scaler.update(predicted_return=0.05, actual_return=None)
        
        assert scaler.null_rejections == 1
        assert len(scaler.residuals) == 0
    
    def test_nan_rejection(self):
        """Test NaN rejection."""
        scaler = ConformalPositionScaler()
        
        scaler.update(predicted_return=float('nan'), actual_return=0.05)
        
        assert scaler.nan_rejections == 1
        assert len(scaler.residuals) == 0
    
    def test_valid_update(self):
        """Test valid residual update."""
        scaler = ConformalPositionScaler()
        
        scaler.update(predicted_return=0.05, actual_return=0.06)
        
        assert len(scaler.residuals) == 1
        assert scaler.residuals[0] == pytest.approx(0.01)
        assert scaler.total_updates == 1
    
    def test_scaling_with_samples(self):
        """Test scaling with sufficient samples."""
        scaler = ConformalPositionScaler(min_samples=5)
        
        # Add 10 samples
        for i in range(10):
            scaler.update(predicted_return=0.05, actual_return=0.05 + (i * 0.01))
        
        position, diagnostics = scaler.scale_position(base_position_usd=10000)
        
        assert position <= 10000
        assert diagnostics['scale_factor'] <= 1.0
        assert diagnostics['status'] == 'active'
        assert diagnostics['samples'] == 10
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        scaler = ConformalPositionScaler()
        
        for i in range(5):
            scaler.update(0.05, 0.06)
        
        stats = scaler.get_statistics()
        
        assert stats['samples'] == 5
        assert 'mean_residual' in stats
        assert 'std_residual' in stats
        assert stats['status'] == 'active'
    
    def test_reset(self):
        """Test scaler reset."""
        scaler = ConformalPositionScaler()
        scaler.update(0.05, 0.06)
        scaler.reset()
        
        assert len(scaler.residuals) == 0
        assert scaler.total_updates == 0
        assert scaler.null_rejections == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
