"""
Unit tests for Multi-Asset Kelly Allocator
"""

import pytest
import sys
import numpy as np
sys.path.insert(0, 'src')

from portfolio.multi_asset_kelly import MultiAssetKellyAllocator


class TestMultiAssetKellyAllocator:
    """Test suite for Multi-Asset Kelly Allocator."""
    
    def test_initialization(self):
        """Test allocator initialization."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            kelly_fraction=0.25
        )
        
        assert allocator.portfolio_value == 100000
        assert allocator.kelly_fraction == 0.25
    
    def test_single_asset_allocation(self):
        """Test allocation with single asset."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            kelly_fraction=0.25
        )
        
        allocations, diag = allocator.allocate(
            expected_returns={'BTC-USD': 0.20},
            volatilities={'BTC-USD': 0.05}
        )
        
        assert 'BTC-USD' in allocations
        assert allocations['BTC-USD'] > 0
        assert diag['status'] == 'success'
    
    def test_multi_asset_allocation(self):
        """Test allocation with multiple assets."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            kelly_fraction=0.25
        )
        
        allocations, diag = allocator.allocate(
            expected_returns={'BTC-USD': 0.20, 'ETH-USD': 0.15, 'SOL-USD': 0.25},
            volatilities={'BTC-USD': 0.05, 'ETH-USD': 0.06, 'SOL-USD': 0.08}
        )
        
        assert len(allocations) == 3
        assert diag['n_assets'] == 3
        assert diag['total_allocated_usd'] > 0
    
    def test_position_limit(self):
        """Test that position limits are enforced."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            max_position_pct=0.10,
            kelly_fraction=0.5  # Aggressive to trigger limit
        )
        
        allocations, diag = allocator.allocate(
            expected_returns={'BTC-USD': 0.50},  # Very high return
            volatilities={'BTC-USD': 0.02}       # Low vol
        )
        
        # Should be capped at 10% of 100k = 10k
        assert allocations['BTC-USD'] <= 10000
    
    def test_utilization_limit(self):
        """Test that portfolio utilization limit is enforced."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            max_portfolio_utilization=0.50,
            kelly_fraction=0.5
        )
        
        allocations, diag = allocator.allocate(
            expected_returns={'BTC-USD': 0.30, 'ETH-USD': 0.30, 'SOL-USD': 0.30},
            volatilities={'BTC-USD': 0.02, 'ETH-USD': 0.02, 'SOL-USD': 0.02}
        )
        
        total = sum(allocations.values())
        assert total <= 50000  # 50% of 100k
    
    def test_correlation_matrix(self):
        """Test allocation with correlation matrix."""
        allocator = MultiAssetKellyAllocator(
            portfolio_value=100000,
            kelly_fraction=0.25
        )
        
        # Highly correlated assets
        corr = np.array([
            [1.0, 0.9],
            [0.9, 1.0]
        ])
        
        allocations, diag = allocator.allocate(
            expected_returns={'BTC-USD': 0.20, 'ETH-USD': 0.15},
            volatilities={'BTC-USD': 0.05, 'ETH-USD': 0.06},
            correlation_matrix=corr,
            symbols=['BTC-USD', 'ETH-USD']
        )
        
        assert diag['status'] == 'success'
    
    def test_rebalancing_signals(self):
        """Test rebalancing signal generation."""
        allocator = MultiAssetKellyAllocator(portfolio_value=100000)
        
        current = {'BTC-USD': 5000, 'ETH-USD': 3000}
        target = {'BTC-USD': 7000, 'ETH-USD': 2000}
        
        trades, diag = allocator.get_rebalancing_signals(
            current_positions=current,
            target_allocations=target,
            rebalance_threshold=0.10
        )
        
        # BTC needs +2000 (40% change), ETH needs -1000 (33% change)
        assert trades.get('BTC-USD', 0) > 0  # Buy more
        assert trades.get('ETH-USD', 0) < 0  # Sell some
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        allocator = MultiAssetKellyAllocator(portfolio_value=100000)
        
        stats = allocator.get_statistics()
        
        assert 'portfolio_value' in stats
        assert 'kelly_fraction' in stats
        assert 'total_allocations' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
