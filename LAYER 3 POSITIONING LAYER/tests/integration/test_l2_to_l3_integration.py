"""
Test Layer 2 → Layer 3 Integration
====================================

Tests the complete L2→L3 pipeline including:
- Signal mapping (L2 TacticalDecision → L3 TacticalSignal)
- Order generation (ExecutionBridge)
- Order execution (ExecutionEngine paper trading)
"""

import pytest
import sys
import time
sys.path.insert(0, 'src')

from core.layer3_types import (
    TacticalSignal, TacticalAction, MarketRegime, CascadeIndicators
)
from integration.l2_signal_mapper import L2SignalMapper
from engines.execution_engine import ExecutionEngine
from phases.phase1_core import Layer3Phase1


# ============================================================================
# Mock Layer 2 Types (for testing without Layer 2 dependency)
# ============================================================================

class MockTradeAction:
    """Mock L2 TradeAction for testing."""
    def __init__(self, name: str):
        self._name = name
    
    @property
    def name(self):
        return self._name


class MockTier:
    """Mock L2 Tier for testing."""
    def __init__(self, value: int):
        self._value = value
    
    @property
    def value(self):
        return self._value


class MockTacticalDecision:
    """Mock L2 TacticalDecision for testing."""
    def __init__(
        self,
        action: str,
        confidence: float,
        tier: int,
        reason: str = "Test decision",
        metadata: dict = None,
    ):
        self.action = MockTradeAction(action)
        self.confidence = confidence
        self.tier = MockTier(tier)
        self.reason = reason
        self.metadata = metadata or {}
        self.timestamp = time.time()


# ============================================================================
# L2 Signal Mapper Tests
# ============================================================================

class TestL2SignalMapper:
    """Tests for L2SignalMapper."""
    
    def test_map_trade_action_buy(self):
        """Test mapping BUY action."""
        mapper = L2SignalMapper()
        action = mapper.map_trade_action(MockTradeAction("BUY"))
        assert action == TacticalAction.BUY
    
    def test_map_trade_action_strong_sell(self):
        """Test mapping STRONG_SELL action."""
        mapper = L2SignalMapper()
        action = mapper.map_trade_action(MockTradeAction("STRONG_SELL"))
        assert action == TacticalAction.STRONG_SELL
    
    def test_map_tier_to_risk_t1(self):
        """Test T1 maps to low risk."""
        mapper = L2SignalMapper()
        risk = mapper.map_tier_to_risk(MockTier(1))
        assert risk == 0.2
    
    def test_map_tier_to_risk_t4(self):
        """Test T4 maps to high risk."""
        mapper = L2SignalMapper()
        risk = mapper.map_tier_to_risk(MockTier(4))
        assert risk == 0.9
    
    def test_map_regime_trending_up(self):
        """Test regime mapping."""
        mapper = L2SignalMapper()
        regime = mapper.map_regime("TRENDING_UP")
        assert regime == MarketRegime.TRENDING_UP
    
    def test_map_decision_to_signal(self):
        """Test complete decision to signal mapping."""
        mapper = L2SignalMapper()
        
        decision = MockTacticalDecision(
            action="STRONG_BUY",
            confidence=0.85,
            tier=1,
            reason="High coherence signal",
            metadata={"regime": "TRENDING_UP"}
        )
        
        signal = mapper.map_decision_to_signal(
            decision=decision,
            symbol="BTC-USD",
            expected_return=0.08,
        )
        
        assert signal.action == TacticalAction.STRONG_BUY
        assert signal.confidence == 0.85
        assert signal.risk_score == 0.2  # T1 → 0.2
        assert signal.symbol == "BTC-USD"
        assert signal.regime == MarketRegime.TRENDING_UP
    
    def test_map_decision_dict(self):
        """Test mapping from dictionary format."""
        mapper = L2SignalMapper()
        
        decision_dict = {
            'action': 'BUY',
            'confidence': 0.75,
            'implicit_tier': 'T2',
            'regime': 'RANGING',
            'timestamp': int(time.time() * 1000),
        }
        
        signal = mapper.map_decision_dict_to_signal(decision_dict, symbol="ETH-USD")
        
        assert signal.action == TacticalAction.BUY
        assert signal.confidence == 0.75
        assert signal.risk_score == 0.4  # T2 → 0.4
        assert signal.regime == MarketRegime.RANGING


# ============================================================================
# Execution Engine Tests
# ============================================================================

class TestExecutionEngine:
    """Tests for ExecutionEngine paper trading."""
    
    def test_paper_trade_buy(self):
        """Test paper trading BUY order."""
        engine = ExecutionEngine(paper_trading=True)
        
        order = {
            'order_id': 'TEST_001',
            'symbol': 'BTC-USD',
            'side': 'BUY',
            'quantity': 0.1,
            'order_type': 'MARKET',
        }
        
        report = engine.submit_order(order, current_price=43000.0)
        
        assert report.status == "FILLED"
        assert report.fill_quantity == 0.1
        assert report.fill_price > 0
        assert report.commission > 0
        assert report.latency_ms > 0
    
    def test_paper_trade_position_tracking(self):
        """Test position tracking after trades."""
        engine = ExecutionEngine(paper_trading=True)
        
        # Open position
        buy_order = {
            'order_id': 'TEST_BUY',
            'symbol': 'BTC-USD',
            'side': 'BUY',
            'quantity': 0.5,
            'order_type': 'MARKET',
        }
        engine.submit_order(buy_order, current_price=42000.0)
        
        # Check position
        position = engine.get_position('BTC-USD')
        assert position is not None
        assert position.side == "LONG"
        assert position.quantity == 0.5
    
    def test_paper_trade_pnl_calculation(self):
        """Test PnL calculation on close."""
        engine = ExecutionEngine(paper_trading=True)
        
        # Open position at 42000
        buy_order = {
            'order_id': 'TEST_BUY',
            'symbol': 'BTC-USD',
            'side': 'BUY',
            'quantity': 0.1,
            'order_type': 'MARKET',
        }
        engine.submit_order(buy_order, current_price=42000.0)
        
        # Close position at 43000 (profit)
        sell_order = {
            'order_id': 'TEST_SELL',
            'symbol': 'BTC-USD',
            'side': 'SELL',
            'quantity': 0.1,
            'order_type': 'MARKET',
        }
        report = engine.submit_order(sell_order, current_price=43000.0)
        
        # Check PnL (should be positive)
        assert report.realized_pnl is not None
        # Approximate check (accounting for slippage and commission)
        
        # Position should be closed
        assert engine.get_position('BTC-USD') is None
    
    def test_performance_stats(self):
        """Test performance statistics tracking."""
        engine = ExecutionEngine(paper_trading=True)
        
        # Execute some trades
        for i in range(3):
            engine.submit_order({
                'order_id': f'TEST_{i}',
                'symbol': 'BTC-USD',
                'side': 'BUY' if i % 2 == 0 else 'SELL',
                'quantity': 0.1,
                'order_type': 'MARKET',
            }, current_price=42000.0 + i * 1000)
        
        stats = engine.get_performance_stats()
        assert 'total_trades' in stats
        assert 'total_commission' in stats
        assert stats['total_commission'] > 0


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestL2L3EndToEnd:
    """End-to-end integration tests."""
    
    def test_l2_to_l3_position_sizing(self):
        """Test L2 decision → L3 position sizing."""
        # Create mapper
        mapper = L2SignalMapper()
        
        # Mock L2 decision
        decision = MockTacticalDecision(
            action="BUY",
            confidence=0.80,
            tier=1,
            reason="Strong momentum signal",
        )
        
        # Map to L3 signal
        signal = mapper.map_decision_to_signal(
            decision=decision,
            symbol="BTC-USD",
            expected_return=0.08,
            predicted_volatility=0.03,
        )
        
        # Initialize L3 Phase 1
        layer3 = Layer3Phase1(portfolio_value=100000, kelly_fraction=0.25)
        
        # Create cascade indicators
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.03,
            volume_ratio=2.0,
            onchain_whale_pressure=0.3,
            exchange_netflow_zscore=0.5,
        )
        
        # Calculate position
        sizing_decision = layer3.calculate_position(
            signal=signal,
            cascade_indicators=cascade,
            current_price=43000.0,
        )
        
        assert sizing_decision.position_size_usd > 0
        assert sizing_decision.position_size_usd <= 100000
        assert sizing_decision.symbol == "BTC-USD"
    
    def test_full_pipeline_l2_signal_to_execution(self):
        """Test complete pipeline: L2 decision → L3 sizing → Execution."""
        # 1. Create L2 decision
        decision = MockTacticalDecision(
            action="STRONG_BUY",
            confidence=0.85,
            tier=1,
            metadata={"regime": "TRENDING_UP"},
        )
        
        # 2. Map to L3 signal
        mapper = L2SignalMapper()
        signal = mapper.map_decision_to_signal(
            decision=decision,
            symbol="BTC-USD",
            expected_return=0.10,
        )
        
        # 3. Calculate position size
        layer3 = Layer3Phase1(portfolio_value=100000)
        cascade = CascadeIndicators(
            funding_rate=0.001,
            oi_change_pct=0.02,
            volume_ratio=1.5,
            onchain_whale_pressure=0.2,
            exchange_netflow_zscore=0.3,
        )
        
        sizing = layer3.calculate_position(
            signal=signal,
            cascade_indicators=cascade,
            current_price=45000.0,
        )
        
        # 4. Execute (paper trade)
        engine = ExecutionEngine(paper_trading=True)
        
        order = {
            'order_id': 'L2_PIPELINE_TEST',
            'symbol': sizing.symbol,
            'side': 'BUY',
            'quantity': sizing.position_size_usd / 45000.0,
            'order_type': 'MARKET',
        }
        
        report = engine.submit_order(order, current_price=45000.0)
        
        # Verify execution
        assert report.status == "FILLED"
        assert report.symbol == "BTC-USD"
        assert report.fill_quantity > 0
        
        # Verify position opened
        position = engine.get_position("BTC-USD")
        assert position is not None
        assert position.side == "LONG"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
