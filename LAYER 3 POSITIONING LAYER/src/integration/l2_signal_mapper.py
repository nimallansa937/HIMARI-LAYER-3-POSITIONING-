"""
HIMARI OPUS V2 - Layer 2 Signal Mapper
=======================================

Maps Layer 2 TacticalDecision to Layer 3 TacticalSignal format.

Signal Mapping:
- TradeAction (L2) → TacticalAction (L3)
- Tier (L2) → risk_score (L3)
- RegimeLabel (L2) → MarketRegime (L3)

Version: 3.1 Enhanced
"""

from typing import Optional, Dict, Any
import logging
import time

# Handle both module and script imports
try:
    from core.layer3_types import TacticalSignal, TacticalAction, MarketRegime
except ImportError:
    from ..core.layer3_types import TacticalSignal, TacticalAction, MarketRegime

logger = logging.getLogger(__name__)


class L2SignalMapper:
    """Maps Layer 2 TacticalDecision to Layer 3 TacticalSignal format."""
    
    # Tier to risk_score mapping
    # T1 = auto-execute (low risk), T4 = emergency (high risk)
    TIER_TO_RISK_SCORE = {
        1: 0.2,   # T1: High confidence, low risk
        2: 0.4,   # T2: Medium confidence
        3: 0.6,   # T3: Low confidence, higher risk
        4: 0.9,   # T4: Emergency, very high risk
    }
    
    # TradeAction to TacticalAction mapping
    TRADE_ACTION_MAP = {
        'STRONG_BUY': TacticalAction.STRONG_BUY,
        'BUY': TacticalAction.BUY,
        'HOLD': TacticalAction.HOLD,
        'SELL': TacticalAction.SELL,
        'STRONG_SELL': TacticalAction.STRONG_SELL,
    }
    
    # RegimeLabel to MarketRegime mapping
    REGIME_MAP = {
        'TRENDING_UP': MarketRegime.TRENDING_UP,
        'TRENDING_DOWN': MarketRegime.TRENDING_DOWN,
        'RANGING': MarketRegime.RANGING,
        'HIGH_VOLATILITY': MarketRegime.HIGH_VOLATILITY,
        'CRISIS': MarketRegime.CRISIS,
        'CRISIS_FLIGHT': MarketRegime.CRISIS,  # Legacy mapping
    }
    
    def __init__(self, default_strategy_id: str = "tactical_l2"):
        """
        Initialize signal mapper.
        
        Args:
            default_strategy_id: Default strategy ID if not provided in decision
        """
        self.default_strategy_id = default_strategy_id
    
    def map_trade_action(self, action) -> TacticalAction:
        """
        Map Layer 2 TradeAction to Layer 3 TacticalAction.
        
        Args:
            action: TradeAction enum or string
            
        Returns:
            TacticalAction enum
        """
        # Handle both enum and string
        action_name = action.name if hasattr(action, 'name') else str(action).upper()
        return self.TRADE_ACTION_MAP.get(action_name, TacticalAction.HOLD)
    
    def map_tier_to_risk(self, tier) -> float:
        """
        Map Layer 2 Tier to Layer 3 risk_score.
        
        Args:
            tier: Tier enum or int
            
        Returns:
            risk_score float [0, 1]
        """
        # Handle both enum and int
        tier_value = tier.value if hasattr(tier, 'value') else int(tier)
        return self.TIER_TO_RISK_SCORE.get(tier_value, 0.5)
    
    def map_regime(self, regime_label) -> MarketRegime:
        """
        Map Layer 2 RegimeLabel to Layer 3 MarketRegime.
        
        Args:
            regime_label: RegimeLabel enum or string
            
        Returns:
            MarketRegime enum
        """
        # Handle both enum and string
        label = regime_label.value if hasattr(regime_label, 'value') else str(regime_label).upper()
        return self.REGIME_MAP.get(label, MarketRegime.RANGING)
    
    def map_decision_to_signal(
        self,
        decision,
        symbol: str = "BTC-USD",
        regime_label = None,
        strategy_id: Optional[str] = None,
        expected_return: Optional[float] = None,
        predicted_volatility: Optional[float] = None,
        sentiment_score: Optional[float] = None,
    ) -> TacticalSignal:
        """
        Map Layer 2 TacticalDecision to Layer 3 TacticalSignal.
        
        Args:
            decision: TacticalDecision from Layer 2
            symbol: Trading symbol (e.g., "BTC-USD")
            regime_label: Optional RegimeLabel if not embedded in decision
            strategy_id: Override strategy ID
            expected_return: Expected return for position sizing
            predicted_volatility: Predicted volatility for sizing
            sentiment_score: Optional sentiment score [0, 1]
            
        Returns:
            TacticalSignal for Layer 3 processing
        """
        try:
            # Map action
            action = self.map_trade_action(decision.action)
            
            # Map tier to risk score
            risk_score = self.map_tier_to_risk(decision.tier)
            
            # Map regime - try from risk_context first, then parameter
            regime = MarketRegime.RANGING
            if hasattr(decision, 'metadata') and decision.metadata:
                regime_str = decision.metadata.get('regime', None)
                if regime_str:
                    regime = self.map_regime(regime_str)
            elif regime_label:
                regime = self.map_regime(regime_label)
            
            # Convert timestamp to nanoseconds
            timestamp = decision.timestamp if hasattr(decision, 'timestamp') else time.time()
            timestamp_ns = int(timestamp * 1e9) if timestamp < 1e12 else int(timestamp * 1e6)
            
            # Build signal
            signal = TacticalSignal(
                strategy_id=strategy_id or self.default_strategy_id,
                symbol=symbol,
                action=action,
                confidence=decision.confidence,
                risk_score=risk_score,
                regime=regime,
                timestamp_ns=timestamp_ns,
                expected_return=expected_return,
                predicted_volatility=predicted_volatility,
                sentiment_score=sentiment_score,
            )
            
            logger.debug(
                f"Mapped L2 decision to L3 signal: "
                f"action={action.value}, confidence={decision.confidence:.2f}, "
                f"risk_score={risk_score:.2f}, regime={regime.value}"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error mapping L2 decision to L3 signal: {e}")
            # Return safe default signal
            return TacticalSignal(
                strategy_id=strategy_id or self.default_strategy_id,
                symbol=symbol,
                action=TacticalAction.HOLD,
                confidence=0.0,
                risk_score=0.9,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
            )
    
    def map_decision_dict_to_signal(
        self,
        decision_dict: Dict[str, Any],
        symbol: str = "BTC-USD",
        strategy_id: Optional[str] = None,
    ) -> TacticalSignal:
        """
        Map Layer 2 decision dictionary to Layer 3 TacticalSignal.
        
        Useful when reading from Redis or JSON serialization.
        
        Args:
            decision_dict: Dictionary with decision fields
            symbol: Trading symbol
            strategy_id: Override strategy ID
            
        Returns:
            TacticalSignal for Layer 3 processing
        """
        try:
            # Extract fields from dictionary
            action_str = decision_dict.get('action', 'HOLD')
            action = self.TRADE_ACTION_MAP.get(action_str.upper(), TacticalAction.HOLD)
            
            # Map tier string to risk score
            tier_str = decision_dict.get('implicit_tier', decision_dict.get('tier', 'T2'))
            tier_num = int(tier_str.replace('T', '')) if isinstance(tier_str, str) else tier_str
            risk_score = self.TIER_TO_RISK_SCORE.get(tier_num, 0.5)
            
            # Map regime
            regime_str = decision_dict.get('regime', 'RANGING')
            regime = self.REGIME_MAP.get(regime_str.upper(), MarketRegime.RANGING)
            
            # Timestamp
            timestamp_ms = decision_dict.get('timestamp', int(time.time() * 1000))
            timestamp_ns = timestamp_ms * 1_000_000  # ms to ns
            
            return TacticalSignal(
                strategy_id=strategy_id or self.default_strategy_id,
                symbol=symbol,
                action=action,
                confidence=decision_dict.get('confidence', 0.5),
                risk_score=risk_score,
                regime=regime,
                timestamp_ns=timestamp_ns,
            )
            
        except Exception as e:
            logger.error(f"Error mapping decision dict to signal: {e}")
            return TacticalSignal(
                strategy_id=strategy_id or self.default_strategy_id,
                symbol=symbol,
                action=TacticalAction.HOLD,
                confidence=0.0,
                risk_score=0.9,
                regime=MarketRegime.RANGING,
                timestamp_ns=time.time_ns(),
            )
