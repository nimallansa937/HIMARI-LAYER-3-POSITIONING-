"""
HIMARI OPUS 2 - Layer 3 Input Validator
========================================

Validates all inputs from Layer 2 before processing.
Implements strict validation per CLAUDE Guide Part III.

Version: 1.0
"""

import time
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class Layer3InputValidator:
    """Validates all inputs from Layer 2 before processing."""
    
    REQUIRED_FIELDS = [
        'signal_id', 'timestamp', 'symbol', 'direction', 
        'confidence', 'strategy_id'
    ]
    
    REQUIRED_MARKET_CONTEXT = [
        'realized_vol_5d', 'realized_vol_20d', 'funding_rate',
        'open_interest', 'open_interest_delta_1h', 'volume_24h',
        'volume_spike_ratio', 'bid_ask_spread', 'btc_correlation'
    ]
    
    REQUIRED_PORTFOLIO_STATE = [
        'portfolio_equity', 'cash_available', 'daily_pnl',
        'daily_pnl_pct', 'drawdown_from_hwm'
    ]
    
    # Staleness threshold in milliseconds
    STALENESS_THRESHOLD_MS = 5000
    
    def validate(self, signal: dict) -> Tuple[bool, List[str]]:
        """
        Validate signal from Layer 2.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in signal:
                errors.append(f"Missing required field: {field}")
        
        # Check market context
        market_ctx = signal.get('market_context', {})
        for field in self.REQUIRED_MARKET_CONTEXT:
            if field not in market_ctx:
                errors.append(f"Missing market context: {field}")
        
        # Check portfolio state
        portfolio = signal.get('portfolio_state', {})
        for field in self.REQUIRED_PORTFOLIO_STATE:
            if field not in portfolio:
                errors.append(f"Missing portfolio state: {field}")
        
        # Validate ranges
        if 'confidence' in signal:
            if not 0.0 <= signal['confidence'] <= 1.0:
                errors.append(f"Confidence out of range: {signal['confidence']}")
        
        if 'direction' in signal:
            if signal['direction'] not in ['LONG', 'SHORT', 'FLAT']:
                errors.append(f"Invalid direction: {signal['direction']}")
        
        # Validate data freshness
        if 'timestamp' in signal:
            age_ms = time.time() * 1000 - signal['timestamp']
            if age_ms > self.STALENESS_THRESHOLD_MS:
                errors.append(f"Signal too stale: {age_ms:.0f}ms old")
        
        # Log validation result
        if errors:
            logger.warning(f"Signal validation failed: {errors}")
        
        return len(errors) == 0, errors
    
    def validate_with_defaults(self, signal: dict) -> Tuple[dict, List[str]]:
        """
        Validate signal and apply defaults for missing optional fields.
        
        Returns:
            Tuple of (signal_with_defaults, list_of_warnings)
        """
        warnings = []
        result = signal.copy()
        
        # Apply defaults for optional fields
        if 'market_context' not in result:
            result['market_context'] = {}
        
        market_ctx = result['market_context']
        
        # Default ATR if missing
        if 'atr_14' not in market_ctx:
            if 'realized_vol_5d' in market_ctx:
                # Estimate ATR from volatility
                market_ctx['atr_14'] = market_ctx['realized_vol_5d'] * 0.02
                warnings.append("ATR estimated from volatility")
        
        # Default current_price if missing
        if 'current_price' not in market_ctx:
            market_ctx['current_price'] = 0.0
            warnings.append("Current price missing, defaulted to 0")
        
        # Default volume spike ratio
        if 'volume_spike_ratio' not in market_ctx:
            market_ctx['volume_spike_ratio'] = 1.0
            warnings.append("Volume spike ratio missing, defaulted to 1.0")
        
        # Default regime if missing
        if 'regime' not in result:
            result['regime'] = 'NORMAL'
            warnings.append("Regime missing, defaulted to NORMAL")
        
        if 'regime_confidence' not in result:
            result['regime_confidence'] = 0.5
            warnings.append("Regime confidence missing, defaulted to 0.5")
        
        return result, warnings


class Layer3RejectionResponse:
    """Response sent to Layer 2 when signal is rejected."""
    
    def __init__(self, signal_id: str, errors: List[str]):
        self.signal_id = signal_id
        self.status = "REJECTED"
        self.errors = errors
        self.timestamp = int(time.time() * 1000)
        self.action = "NO_TRADE"
        self.reason = "INPUT_VALIDATION_FAILED"
    
    def to_dict(self) -> dict:
        """Serialize rejection response."""
        return {
            'signal_id': self.signal_id,
            'status': self.status,
            'errors': self.errors,
            'timestamp': self.timestamp,
            'action': self.action,
            'reason': self.reason
        }
