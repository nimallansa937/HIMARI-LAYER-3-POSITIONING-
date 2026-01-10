"""
HIMARI OPUS V2 - Sentiment-Aware Position Sizer
================================================

Integrates Layer 2 sentiment signals for position sizing adjustment.

Features:
- Sentiment boost/reduction based on confidence
- Configurable weight (default 15%)
- Graceful degradation when sentiment unavailable
- Comprehensive diagnostics

Version: 3.1 Enhanced
"""

from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SentimentAwareSizer:
    """
    Sentiment-aware position sizing adjustment.
    
    Applies sentiment adjustment based on Layer 2 sentiment signals:
    - Positive sentiment (>0.6): Boost position up to 15%
    - Negative sentiment (<0.4): Reduce position up to 15%
    - Neutral sentiment (0.4-0.6): No adjustment
    """
    
    def __init__(
        self,
        sentiment_weight: float = 0.15,
        neutral_low: float = 0.4,
        neutral_high: float = 0.6,
        enabled: bool = True
    ):
        """
        Initialize sentiment sizer.
        
        Args:
            sentiment_weight: Maximum adjustment weight (default 15%)
            neutral_low: Lower bound of neutral zone
            neutral_high: Upper bound of neutral zone
            enabled: Whether sentiment adjustment is enabled
        """
        self.sentiment_weight = sentiment_weight
        self.neutral_low = neutral_low
        self.neutral_high = neutral_high
        self.enabled = enabled
        
        # Tracking
        self.total_adjustments = 0
        self.positive_adjustments = 0
        self.negative_adjustments = 0
        self.neutral_skips = 0
        self.missing_sentiment_count = 0
    
    def adjust_for_sentiment(
        self,
        base_position_usd: float,
        sentiment_score: Optional[float] = None,
        sentiment_confidence: Optional[float] = None
    ) -> Tuple[float, float, Dict]:
        """
        Adjust position size based on sentiment.
        
        Args:
            base_position_usd: Base position size in USD
            sentiment_score: Sentiment score [0, 1] where 0=bearish, 1=bullish
            sentiment_confidence: Confidence in sentiment signal [0, 1]
            
        Returns:
            Tuple of (adjusted_position_usd, multiplier, diagnostics)
        """
        diagnostics = {
            'enabled': self.enabled,
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence,
            'base_position_usd': base_position_usd
        }
        
        # Check if enabled
        if not self.enabled:
            diagnostics['status'] = 'disabled'
            return base_position_usd, 1.0, diagnostics
        
        # Check if sentiment available
        if sentiment_score is None:
            self.missing_sentiment_count += 1
            diagnostics['status'] = 'no_sentiment'
            return base_position_usd, 1.0, diagnostics
        
        # Scale adjustment by confidence if available
        confidence_scale = sentiment_confidence if sentiment_confidence is not None else 1.0
        
        # Calculate adjustment
        if sentiment_score > self.neutral_high:
            # Bullish: boost position
            boost_strength = (sentiment_score - self.neutral_high) / (1.0 - self.neutral_high)
            multiplier = 1.0 + (self.sentiment_weight * boost_strength * confidence_scale)
            self.positive_adjustments += 1
            diagnostics['adjustment_type'] = 'boost'
        
        elif sentiment_score < self.neutral_low:
            # Bearish: reduce position
            reduction_strength = (self.neutral_low - sentiment_score) / self.neutral_low
            multiplier = 1.0 - (self.sentiment_weight * reduction_strength * confidence_scale)
            self.negative_adjustments += 1
            diagnostics['adjustment_type'] = 'reduction'
        
        else:
            # Neutral: no adjustment
            multiplier = 1.0
            self.neutral_skips += 1
            diagnostics['adjustment_type'] = 'neutral'
        
        self.total_adjustments += 1
        
        adjusted_position_usd = base_position_usd * multiplier
        
        diagnostics['multiplier'] = multiplier
        diagnostics['adjusted_position_usd'] = adjusted_position_usd
        diagnostics['status'] = 'applied'
        
        if multiplier != 1.0:
            change_pct = (multiplier - 1.0) * 100
            logger.debug(
                f"Sentiment adjustment: {change_pct:+.1f}% "
                f"(score={sentiment_score:.2f}, conf={confidence_scale:.2f})"
            )
        
        return adjusted_position_usd, multiplier, diagnostics
    
    def get_statistics(self) -> Dict:
        """Get sentiment adjustment statistics."""
        return {
            'enabled': self.enabled,
            'sentiment_weight': self.sentiment_weight,
            'total_adjustments': self.total_adjustments,
            'positive_adjustments': self.positive_adjustments,
            'negative_adjustments': self.negative_adjustments,
            'neutral_skips': self.neutral_skips,
            'missing_sentiment_count': self.missing_sentiment_count,
            'adjustment_rate': (
                (self.positive_adjustments + self.negative_adjustments) / 
                max(1, self.total_adjustments)
            )
        }
    
    def set_enabled(self, enabled: bool):
        """Enable or disable sentiment adjustment."""
        self.enabled = enabled
        logger.info(f"Sentiment adjustment {'enabled' if enabled else 'disabled'}")
    
    def update_weight(self, weight: float):
        """Update sentiment weight."""
        if 0.0 <= weight <= 0.5:
            self.sentiment_weight = weight
            logger.info(f"Sentiment weight updated to {weight:.1%}")
        else:
            logger.error(f"Invalid sentiment weight: {weight}. Must be in [0, 0.5]")
