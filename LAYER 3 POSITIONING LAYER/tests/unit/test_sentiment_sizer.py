"""
Unit tests for Sentiment-Aware Sizer
"""

import pytest
import sys
sys.path.insert(0, 'src')

from engines.sentiment_sizer import SentimentAwareSizer


class TestSentimentAwareSizer:
    """Test suite for Sentiment-Aware Position Sizer."""
    
    def test_initialization(self):
        """Test sizer initialization."""
        sizer = SentimentAwareSizer(sentiment_weight=0.15)
        
        assert sizer.sentiment_weight == 0.15
        assert sizer.neutral_low == 0.4
        assert sizer.neutral_high == 0.6
        assert sizer.enabled == True
    
    def test_disabled_returns_unchanged(self):
        """Test that disabled sizer returns unchanged position."""
        sizer = SentimentAwareSizer(enabled=False)
        
        adjusted, multiplier, diagnostics = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.9
        )
        
        assert adjusted == 10000
        assert multiplier == 1.0
        assert diagnostics['status'] == 'disabled'
    
    def test_missing_sentiment_returns_unchanged(self):
        """Test that missing sentiment returns unchanged position."""
        sizer = SentimentAwareSizer()
        
        adjusted, multiplier, diagnostics = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=None
        )
        
        assert adjusted == 10000
        assert multiplier == 1.0
        assert diagnostics['status'] == 'no_sentiment'
        assert sizer.missing_sentiment_count == 1
    
    def test_bullish_sentiment_boost(self):
        """Test bullish sentiment boosts position."""
        sizer = SentimentAwareSizer(sentiment_weight=0.15)
        
        adjusted, multiplier, diagnostics = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.9  # Very bullish
        )
        
        assert adjusted > 10000
        assert multiplier > 1.0
        assert diagnostics['adjustment_type'] == 'boost'
        assert sizer.positive_adjustments == 1
    
    def test_bearish_sentiment_reduction(self):
        """Test bearish sentiment reduces position."""
        sizer = SentimentAwareSizer(sentiment_weight=0.15)
        
        adjusted, multiplier, diagnostics = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.1  # Very bearish
        )
        
        assert adjusted < 10000
        assert multiplier < 1.0
        assert diagnostics['adjustment_type'] == 'reduction'
        assert sizer.negative_adjustments == 1
    
    def test_neutral_sentiment_no_change(self):
        """Test neutral sentiment doesn't change position."""
        sizer = SentimentAwareSizer()
        
        adjusted, multiplier, diagnostics = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.5  # Neutral
        )
        
        assert adjusted == 10000
        assert multiplier == 1.0
        assert diagnostics['adjustment_type'] == 'neutral'
        assert sizer.neutral_skips == 1
    
    def test_confidence_scaling(self):
        """Test that sentiment confidence scales adjustment."""
        sizer = SentimentAwareSizer(sentiment_weight=0.15)
        
        # Full confidence
        adjusted_full, mult_full, _ = sizer.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.9,
            sentiment_confidence=1.0
        )
        
        # Half confidence
        sizer2 = SentimentAwareSizer(sentiment_weight=0.15)
        adjusted_half, mult_half, _ = sizer2.adjust_for_sentiment(
            base_position_usd=10000,
            sentiment_score=0.9,
            sentiment_confidence=0.5
        )
        
        # Full confidence should have larger adjustment
        assert (mult_full - 1.0) > (mult_half - 1.0)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        sizer = SentimentAwareSizer()
        
        sizer.adjust_for_sentiment(10000, 0.9)  # Bullish
        sizer.adjust_for_sentiment(10000, 0.1)  # Bearish
        sizer.adjust_for_sentiment(10000, 0.5)  # Neutral
        
        stats = sizer.get_statistics()
        
        assert stats['total_adjustments'] == 3
        assert stats['positive_adjustments'] == 1
        assert stats['negative_adjustments'] == 1
        assert stats['neutral_skips'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
