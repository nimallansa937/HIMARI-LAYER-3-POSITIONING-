"""
HIMARI Layer 3 - Market Data Module
====================================

Live market data feeds for order execution.
"""

from .price_feed import (
    PriceFeed,
    get_price_feed,
    get_current_price,
)

__all__ = [
    'PriceFeed',
    'get_price_feed',
    'get_current_price',
]
