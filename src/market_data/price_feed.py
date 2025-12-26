"""
HIMARI Layer 3 - Live Price Feed
==================================

Fetches real-time market prices for order execution.
Uses Binance public API (no authentication required).

Features:
- REST API for current prices
- Symbol mapping (BTC-USD -> BTCUSDT)
- Caching to avoid rate limits
- Fallback to simulated prices if API fails

Version: 1.0
"""

import requests
import time
import logging
from typing import Optional, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class PriceFeed:
    """
    Live price feed for crypto assets.

    Uses Binance public API for real-time prices.
    No authentication required.
    """

    # Symbol mapping: HIMARI format -> Binance format
    SYMBOL_MAP = {
        'BTC-USD': 'BTCUSDT',
        'BTC-USDT': 'BTCUSDT',
        'BTCUSDT': 'BTCUSDT',
        'ETH-USD': 'ETHUSDT',
        'ETH-USDT': 'ETHUSDT',
        'ETHUSDT': 'ETHUSDT',
        'SOL-USD': 'SOLUSDT',
        'SOL-USDT': 'SOLUSDT',
        'SOLUSDT': 'SOLUSDT',
    }

    # Fallback prices if API fails (for testing)
    FALLBACK_PRICES = {
        'BTCUSDT': 87000.0,
        'ETHUSDT': 2500.0,
        'SOLUSDT': 150.0,
    }

    def __init__(
        self,
        cache_seconds: int = 5,
        timeout_seconds: int = 3,
        use_fallback: bool = True
    ):
        """
        Initialize price feed.

        Args:
            cache_seconds: How long to cache prices (default: 5s)
            timeout_seconds: API request timeout (default: 3s)
            use_fallback: Use fallback prices if API fails (default: True)
        """
        self.cache_seconds = cache_seconds
        self.timeout_seconds = timeout_seconds
        self.use_fallback = use_fallback

        # Price cache: {symbol: {'price': float, 'timestamp': float}}
        self._cache: Dict[str, Dict] = {}

        # API endpoints
        self.base_url = "https://api.binance.com/api/v3"

        # Statistics
        self.api_calls = 0
        self.cache_hits = 0
        self.api_errors = 0

        logger.info(
            f"PriceFeed initialized (cache={cache_seconds}s, "
            f"timeout={timeout_seconds}s, fallback={'on' if use_fallback else 'off'})"
        )

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Binance format.

        Args:
            symbol: Symbol in HIMARI format (e.g., "BTC-USD")

        Returns:
            Symbol in Binance format (e.g., "BTCUSDT")
        """
        symbol_upper = symbol.upper()
        return self.SYMBOL_MAP.get(symbol_upper, symbol_upper)

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD", "BTCUSDT")

        Returns:
            Current price as float, or None if unavailable
        """
        # Normalize symbol
        binance_symbol = self._normalize_symbol(symbol)

        # Check cache
        current_time = time.time()
        if binance_symbol in self._cache:
            cached_data = self._cache[binance_symbol]
            age = current_time - cached_data['timestamp']

            if age < self.cache_seconds:
                self.cache_hits += 1
                logger.debug(
                    f"Cache hit: {binance_symbol} = ${cached_data['price']:,.2f} "
                    f"(age: {age:.1f}s)"
                )
                return cached_data['price']

        # Fetch from API
        price = self._fetch_from_api(binance_symbol)

        if price is not None:
            # Update cache
            self._cache[binance_symbol] = {
                'price': price,
                'timestamp': current_time
            }
            return price

        # Fallback if API failed
        if self.use_fallback:
            fallback_price = self.FALLBACK_PRICES.get(binance_symbol)
            if fallback_price:
                logger.warning(
                    f"Using fallback price for {binance_symbol}: ${fallback_price:,.2f}"
                )
                return fallback_price

        logger.error(f"Failed to get price for {binance_symbol}")
        return None

    def _fetch_from_api(self, symbol: str) -> Optional[float]:
        """
        Fetch price from Binance API.

        Args:
            symbol: Symbol in Binance format (e.g., "BTCUSDT")

        Returns:
            Price as float, or None if failed
        """
        url = f"{self.base_url}/ticker/price"
        params = {'symbol': symbol}

        try:
            self.api_calls += 1
            response = requests.get(url, params=params, timeout=self.timeout_seconds)
            response.raise_for_status()

            data = response.json()
            price = float(data['price'])

            logger.debug(f"API fetch: {symbol} = ${price:,.2f}")
            return price

        except requests.exceptions.Timeout:
            self.api_errors += 1
            logger.error(f"API timeout for {symbol}")
            return None

        except requests.exceptions.RequestException as e:
            self.api_errors += 1
            logger.error(f"API error for {symbol}: {e}")
            return None

        except (KeyError, ValueError) as e:
            self.api_errors += 1
            logger.error(f"Invalid API response for {symbol}: {e}")
            return None

    def get_multiple_prices(self, symbols: list) -> Dict[str, float]:
        """
        Get prices for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary mapping symbols to prices
        """
        prices = {}
        for symbol in symbols:
            price = self.get_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    def clear_cache(self):
        """Clear price cache."""
        self._cache.clear()
        logger.info("Price cache cleared")

    def get_statistics(self) -> Dict:
        """Get price feed statistics."""
        total_requests = self.api_calls + self.cache_hits
        cache_hit_rate = (
            self.cache_hits / total_requests if total_requests > 0 else 0.0
        )

        return {
            'api_calls': self.api_calls,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': cache_hit_rate,
            'api_errors': self.api_errors,
            'cached_symbols': list(self._cache.keys()),
            'cache_size': len(self._cache),
        }


# Global price feed instance (singleton)
_global_price_feed: Optional[PriceFeed] = None


def get_price_feed(
    cache_seconds: int = 5,
    timeout_seconds: int = 3,
    use_fallback: bool = True
) -> PriceFeed:
    """
    Get global price feed instance (singleton pattern).

    Args:
        cache_seconds: Cache duration
        timeout_seconds: API timeout
        use_fallback: Use fallback prices on error

    Returns:
        PriceFeed instance
    """
    global _global_price_feed

    if _global_price_feed is None:
        _global_price_feed = PriceFeed(
            cache_seconds=cache_seconds,
            timeout_seconds=timeout_seconds,
            use_fallback=use_fallback
        )

    return _global_price_feed


def get_current_price(symbol: str) -> Optional[float]:
    """
    Convenience function to get current price.

    Args:
        symbol: Trading symbol

    Returns:
        Current price or None
    """
    feed = get_price_feed()
    return feed.get_price(symbol)


# =============================================================================
# Testing
# =============================================================================

def test_price_feed():
    """Test price feed functionality."""
    print("=" * 80)
    print("HIMARI LAYER 3 - Price Feed Test")
    print("=" * 80)
    print()

    # Initialize price feed
    print("Initializing price feed...")
    feed = PriceFeed(cache_seconds=5, timeout_seconds=3, use_fallback=True)
    print()

    # Test single price fetch
    print("Test 1: Fetch BTC price")
    print("-" * 60)
    btc_price = feed.get_price("BTC-USD")
    if btc_price:
        print(f"  BTC-USD:  ${btc_price:,.2f}")
    else:
        print(f"  BTC-USD:  FAILED")
    print()

    # Test multiple symbols
    print("Test 2: Fetch multiple symbols")
    print("-" * 60)
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    prices = feed.get_multiple_prices(symbols)
    for symbol, price in prices.items():
        print(f"  {symbol:10s} ${price:,.2f}")
    print()

    # Test cache
    print("Test 3: Cache performance")
    print("-" * 60)
    start_time = time.time()
    cached_price = feed.get_price("BTC-USD")  # Should hit cache
    elapsed_ms = (time.time() - start_time) * 1000
    print(f"  Cached BTC: ${cached_price:,.2f}")
    print(f"  Fetch Time: {elapsed_ms:.2f} ms")
    print()

    # Statistics
    print("Test 4: Statistics")
    print("-" * 60)
    stats = feed.get_statistics()
    print(f"  API Calls:      {stats['api_calls']}")
    print(f"  Cache Hits:     {stats['cache_hits']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.1%}")
    print(f"  API Errors:     {stats['api_errors']}")
    print(f"  Cached Symbols: {', '.join(stats['cached_symbols'])}")
    print()

    # Test symbol normalization
    print("Test 5: Symbol normalization")
    print("-" * 60)
    test_symbols = ["BTC-USD", "btc-usd", "BTCUSDT", "btcusdt"]
    for sym in test_symbols:
        normalized = feed._normalize_symbol(sym)
        print(f"  {sym:10s} -> {normalized}")
    print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_price_feed()
