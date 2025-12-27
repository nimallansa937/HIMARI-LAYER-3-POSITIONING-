"""
Multi-Asset Trading Environment for HIMARI Layer 3
Trains on multiple cryptocurrencies simultaneously to learn cross-asset correlations
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import requests
from datetime import datetime, timedelta


@dataclass
class MultiAssetEnvConfig:
    """Configuration for multi-asset trading environment."""
    symbols: List[str] = None  # e.g., ['BTC-USD', 'ETH-USD', 'SOL-USD']
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5
    commission_rate: float = 0.001
    slippage_bps: float = 5.0
    reward_window: int = 10
    max_steps: int = 500
    use_live_prices: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']


class MultiAssetTradingEnv:
    """
    Multi-asset trading environment.

    State space: [market features x num_assets]
    Action space: [position multipliers x num_assets] continuous [0, 2]
    """

    def __init__(self, config: MultiAssetEnvConfig):
        self.config = config
        self.num_assets = len(config.symbols)

        # State dimension: 16 features per asset + 4 correlation features
        self.state_dim = 16 * self.num_assets + 4
        self.action_dim = self.num_assets

        # Initialize price data
        self.prices = {symbol: [] for symbol in config.symbols}
        self.current_step = 0

        # Portfolio state
        self.capital = config.initial_capital
        self.positions = {symbol: 0.0 for symbol in config.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in config.symbols}

        # Performance tracking
        self.episode_returns = []
        self.recent_returns = []

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Fetch fresh price data if using live prices
        if self.config.use_live_prices:
            self._fetch_live_prices()
        else:
            self._generate_synthetic_prices()

        # Reset portfolio
        self.capital = self.config.initial_capital
        self.positions = {symbol: 0.0 for symbol in self.config.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.config.symbols}
        self.current_step = 0
        self.recent_returns = []

        return self._get_state()

    def _fetch_live_prices(self):
        """Fetch live price data from CoinGecko."""
        for symbol in self.config.symbols:
            # Convert symbol format (BTC-USD -> bitcoin)
            coin_id = self._symbol_to_coin_id(symbol)

            try:
                # Fetch last 500 hourly candles
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': '21',  # ~500 hours
                    'interval': 'hourly'
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()

                data = response.json()
                prices = [p[1] for p in data['prices']]
                self.prices[symbol] = prices[-self.config.max_steps:]

            except Exception as e:
                print(f"Failed to fetch {symbol} prices: {e}, using synthetic")
                self._generate_synthetic_prices_for_symbol(symbol)

    def _symbol_to_coin_id(self, symbol: str) -> str:
        """Convert trading symbol to CoinGecko coin ID."""
        mapping = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum',
            'SOL-USD': 'solana',
            'BNB-USD': 'binancecoin',
            'XRP-USD': 'ripple',
            'ADA-USD': 'cardano',
            'AVAX-USD': 'avalanche-2',
            'DOT-USD': 'polkadot',
        }
        return mapping.get(symbol, 'bitcoin')

    def _generate_synthetic_prices(self):
        """Generate synthetic price data for all assets."""
        for symbol in self.config.symbols:
            self._generate_synthetic_prices_for_symbol(symbol)

    def _generate_synthetic_prices_for_symbol(self, symbol: str):
        """Generate synthetic price data with realistic correlations."""
        base_price = {'BTC-USD': 87000, 'ETH-USD': 3200, 'SOL-USD': 180}.get(symbol, 1000)
        volatility = 0.02

        prices = [base_price]
        for _ in range(self.config.max_steps):
            # GBM with mean reversion
            drift = -0.0001 * (prices[-1] - base_price) / base_price
            shock = np.random.normal(drift, volatility)
            new_price = prices[-1] * (1 + shock)
            prices.append(max(new_price, base_price * 0.5))

        self.prices[symbol] = prices

    def _get_state(self) -> np.ndarray:
        """
        Get current environment state.

        Returns 16 features per asset + 4 correlation features:
        Per asset (16 features):
        - Price (normalized)
        - Returns: 1h, 4h, 24h
        - Volatility: 1h, 4h, 24h
        - RSI (14-period)
        - MACD, Signal, Histogram
        - Volume ratio
        - Current position
        - Entry price distance
        - Unrealized P&L
        - Asset allocation %

        Correlation features (4 features):
        - BTC-ETH correlation
        - BTC-SOL correlation
        - ETH-SOL correlation
        - Portfolio concentration
        """
        state_features = []

        # Per-asset features
        for symbol in self.config.symbols:
            prices = self.prices[symbol]
            current_price = prices[self.current_step] if self.current_step < len(prices) else prices[-1]

            # Normalize price
            price_norm = current_price / 100000.0

            # Returns
            returns_1h = self._calc_return(symbol, 1)
            returns_4h = self._calc_return(symbol, 4)
            returns_24h = self._calc_return(symbol, 24)

            # Volatility
            vol_1h = self._calc_volatility(symbol, 1)
            vol_4h = self._calc_volatility(symbol, 4)
            vol_24h = self._calc_volatility(symbol, 24)

            # Technical indicators
            rsi = self._calc_rsi(symbol, 14)
            macd, signal, histogram = self._calc_macd(symbol)

            # Volume proxy (using price volatility)
            volume_ratio = vol_1h / (vol_24h + 1e-8)

            # Position features
            position_size = self.positions[symbol] / self.capital
            entry_dist = (current_price - self.entry_prices[symbol]) / (current_price + 1e-8) if self.entry_prices[symbol] > 0 else 0
            unrealized_pnl = (current_price - self.entry_prices[symbol]) * self.positions[symbol] if self.entry_prices[symbol] > 0 else 0
            unrealized_pnl_norm = unrealized_pnl / self.capital

            # Portfolio allocation
            total_position_value = sum(
                self.prices[s][min(self.current_step, len(self.prices[s])-1)] * self.positions[s]
                for s in self.config.symbols
            )
            allocation_pct = (current_price * self.positions[symbol]) / (total_position_value + 1e-8) if total_position_value > 0 else 0

            # Combine features
            asset_state = [
                price_norm, returns_1h, returns_4h, returns_24h,
                vol_1h, vol_4h, vol_24h, rsi,
                macd, signal, histogram, volume_ratio,
                position_size, entry_dist, unrealized_pnl_norm, allocation_pct
            ]
            state_features.extend(asset_state)

        # Cross-asset correlation features
        if self.num_assets >= 2:
            corr_btc_eth = self._calc_correlation('BTC-USD', 'ETH-USD', 24) if 'BTC-USD' in self.config.symbols and 'ETH-USD' in self.config.symbols else 0
            corr_btc_sol = self._calc_correlation('BTC-USD', 'SOL-USD', 24) if 'BTC-USD' in self.config.symbols and 'SOL-USD' in self.config.symbols else 0
            corr_eth_sol = self._calc_correlation('ETH-USD', 'SOL-USD', 24) if 'ETH-USD' in self.config.symbols and 'SOL-USD' in self.config.symbols else 0
        else:
            corr_btc_eth = corr_btc_sol = corr_eth_sol = 0

        # Portfolio concentration (Herfindahl index)
        total_value = sum(
            self.prices[s][min(self.current_step, len(self.prices[s])-1)] * self.positions[s]
            for s in self.config.symbols
        )
        if total_value > 0:
            concentrations = [
                (self.prices[s][min(self.current_step, len(self.prices[s])-1)] * self.positions[s] / total_value) ** 2
                for s in self.config.symbols
            ]
            concentration = sum(concentrations)
        else:
            concentration = 0

        correlation_features = [corr_btc_eth, corr_btc_sol, corr_eth_sol, concentration]
        state_features.extend(correlation_features)

        return np.array(state_features, dtype=np.float32)

    def _calc_return(self, symbol: str, periods: int) -> float:
        """Calculate return over N periods."""
        prices = self.prices[symbol]
        if self.current_step < periods:
            return 0.0
        return (prices[self.current_step] - prices[self.current_step - periods]) / prices[self.current_step - periods]

    def _calc_volatility(self, symbol: str, periods: int) -> float:
        """Calculate volatility over N periods."""
        prices = self.prices[symbol]
        if self.current_step < periods:
            return 0.0
        recent_prices = prices[max(0, self.current_step - periods):self.current_step + 1]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        return np.std(returns) if len(returns) > 0 else 0.0

    def _calc_rsi(self, symbol: str, periods: int = 14) -> float:
        """Calculate RSI indicator."""
        prices = self.prices[symbol]
        if self.current_step < periods:
            return 0.5
        recent_prices = prices[max(0, self.current_step - periods):self.current_step + 1]
        deltas = np.diff(recent_prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return rsi

    def _calc_macd(self, symbol: str) -> Tuple[float, float, float]:
        """Calculate MACD indicators."""
        # Simplified MACD
        fast_period = 12
        slow_period = 26
        signal_period = 9

        prices = self.prices[symbol]
        if self.current_step < slow_period:
            return 0.0, 0.0, 0.0

        recent_prices = np.array(prices[max(0, self.current_step - slow_period):self.current_step + 1])

        if len(recent_prices) < slow_period:
            return 0.0, 0.0, 0.0

        ema_fast = recent_prices[-fast_period:].mean()
        ema_slow = recent_prices[-slow_period:].mean()
        macd = (ema_fast - ema_slow) / (ema_slow + 1e-8)

        # Simplified signal line
        signal = macd * 0.8  # Approximation
        histogram = macd - signal

        return macd, signal, histogram

    def _calc_correlation(self, symbol1: str, symbol2: str, periods: int) -> float:
        """Calculate price correlation between two assets."""
        if self.current_step < periods:
            return 0.0

        prices1 = np.array(self.prices[symbol1][max(0, self.current_step - periods):self.current_step + 1])
        prices2 = np.array(self.prices[symbol2][max(0, self.current_step - periods):self.current_step + 1])

        if len(prices1) < 2 or len(prices2) < 2:
            return 0.0

        returns1 = np.diff(prices1) / prices1[:-1]
        returns2 = np.diff(prices2) / prices2[:-1]

        if len(returns1) == 0 or len(returns2) == 0:
            return 0.0

        corr = np.corrcoef(returns1, returns2)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one environment step.

        Args:
            actions: Array of position multipliers [0, 2] for each asset

        Returns:
            next_state, reward, done, info
        """
        # Execute trades for each asset
        total_pnl = 0.0

        for i, symbol in enumerate(self.config.symbols):
            multiplier = np.clip(actions[i], 0.0, 2.0)
            current_price = self.prices[symbol][min(self.current_step, len(self.prices[symbol])-1)]

            # Calculate target position
            base_position = self.capital * self.config.max_position_pct
            target_position = base_position * multiplier / current_price

            # Execute trade
            position_change = target_position - self.positions[symbol]

            if abs(position_change) > 1e-6:
                # Apply commission and slippage
                trade_cost = abs(position_change * current_price) * self.config.commission_rate
                slippage = abs(position_change * current_price) * (self.config.slippage_bps / 10000.0)

                # Update position
                self.capital -= trade_cost + slippage
                self.positions[symbol] = target_position
                self.entry_prices[symbol] = current_price

        # Move to next step
        self.current_step += 1

        # Calculate step P&L
        for symbol in self.config.symbols:
            if self.positions[symbol] != 0 and self.entry_prices[symbol] > 0:
                current_price = self.prices[symbol][min(self.current_step, len(self.prices[symbol])-1)]
                pnl = (current_price - self.entry_prices[symbol]) * self.positions[symbol]
                total_pnl += pnl

        # Calculate reward
        step_return = total_pnl / self.capital if self.capital > 0 else 0
        self.recent_returns.append(step_return)

        # Sharpe-based reward
        if len(self.recent_returns) >= self.config.reward_window:
            recent_window = self.recent_returns[-self.config.reward_window:]
            mean_return = np.mean(recent_window)
            std_return = np.std(recent_window)
            sharpe = mean_return / (std_return + 1e-8)
            reward = sharpe
        else:
            reward = step_return

        # Check if done
        done = (
            self.current_step >= self.config.max_steps - 1 or
            self.capital < self.config.initial_capital * 0.5
        )

        # Info
        info = {
            'capital': self.capital,
            'pnl': total_pnl,
            'positions': self.positions.copy(),
            'step': self.current_step,
        }

        next_state = self._get_state()

        return next_state, reward, done, info
