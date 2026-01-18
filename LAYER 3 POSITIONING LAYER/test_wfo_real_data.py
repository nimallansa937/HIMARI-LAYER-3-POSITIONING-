"""
HIMARI Layer 3: WFO Model Real Data Validation
===============================================

Tests WFO-trained PPO model on real BTC hourly data using the 
same methodology as test_real_data_ccxt.py (regime detection, 
bounded delta position sizing, etc.)

Usage:
    python test_wfo_real_data.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'rl'))

try:
    import ccxt
except ImportError:
    print("❌ CCXT not installed. Run: pip install ccxt")
    sys.exit(1)

import torch

# Import PPO agent
from rl.ppo_agent import PPOAgent, PPOConfig


# =============================================================================
# CONSTANTS (From test_hybrid_strategy.py V2.1)
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)  # ~77.76
COMMISSION_RATE = 0.001  # 0.1%


# =============================================================================
# Download Real Data from Binance via CCXT
# =============================================================================

def download_btc_hourly(days: int = 730) -> np.ndarray:
    """Download BTC/USDT hourly data from Binance."""
    print(f"\n📥 Downloading {days} days of BTC/USDT hourly data from Binance...")
    
    exchange = ccxt.binance({'enableRateLimit': True})
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 3600000
            
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            pct = len(all_ohlcv) / (days * 24) * 100
            print(f"   ⏳ {pct:.0f}% | Downloaded to: {current_date.strftime('%Y-%m-%d %H:%M')}")
            
            if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000) - 3600000:
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"   ⚠️ Error: {e}, retrying...")
            time.sleep(1)
            continue
    
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    prices = df['close'].values
    
    print(f"✅ Downloaded {len(prices)} hourly candles")
    print(f"   📅 {datetime.fromtimestamp(all_ohlcv[0][0]/1000).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(all_ohlcv[-1][0]/1000).strftime('%Y-%m-%d')}")
    
    return prices


# =============================================================================
# Market Regime Detection
# =============================================================================

class MarketRegime(Enum):
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"
    CASCADE = "CASCADE"


class RegimeDetector:
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.returns_history: List[float] = []
        self.volatility_history: List[float] = []
    
    def update(self, return_value: float):
        self.returns_history.append(return_value)
        if len(self.returns_history) >= 20:
            vol = np.std(self.returns_history[-20:]) * HOURLY_ANNUALIZATION
            self.volatility_history.append(vol)
        if len(self.returns_history) > self.lookback * 2:
            self.returns_history.pop(0)
        if len(self.volatility_history) > self.lookback:
            self.volatility_history.pop(0)
    
    def detect(self) -> Tuple[str, MarketRegime]:
        if len(self.returns_history) < self.lookback:
            return "neutral", MarketRegime.NORMAL
        
        returns_20 = np.sum(self.returns_history[-20:])
        current_vol = self.volatility_history[-1] if self.volatility_history else 0.3
        
        if current_vol > 1.2:
            return "neutral", MarketRegime.CASCADE
        elif current_vol > 0.8:
            return "neutral", MarketRegime.CRISIS
        elif current_vol > 0.6:
            market_regime = MarketRegime.HIGH_VOL
        else:
            market_regime = MarketRegime.NORMAL
        
        if returns_20 > 0.03 and market_regime == MarketRegime.NORMAL:
            return "bull", market_regime
        elif returns_20 < -0.03:
            return "bear", market_regime
        else:
            return "neutral", market_regime


# =============================================================================
# Bounded Delta Position Sizer
# =============================================================================

@dataclass
class BoundedDeltaConfig:
    max_rl_delta: float = 0.30
    target_volatility: float = 0.15
    vol_lookback: int = 20
    leverage_caps: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 2.0, "HIGH_VOL": 1.5, "CRISIS": 1.0, "CASCADE": 0.0
    })


class BoundedDeltaPositionSizer:
    def __init__(self, config: BoundedDeltaConfig = None):
        self.config = config or BoundedDeltaConfig()
        self.returns_history: List[float] = []
    
    def update_returns(self, return_value: float):
        self.returns_history.append(return_value)
        if len(self.returns_history) > 100:
            self.returns_history.pop(0)
    
    def calculate_volatility_target_position(self, capital: float) -> float:
        if len(self.returns_history) < self.config.vol_lookback:
            return capital * 0.25
        
        realized_vol = np.std(self.returns_history[-self.config.vol_lookback:]) * HOURLY_ANNUALIZATION
        realized_vol = max(realized_vol, 0.05)
        
        vol_scalar = self.config.target_volatility / realized_vol
        vol_scalar = np.clip(vol_scalar, 0.1, 3.0)
        
        base_position = capital * 0.02 * vol_scalar
        base_position = min(base_position, capital * 0.5)
        
        return base_position
    
    def apply_bounded_delta(
        self,
        raw_rl_output: float,
        capital: float,
        regime: MarketRegime
    ) -> Tuple[float, Dict]:
        base_position = self.calculate_volatility_target_position(capital)
        
        clipped_rl = np.clip(raw_rl_output, -1.0, 1.0)
        rl_delta = clipped_rl * self.config.max_rl_delta
        
        adjusted_position = base_position * (1.0 + rl_delta)
        
        leverage_cap = self.config.leverage_caps.get(regime.value, 1.0)
        max_position = capital * leverage_cap
        
        rl_disabled = regime in [MarketRegime.CRISIS, MarketRegime.CASCADE]
        
        if rl_disabled:
            final_position = min(base_position * 0.5, max_position)
        else:
            final_position = min(adjusted_position, max_position)
        
        final_position = max(0, final_position)
        
        return final_position, {'rl_disabled': rl_disabled}


# =============================================================================
# Feature Engineering
# =============================================================================

class FeatureEngineer:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.prices = []
        self.returns_history = []
    
    def update(self, price: float):
        self.prices.append(price)
        if len(self.prices) > 1:
            ret = (price - self.prices[-2]) / self.prices[-2]
            self.returns_history.append(ret)
        if len(self.prices) > 200:
            self.prices.pop(0)
        if len(self.returns_history) > 100:
            self.returns_history.pop(0)
    
    def get_state(
        self,
        current_position_usd: float,
        current_capital: float,
        regime_name: str
    ) -> np.ndarray:
        """Generate 16-dim state vector for PPO agent."""
        state = np.zeros(16, dtype=np.float32)
        
        if len(self.returns_history) < 20:
            return state
        
        recent_returns = self.returns_history[-20:]
        volatility = np.std(recent_returns) * HOURLY_ANNUALIZATION
        momentum_1h = self.returns_history[-1] if self.returns_history else 0
        momentum_4h = np.sum(self.returns_history[-4:]) if len(self.returns_history) >= 4 else 0
        total_pnl_pct = (current_capital - self.initial_capital) / self.initial_capital
        
        # State encoding (matching PPO agent expectations)
        state[0] = 0.5 + (momentum_1h * 10)  # Signal confidence proxy
        state[1] = 0  # Signal action
        state[2] = 0  # Signal tier
        state[3] = 1  # Signal strength
        state[4] = 1  # Market aligned
        state[5] = 0  # L2 confidence
        state[6] = 0  # L2 direction
        state[7] = 0  # L2 leverage
        
        max_position = self.initial_capital * 0.5
        state[8] = min(abs(current_position_usd) / max_position, 1.0)  # Position size norm
        state[9] = 1 if current_position_usd > 0 else (-1 if current_position_usd < 0 else 0)  # Position direction
        state[10] = np.clip(total_pnl_pct * 2, -1, 1)  # PnL normalized
        state[11] = np.clip(momentum_1h * 10, -1, 1)  # 1h momentum
        state[12] = np.clip(momentum_4h * 5, -1, 1)  # 4h momentum
        state[13] = np.clip(volatility / 0.1, 0, 1)  # Volatility
        state[14] = 0.5  # Current drawdown placeholder
        
        cascade_risk_map = {"bull": 0.1, "bear": 0.4, "neutral": 0.2}
        state[15] = cascade_risk_map.get(regime_name, 0.2)  # Cascade risk
        
        return state


# =============================================================================
# Kelly Momentum Position (for bull regime)
# =============================================================================

def kelly_momentum_position(returns: List[float], capital: float, max_position_pct: float = 0.5) -> float:
    if len(returns) < 20:
        return capital * 0.25
    
    recent_returns = returns[-20:]
    wins = [r for r in recent_returns if r > 0]
    losses = [r for r in recent_returns if r < 0]
    
    if not losses:
        kelly = 0.25
    elif not wins:
        kelly = 0.05
    else:
        win_rate = len(wins) / len(recent_returns)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss > 0:
            kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        else:
            kelly = 0.25
        kelly = np.clip(kelly, 0, 0.5)
    
    momentum = np.sum(recent_returns)
    momentum_factor = 1 + np.clip(momentum * 2, -0.3, 0.3)
    
    position = capital * kelly * momentum_factor * max_position_pct
    return min(position, capital * max_position_pct)


# =============================================================================
# Main Backtest
# =============================================================================

def run_wfo_backtest(
    agent: PPOAgent,
    prices: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict:
    """Run WFO model backtest using same methodology as test_real_data_ccxt.py"""
    
    print("\n🚀 Running WFO Model Backtest...")
    print("=" * 60)
    
    regime_detector = RegimeDetector(lookback=50)
    position_sizer = BoundedDeltaPositionSizer()
    feature_eng = FeatureEngineer(initial_capital=initial_capital)
    
    capital = initial_capital
    position_usd = 0.0
    prev_position_pct = 0.0
    returns_list = []
    regime_counts = {"bull": 0, "bear": 0, "neutral": 0}
    rl_disabled_count = 0
    total_commission = 0.0
    
    for t in range(1, len(prices)):
        ret = (prices[t] - prices[t-1]) / prices[t-1]
        
        # Progress logging
        if t % 1752 == 0:
            pct = t / len(prices) * 100
            regime_name, _ = regime_detector.detect()
            print(f"   ⏳ {pct:.0f}% | Capital: ${capital:,.0f} | Regime: {regime_name.upper()}")
        
        # Update trackers
        regime_detector.update(ret)
        position_sizer.update_returns(ret)
        feature_eng.update(prices[t])
        
        # Warmup period
        if t < 60:
            returns_list.append(0)
            continue
        
        # Detect regime
        regime_name, market_regime = regime_detector.detect()
        regime_counts[regime_name] += 1
        
        # Get state
        state = feature_eng.get_state(
            current_position_usd=position_usd,
            current_capital=capital,
            regime_name=regime_name
        )
        
        # Position sizing based on regime
        if regime_name == "bull":
            target_position = kelly_momentum_position(
                feature_eng.returns_history,
                capital
            )
        else:
            # Use PPO agent for non-bull regimes
            action, _ = agent.get_action(state, deterministic=True)
            # Convert action [0, 2] to raw RL output [-1, 1]
            raw_rl_output = (action - 1.0)
            
            target_position, debug = position_sizer.apply_bounded_delta(
                raw_rl_output=raw_rl_output,
                capital=capital,
                regime=market_regime
            )
            
            if debug['rl_disabled']:
                rl_disabled_count += 1
        
        # Commission
        position_pct = target_position / capital if capital > 0 else 0
        position_change = abs(position_pct - prev_position_pct)
        commission = position_change * COMMISSION_RATE
        total_commission += commission * capital
        
        # Strategy return
        strategy_return = position_pct * ret - commission
        
        capital *= (1 + strategy_return)
        position_usd = target_position
        prev_position_pct = position_pct
        returns_list.append(strategy_return)
    
    # Calculate metrics
    returns_arr = np.array(returns_list)
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - initial_capital) / initial_capital
    
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = np.max(drawdowns)
    
    total_periods = sum(regime_counts.values())
    
    # Buy & Hold comparison
    bh_return = (prices[-1] - prices[0]) / prices[0]
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'total_commission': total_commission,
        'buy_hold_return': bh_return,
        'alpha': total_return - bh_return,
        'regime_distribution': {k: v / total_periods for k, v in regime_counts.items()},
        'rl_disabled_pct': rl_disabled_count / total_periods if total_periods > 0 else 0,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 3: WFO MODEL REAL DATA VALIDATION")
    print("=" * 70)
    print()
    print("⚠️ This test uses REAL BTC/USDT hourly data from Binance")
    print("   Using same methodology as test_real_data_ccxt.py")
    print()
    
    # Load WFO model
    model_path = "../wfo_models/checkpoints/rl_policy_final.pt"
    if not os.path.exists(model_path):
        alt_paths = [
            "wfo_models/checkpoints/rl_policy_final.pt",
            "../wfo_models/checkpoints/rl_policy_final.pt",
            "../../wfo_models/checkpoints/rl_policy_final.pt",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                model_path = p
                break
    
    print(f"📦 Loading WFO model from: {model_path}")
    
    config = PPOConfig(state_dim=16, hidden_dim=64)
    agent = PPOAgent(config, device='cpu')
    agent.load(model_path)
    agent.eval_mode()
    print("✅ Model loaded successfully!")
    
    # Download real data
    prices = download_btc_hourly(days=730)
    
    # Run backtest
    results = run_wfo_backtest(agent, prices)
    
    # Print results
    print("\n" + "=" * 70)
    print("WFO MODEL RESULTS (Full Methodology)")
    print("=" * 70)
    print(f"  📈 Total Return:     {results['total_return']*100:+.2f}%")
    print(f"  📊 Sharpe Ratio:     {results['sharpe']:.4f}")
    print(f"  📉 Max Drawdown:     {results['max_drawdown']*100:.2f}%")
    print(f"  💰 Final Capital:    ${results['final_capital']:,.2f}")
    print(f"  💸 Total Commission: ${results['total_commission']:,.2f}")
    print()
    print(f"  📊 Buy & Hold:       {results['buy_hold_return']*100:+.2f}%")
    print(f"  ⭐ Alpha:            {results['alpha']*100:+.2f}%")
    print()
    print(f"  🎯 Regime Distribution:")
    for regime, pct in results['regime_distribution'].items():
        print(f"      {regime.upper():10s}: {pct*100:.1f}%")
    print(f"  🛡️ RL Disabled:      {results['rl_disabled_pct']*100:.1f}%")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
