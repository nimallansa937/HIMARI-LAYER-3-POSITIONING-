"""
HIMARI Layer 3: Real Data Validation
=====================================

Downloads real BTC/USDT hourly data from Binance using CCXT
and runs the Hybrid Strategy V2.1 backtest for validation.

Usage:
    python test_real_data_ccxt.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Fix import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ccxt
except ImportError:
    print("❌ CCXT not installed. Run: pip install ccxt")
    sys.exit(1)

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum

# =============================================================================
# CONSTANTS (From test_hybrid_strategy.py V2.1)
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)  # ~77.76
COMMISSION_RATE = 0.001  # 0.1%


# =============================================================================
# Download Real Data from Binance via CCXT
# =============================================================================

def download_btc_hourly(days: int = 730) -> np.ndarray:
    """
    Download BTC/USDT hourly OHLCV data from Binance.
    
    Args:
        days: Number of days to download (default 730 = 2 years)
        
    Returns:
        numpy array of close prices
    """
    print(f"\n📥 Downloading {days} days of BTC/USDT hourly data from Binance...")
    
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Calculate time range
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    
    all_ohlcv = []
    limit = 1000  # Binance max per request
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=limit)
            
            if not ohlcv:
                break
                
            all_ohlcv.extend(ohlcv)
            
            # Update since to last timestamp + 1 hour
            since = ohlcv[-1][0] + 3600000
            
            # Progress
            current_date = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            pct = len(all_ohlcv) / (days * 24) * 100
            print(f"   ⏳ {pct:.0f}% | Downloaded to: {current_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Check if we've reached current time
            if ohlcv[-1][0] >= int(datetime.now().timestamp() * 1000) - 3600000:
                break
                
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"   ⚠️ Error: {e}, retrying...")
            time.sleep(1)
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    print(f"   ✅ Downloaded {len(df):,} hourly candles")
    print(f"   📅 Range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Save to CSV for future use
    csv_path = os.path.join(os.path.dirname(__file__), 'btc_hourly_real.csv')
    df.to_csv(csv_path, index=False)
    print(f"   💾 Saved to: {csv_path}")
    
    return df['close'].values


# =============================================================================
# Copy of V2.1 Strategy Components
# =============================================================================

@dataclass
class TrainingConfig:
    state_dim: int = 16
    hidden_dim: int = 128
    lstm_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    checkpoint_interval: int = 50000
    patience: int = 3
    initial_capital: float = 100000.0
    max_position_pct: float = 0.5
    commission_rate: float = 0.001

import __main__
__main__.TrainingConfig = TrainingConfig


class MarketRegime(Enum):
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"
    CASCADE = "CASCADE"


class LSTMPPONetworkV2(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.lstm = nn.LSTM(
            config.hidden_dim, config.hidden_dim, config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 2)
        )
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.actor(out[:, -1, :])[:, 0:1]


def load_model(path: str) -> nn.Module:
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', TrainingConfig())
    model = LSTMPPONetworkV2(config)
    model.load_state_dict(checkpoint['network_state_dict'], strict=True)
    model.eval()
    return model


def predict_rl_raw(model: nn.Module, state: np.ndarray) -> float:
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = model(state_t)
        return torch.tanh(action).item()


@dataclass
class BoundedDeltaConfig:
    max_rl_delta: float = 0.30
    target_volatility: float = 0.15
    vol_lookback: int = 20
    leverage_caps: Dict[str, float] = field(default_factory=lambda: {
        "NORMAL": 2.0,
        "HIGH_VOL": 1.5,
        "CRISIS": 1.0,
        "CASCADE": 0.0
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
        returns_50 = np.sum(self.returns_history[-self.lookback:])
        
        current_vol = self.volatility_history[-1] if self.volatility_history else 0.3
        
        if current_vol > 1.2:
            return "neutral", MarketRegime.CASCADE
        elif current_vol > 0.8:
            return "neutral", MarketRegime.CRISIS
        elif current_vol > 0.6:
            market_regime = MarketRegime.HIGH_VOL
        else:
            market_regime = MarketRegime.NORMAL
        
        momentum_score = returns_20 * 0.6 + returns_50 * 0.4
        
        if momentum_score > 0.03:
            return "bull", market_regime
        elif momentum_score < -0.03:
            return "bear", market_regime
        else:
            return "neutral", market_regime


class HybridFeatureEngineer:
    def __init__(self, initial_capital: float = 100000.0):
        self.price_history: List[float] = []
        self.returns_history: List[float] = []
        self.peak_capital = initial_capital
        self.max_history = 500
    
    def update(self, price: float):
        if self.price_history:
            ret = (price - self.price_history[-1]) / self.price_history[-1]
            self.returns_history.append(ret)
            if len(self.returns_history) > self.max_history:
                self.returns_history.pop(0)
        
        self.price_history.append(price)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
    
    def get_state(
        self,
        current_position_usd: float,
        current_capital: float,
        initial_capital: float,
        regime_name: str
    ) -> np.ndarray:
        state = np.zeros(16, dtype=np.float32)
        
        if len(self.returns_history) < 20:
            return state
        
        self.peak_capital = max(self.peak_capital, current_capital)
        total_pnl_pct = (current_capital - initial_capital) / initial_capital
        
        momentum_1h = np.sum(self.returns_history[-60:]) if len(self.returns_history) >= 60 else np.sum(self.returns_history)
        momentum_4h = np.sum(self.returns_history[-240:]) if len(self.returns_history) >= 240 else momentum_1h
        
        volatility = np.std(self.returns_history[-20:]) * HOURLY_ANNUALIZATION
        
        state[0] = 0.75
        state[1] = 0
        state[2] = 0
        state[3] = 1
        state[4] = 1
        state[5] = 0
        state[6] = 0
        state[7] = 0
        
        max_position = initial_capital * 0.5
        state[8] = min(abs(current_position_usd) / max_position, 1.0)
        state[9] = 1 if current_position_usd > 0 else (-1 if current_position_usd < 0 else 0)
        state[10] = np.clip(total_pnl_pct * 2, -1, 1)
        state[11] = np.clip(momentum_1h * 10, -1, 1)
        state[12] = np.clip(momentum_4h * 5, -1, 1)
        state[13] = np.clip(volatility / 0.1, 0, 1)
        state[14] = 0.5
        
        cascade_risk_map = {"bull": 0.1, "bear": 0.4, "neutral": 0.2}
        state[15] = cascade_risk_map.get(regime_name, 0.2)
        
        return state


def kelly_momentum_position(
    returns: List[float],
    capital: float,
    max_position_pct: float = 0.5
) -> float:
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


def run_hybrid_backtest(
    models: List[nn.Module],
    prices: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict:
    """Run hybrid strategy backtest on real data."""
    
    regime_detector = RegimeDetector(lookback=50)
    position_sizer = BoundedDeltaPositionSizer()
    feature_eng = HybridFeatureEngineer(initial_capital=initial_capital)
    
    capital = initial_capital
    position_usd = 0.0
    prev_position_pct = 0.0
    returns_list = []
    regime_counts = {"bull": 0, "bear": 0, "neutral": 0}
    rl_disabled_count = 0
    total_commission = 0.0
    
    for t in range(1, len(prices)):
        ret = (prices[t] - prices[t-1]) / prices[t-1]
        
        if t % 1752 == 0:
            pct = t / len(prices) * 100
            regime_name, _ = regime_detector.detect()
            print(f"   ⏳ {pct:.0f}% | Capital: ${capital:,.0f} | Regime: {regime_name.upper()}")
        
        regime_detector.update(ret)
        position_sizer.update_returns(ret)
        feature_eng.update(prices[t])
        
        if t < 60:
            returns_list.append(0)
            continue
        
        regime_name, market_regime = regime_detector.detect()
        regime_counts[regime_name] += 1
        
        state = feature_eng.get_state(
            current_position_usd=position_usd,
            current_capital=capital,
            initial_capital=initial_capital,
            regime_name=regime_name
        )
        
        if regime_name == "bull":
            target_position = kelly_momentum_position(
                feature_eng.returns_history,
                capital
            )
        else:
            rl_outputs = [predict_rl_raw(model, state) for model in models]
            ensemble_output = np.mean(rl_outputs)
            
            target_position, debug = position_sizer.apply_bounded_delta(
                raw_rl_output=ensemble_output,
                capital=capital,
                regime=market_regime
            )
            
            if debug['rl_disabled']:
                rl_disabled_count += 1
        
        position_pct = target_position / capital if capital > 0 else 0
        position_change = abs(position_pct - prev_position_pct)
        commission = position_change * COMMISSION_RATE
        total_commission += commission * capital
        
        strategy_return = position_pct * ret - commission
        
        capital *= (1 + strategy_return)
        position_usd = target_position
        prev_position_pct = position_pct
        returns_list.append(strategy_return)
    
    returns_arr = np.array(returns_list)
    
    sharpe = np.mean(returns_arr) / (np.std(returns_arr) + 1e-8) * HOURLY_ANNUALIZATION
    total_return = (capital - initial_capital) / initial_capital
    
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = np.max(drawdowns)
    
    total_periods = sum(regime_counts.values())
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'final_capital': capital,
        'total_commission': total_commission,
        'regime_distribution': {
            k: v / total_periods for k, v in regime_counts.items()
        },
        'rl_disabled_pct': rl_disabled_count / total_periods if total_periods > 0 else 0,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIMARI Layer 3: REAL DATA VALIDATION")
    print("=" * 70)
    print()
    print("⚠️ This test uses REAL BTC/USDT hourly data from Binance")
    print("   Expected Sharpe: 0.8 - 1.5 (realistic)")
    print()
    
    # Load models
    print("Loading RL ensemble models...")
    models_dir = "C:/Users/chari/OneDrive/Documents/HIMARI OPUS 2/LAYER 3 V1/LAYER 3 TRAINED ESSEMBLE MODLES - Copy"
    model_files = [f"model_{i}.pt" for i in range(1, 6)]
    
    models = []
    for f in model_files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            try:
                model = load_model(path)
                models.append(model)
                print(f"   ✅ {f}")
            except Exception as e:
                print(f"   ❌ {f}: {e}")
        else:
            print(f"   ⚠️ {f} not found")
    
    if not models:
        print("\n⚠️ No models loaded, using dummy models")
        models = [LSTMPPONetworkV2(TrainingConfig()) for _ in range(5)]
    
    # Download real data
    prices = download_btc_hourly(days=730)
    
    # Run backtest
    print("\n" + "=" * 70)
    print("BACKTESTING ON REAL BTC DATA")
    print("=" * 70)
    
    results = run_hybrid_backtest(models, prices)
    
    # Buy & Hold comparison
    bh_return = (prices[-1] - prices[0]) / prices[0]
    bh_returns = np.diff(prices) / prices[:-1]
    bh_sharpe = np.mean(bh_returns) / (np.std(bh_returns) + 1e-8) * HOURLY_ANNUALIZATION
    
    cumulative_bh = np.cumsum(bh_returns)
    running_max_bh = np.maximum.accumulate(cumulative_bh)
    bh_max_dd = np.max(running_max_bh - cumulative_bh)
    
    # Results
    print("\n" + "=" * 70)
    print("REAL DATA RESULTS")
    print("=" * 70)
    
    print(f"\n{'Metric':<35} {'Hybrid':<15} {'Buy & Hold':<15}")
    print("-" * 65)
    print(f"{'Sharpe Ratio':<35} {results['sharpe']:+.4f}       {bh_sharpe:+.4f}")
    print(f"{'Total Return':<35} {results['total_return']*100:+.2f}%        {bh_return*100:+.2f}%")
    print(f"{'Max Drawdown':<35} {results['max_drawdown']*100:.2f}%         {bh_max_dd*100:.2f}%")
    print(f"{'Final Capital':<35} ${results['final_capital']:,.0f}")
    print(f"{'Total Commission':<35} ${results['total_commission']:,.0f}")
    
    print(f"\n📊 Regime Distribution:")
    for regime, pct in results['regime_distribution'].items():
        print(f"   {regime.capitalize():<10} {pct*100:>6.1f}%")
    
    print(f"\n🛡️ RL Disabled (Crisis): {results['rl_disabled_pct']*100:.1f}%")
    
    # Verdict
    print("\n" + "=" * 70)
    print("REAL DATA VERDICT")
    print("=" * 70)
    
    if results['sharpe'] > 1.0 and results['max_drawdown'] < 0.10:
        print(f"\n🎉 PRODUCTION READY!")
        print(f"   Sharpe: {results['sharpe']:+.4f} (>1.0 ✅)")
        print(f"   Max DD: {results['max_drawdown']*100:.2f}% (<10% ✅)")
    elif results['sharpe'] > 0.5:
        print(f"\n⚠️ ACCEPTABLE - Minor tuning needed")
        print(f"   Sharpe: {results['sharpe']:+.4f}")
        print(f"   Max DD: {results['max_drawdown']*100:.2f}%")
    else:
        print(f"\n❌ NEEDS WORK")
        print(f"   Sharpe: {results['sharpe']:+.4f}")
        print(f"   Max DD: {results['max_drawdown']*100:.2f}%")
    
    if results['sharpe'] > bh_sharpe:
        diff = results['sharpe'] - bh_sharpe
        print(f"\n🎯 BEATS Buy & Hold by {diff:+.4f} Sharpe points!")
    else:
        diff = bh_sharpe - results['sharpe']
        print(f"\n📉 Underperforms Buy & Hold by {diff:.4f} Sharpe points")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
