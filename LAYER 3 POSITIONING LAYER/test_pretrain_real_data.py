"""
HIMARI Layer 3: Pretrain Model Real Data Validation
===================================================

Tests pre-trained LSTM-PPO ensemble models on real BTC hourly data using 
the EXACT same methodology as test_real_data_ccxt.py.

Usage:
    python test_pretrain_real_data.py
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import ccxt
except ImportError:
    print("❌ CCXT not installed. Run: pip install ccxt")
    sys.exit(1)

import torch
import torch.nn as nn


# =============================================================================
# CONSTANTS (From test_hybrid_strategy.py V2.1)
# =============================================================================

HOURLY_ANNUALIZATION = np.sqrt(252 * 24)  # ~77.76
COMMISSION_RATE = 0.001  # 0.1%


# =============================================================================
# Download Real Data from Binance via CCXT
# =============================================================================

def download_btc_hourly(days: int = 730) -> np.ndarray:
    """Download BTC/USDT hourly OHLCV data from Binance."""
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
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    prices = df['close'].values
    
    print(f"✅ Downloaded {len(prices)} hourly candles")
    print(f"   📅 {df['datetime'].iloc[0].strftime('%Y-%m-%d')} to {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
    
    return prices


# =============================================================================
# Market Regime Detection (Same as test_real_data_ccxt.py)
# =============================================================================

class MarketRegime(Enum):
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"
    CASCADE = "CASCADE"


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
        
        if returns_20 > 0.03 and market_regime == MarketRegime.NORMAL:
            return "bull", market_regime
        elif returns_20 < -0.03:
            return "bear", market_regime
        else:
            return "neutral", market_regime


# =============================================================================
# LSTM-PPO Network (Same as test_real_data_ccxt.py)
# =============================================================================

@dataclass
class TrainingConfig:
    state_dim: int = 16
    hidden_dim: int = 128  # Match pretrain model
    lstm_layers: int = 2   # Match pretrain model
    learning_rate: float = 3e-4


class LSTMPPONetworkV2(nn.Module):
    """LSTM + PPO network matching pretrain checkpoint architecture."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Input projection (matches checkpoint: input_proj.0.weight [128, 16])
        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # LSTM (matches checkpoint: 2 layers, hidden=128)
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # Actor (matches checkpoint: actor.0 [64, 128], actor.3 [2, 64])
        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)  # Outputs mean and log_std
        )
        
        # Critic (matches checkpoint: critic.0 [64, 128], critic.3 [1, 64])
        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.hidden = None
    
    def reset_hidden(self, batch_size: int = 1):
        device = next(self.parameters()).device
        self.hidden = (
            torch.zeros(self.config.lstm_layers, batch_size, self.config.hidden_dim).to(device),
            torch.zeros(self.config.lstm_layers, batch_size, self.config.hidden_dim).to(device)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hidden is None:
            self.reset_hidden(x.size(0))
        
        # Input projection
        x = self.input_proj(x)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())
        
        features = lstm_out[:, -1, :]
        
        # Actor outputs mean and log_std
        actor_out = self.actor(features)
        mean = torch.tanh(actor_out[:, 0:1])  # Tanh to bound action
        
        value = self.critic(features)
        
        return mean, value


def predict_rl_raw(model: nn.Module, state: np.ndarray) -> float:
    """Get raw RL output from model."""
    model.eval()
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, _ = model(state_tensor)
        return float(action.item())


# =============================================================================
# Feature Engineering (Same as test_real_data_ccxt.py)
# =============================================================================

class HybridFeatureEngineer:
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
        initial_capital: float,
        regime_name: str
    ) -> np.ndarray:
        state = np.zeros(16, dtype=np.float32)
        
        if len(self.returns_history) < 20:
            return state
        
        recent_returns = self.returns_history[-20:]
        volatility = np.std(recent_returns) * HOURLY_ANNUALIZATION
        momentum_1h = self.returns_history[-1] if self.returns_history else 0
        momentum_4h = np.sum(self.returns_history[-4:]) if len(self.returns_history) >= 4 else 0
        total_pnl_pct = (current_capital - initial_capital) / initial_capital
        
        # Feature encoding matching LSTM-PPO expectations
        state[0] = 0
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


# =============================================================================
# Backtest (Same as test_real_data_ccxt.py)
# =============================================================================

def run_hybrid_backtest(
    models: List[nn.Module],
    prices: np.ndarray,
    initial_capital: float = 100000.0
) -> Dict:
    """Run hybrid strategy backtest on real data."""
    
    print("\n🚀 Running Pretrain Ensemble Backtest...")
    print("=" * 60)
    
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
    
    # Reset LSTM hidden states
    for model in models:
        model.reset_hidden(1)
    
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
    print("HIMARI Layer 3: PRETRAIN ENSEMBLE REAL DATA VALIDATION")
    print("=" * 70)
    print()
    print("⚠️ This test uses REAL BTC/USDT hourly data from Binance")
    print("   Using EXACT same methodology as test_real_data_ccxt.py")
    print()
    
    # Load pretrain ensemble models
    models_dir = "../pretrain_models/balanced_ensemble"
    model_folders = [f"model_{i}_seed_{seed}" for i, seed in enumerate([42, 123, 456, 789, 1024], 1)]
    
    print("📦 Loading pretrain ensemble models...")
    models = []
    
    for folder in model_folders:
        model_path = os.path.join(models_dir, folder, "best_model.pt")
        if os.path.exists(model_path):
            model = LSTMPPONetworkV2(TrainingConfig())
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Handle different checkpoint formats
            if 'network_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['network_state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'policy_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['policy_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            models.append(model)
            print(f"   ✅ Loaded: {folder}")
        else:
            print(f"   ⚠️ Not found: {model_path}")
    
    if not models:
        print("❌ No models found! Check the model path.")
        return
    
    print(f"\n✅ Loaded {len(models)} ensemble models")
    
    # Download real data
    prices = download_btc_hourly(days=730)
    
    # Run backtest
    results = run_hybrid_backtest(models, prices)
    
    # Print results
    print("\n" + "=" * 70)
    print("PRETRAIN ENSEMBLE RESULTS")
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
