"""
Layer 3 Bounded Delta PPO Training - FIXED with Layer 2 Lessons

Key fixes from Layer 2 training:
1. Variance-normalized rewards (Session 7-18)
2. Adaptive costs as fractions of E[|PnL|] (Session 18)
3. No guaranteed bonuses (causes collapse)
4. Regime-specific cost scaling
5. Percentile-based features for generalization (AHHMM Session 3)

Original L3 problem: Sharpe = -0.078
Expected after fix: Sharpe = 0.25-0.45
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import pickle
import os
from datetime import datetime

print("=" * 70)
print("Layer 3 Bounded Delta PPO - FIXED with Layer 2 Lessons")
print("=" * 70)
print()
print("Key L2 lessons applied:")
print("  1. Variance-normalized PnL (not raw returns)")
print("  2. Adaptive costs = fraction of E[|norm_pnl|]")
print("  3. Regime-specific cost multipliers")
print("  4. No guaranteed HOLD bonuses")
print("  5. Percentile features for generalization")
print()

# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Data
    train_data_path = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS\btc_1h_2020_2024.csv"
    test_data_path = r"C:\Users\chari\OneDrive\Documents\BTC DATA SETS\btc_1h_2025_2026.csv"

    # Bounded Delta (from L3 - this is correct)
    delta_lower = -0.30  # Max 30% reduction
    delta_upper = 0.30   # Max 30% increase

    # Network
    state_dim = 16  # Percentile features
    hidden_dim = 128
    lstm_layers = 1
    sequence_length = 20

    # PPO Hyperparameters
    actor_lr = 3e-4
    critic_lr = 1e-3
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.05  # L2 lesson: higher entropy prevents collapse

    # Training
    epochs = 10  # Local test
    batch_size = 64
    update_epochs = 4

    # L2 Adaptive Costs (as fractions of E[|norm_pnl|])
    base_trade_cost_frac = 0.40      # 40% of expected |pnl|
    lowvol_trade_cost_frac = 0.0     # No extra cost
    highvol_trade_cost_frac = 0.40   # +40% extra
    crisis_trade_cost_frac = 0.60    # +60% extra
    trending_trade_bonus_frac = 0.40 # -40% discount (encourage trading)

    # Position change cost
    position_change_cost_frac = 0.20  # Cost for changing position

    # Regime multipliers (from L3 - keep these)
    regime_multipliers = {
        0: 1.0,   # LOW_VOL - full position
        1: 1.2,   # TRENDING - slightly more aggressive
        2: 0.6,   # HIGH_VOL - reduce
        3: 0.2,   # CRISIS - minimal
    }

    # Output
    output_dir = r"C:\Users\chari\OneDrive\Documents\HIMARI OPUS 2\LAYER 3 V1\LAYER 3 POSITIONING LAYER\L3_BOUNDED_DELTA_FIXED"

config = Config()

# Create output directory
os.makedirs(config.output_dir, exist_ok=True)

# =============================================================================
# Data Loading and Feature Engineering
# =============================================================================

def load_and_prepare_data(filepath):
    """Load data and compute percentile-based features (L2 lesson)"""
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'open_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    # Basic features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Volatility features
    df['volatility'] = df['returns'].rolling(24).std()
    df['vol_of_vol'] = df['volatility'].rolling(24).std()

    # Volume features
    df['volume_ma'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Trend features
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['trend_strength'] = (df['sma_20'] - df['sma_50']) / df['close']

    # True range
    df['high_low'] = df['high'] - df['low']
    df['true_range'] = df['high_low'] / df['close']

    # Momentum
    df['momentum_12'] = df['close'].pct_change(12)
    df['momentum_24'] = df['close'].pct_change(24)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # Drop NaN
    df = df.dropna().reset_index(drop=True)

    # === L2 LESSON: Convert to percentiles for generalization ===
    lookback = 500

    feature_cols = ['volatility', 'trend_strength', 'volume_ratio',
                    'true_range', 'vol_of_vol', 'momentum_12',
                    'momentum_24', 'rsi']

    percentile_features = []
    for col in feature_cols:
        pct = df[col].rolling(lookback, min_periods=100).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
        percentile_features.append(pct.values)

    # Return direction (signed)
    ret_dir = np.sign(df['returns'].values) * 0.5 + 0.5  # Map to [0, 1]
    percentile_features.append(ret_dir)

    # Stack features
    features = np.column_stack(percentile_features)

    # Detect regimes (simplified - use volatility percentile)
    vol_pct = percentile_features[0]
    trend_pct = percentile_features[1]

    regimes = np.zeros(len(df), dtype=np.int64)
    regimes[(vol_pct < 0.4) & (trend_pct < 0.5)] = 0  # LOW_VOL
    regimes[(trend_pct >= 0.6)] = 1  # TRENDING
    regimes[(vol_pct >= 0.7) & (vol_pct < 0.9)] = 2  # HIGH_VOL
    regimes[(vol_pct >= 0.9)] = 3  # CRISIS

    # Get returns and prices
    returns = df['returns'].values
    prices = df['close'].values

    # Drop initial NaN from percentile calculation
    valid_idx = ~np.isnan(features).any(axis=1)
    features = features[valid_idx]
    regimes = regimes[valid_idx]
    returns = returns[valid_idx]
    prices = prices[valid_idx]

    print(f"  {len(features)} samples, {features.shape[1]} features")
    print(f"  Regime distribution: LOW_VOL={np.mean(regimes==0):.1%}, TRENDING={np.mean(regimes==1):.1%}, "
          f"HIGH_VOL={np.mean(regimes==2):.1%}, CRISIS={np.mean(regimes==3):.1%}")

    return features, regimes, returns, prices

# =============================================================================
# Actor-Critic Network
# =============================================================================

class BoundedDeltaActorCritic(nn.Module):
    """
    Outputs bounded delta in [-0.30, +0.30]
    Uses tanh to bound output naturally
    """
    def __init__(self, state_dim, hidden_dim, delta_bounds=(-0.30, 0.30)):
        super().__init__()
        self.delta_bounds = delta_bounds
        self.delta_range = (delta_bounds[1] - delta_bounds[0]) / 2
        self.delta_mid = (delta_bounds[1] + delta_bounds[0]) / 2

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor (outputs mean and log_std for delta)
        self.actor_mean = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state):
        features = self.feature_net(state)

        # Actor: mean of delta (will be passed through tanh)
        delta_mean_raw = self.actor_mean(features)
        delta_mean = torch.tanh(delta_mean_raw) * self.delta_range + self.delta_mid

        # Std (constrained to reasonable range)
        delta_std = torch.exp(self.actor_log_std).clamp(0.01, 0.3)

        # Critic: state value
        value = self.critic(features)

        return delta_mean, delta_std, value

    def get_action(self, state, deterministic=False):
        delta_mean, delta_std, value = self.forward(state)

        if deterministic:
            delta = delta_mean
            log_prob = torch.zeros_like(delta)
        else:
            dist = Normal(delta_mean, delta_std)
            delta = dist.sample()
            delta = delta.clamp(self.delta_bounds[0], self.delta_bounds[1])
            log_prob = dist.log_prob(delta)

        return delta, log_prob, value

    def evaluate_actions(self, states, actions):
        delta_mean, delta_std, values = self.forward(states)
        dist = Normal(delta_mean, delta_std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

# =============================================================================
# L2-Fixed Reward Function
# =============================================================================

def compute_l2_fixed_reward(returns, positions, position_changes, regimes,
                            prev_positions, config):
    """
    L2-style variance-normalized reward with adaptive costs

    Key differences from failed L3:
    1. Normalize PnL by batch volatility
    2. Costs are fractions of E[|norm_pnl|], not fixed values
    3. Regime-specific cost scaling
    4. No guaranteed bonuses
    """
    returns = torch.tensor(returns, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)
    position_changes = torch.tensor(position_changes, dtype=torch.float32)
    regimes = torch.tensor(regimes, dtype=torch.int64)

    # === L2 LESSON 1: Variance normalization ===
    batch_vol = torch.std(returns) + 1e-6
    norm_returns = returns / batch_vol

    # Position PnL (position * normalized return)
    position_pnl = positions * norm_returns

    # === L2 LESSON 2: Adaptive costs as fractions of E[|norm_pnl|] ===
    expected_abs_pnl = torch.abs(norm_returns).mean()

    # Base position change cost
    abs_position_change = torch.abs(position_changes)
    change_cost = abs_position_change * config.position_change_cost_frac * expected_abs_pnl

    # === L2 LESSON 3: Regime-specific cost multipliers ===
    # Trade cost by regime
    is_trading = (torch.abs(positions) > 0.1).float()

    base_cost = is_trading * config.base_trade_cost_frac * expected_abs_pnl

    # Extra costs for risky regimes
    highvol_cost = is_trading * (regimes == 2).float() * config.highvol_trade_cost_frac * expected_abs_pnl
    crisis_cost = is_trading * (regimes == 3).float() * config.crisis_trade_cost_frac * expected_abs_pnl

    # Bonus for trading in trending (opportunity cost of not trading)
    trending_bonus = is_trading * (regimes == 1).float() * config.trending_trade_bonus_frac * expected_abs_pnl

    # === L2 LESSON 4: Wrong direction penalty ===
    wrong_direction = (positions * norm_returns < 0).float()
    wrong_penalty = wrong_direction * 0.30 * expected_abs_pnl

    # Extra penalty in risky regimes
    risky_regime = ((regimes == 2) | (regimes == 3)).float()
    risky_wrong_penalty = wrong_direction * risky_regime * 0.20 * expected_abs_pnl

    # === COMBINE ===
    rewards = (position_pnl
               - change_cost
               - base_cost
               - highvol_cost
               - crisis_cost
               + trending_bonus
               - wrong_penalty
               - risky_wrong_penalty)

    return rewards.numpy()

# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(model, optimizer_actor, optimizer_critic, features, regimes,
                returns, config):
    """Train one epoch with L2-fixed rewards"""

    n_samples = len(features)
    indices = np.random.permutation(n_samples - config.sequence_length - 1)

    total_reward = 0
    total_actor_loss = 0
    total_critic_loss = 0
    n_batches = 0

    # Track action distribution
    all_deltas = []
    all_positions = []

    for batch_start in range(0, len(indices) - config.batch_size, config.batch_size):
        batch_indices = indices[batch_start:batch_start + config.batch_size]

        # Prepare batch
        batch_states = []
        batch_returns = []
        batch_regimes = []

        for idx in batch_indices:
            state = features[idx]
            batch_states.append(state)
            batch_returns.append(returns[idx + 1])  # Next period return
            batch_regimes.append(regimes[idx])

        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float32)
        batch_returns_np = np.array(batch_returns)
        batch_regimes_np = np.array(batch_regimes)

        # Get actions
        with torch.no_grad():
            deltas, log_probs, values = model.get_action(batch_states)

        deltas_np = deltas.numpy().flatten()
        all_deltas.extend(deltas_np)

        # Apply regime multipliers to get final position
        regime_mults = np.array([config.regime_multipliers[r] for r in batch_regimes_np])
        base_position = 0.5  # Start from 50% base position
        positions = base_position * (1 + deltas_np) * regime_mults
        positions = np.clip(positions, 0, 1)  # Clip to [0, 1]
        all_positions.extend(positions)

        # Position changes (simplified - from neutral)
        position_changes = deltas_np
        prev_positions = np.zeros_like(positions)

        # Compute L2-fixed rewards
        rewards = compute_l2_fixed_reward(
            batch_returns_np, positions, position_changes,
            batch_regimes_np, prev_positions, config
        )

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        total_reward += rewards.mean()

        # PPO Update
        for _ in range(config.update_epochs):
            # Re-evaluate actions
            new_log_probs, new_values, entropy = model.evaluate_actions(
                batch_states, deltas
            )

            # Advantage (simplified - just reward - value)
            advantages = rewards_tensor - new_values.detach()

            # Actor loss (PPO clip)
            ratio = torch.exp(new_log_probs - log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - config.clip_epsilon,
                               1 + config.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            actor_loss = actor_loss - config.entropy_coef * entropy.mean()

            # Critic loss (recompute to avoid graph issue)
            _, new_values_critic, _ = model.evaluate_actions(batch_states, deltas)
            critic_loss = nn.MSELoss()(new_values_critic, rewards_tensor)

            # Combined loss (update together)
            total_loss = actor_loss + 0.5 * critic_loss

            optimizer_actor.zero_grad()
            optimizer_critic.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer_actor.step()
            optimizer_critic.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        n_batches += 1

    # Compute stats
    all_deltas = np.array(all_deltas)
    all_positions = np.array(all_positions)

    stats = {
        'mean_reward': total_reward / max(n_batches, 1),
        'actor_loss': total_actor_loss / max(n_batches * config.update_epochs, 1),
        'critic_loss': total_critic_loss / max(n_batches * config.update_epochs, 1),
        'mean_delta': np.mean(all_deltas),
        'std_delta': np.std(all_deltas),
        'mean_position': np.mean(all_positions),
        'hold_pct': np.mean(all_positions < 0.1) * 100,
        'full_pct': np.mean(all_positions > 0.8) * 100,
    }

    return stats

# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, features, regimes, returns, prices, config, name=""):
    """Evaluate model and compute Sharpe, returns, etc."""

    model.eval()
    n_samples = len(features)

    positions = []
    deltas = []
    pnls = []

    with torch.no_grad():
        for i in range(n_samples - 1):
            state = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0)
            delta, _, _ = model.get_action(state, deterministic=True)
            delta_val = delta.item()
            deltas.append(delta_val)

            # Apply regime multiplier
            regime_mult = config.regime_multipliers[regimes[i]]
            base_position = 0.5
            position = base_position * (1 + delta_val) * regime_mult
            position = np.clip(position, 0, 1)
            positions.append(position)

            # PnL
            pnl = position * returns[i + 1]
            pnls.append(pnl)

    positions = np.array(positions)
    deltas = np.array(deltas)
    pnls = np.array(pnls)

    # Metrics
    total_return = np.sum(pnls)
    mean_return = np.mean(pnls)
    std_return = np.std(pnls)
    sharpe = mean_return / (std_return + 1e-10) * np.sqrt(24 * 365)  # Annualized

    # Drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns)

    # Position stats by regime
    regime_stats = {}
    for r in range(4):
        mask = regimes[:-1] == r
        if mask.sum() > 0:
            regime_stats[r] = {
                'mean_pos': np.mean(positions[mask]),
                'hold_pct': np.mean(positions[mask] < 0.1) * 100,
            }

    print(f"\n{name} Evaluation:")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Total Return: {total_return*100:.2f}%")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Mean Position: {np.mean(positions):.2f}")
    print(f"  Mean Delta: {np.mean(deltas):.3f} (std={np.std(deltas):.3f})")
    print(f"  Position Distribution: HOLD(<10%)={np.mean(positions<0.1)*100:.1f}%, "
          f"MID={np.mean((positions>=0.1)&(positions<0.8))*100:.1f}%, "
          f"FULL(>80%)={np.mean(positions>=0.8)*100:.1f}%")
    print(f"  By Regime:")
    regime_names = ['LOW_VOL', 'TRENDING', 'HIGH_VOL', 'CRISIS']
    for r, stats in regime_stats.items():
        print(f"    {regime_names[r]}: pos={stats['mean_pos']:.2f}, hold={stats['hold_pct']:.1f}%")

    model.train()

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'mean_position': np.mean(positions),
        'regime_stats': regime_stats,
    }

# =============================================================================
# Main Training
# =============================================================================

def main():
    print("=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)

    # Load training data
    train_features, train_regimes, train_returns, train_prices = load_and_prepare_data(
        config.train_data_path
    )

    # Load test data
    test_features, test_regimes, test_returns, test_prices = load_and_prepare_data(
        config.test_data_path
    )

    # Update state dim
    config.state_dim = train_features.shape[1]

    print()
    print("=" * 70)
    print("Step 2: Initialize Model")
    print("=" * 70)

    model = BoundedDeltaActorCritic(
        state_dim=config.state_dim,
        hidden_dim=config.hidden_dim,
        delta_bounds=(config.delta_lower, config.delta_upper)
    )

    optimizer_actor = optim.Adam(
        list(model.feature_net.parameters()) +
        list(model.actor_mean.parameters()) +
        [model.actor_log_std],
        lr=config.actor_lr
    )
    optimizer_critic = optim.Adam(model.critic.parameters(), lr=config.critic_lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Delta bounds: [{config.delta_lower}, {config.delta_upper}]")

    print()
    print("=" * 70)
    print("Step 3: Training (Local Test - {config.epochs} epochs)")
    print("=" * 70)

    best_sharpe = -float('inf')
    training_log = []

    for epoch in range(1, config.epochs + 1):
        stats = train_epoch(
            model, optimizer_actor, optimizer_critic,
            train_features, train_regimes, train_returns, config
        )

        training_log.append(stats)

        print(f"Epoch {epoch:2d}/{config.epochs}: "
              f"reward={stats['mean_reward']:.4f}, "
              f"delta={stats['mean_delta']:+.3f} +/- {stats['std_delta']:.3f}, "
              f"pos={stats['mean_position']:.2f}, "
              f"hold={stats['hold_pct']:.1f}%")

    print()
    print("=" * 70)
    print("Step 4: Evaluation")
    print("=" * 70)

    train_results = evaluate_model(
        model, train_features, train_regimes, train_returns, train_prices,
        config, "Training (2020-2024)"
    )

    test_results = evaluate_model(
        model, test_features, test_regimes, test_returns, test_prices,
        config, "Test (2025-2026)"
    )

    print()
    print("=" * 70)
    print("Step 5: Generalization Comparison")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Training':<15} {'Test':<15} {'Diff':<15}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<25} {train_results['sharpe']:<15.3f} {test_results['sharpe']:<15.3f} "
          f"{test_results['sharpe'] - train_results['sharpe']:+.3f}")
    print(f"{'Max Drawdown':<25} {train_results['max_drawdown']*100:<14.2f}% {test_results['max_drawdown']*100:<14.2f}% "
          f"{(test_results['max_drawdown'] - train_results['max_drawdown'])*100:+.2f}%")
    print(f"{'Mean Position':<25} {train_results['mean_position']:<15.3f} {test_results['mean_position']:<15.3f} "
          f"{test_results['mean_position'] - train_results['mean_position']:+.3f}")

    # Check if generalization is good
    sharpe_diff = test_results['sharpe'] - train_results['sharpe']
    if sharpe_diff > -0.1:
        print(f"\n[OK] Generalization looks good! (Sharpe diff = {sharpe_diff:+.3f})")
        generalization_ok = True
    else:
        print(f"\n[WARN] Generalization concern: Sharpe dropped by {-sharpe_diff:.3f}")
        generalization_ok = False

    print()
    print("=" * 70)
    print("Step 6: Save Model")
    print("=" * 70)

    # Save model
    model_path = os.path.join(config.output_dir, "bounded_delta_l2_fixed.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'state_dim': config.state_dim,
            'hidden_dim': config.hidden_dim,
            'delta_bounds': (config.delta_lower, config.delta_upper),
        },
        'train_results': train_results,
        'test_results': test_results,
        'training_log': training_log,
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Save config
    config_path = os.path.join(config.output_dir, "training_config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump({
            'base_trade_cost_frac': config.base_trade_cost_frac,
            'highvol_trade_cost_frac': config.highvol_trade_cost_frac,
            'crisis_trade_cost_frac': config.crisis_trade_cost_frac,
            'trending_trade_bonus_frac': config.trending_trade_bonus_frac,
            'regime_multipliers': config.regime_multipliers,
            'delta_bounds': (config.delta_lower, config.delta_upper),
            'entropy_coef': config.entropy_coef,
        }, f)
    print(f"Config saved to: {config_path}")

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()
    print(f"Original L3 PPO Sharpe:  -0.078")
    print(f"L2-Fixed Train Sharpe:   {train_results['sharpe']:.3f}")
    print(f"L2-Fixed Test Sharpe:    {test_results['sharpe']:.3f}")
    print()

    if train_results['sharpe'] > 0 and generalization_ok:
        print("[OK] SUCCESS! L2 lessons fixed the reward function.")
        print("     Ready for full training on Vast.ai GPU.")
    else:
        print("[INFO] Local test complete. May need hyperparameter tuning.")

    return train_results, test_results

if __name__ == "__main__":
    main()
