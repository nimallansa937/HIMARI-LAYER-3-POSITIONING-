"""
HIMARI Layer 3: Bounded Delta Training V2
==========================================

V2 Improvements:
1. Auto-scaling rewards based on observed return magnitudes
2. Reward clipping to prevent explosions
3. Running statistics for normalization
4. Enhanced diagnostic logging
5. Robust handling of edge cases

Author: Claude
Date: 2026-01-10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
import pickle
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Optional W&B
WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class BoundedDeltaConfig:
    """Configuration for bounded delta reward shaping - V2 with auto-scaling."""

    # Delta bounds
    delta_lower: float = -0.30
    delta_upper: float = 0.30

    # Reward weights (normalized to sum ~1.0)
    w_risk_adjusted: float = 0.40
    w_delta_efficiency: float = 0.25
    w_drawdown_penalty: float = 0.15
    w_regime_compliance: float = 0.10
    w_smoothness: float = 0.05
    w_survival_bonus: float = 0.05

    # Risk parameters
    target_return: float = 0.0
    drawdown_threshold: float = 0.05
    max_drawdown_cap: float = 0.20

    # Regime-optimal deltas
    regime_optimal_delta: Dict[str, float] = None

    # V2: Auto-scaling parameters
    reward_clip_min: float = -10.0
    reward_clip_max: float = 10.0
    return_scale_window: int = 500  # Rolling window for return scaling
    adaptive_scaling: bool = True
    min_scale_value: float = 1e-6  # Prevent division by zero

    def __post_init__(self):
        if self.regime_optimal_delta is None:
            self.regime_optimal_delta = {
                "bull": 0.20,
                "bear": -0.15,
                "ranging": 0.0,
                "mixed": 0.0,
                "crisis": -0.30,
                "crash": -0.30,
                "volatility_cluster": -0.10,
                "cascade": -0.30
            }


@dataclass
class Tier1Config:
    """Configuration for Tier 1 Volatility Targeting (Deterministic Base)."""
    target_vol_annual: float = 0.15
    lookback_short: int = 5
    lookback_long: int = 20
    base_fraction: float = 0.5  # Half-Kelly (conservative)
    min_position_pct: float = 0.01
    max_position_pct: float = 0.10
    atr_period: int = 14
    atr_stop_multiplier: float = 2.0


@dataclass
class NetworkConfig:
    """Neural network hyperparameters."""
    state_dim: int = 32
    hidden_dim: int = 128
    lstm_layers: int = 2
    sequence_length: int = 20

    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    gamma: float = 0.99
    lambda_gae: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    buffer_size: int = 2048
    batch_size: int = 256
    update_epochs: int = 4

    dropout: float = 0.1


# ============================================================================
# TIER 1: VOLATILITY TARGETING (DETERMINISTIC BASE)
# ============================================================================

class VolatilityTargeter:
    """
    Tier 1: Deterministic volatility-targeted position sizing.

    This computes the BASE position that the RL delta will adjust.
    Pure arithmetic - no neural networks in the critical path.
    """

    def __init__(self, config: Tier1Config):
        self.config = config
        self._vol_buffer = deque(maxlen=config.lookback_long)

    def compute_base_position(
        self,
        portfolio_equity: float,
        realized_vol: float,
        regime: str
    ) -> Tuple[float, Dict]:
        """
        Compute base position size using volatility targeting.

        Formula: position = (target_vol / realized_vol) * base_fraction * equity

        Returns:
            base_position_pct: Position as fraction of equity [0.01, 0.10]
            diagnostics: Dict with computation details
        """
        # Update volatility buffer
        self._vol_buffer.append(realized_vol)

        # Compute blended volatility (short + long lookback)
        if len(self._vol_buffer) >= self.config.lookback_short:
            short_vol = np.mean(list(self._vol_buffer)[-self.config.lookback_short:])
            long_vol = np.mean(list(self._vol_buffer))
            blended_vol = 0.7 * short_vol + 0.3 * long_vol
        else:
            blended_vol = realized_vol

        # Prevent division by zero
        blended_vol = max(blended_vol, 0.001)

        # Core volatility targeting formula
        raw_position = (self.config.target_vol_annual / blended_vol) * self.config.base_fraction

        # Clamp to bounds
        clamped_position = np.clip(
            raw_position,
            self.config.min_position_pct,
            self.config.max_position_pct
        )

        diagnostics = {
            "realized_vol": realized_vol,
            "blended_vol": blended_vol,
            "raw_position_pct": raw_position,
            "clamped_position_pct": clamped_position,
            "vol_buffer_len": len(self._vol_buffer)
        }

        return clamped_position, diagnostics

    def reset(self):
        """Reset volatility buffer for new episode."""
        self._vol_buffer.clear()


# ============================================================================
# RUNNING STATISTICS FOR NORMALIZATION
# ============================================================================

class RunningStats:
    """Track running mean and std for normalization."""

    def __init__(self, window_size: int = 500):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 1.0

    def update(self, value: float):
        """Add new value and update stats."""
        self.values.append(value)
        if len(self.values) >= 2:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values)
            if self.std < 1e-6:
                self.std = 1.0

    def normalize(self, value: float) -> float:
        """Normalize value using running stats."""
        return (value - self.mean) / self.std

    def get_stats(self) -> Tuple[float, float]:
        """Return current mean and std."""
        return self.mean, self.std


# ============================================================================
# REWARD FUNCTION V2 - ADAPTIVE SCALING
# ============================================================================

class BoundedDeltaRewardV2:
    """
    V2 Bounded Delta Reward Function with Adaptive Scaling.

    Key improvements:
    - Auto-scales rewards based on observed return magnitudes
    - Clips rewards to prevent explosions
    - Uses running statistics for normalization
    - More robust to different data scales
    """

    def __init__(self, config: BoundedDeltaConfig):
        self.config = config
        self._returns_history: List[float] = []
        self._delta_history: List[float] = []
        self._peak_equity: float = 1.0
        self._current_equity: float = 1.0

        # V2: Running statistics for adaptive scaling
        self.return_stats = RunningStats(config.return_scale_window)
        self.reward_stats = RunningStats(config.return_scale_window)

        # Diagnostics
        self.step_count = 0
        self.extreme_reward_count = 0

    def compute_reward(
        self,
        raw_delta: float,
        base_position: float,
        realized_return: float,
        regime: str,
        volatility: float,
        prev_delta: float = 0.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward with adaptive scaling.

        Returns:
            total_reward: Clipped and normalized reward
            components: Dict of individual components + diagnostics
        """

        self.step_count += 1

        # 1. Bound the delta
        bounded_delta = float(np.clip(
            np.tanh(raw_delta) * 0.30,
            self.config.delta_lower,
            self.config.delta_upper
        ))

        # 2. Compute position return
        actual_position = base_position * (1.0 + bounded_delta)
        position_return = actual_position * realized_return

        # Update running stats BEFORE computing reward
        self.return_stats.update(abs(position_return))

        # Track equity
        self._returns_history.append(position_return)
        self._delta_history.append(bounded_delta)
        self._current_equity *= (1 + position_return)
        self._peak_equity = max(self._peak_equity, self._current_equity)

        drawdown = 1.0 - (self._current_equity / self._peak_equity) if self._peak_equity > 0 else 0.0

        # ===== REWARD COMPONENTS (with adaptive scaling) =====

        r_risk_adjusted = self._compute_sortino_reward(position_return)
        r_delta_efficiency = self._compute_delta_efficiency(bounded_delta, realized_return, regime)
        r_drawdown = self._compute_drawdown_penalty(drawdown)
        r_regime = self._compute_regime_compliance(bounded_delta, regime)
        r_smoothness = self._compute_smoothness_penalty(bounded_delta, prev_delta)
        r_survival = self._compute_survival_bonus(drawdown)

        # Combine components
        total_reward = (
            self.config.w_risk_adjusted * r_risk_adjusted +
            self.config.w_delta_efficiency * r_delta_efficiency +
            self.config.w_drawdown_penalty * r_drawdown +
            self.config.w_regime_compliance * r_regime +
            self.config.w_smoothness * r_smoothness +
            self.config.w_survival_bonus * r_survival
        )

        # V2: Clip reward to prevent explosions
        total_reward_clipped = np.clip(
            total_reward,
            self.config.reward_clip_min,
            self.config.reward_clip_max
        )

        # Track extreme rewards
        if abs(total_reward) > self.config.reward_clip_max:
            self.extreme_reward_count += 1

        # Update reward stats
        self.reward_stats.update(total_reward_clipped)

        components = {
            "risk_adjusted": r_risk_adjusted,
            "delta_efficiency": r_delta_efficiency,
            "drawdown_penalty": r_drawdown,
            "regime_compliance": r_regime,
            "smoothness": r_smoothness,
            "survival_bonus": r_survival,
            "bounded_delta": bounded_delta,
            "position_return": position_return,
            "drawdown": drawdown,
            "equity": self._current_equity,
            "total_reward": total_reward_clipped,
            "total_reward_unclipped": total_reward,
            "return_scale_mean": self.return_stats.mean,
            "return_scale_std": self.return_stats.std,
        }

        return total_reward_clipped, components

    def _compute_sortino_reward(self, position_return: float) -> float:
        """Sortino-style reward with adaptive scaling."""
        excess_return = position_return - self.config.target_return

        # V2: Adaptive scaling based on observed return magnitudes
        if self.config.adaptive_scaling and len(self.return_stats.values) > 10:
            scale = max(self.return_stats.std, self.config.min_scale_value)
            normalized_return = excess_return / scale
        else:
            # Fallback to fixed scaling
            normalized_return = excess_return * 100

        if normalized_return >= 0:
            return normalized_return
        else:
            # 2x penalty for losses
            return normalized_return * 2.0

    def _compute_delta_efficiency(
        self,
        delta: float,
        realized_return: float,
        regime: str
    ) -> float:
        """Reward delta choices that align with outcomes - with scaling."""

        if regime in ["crisis", "cascade", "volatility_cluster", "crash"]:
            # In crisis: reward reduction
            if delta < 0:
                return 0.5 * abs(delta)
            else:
                return -1.0

        # Normal: reward alignment with bounded scaling
        delta_sign = np.sign(delta)
        return_sign = np.sign(realized_return)

        alignment = abs(delta * realized_return)

        # V2: Adaptive scaling
        if self.config.adaptive_scaling and len(self.return_stats.values) > 10:
            scale = max(self.return_stats.std, self.config.min_scale_value)
            alignment_scaled = alignment / scale
        else:
            alignment_scaled = alignment * 10  # Fixed fallback

        # Clip to reasonable range
        alignment_scaled = np.clip(alignment_scaled, 0, 5.0)

        if delta_sign == return_sign:
            return alignment_scaled
        elif delta_sign == 0:
            return 0.1
        else:
            return -alignment_scaled * 0.5

    def _compute_drawdown_penalty(self, drawdown: float) -> float:
        """Asymmetric penalty for drawdowns."""
        if drawdown <= self.config.drawdown_threshold:
            return 0.0

        excess_dd = drawdown - self.config.drawdown_threshold
        normalized_dd = min(excess_dd / self.config.max_drawdown_cap, 1.0)

        # Quadratic penalty (bounded to -2.0 max)
        return -(normalized_dd ** 2) * 2.0

    def _compute_regime_compliance(self, delta: float, regime: str) -> float:
        """Reward regime-aligned deltas."""
        optimal_delta = self.config.regime_optimal_delta.get(regime, 0.0)
        deviation = abs(delta - optimal_delta)

        # Normalized to [-1, 1] range
        return 1.0 - (deviation / 0.60)

    def _compute_smoothness_penalty(
        self,
        current_delta: float,
        prev_delta: float
    ) -> float:
        """Penalize erratic changes."""
        delta_change = abs(current_delta - prev_delta)

        if delta_change < 0.05:
            return 0.0
        else:
            # Bounded penalty
            return -min((delta_change - 0.05) * 2.0, 1.0)

    def _compute_survival_bonus(self, drawdown: float) -> float:
        """Bonus for avoiding ruin."""
        if drawdown < 0.05:
            return 0.5
        elif drawdown < 0.10:
            return 0.2
        elif drawdown < 0.15:
            return 0.0
        else:
            return -0.5

    def get_episode_sharpe(self) -> float:
        """Calculate Sharpe ratio."""
        if len(self._returns_history) < 2:
            return 0.0

        returns = np.array(self._returns_history)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)

        if std_ret < 1e-8:
            return 0.0

        sharpe = mean_ret / std_ret
        return float(sharpe)

    def get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic statistics."""
        return {
            "step_count": self.step_count,
            "extreme_reward_pct": 100 * self.extreme_reward_count / max(self.step_count, 1),
            "return_scale_mean": self.return_stats.mean,
            "return_scale_std": self.return_stats.std,
            "reward_mean": self.reward_stats.mean,
            "reward_std": self.reward_stats.std,
        }

    def reset(self):
        """Reset episode tracking (keep running stats)."""
        self._returns_history = []
        self._delta_history = []
        self._peak_equity = 1.0
        self._current_equity = 1.0


# ============================================================================
# NEURAL NETWORK (same as V1)
# ============================================================================

class LSTMPolicyNetwork(nn.Module):
    """LSTM-based policy for bounded delta."""

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0
        )

        self.actor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 2)
        )

        self.critic = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, hidden=None):
        """Forward pass through network."""
        batch_size = x.size(0)

        x_proj = self.input_proj(x)
        lstm_out, hidden_new = self.lstm(x_proj, hidden)
        last_hidden = lstm_out[:, -1, :]

        actor_out = self.actor(last_hidden)
        value = self.critic(last_hidden)

        return actor_out, value, hidden_new

    def get_action(self, state, hidden=None, deterministic=False):
        """Sample action from policy."""
        actor_out, value, hidden_new = self.forward(state, hidden)

        mean = actor_out[:, 0]
        log_std = actor_out[:, 1]
        std = torch.exp(torch.clamp(log_std, -20, 2))

        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        log_prob = torch.distributions.Normal(mean, std).log_prob(action)

        return action, log_prob, value, hidden_new

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update."""
        batch_size = states.size(0)
        hidden = None

        actor_out, values, _ = self.forward(states, hidden)

        mean = actor_out[:, 0:1]
        log_std = actor_out[:, 1:2]
        std = torch.exp(torch.clamp(log_std, -20, 2))

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values, entropy


# ============================================================================
# PPO TRAINER (same as V1 but with V2 reward)
# ============================================================================

class BoundedDeltaPPOTrainerV2:
    """PPO trainer with V2 reward function."""

    def __init__(
        self,
        net_config: NetworkConfig,
        delta_config: BoundedDeltaConfig,
        tier1_config: Tier1Config,
        device: str = "cpu"
    ):
        self.net_config = net_config
        self.delta_config = delta_config
        self.device = device

        # Components
        self.policy = LSTMPolicyNetwork(net_config).to(device)
        self.tier1 = VolatilityTargeter(tier1_config)
        self.reward_fn = BoundedDeltaRewardV2(delta_config)  # V2 reward

        # Optimizer
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': net_config.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': net_config.critic_lr},
            {'params': self.policy.lstm.parameters(), 'lr': net_config.actor_lr},
            {'params': self.policy.input_proj.parameters(), 'lr': net_config.actor_lr},
        ])

        self.buffer = RolloutBuffer(net_config.buffer_size)

        self.prev_delta = 0.0
        self.hidden = None

    def select_action(
        self,
        state_seq: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[float, float, float, float]:
        """Select action."""
        with torch.no_grad():
            states = torch.FloatTensor(state_seq).unsqueeze(0).to(self.device)
            action, log_prob, value, self.hidden = self.policy.get_action(
                states, self.hidden, deterministic
            )

            raw_delta = action.item() if action.dim() == 0 else action.cpu().numpy().item()
            bounded_delta = float(np.clip(np.tanh(raw_delta) * 0.30, -0.30, 0.30))

            value_scalar = value.item() if value.dim() == 0 else value.cpu().numpy().flatten()[0]
            log_prob_scalar = log_prob.item() if log_prob.dim() == 0 else log_prob.cpu().numpy().flatten()[0]

            return bounded_delta, raw_delta, float(value_scalar), float(log_prob_scalar)

    def step(
        self,
        state_seq: np.ndarray,
        market_return: float,
        volatility: float,
        regime: str,
        portfolio_equity: float = 100000.0
    ) -> Tuple[float, Dict]:
        """Execute training step."""
        base_position, tier1_diag = self.tier1.compute_base_position(
            portfolio_equity, volatility, regime
        )

        bounded_delta, raw_delta, value, log_prob = self.select_action(state_seq)

        reward, reward_components = self.reward_fn.compute_reward(
            raw_delta=raw_delta,
            base_position=base_position,
            realized_return=market_return,
            regime=regime,
            volatility=volatility,
            prev_delta=self.prev_delta
        )

        self.buffer.add(
            state=state_seq,
            action=raw_delta,
            reward=reward,
            value=value,
            log_prob=log_prob
        )

        self.prev_delta = bounded_delta

        info = {
            "bounded_delta": bounded_delta,
            "raw_delta": raw_delta,
            "base_position": base_position,
            "reward": reward,
            "value": value,
            **reward_components,
            **tier1_diag
        }

        return bounded_delta, info

    def end_episode(self, final_value: float = 0.0) -> Dict:
        """End episode and compute returns."""
        episode_sharpe = self.reward_fn.get_episode_sharpe()
        final_equity = self.reward_fn._current_equity

        self.buffer.finish_episode(final_value)

        self.tier1.reset()
        self.reward_fn.reset()
        self.prev_delta = 0.0
        self.hidden = None

        return {
            "episode_sharpe": episode_sharpe,
            "final_equity": final_equity
        }

    def update(self) -> Dict[str, float]:
        """PPO update."""
        if len(self.buffer) < self.net_config.batch_size:
            return {}

        states, actions, returns, advantages, old_log_probs = self.buffer.get()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(-1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(-1).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0

        for _ in range(self.net_config.update_epochs):
            log_probs, values, entropy = self.policy.evaluate_actions(states, actions)

            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.net_config.clip_epsilon,
                1 + self.net_config.clip_epsilon
            ) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy_bonus = entropy.mean()

            loss = (
                policy_loss +
                self.net_config.value_coef * value_loss -
                self.net_config.entropy_coef * entropy_bonus
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.net_config.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy_bonus.item()

        self.buffer.clear()

        n = self.net_config.update_epochs
        return {
            "total_loss": total_loss / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
            "entropy": entropy_sum / n
        }

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "net_config": asdict(self.net_config),
            "delta_config": asdict(self.delta_config),
            "reward_diagnostics": self.reward_fn.get_diagnostics(),
        }, path)

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ============================================================================
# ROLLOUT BUFFER (same as V1)
# ============================================================================

class RolloutBuffer:
    """Buffer for rollout storage."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.clear()

    def add(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        value: float,
        log_prob: float
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def finish_episode(self, final_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns/advantages and accumulate."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [final_value])

        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae

        returns = advantages + values[:-1]

        self.returns.extend(returns.tolist())
        self.advantages.extend(advantages.tolist())

        self.rewards = []
        self.values = []

    def get(self) -> Tuple[np.ndarray, ...]:
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.returns),
            np.array(self.advantages),
            np.array(self.log_probs)
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []

    def __len__(self):
        return len(self.states)


# ============================================================================
# SYNTHETIC ENVIRONMENT (same as V1)
# ============================================================================

class SyntheticTradingEnv:
    """Synthetic trading environment."""

    def __init__(self, data_path: str, seq_length: int = 20):
        self.seq_length = seq_length
        self.scenarios = self._load_scenarios(data_path)
        self.current_scenario = None
        self.step_idx = 0

    def _load_scenarios(self, path: str) -> List[Dict]:
        """Load scenarios from pickle."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                scenarios = pickle.load(f)
            logging.info(f"✅ Loaded {len(scenarios)} synthetic scenarios")
            return scenarios
        else:
            logging.warning(f"⚠️  No scenarios at {path}, generating defaults")
            return self._generate_default_scenarios(500)

    def _generate_default_scenarios(self, n: int) -> List[Dict]:
        """Generate default scenarios."""
        scenarios = []
        regime_types = ["bull", "bear", "mixed", "crash", "volatility_cluster"]

        for i in range(n):
            regime = random.choice(regime_types)
            length = random.randint(500, 1500)

            if regime == "bull":
                drift = 0.0002
                vol = 0.02
            elif regime == "bear":
                drift = -0.0001
                vol = 0.025
            elif regime == "crash":
                drift = -0.001
                vol = 0.05
            elif regime == "volatility_cluster":
                drift = 0.0
                vol = 0.04
            else:
                drift = 0.0
                vol = 0.03

            returns = np.random.normal(drift, vol, length)
            prices = 100 * np.cumprod(1 + returns)
            volatility = np.abs(returns) * np.sqrt(252)

            features = np.random.randn(length, 32) * 0.1
            features[:, 0] = volatility
            features[:, 1] = np.convolve(returns, np.ones(20)/20, mode='same')

            scenarios.append({
                "type": regime,
                "prices": prices,
                "returns": returns,
                "vols": volatility,
                "features": features
            })

        return scenarios

    def reset(self) -> Tuple[np.ndarray, str]:
        """Reset to new scenario."""
        self.current_scenario = random.choice(self.scenarios)
        self.step_idx = self.seq_length

        initial_state = self.current_scenario["features"][:self.seq_length]
        regime_type = self.current_scenario.get("type", "mixed")
        return initial_state, regime_type

    def step(self) -> Tuple[np.ndarray, float, float, bool, str]:
        """Step forward in scenario."""
        if self.step_idx >= len(self.current_scenario["returns"]) - 1:
            done = True
            next_state = self.current_scenario["features"][self.step_idx - self.seq_length:self.step_idx]
            market_return = 0.0
            volatility = 0.5
        else:
            market_return = self.current_scenario["returns"][self.step_idx]
            volatility = self.current_scenario["vols"][self.step_idx]

            self.step_idx += 1

            start_idx = max(0, self.step_idx - self.seq_length)
            end_idx = self.step_idx
            next_state = self.current_scenario["features"][start_idx:end_idx]

            if next_state.shape[0] < self.seq_length:
                padding = np.zeros((self.seq_length - next_state.shape[0], 32))
                next_state = np.vstack([padding, next_state])

            done = (self.step_idx >= len(self.current_scenario["returns"]) - 1)

        regime = self.current_scenario.get("type", "mixed")
        return next_state, market_return, volatility, done, regime


# ============================================================================
# TRAINING SCRIPT
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to synthetic scenarios (.pkl)")
    parser.add_argument("--steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--output", type=str, default="./output/bounded_delta_v2", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)

    # W&B
    if WANDB_AVAILABLE:
        wandb.init(
            project="himari-layer3-bounded-delta-v2",
            name=f"v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "total_steps": args.steps,
                "device": args.device,
                "architecture": "LSTM-PPO-BoundedDelta-V2",
                "delta_bounds": [-0.30, 0.30],
                "adaptive_scaling": True,
            }
        )
        logging.info("✅ W&B logging enabled")

    # Initialize
    net_config = NetworkConfig()
    delta_config = BoundedDeltaConfig()
    tier1_config = Tier1Config()

    trainer = BoundedDeltaPPOTrainerV2(
        net_config=net_config,
        delta_config=delta_config,
        tier1_config=tier1_config,
        device=args.device
    )

    if args.checkpoint and os.path.exists(args.checkpoint):
        trainer.load(args.checkpoint)
        logging.info(f"✅ Loaded checkpoint: {args.checkpoint}")

    env = SyntheticTradingEnv(args.data, seq_length=net_config.sequence_length)

    # Metrics
    total_steps = 0
    episode = 0
    recent_rewards = deque(maxlen=100)
    recent_sharpes = deque(maxlen=100)

    logging.info("=" * 60)
    logging.info("Starting Bounded Delta Training V2")
    logging.info("=" * 60)
    logging.info(f"Target steps: {args.steps:,}")
    logging.info(f"Scenarios loaded: {len(env.scenarios)}")
    logging.info(f"Delta bounds: [{delta_config.delta_lower}, {delta_config.delta_upper}]")
    logging.info(f"Adaptive scaling: {delta_config.adaptive_scaling}")
    logging.info(f"Reward clip: [{delta_config.reward_clip_min}, {delta_config.reward_clip_max}]")
    logging.info(f"Device: {args.device}")
    logging.info("=" * 60)

    # Training loop
    while total_steps < args.steps:
        state, regime = env.reset()
        episode_reward = 0.0
        episode_steps = 0

        done = False
        while not done:
            next_state, market_return, volatility, done, regime = env.step()

            bounded_delta, info = trainer.step(
                state_seq=state,
                market_return=market_return,
                volatility=volatility,
                regime=regime
            )

            episode_reward += info["reward"]
            episode_steps += 1
            total_steps += 1
            state = next_state

            if total_steps % 50_000 == 0:
                ckpt_path = os.path.join(args.output, f"checkpoint_{total_steps}.pt")
                trainer.save(ckpt_path)
                logging.info(f"💾 Checkpoint: {ckpt_path}")

        # End episode
        episode_stats = trainer.end_episode()
        episode_sharpe = episode_stats["episode_sharpe"]

        episode += 1
        recent_rewards.append(episode_reward)
        recent_sharpes.append(episode_sharpe)

        # PPO update after episode
        if len(trainer.buffer) >= net_config.batch_size:
            losses = trainer.update()
            if WANDB_AVAILABLE and losses:
                wandb.log({
                    "loss/total": losses["total_loss"],
                    "loss/policy": losses["policy_loss"],
                    "loss/value": losses["value_loss"],
                    "loss/entropy": losses["entropy"],
                }, step=total_steps)

        # Logging
        if episode % 10 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_sharpe = np.mean(recent_sharpes)
            progress = 100 * total_steps / args.steps

            # V2: Diagnostic info
            diagnostics = trainer.reward_fn.get_diagnostics()

            logging.info(
                f"Ep {episode:5d} | Steps: {total_steps:7d}/{args.steps} ({progress:5.1f}%) | "
                f"Reward: {episode_reward:7.2f} | Sharpe: {episode_sharpe:6.3f} | "
                f"RetScale: {diagnostics['return_scale_std']:.6f} | "
                f"Extreme: {diagnostics['extreme_reward_pct']:.1f}%"
            )

            if WANDB_AVAILABLE:
                wandb.log({
                    "episode": episode,
                    "total_steps": total_steps,
                    "progress_pct": progress,
                    "episode_reward": episode_reward,
                    "episode_sharpe": episode_sharpe,
                    "avg_reward_100": avg_reward,
                    "avg_sharpe_100": avg_sharpe,
                    **{f"diagnostic/{k}": v for k, v in diagnostics.items()},
                }, step=total_steps)

    # Final save
    final_path = os.path.join(args.output, "bounded_delta_v2_final.pt")
    trainer.save(final_path)

    avg_reward = np.mean(recent_rewards)
    avg_sharpe = np.mean(recent_sharpes)

    logging.info("=" * 60)
    logging.info("Bounded Delta Training V2 Complete!")
    logging.info("=" * 60)
    logging.info(f"Total episodes: {episode}")
    logging.info(f"Total steps: {total_steps:,}")
    logging.info(f"Final avg reward: {avg_reward:.3f}")
    logging.info(f"Final avg Sharpe: {avg_sharpe:.3f}")
    logging.info(f"Model saved: {final_path}")
    logging.info("=" * 60)

    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
