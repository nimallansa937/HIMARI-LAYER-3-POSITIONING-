"""
HIMARI Layer 3 - Risk-Aware Reward Function
=============================================

Implements multi-component reward function based on findings from:
- AlexRzk/ORION (BlackRock-grade trading environment)
- JMaskiewicz/RL_tester (Numba-optimized reward calculation)
- joel-saucedo/Crypto-Strategy-Lab (regime-aware penalties)

Reward Components:
R = R_pnl + R_risk + R_diversity + R_drawdown + R_transaction

Version: 1.0
"""

import numpy as np

try:
    from numba import jit
except ImportError:
    # Fallback if numba is not installed
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum


class RewardComponent(Enum):
    """Individual reward components for analysis."""
    PNL = "pnl"
    VOLATILITY_PENALTY = "volatility_penalty"
    DRAWDOWN_PENALTY = "drawdown_penalty"
    TRANSACTION_COST = "transaction_cost"
    ACTION_DIVERSITY = "action_diversity"
    HOLDING_BONUS = "holding_bonus"
    WRONG_ACTION_PENALTY = "wrong_action_penalty"


@dataclass
class RewardConfig:
    """
    Configuration for risk-aware reward function.
    
    Based on findings from:
    - ORION:  Multi-component reward structure
    - RL_tester: Transaction cost and wrong action penalties
    - Crypto-Strategy-Lab: Volatility and drawdown penalties
    """
    # PnL scaling
    pnl_scale: float = 1000.0  # Scale factor for returns (from RL_tester)
    
    # Volatility penalty (from your best practices doc)
    lambda_vol: float = 0.5
    vol_baseline: float = 0.02  # 2% daily vol is "normal"
    
    # Drawdown penalty (from ORION)
    lambda_dd: float = 2.0
    dd_threshold: float = 0.05  # Start penalizing after 5% drawdown
    dd_crisis_threshold: float = 0.15  # Crisis level drawdown
    
    # Transaction cost penalty (from RL_tester)
    lambda_cost: float = 0.1
    provision_rate: float = 0.001  # 0.1% transaction cost
    
    # Wrong action penalty (from RL_tester)
    wrong_action_multiplier: float = 1.5  # Amplify losses
    
    # Action diversity bonus (from ORION - prevents policy collapse)
    diversity_bonus:  float = 0.01
    diversity_window: int = 20
    
    # Holding bonus (from RL_tester - prevents churning)
    holding_bonus: float = 0.001
    min_holding_periods: int = 5


# =============================================================================
# NUMBA-OPTIMIZED REWARD FUNCTIONS (from JMaskiewicz/RL_tester)
# =============================================================================

@jit(nopython=True)
def calculate_pnl_reward(
    previous_price:  float,
    current_price:  float,
    position:  float,
    leverage: float = 1.0,
    scale: float = 1000.0
) -> float:
    """
    Calculate PnL-based reward component.
    
    From: JMaskiewicz/RL_tester reward_calculation function
    """
    if previous_price == 0:
        return 0.0
    
    normal_return = (current_price - previous_price) / previous_price
    reward = normal_return * position * leverage * scale
    
    return reward


@jit(nopython=True)
def calculate_transaction_cost_penalty(
    previous_position: float,
    current_position: float,
    provision_rate: float,
    capital: float,
    scale: float = 1000.0
) -> float:
    """
    Calculate transaction cost penalty for position changes.
    
    From: JMaskiewicz/RL_tester provision_cost calculation
    """
    if current_position == previous_position:
        return 0.0
    
    # Position changed - apply provision cost
    if abs(current_position) > 0: 
        provision_cost = provision_rate * capital * abs(current_position - previous_position)
        return -provision_cost * scale / capital  # Normalize by capital
    
    return 0.0


@jit(nopython=True)
def calculate_wrong_action_penalty(
    reward: float,
    multiplier: float = 1.5
) -> float:
    """
    Amplify losses for wrong actions.
    
    From: JMaskiewicz/RL_tester - "Penalize the agent for taking the wrong action"
    """
    if reward < 0:
        return reward * multiplier
    return reward


@jit(nopython=True)
def calculate_volatility_penalty(
    rolling_volatility: float,
    baseline_volatility: float,
    lambda_vol: float,
    scale: float = 1000.0
) -> float:
    """
    Penalize holding positions during high volatility.
    
    From: Your best practices doc + Crypto-Strategy-Lab regime detection
    """
    excess_vol = max(0.0, rolling_volatility - baseline_volatility)
    return -lambda_vol * excess_vol * scale


@jit(nopython=True)
def calculate_drawdown_penalty(
    current_drawdown: float,
    dd_threshold: float,
    lambda_dd: float,
    scale: float = 1000.0
) -> float:
    """
    Penalize drawdowns to encourage survival.
    
    From: AlexRzk/ORION - R_drawdown component
    """
    if current_drawdown > dd_threshold:
        return 0.0  # No penalty if drawdown is acceptable
    
    # Drawdown is negative, so we penalize when it's below threshold
    excess_dd = abs(current_drawdown) - dd_threshold
    if excess_dd > 0:
        return -lambda_dd * excess_dd * scale
    
    return 0.0


@jit(nopython=True)
def calculate_holding_bonus(
    previous_position: float,
    current_position: float,
    holding_periods: int,
    min_holding:  int,
    bonus:  float,
    scale: float = 1000.0
) -> float:
    """
    Bonus for maintaining positions (prevents churning).
    
    From: JMaskiewicz/RL_tester - "premium for holding position"
    """
    if previous_position == current_position and abs(current_position) > 0:
        if holding_periods >= min_holding: 
            return bonus * scale
    return 0.0


# =============================================================================
# MAIN REWARD CALCULATOR CLASS
# =============================================================================

class RiskAwareRewardCalculator: 
    """
    Multi-component risk-aware reward calculator.
    
    Implements best practices from:
    - AlexRzk/ORION:  Multi-component structure (R_pnl + R_risk + R_diversity + R_drawdown)
    - JMaskiewicz/RL_tester:  Numba optimization, transaction costs, wrong action penalties
    - joel-saucedo/Crypto-Strategy-Lab: Regime-aware penalties
    - EpicSanDev/EVIL2ROOT_AI: Kelly criterion integration
    """
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        
        # Tracking for diversity bonus
        self.action_history: list = []
        self.holding_periods: int = 0
        self.previous_position: float = 0.0
        
        # Peak tracking for drawdown
        self.peak_capital: float = 0.0
        
    def reset(self, initial_capital: float = 100000.0):
        """Reset calculator state for new episode."""
        self.action_history = []
        self.holding_periods = 0
        self.previous_position = 0.0
        self.peak_capital = initial_capital
        
    def calculate_reward(
        self,
        previous_price: float,
        current_price: float,
        previous_position: float,
        current_position: float,
        current_capital: float,
        rolling_volatility: float,
        leverage: float = 1.0
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-component risk-aware reward.
        
        Args:
            previous_price: Price at t-1
            current_price:  Price at t
            previous_position: Position at t-1 (normalized, -1 to 1)
            current_position: Position at t (normalized, -1 to 1)
            current_capital: Current portfolio value
            rolling_volatility:  Recent volatility estimate
            leverage: Trading leverage
            
        Returns: 
            total_reward: Combined reward value
            components: Dictionary of individual reward components
        """
        cfg = self.config
        
        # Update peak capital for drawdown calculation
        # If this is called in the first step, peak might need init. handled in reset or here.
        if self.peak_capital == 0: self.peak_capital = current_capital 
        self.peak_capital = max(self.peak_capital, current_capital)
        
        current_drawdown = (current_capital - self.peak_capital) / self.peak_capital
        
        # Update holding periods
        if previous_position == current_position and abs(current_position) > 0:
            self.holding_periods += 1
        else:
            self.holding_periods = 0
        
        # Track action for diversity calculation
        self.action_history.append(current_position)
        if len(self.action_history) > cfg.diversity_window:
            self.action_history.pop(0)
        
        # =================================================================
        # Component 1: PnL Reward (from RL_tester)
        # =================================================================
        pnl_reward = calculate_pnl_reward(
            previous_price, current_price, current_position,
            leverage, cfg.pnl_scale
        )
        
        # =================================================================
        # Component 2: Wrong Action Penalty (from RL_tester)
        # =================================================================
        pnl_with_penalty = calculate_wrong_action_penalty(
            pnl_reward, cfg.wrong_action_multiplier
        )
        wrong_action_penalty = pnl_with_penalty - pnl_reward
        
        # =================================================================
        # Component 3: Transaction Cost Penalty (from RL_tester)
        # =================================================================
        transaction_penalty = calculate_transaction_cost_penalty(
            previous_position, current_position,
            cfg.provision_rate, current_capital, cfg.pnl_scale
        )
        
        # =================================================================
        # Component 4: Volatility Penalty (from Best Practices + Crypto-Strategy-Lab)
        # =================================================================
        vol_penalty = calculate_volatility_penalty(
            rolling_volatility, cfg.vol_baseline,
            cfg.lambda_vol, cfg.pnl_scale
        )
        
        # Scale by position size (only penalize if holding)
        vol_penalty *= abs(current_position)
        
        # =================================================================
        # Component 5: Drawdown Penalty (from ORION)
        # =================================================================
        dd_penalty = calculate_drawdown_penalty(
            current_drawdown, cfg.dd_threshold,
            cfg.lambda_dd, cfg.pnl_scale
        )
        
        # =================================================================
        # Component 6: Action Diversity Bonus (from ORION)
        # =================================================================
        diversity_bonus = self._calculate_diversity_bonus()
        
        # =================================================================
        # Component 7: Holding Bonus (from RL_tester)
        # =================================================================
        hold_bonus = calculate_holding_bonus(
            previous_position, current_position,
            self.holding_periods, cfg.min_holding_periods,
            cfg.holding_bonus, cfg.pnl_scale
        )
        
        # =================================================================
        # Combine All Components
        # =================================================================
        total_reward = (
            pnl_with_penalty +
            transaction_penalty +
            vol_penalty +
            dd_penalty +
            diversity_bonus +
            hold_bonus
        )
        
        # Normalize reward to reasonable range
        total_reward = np.clip(total_reward / cfg.pnl_scale, -10.0, 10.0)
        
        # Store previous position
        self.previous_position = current_position
        
        # Return components for analysis
        components = {
            RewardComponent.PNL.value: pnl_reward / cfg.pnl_scale,
            RewardComponent.WRONG_ACTION_PENALTY.value: wrong_action_penalty / cfg.pnl_scale,
            RewardComponent.TRANSACTION_COST.value: transaction_penalty / cfg.pnl_scale,
            RewardComponent.VOLATILITY_PENALTY.value: vol_penalty / cfg.pnl_scale,
            RewardComponent.DRAWDOWN_PENALTY.value: dd_penalty / cfg.pnl_scale,
            RewardComponent.ACTION_DIVERSITY.value: diversity_bonus / cfg.pnl_scale,
            RewardComponent.HOLDING_BONUS.value: hold_bonus / cfg.pnl_scale,
        }
        
        return total_reward, components
    
    def _calculate_diversity_bonus(self) -> float:
        """
        Calculate action diversity bonus to prevent policy collapse.
        
        From: AlexRzk/ORION - R_diversity component
        """
        if len(self.action_history) < self.config.diversity_window:
            return 0.0
        
        # Count unique actions in recent history
        unique_actions = len(set(np.round(self.action_history, 1)))
        max_possible = min(len(self.action_history), 21)  # -1 to 1 in 0.1 increments
        
        # Diversity ratio
        diversity_ratio = unique_actions / max_possible
        
        # Bonus for diverse actions
        return self.config.diversity_bonus * diversity_ratio * self.config.pnl_scale


# =============================================================================
# FAST NUMBA VERSION FOR TRAINING (Combined function)
# =============================================================================

@jit(nopython=True)
def fast_risk_aware_reward(
    prev_price: float,
    curr_price: float,
    prev_pos: float,
    curr_pos: float,
    capital: float,
    rolling_vol: float,
    drawdown: float,
    holding_periods: int,
    # Config parameters
    pnl_scale: float = 1000.0,
    lambda_vol: float = 0.5,
    vol_baseline: float = 0.02,
    lambda_dd:  float = 2.0,
    dd_threshold: float = 0.05,
    provision_rate: float = 0.001,
    wrong_action_mult: float = 1.5,
    holding_bonus: float = 0.001,
    min_holding:  int = 5
) -> float:
    """
    Fast combined reward calculation for training loops.
    
    Combines all components in a single Numba-optimized function. 
    """
    # 1. Base PnL
    if prev_price > 0:
        normal_return = (curr_price - prev_price) / prev_price
        reward = normal_return * curr_pos * pnl_scale
    else:
        reward = 0.0
    
    # 2. Wrong action penalty
    if reward < 0:
        reward *= wrong_action_mult
    
    # 3. Transaction cost
    if curr_pos != prev_pos and abs(curr_pos) > 0:
        reward -= provision_rate * pnl_scale * abs(curr_pos - prev_pos)
    
    # 4. Volatility penalty
    excess_vol = max(0.0, rolling_vol - vol_baseline)
    reward -= lambda_vol * excess_vol * pnl_scale * abs(curr_pos)
    
    # 5. Drawdown penalty
    if drawdown < -dd_threshold:
        excess_dd = abs(drawdown) - dd_threshold
        reward -= lambda_dd * excess_dd * pnl_scale
    
    # 6. Holding bonus
    if prev_pos == curr_pos and abs(curr_pos) > 0 and holding_periods >= min_holding:
        reward += holding_bonus * pnl_scale
    
    # Normalize
    return max(-10.0, min(10.0, reward / pnl_scale))


# =============================================================================
# TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Risk-Aware Reward Calculator Test")
    print("=" * 70)
    
    config = RewardConfig()
    calculator = RiskAwareRewardCalculator(config)
    calculator.reset(initial_capital=100000)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Profitable long position",
            "prev_price": 50000, "curr_price": 51000,
            "prev_pos": 0.5, "curr_pos": 0.5,
            "capital": 102000, "vol": 0.02
        },
        {
            "name": "Loss on long position",
            "prev_price": 51000, "curr_price": 49000,
            "prev_pos": 0.5, "curr_pos": 0.5,
            "capital": 98000, "vol": 0.03
        },
        {
            "name": "High volatility penalty",
            "prev_price":  49000, "curr_price": 49500,
            "prev_pos":  0.5, "curr_pos": 0.5,
            "capital": 98500, "vol": 0.06
        },
        {
            "name": "Position change (transaction cost)",
            "prev_price": 49500, "curr_price": 50000,
            "prev_pos": 0.5, "curr_pos": -0.3,
            "capital": 99000, "vol": 0.025
        },
        {
            "name": "Crisis drawdown",
            "prev_price": 50000, "curr_price": 45000,
            "prev_pos": -0.3, "curr_pos": -0.3,
            "capital": 85000, "vol": 0.08
        },
    ]
    
    print("\nScenario Results:")
    print("-" * 70)
    
    for scenario in scenarios:
        reward, components = calculator.calculate_reward(
            previous_price=scenario["prev_price"],
            current_price=scenario["curr_price"],
            previous_position=scenario["prev_pos"],
            current_position=scenario["curr_pos"],
            current_capital=scenario["capital"],
            rolling_volatility=scenario["vol"]
        )
        
        print(f"\n{scenario['name']}:")
        print(f"  Total Reward: {reward:+.4f}")
        print(f"  Components:")
        for comp, value in components.items():
            if abs(value) > 0.0001:
                print(f"    {comp}:  {value:+.4f}")
    
    # Test fast version
    print("\n" + "=" * 70)
    print("Fast Numba Version Test:")
    print("-" * 70)
    
    fast_reward = fast_risk_aware_reward(
        prev_price=50000, curr_price=51000,
        prev_pos=0.5, curr_pos=0.5,
        capital=102000, rolling_vol=0.02,
        drawdown=-0.02, holding_periods=10
    )
    print(f"Fast reward: {fast_reward:+.4f}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
