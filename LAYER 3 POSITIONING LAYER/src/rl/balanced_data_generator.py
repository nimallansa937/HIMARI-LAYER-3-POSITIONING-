"""
HIMARI Layer 3: Balanced Training Data Generator
=================================================

Generates a balanced dataset of synthetic market scenarios to train the positioning model.
New distribution helps model learn both upside capture and downside protection.

Distribution:
- 40% Bull Trend (Learn to hold/pyramid)
- 40% Bear/Crash (Learn to hedge/exit)
- 20% Ranging/Chop (Learn to reduce activity)

Usage:
    python src/rl/balanced_data_generator.py --output /tmp/synthetic_data/balanced_scenarios.pkl
"""

import numpy as np
import pickle
import argparse
import os
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_garch_process(length: int, omega: float, alpha: float, beta: float, initial_vol: float) -> np.ndarray:
    """Generate volatility using GARCH(1,1) model."""
    vol = np.zeros(length)
    vol[0] = initial_vol
    
    # Generate random shocks
    shocks = np.random.normal(0, 1, length)
    
    returns = np.zeros(length)
    
    for t in range(1, length):
        # GARCH variance update
        var_t = omega + alpha * (returns[t-1]**2) + beta * (vol[t-1]**2)
        vol[t] = np.sqrt(var_t)
        returns[t] = vol[t] * shocks[t]
        
    return returns, vol

def generate_bull_scenario(length: int = 1000) -> Dict:
    """
    Generate a Bull Market scenario.
    Characteristics:
    - Positive drift
    - Lower volatility
    - occasional dips (buy the dip opportunities)
    """
    # GARCH params for stable uptrend
    omega = 1e-6
    alpha = 0.05
    beta = 0.90
    initial_vol = 0.01
    
    returns, _ = generate_garch_process(length, omega, alpha, beta, initial_vol)
    
    # Add positive drift (stronger than random walk)
    drift = np.random.uniform(0.0005, 0.0015, length)  # 0.05% to 0.15% per step
    
    # Add momentum waves
    t = np.linspace(0, 4*np.pi, length)
    momentum = 0.0005 * np.sin(t)
    
    final_returns = returns + drift + momentum
    
    # Add flash dump (buy the dip opportunity)
    if np.random.random() < 0.5:
        idx = np.random.randint(200, length-200)
        final_returns[idx:idx+5] -= 0.02  # 2% drops
    
    return {
        "type": "bull",
        "returns": final_returns,
        "label": "uptrend"
    }

def generate_bear_scenario(length: int = 1000) -> Dict:
    """
    Generate a Bear Market / Crash scenario.
    Characteristics:
    - Negative drift
    - High volatility clusters
    - Sharp drops
    """
    # GARCH params for volatile downtrend
    omega = 5e-6
    alpha = 0.15
    beta = 0.80
    initial_vol = 0.02
    
    returns, _ = generate_garch_process(length, omega, alpha, beta, initial_vol)
    
    # Add negative drift
    drift = np.random.uniform(-0.002, -0.0005, length)
    
    # Add panic spikes
    final_returns = returns + drift
    
    # Crash events
    num_crashes = np.random.randint(1, 4)
    for _ in range(num_crashes):
        idx = np.random.randint(100, length-100)
        width = np.random.randint(3, 10)
        final_returns[idx:idx+width] -= np.random.uniform(0.01, 0.03, width)
        
        # Volatility expansion after crash
        final_returns[idx+width:idx+width+50] *= 2.0
        
    return {
        "type": "bear",
        "returns": final_returns,
        "label": "downtrend"
    }

def generate_ranging_scenario(length: int = 1000) -> Dict:
    """
    Generate a Ranging / Chop scenario.
    Characteristics:
    - Near zero drift
    - Mean reverting
    - Variable volatility
    """
    omega = 2e-6
    alpha = 0.1
    beta = 0.85
    initial_vol = 0.015
    
    returns, _ = generate_garch_process(length, omega, alpha, beta, initial_vol)
    
    # Mean reversion force
    prices = np.cumprod(1 + returns)
    ma = np.convolve(prices, np.ones(50)/50, mode='same')
    
    # Add reversion adjustment
    price_deviation = (prices - ma) / ma
    reversion = -0.1 * price_deviation  # Pull back to mean
    
    final_returns = returns + reversion
    
    return {
        "type": "ranging",
        "returns": final_returns,
        "label": "chop"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000, help="Number of scenarios")
    parser.add_argument("--length", type=int, default=1000, help="Length of each scenario")
    parser.add_argument("--output", type=str, default="data/balanced_scenarios.pkl")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    scenarios = []
    
    # Distribution based on strategy
    n_bull = int(args.count * 0.40)
    n_bear = int(args.count * 0.40)
    n_range = args.count - n_bull - n_bear
    
    logger.info(f"Generating {args.count} scenarios:")
    logger.info(f"  - Bull: {n_bull}")
    logger.info(f"  - Bear: {n_bear}")
    logger.info(f"  - Ranging: {n_range}")
    
    for _ in range(n_bull):
        scenarios.append(generate_bull_scenario(args.length))
        
    for _ in range(n_bear):
        scenarios.append(generate_bear_scenario(args.length))
        
    for _ in range(n_range):
        scenarios.append(generate_ranging_scenario(args.length))
    
    # Shuffle
    np.random.shuffle(scenarios)
    
    with open(args.output, "wb") as f:
        pickle.dump(scenarios, f)
        
    logger.info(f"✅ Saved balanced dataset to {args.output}")

if __name__ == "__main__":
    main()
