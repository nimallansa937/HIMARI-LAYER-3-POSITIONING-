"""
HIMARI OPUS 2 - Tier 4: Hard Constraint Enforcer
=================================================

Enforces non-negotiable position limits.
Implements Tier 4 per CLAUDE Guide Part VII.

These constraints are absolute. No upstream tier can override them.

Version: 1.0
"""

import logging
from typing import Tuple, Dict, Any, List

# Handle imports
try:
    from core.layer3_config import HardConstraintConfig
except ImportError:
    from ..core.layer3_config import HardConstraintConfig

logger = logging.getLogger(__name__)


class HardConstraintEnforcer:
    """
    Tier 4: Enforce non-negotiable position limits.
    
    These constraints are absolute. No upstream tier can override them.
    
    Constraints:
    1. Single position cap: 5% of portfolio
    2. Sector concentration cap: 20% per sector
    3. Correlation aggregation: ρ > 0.7 positions count as one
    4. Leverage caps by regime
    """
    
    def __init__(self, config: HardConstraintConfig = None):
        """
        Initialize hard constraint enforcer.
        
        Args:
            config: HardConstraintConfig with all parameters
        """
        if config is None:
            config = HardConstraintConfig()
        
        self.max_single_position_pct = config.max_single_position_pct  # 5%
        self.max_sector_concentration_pct = config.max_sector_concentration_pct  # 20%
        self.correlation_aggregation_threshold = config.correlation_aggregation_threshold  # 0.7
        
        self.leverage_caps = {
            'NORMAL':   config.leverage_cap_normal,    # 2.0
            'HIGH_VOL': config.leverage_cap_high_vol,  # 1.5
            'CRISIS':   config.leverage_cap_crisis,    # 1.0
            'CASCADE':  config.leverage_cap_cascade    # 0.0
        }
    
    def enforce_constraints(
        self,
        position: float,
        portfolio_equity: float,
        sector_exposures: dict,
        sector: str,
        regime: str,
        correlated_positions: List[dict]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Apply all hard constraints to position.
        
        Args:
            position: Position size from Tier 3
            portfolio_equity: Total portfolio value
            sector_exposures: Current exposure by sector {sector: USD}
            sector: Sector of this position
            regime: Current market regime
            correlated_positions: List of positions with correlation info
            
        Returns:
            Tuple of (constrained_position, diagnostics)
        """
        constraints_hit = []
        original_position = position
        
        # Constraint 1: Single position cap (5% of portfolio)
        max_single = portfolio_equity * self.max_single_position_pct
        if position > max_single:
            position = max_single
            constraints_hit.append('SINGLE_POSITION_CAP')
        
        # Constraint 2: Sector concentration cap (20% of portfolio)
        current_sector_exposure = sector_exposures.get(sector, 0.0)
        max_sector = portfolio_equity * self.max_sector_concentration_pct
        remaining_sector_capacity = max(0, max_sector - current_sector_exposure)
        if position > remaining_sector_capacity:
            position = remaining_sector_capacity
            constraints_hit.append('SECTOR_CONCENTRATION_CAP')
        
        # Constraint 3: Correlation aggregation
        # If this position is highly correlated with existing positions,
        # treat them as a single position for limit purposes
        correlated_exposure = sum(
            pos.get('size', 0) for pos in correlated_positions 
            if pos.get('correlation', 0) > self.correlation_aggregation_threshold
        )
        effective_position = position + correlated_exposure
        if effective_position > max_single:
            position = max(0, max_single - correlated_exposure)
            constraints_hit.append('CORRELATION_AGGREGATION')
        
        # Constraint 4: Leverage cap by regime
        leverage_cap = self.leverage_caps.get(regime, 1.0)
        max_leveraged_position = portfolio_equity * leverage_cap
        if position > max_leveraged_position:
            position = max_leveraged_position
            constraints_hit.append('LEVERAGE_CAP')
        
        # Ensure non-negative
        position = max(0, position)
        
        diagnostics = {
            'original_position': original_position,
            'constrained_position': position,
            'constraints_hit': constraints_hit,
            'num_constraints_hit': len(constraints_hit),
            'leverage_cap': leverage_cap,
            'max_single': max_single,
            'max_sector': max_sector,
            'current_sector_exposure': current_sector_exposure,
            'remaining_sector_capacity': remaining_sector_capacity,
            'correlated_exposure': correlated_exposure,
            'regime': regime,
            'tier': 'HARD_CONSTRAINTS'
        }
        
        if constraints_hit:
            logger.info(
                f"Hard constraints applied: {constraints_hit} "
                f"${original_position:,.2f} → ${position:,.2f}"
            )
        
        return position, diagnostics
    
    def get_sector_for_symbol(self, symbol: str) -> str:
        """
        Map symbol to sector classification.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Sector classification
        """
        sector_map = {
            'BTC': 'L1',
            'ETH': 'L1',
            'SOL': 'L1',
            'AVAX': 'L1',
            'DOGE': 'MEME',
            'SHIB': 'MEME',
            'PEPE': 'MEME',
            'UNI': 'DEFI',
            'AAVE': 'DEFI',
            'LINK': 'ORACLE',
            'MATIC': 'L2',
            'ARB': 'L2',
            'OP': 'L2',
        }
        
        symbol_upper = symbol.upper()
        for token, sector in sector_map.items():
            if token in symbol_upper:
                return sector
        return 'OTHER'
    
    def get_correlated_positions(
        self,
        symbol: str,
        open_positions: List[dict]
    ) -> List[dict]:
        """
        Find positions correlated with the target symbol.
        
        Simplified implementation: return positions in same sector.
        
        Args:
            symbol: Target symbol
            open_positions: List of open positions
            
        Returns:
            List of correlated positions with correlation info
        """
        target_sector = self.get_sector_for_symbol(symbol)
        
        correlated = []
        for pos in open_positions:
            pos_symbol = pos.get('symbol', '')
            pos_sector = self.get_sector_for_symbol(pos_symbol)
            
            if pos_sector == target_sector and pos_symbol != symbol:
                # Same sector positions assumed to have high correlation
                correlated.append({
                    'symbol': pos_symbol,
                    'size': pos.get('size', 0),
                    'correlation': 0.85  # Assumed for same sector
                })
        
        return correlated
