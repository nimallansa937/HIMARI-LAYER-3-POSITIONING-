"""
HIMARI OPUS V2 - Ensemble Position Aggregator V2
=================================================

Aggregate positions from multiple strategies with weight drift tracking
for post-trade attribution analysis.

Features:
- Weight history tracking (last 1000 decisions)
- Drift detection (>20% change triggers alert)
- Correlation penalty application
- CSV export for attribution
- Position limit enforcement

Version: 3.1 Phase 2
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import csv
import time
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class WeightSnapshot:
    """Snapshot of portfolio weights at a point in time."""
    timestamp_ns: int
    weights: Dict[str, float]
    total_allocated_usd: float
    num_strategies: int
    drift_alerts: List[str] = field(default_factory=list)


class EnsemblePositionAggregatorV2:
    """
    Enhanced ensemble position aggregator with weight drift tracking.
    
    Aggregates position sizes from multiple strategies while:
    - Applying correlation penalties to reduce correlated positions
    - Tracking weight evolution over time
    - Detecting sudden weight shifts (>20% change)
    - Exporting weight history for post-trade attribution
    """
    
    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = 0.10,
        max_correlation: float = 0.7,
        history_window: int = 1000,
        drift_alert_threshold: float = 0.20
    ):
        """
        Initialize ensemble aggregator.
        
        Args:
            portfolio_value: Total portfolio value in USD
            max_position_pct: Maximum single position as % of portfolio
            max_correlation: Maximum acceptable correlation
            history_window: Number of weight snapshots to retain
            drift_alert_threshold: Weight change % to trigger alert
        """
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_correlation = max_correlation
        self.drift_alert_threshold = drift_alert_threshold
        
        # Weight drift tracking
        self.weight_history: deque = deque(maxlen=history_window)
        self.last_weights: Dict[str, float] = {}
        
        # Statistics
        self.total_aggregations = 0
        self.drift_alerts_triggered = 0
        self.correlation_penalties_applied = 0
        
        logger.info(
            f"EnsemblePositionAggregatorV2 initialized: "
            f"portfolio=${portfolio_value:,.0f}, max_pos={max_position_pct:.0%}"
        )
    
    def aggregate_positions(
        self,
        strategies: List[Dict],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Aggregate positions from multiple strategies.
        
        Args:
            strategies: List of strategy dicts with:
                - 'id': Strategy identifier
                - 'symbol': Asset symbol
                - 'size': Requested position size USD
                - 'confidence': Strategy confidence [0,1]
            correlation_matrix: NxN correlation matrix (optional)
            
        Returns:
            Tuple of (allocations, diagnostics)
        """
        self.total_aggregations += 1
        timestamp_ns = time.time_ns()
        
        if not strategies:
            return {}, {'status': 'no_strategies'}
        
        # Calculate total requested
        total_requested = sum(s['size'] for s in strategies)
        
        # Build current weights
        current_weights = {}
        position_allocations = {}
        correlation_penalties = {}
        
        for i, strategy in enumerate(strategies):
            strategy_id = strategy['id']
            symbol = strategy['symbol']
            base_size = strategy['size']
            confidence = strategy.get('confidence', 0.5)
            
            # Calculate weight
            weight = base_size / total_requested if total_requested > 0 else 0.0
            current_weights[strategy_id] = weight
            
            # Apply correlation penalty if matrix provided
            if correlation_matrix is not None and i < correlation_matrix.shape[0]:
                penalty = self._calculate_correlation_penalty(
                    i, correlation_matrix, strategies
                )
                if penalty < 1.0:
                    self.correlation_penalties_applied += 1
                    correlation_penalties[symbol] = penalty
            else:
                penalty = 1.0
            
            # Calculate final allocation
            adjusted_size = base_size * penalty * confidence
            
            # Apply position limit
            max_size = self.portfolio_value * self.max_position_pct
            if adjusted_size > max_size:
                adjusted_size = max_size
            
            # Aggregate by symbol (multiple strategies may target same symbol)
            if symbol in position_allocations:
                position_allocations[symbol] += adjusted_size
            else:
                position_allocations[symbol] = adjusted_size
        
        # Detect weight drift
        drift_alerts = self._detect_weight_drift(current_weights)
        
        # Record snapshot
        total_allocated = sum(position_allocations.values())
        snapshot = WeightSnapshot(
            timestamp_ns=timestamp_ns,
            weights=current_weights.copy(),
            total_allocated_usd=total_allocated,
            num_strategies=len(strategies),
            drift_alerts=drift_alerts
        )
        self.weight_history.append(snapshot)
        
        # Update last weights
        self.last_weights = current_weights.copy()
        
        # Diagnostics
        diagnostics = {
            'num_strategies': len(strategies),
            'total_requested_usd': total_requested,
            'total_allocated_usd': total_allocated,
            'utilization_pct': total_allocated / self.portfolio_value * 100,
            'weights': current_weights,
            'correlation_penalties': correlation_penalties,
            'drift_alerts': drift_alerts,
            'status': 'success'
        }
        
        if drift_alerts:
            logger.warning(f"Weight drift alerts: {drift_alerts}")
        
        return position_allocations, diagnostics
    
    def _calculate_correlation_penalty(
        self,
        strategy_idx: int,
        correlation_matrix: np.ndarray,
        strategies: List[Dict]
    ) -> float:
        """
        Calculate correlation penalty for a strategy.
        
        Reduces position size if highly correlated with other strategies.
        """
        penalties = []
        
        for j, other in enumerate(strategies):
            if j != strategy_idx and j < correlation_matrix.shape[1]:
                corr = abs(correlation_matrix[strategy_idx, j])
                
                if corr > self.max_correlation:
                    # Penalty proportional to excess correlation
                    excess = corr - self.max_correlation
                    penalty = max(0.5, 1.0 - excess)  # Min 50% of position
                    penalties.append(penalty)
        
        if penalties:
            return min(penalties)  # Use most restrictive penalty
        return 1.0
    
    def _detect_weight_drift(
        self,
        current_weights: Dict[str, float]
    ) -> List[str]:
        """Detect significant weight changes."""
        alerts = []
        
        if not self.last_weights:
            return alerts
        
        all_strategies = set(current_weights.keys()) | set(self.last_weights.keys())
        
        for strategy_id in all_strategies:
            current = current_weights.get(strategy_id, 0.0)
            previous = self.last_weights.get(strategy_id, 0.0)
            
            if previous > 0:
                change_pct = abs(current - previous) / previous
            elif current > 0:
                change_pct = 1.0  # New strategy = 100% change
            else:
                continue
            
            if change_pct >= self.drift_alert_threshold:
                self.drift_alerts_triggered += 1
                alerts.append(
                    f"{strategy_id}: {previous:.1%} → {current:.1%} ({change_pct:.0%} change)"
                )
        
        return alerts
    
    def export_weight_history(
        self,
        output_path: str,
        format: str = 'csv'
    ) -> bool:
        """
        Export weight history for post-trade attribution.
        
        Args:
            output_path: Path to output file
            format: Export format ('csv' supported)
            
        Returns:
            True if export successful
        """
        if not self.weight_history:
            logger.warning("No weight history to export")
            return False
        
        try:
            if format == 'csv':
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    
                    # Header
                    writer.writerow([
                        'timestamp_ns',
                        'timestamp_iso',
                        'strategy_id',
                        'weight',
                        'total_allocated_usd',
                        'num_strategies'
                    ])
                    
                    # Data rows
                    for snapshot in self.weight_history:
                        timestamp_iso = time.strftime(
                            '%Y-%m-%dT%H:%M:%SZ',
                            time.gmtime(snapshot.timestamp_ns / 1e9)
                        )
                        
                        for strategy_id, weight in snapshot.weights.items():
                            writer.writerow([
                                snapshot.timestamp_ns,
                                timestamp_iso,
                                strategy_id,
                                f"{weight:.6f}",
                                f"{snapshot.total_allocated_usd:.2f}",
                                snapshot.num_strategies
                            ])
            
            logger.info(f"Exported {len(self.weight_history)} snapshots to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export weight history: {e}")
            return False
    
    def get_weight_statistics(self) -> Dict:
        """Get weight distribution statistics."""
        if not self.weight_history:
            return {'snapshots': 0}
        
        # Get all unique strategies
        all_strategies = set()
        for snapshot in self.weight_history:
            all_strategies.update(snapshot.weights.keys())
        
        # Calculate average weights
        avg_weights = {}
        for strategy_id in all_strategies:
            weights = [
                s.weights.get(strategy_id, 0.0)
                for s in self.weight_history
                if strategy_id in s.weights
            ]
            if weights:
                avg_weights[strategy_id] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                }
        
        return {
            'snapshots': len(self.weight_history),
            'unique_strategies': len(all_strategies),
            'average_weights': avg_weights,
            'drift_alerts_triggered': self.drift_alerts_triggered,
            'correlation_penalties_applied': self.correlation_penalties_applied
        }
    
    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'portfolio_value': self.portfolio_value,
            'max_position_pct': self.max_position_pct,
            'max_correlation': self.max_correlation,
            'drift_alert_threshold': self.drift_alert_threshold,
            'total_aggregations': self.total_aggregations,
            'drift_alerts_triggered': self.drift_alerts_triggered,
            'correlation_penalties_applied': self.correlation_penalties_applied,
            'history_size': len(self.weight_history),
            'last_weights': self.last_weights
        }
