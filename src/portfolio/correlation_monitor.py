"""
HIMARI OPUS V2 - Correlation Monitor
=====================================

Portfolio-level correlation matrix computation and monitoring.

Features:
- Rolling correlation calculation
- Cross-asset correlation alerts
- Portfolio diversification score
- Correlation heatmap export

Version: 3.1 Phase 2
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import logging

logger = logging.getLogger(__name__)


class CorrelationMonitor:
    """
    Monitor cross-asset correlations for portfolio risk management.
    
    Tracks rolling correlations and alerts when positions become
    too correlated, reducing diversification benefits.
    """
    
    def __init__(
        self,
        window_size: int = 60,
        correlation_alert_threshold: float = 0.7,
        min_samples: int = 20
    ):
        """
        Initialize correlation monitor.
        
        Args:
            window_size: Rolling window for correlation calculation
            correlation_alert_threshold: Threshold for high correlation alert
            min_samples: Minimum samples needed for valid correlation
        """
        self.window_size = window_size
        self.correlation_alert_threshold = correlation_alert_threshold
        self.min_samples = min_samples
        
        # Return history by symbol
        self.return_history: Dict[str, deque] = {}
        
        # Current correlation matrix
        self.current_correlation: Optional[np.ndarray] = None
        self.symbols: List[str] = []
        
        # Statistics
        self.total_updates = 0
        self.high_correlation_alerts = 0
        
        logger.info(
            f"CorrelationMonitor initialized: window={window_size}, "
            f"alert_threshold={correlation_alert_threshold}"
        )
    
    def update(
        self,
        returns: Dict[str, float]
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Update with new return observations.

        Args:
            returns: {symbol: return} for this period

        Returns:
            Tuple of (correlation_matrix, diagnostics)
        """
        self.total_updates += 1

        # Cleanup inactive symbols (no data in last 2x window_size updates)
        if self.total_updates % (self.window_size * 2) == 0:
            self._cleanup_inactive_symbols(returns.keys())

        # Update return history (filter invalid values)
        for symbol, ret in returns.items():
            # Skip NaN/Inf values
            if ret is None or np.isnan(ret) or np.isinf(ret):
                logger.debug(f"Skipping invalid return for {symbol}: {ret}")
                continue

            if symbol not in self.return_history:
                self.return_history[symbol] = deque(maxlen=self.window_size)
            self.return_history[symbol].append(ret)
        
        # Build correlation matrix if enough data
        symbols = list(self.return_history.keys())
        n_symbols = len(symbols)
        
        if n_symbols < 2:
            return None, {'status': 'insufficient_symbols', 'n_symbols': n_symbols}
        
        # Check minimum samples
        min_len = min(len(self.return_history[s]) for s in symbols)
        if min_len < self.min_samples:
            return None, {
                'status': 'insufficient_data',
                'min_samples': min_len,
                'required': self.min_samples
            }
        
        # Build return matrix [n_samples, n_symbols]
        n_samples = min_len
        return_matrix = np.zeros((n_samples, n_symbols))
        
        for i, symbol in enumerate(symbols):
            returns_list = list(self.return_history[symbol])
            return_matrix[:, i] = returns_list[-n_samples:]
        
        # Calculate correlation matrix
        try:
            self.current_correlation = np.corrcoef(return_matrix.T)
            self.symbols = symbols
        except Exception as e:
            logger.error(f"Correlation calculation failed: {e}")
            return None, {'status': 'calculation_error', 'error': str(e)}
        
        # Detect high correlations
        alerts = self._detect_high_correlations()
        
        # Calculate diversification score
        div_score = self._calculate_diversification_score()
        
        diagnostics = {
            'status': 'success',
            'n_symbols': n_symbols,
            'n_samples': n_samples,
            'high_correlation_alerts': alerts,
            'diversification_score': div_score,
            'avg_correlation': self._get_average_correlation()
        }
        
        return self.current_correlation, diagnostics
    
    def _detect_high_correlations(self) -> List[str]:
        """Detect pairs with high correlation."""
        alerts = []
        
        if self.current_correlation is None:
            return alerts
        
        n = len(self.symbols)
        
        for i in range(n):
            for j in range(i + 1, n):
                corr = abs(self.current_correlation[i, j])
                
                if corr >= self.correlation_alert_threshold:
                    self.high_correlation_alerts += 1
                    alerts.append(
                        f"{self.symbols[i]}/{self.symbols[j]}: {corr:.2f}"
                    )
        
        return alerts
    
    def _calculate_diversification_score(self) -> float:
        """
        Calculate portfolio diversification score.
        
        Score = 1 - average_absolute_correlation
        Higher is better (more diversified)
        """
        if self.current_correlation is None:
            return 1.0
        
        avg_corr = self._get_average_correlation()
        return max(0.0, 1.0 - avg_corr)
    
    def _get_average_correlation(self) -> float:
        """Get average off-diagonal correlation."""
        if self.current_correlation is None:
            return 0.0
        
        n = len(self.symbols)
        if n < 2:
            return 0.0
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(np.abs(self.current_correlation), k=1)
        n_pairs = (n * (n - 1)) / 2
        
        return np.sum(upper_triangle) / n_pairs if n_pairs > 0 else 0.0
    
    def get_correlation_for_pair(
        self,
        symbol1: str,
        symbol2: str
    ) -> Optional[float]:
        """Get correlation between two specific symbols."""
        if self.current_correlation is None:
            return None
        
        if symbol1 not in self.symbols or symbol2 not in self.symbols:
            return None
        
        i = self.symbols.index(symbol1)
        j = self.symbols.index(symbol2)
        
        return float(self.current_correlation[i, j])
    
    def get_matrix_as_dict(self) -> Dict:
        """Get correlation matrix as dictionary."""
        if self.current_correlation is None:
            return {}
        
        result = {}
        for i, sym1 in enumerate(self.symbols):
            result[sym1] = {}
            for j, sym2 in enumerate(self.symbols):
                result[sym1][sym2] = float(self.current_correlation[i, j])
        
        return result
    
    def _cleanup_inactive_symbols(self, active_symbols: set):
        """Remove symbols that haven't been updated recently."""
        inactive = set(self.return_history.keys()) - set(active_symbols)
        for symbol in inactive:
            del self.return_history[symbol]
            logger.debug(f"Removed inactive symbol: {symbol}")

    def get_state(self) -> Dict:
        """Get current state."""
        return {
            'window_size': self.window_size,
            'correlation_alert_threshold': self.correlation_alert_threshold,
            'total_updates': self.total_updates,
            'high_correlation_alerts': self.high_correlation_alerts,
            'n_symbols': len(self.symbols),
            'symbols': self.symbols,
            'diversification_score': self._calculate_diversification_score(),
            'avg_correlation': self._get_average_correlation()
        }
