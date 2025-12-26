"""
HIMARI OPUS V2 - Circuit Breaker for Colab Pro API
===================================================

Circuit breaker with exponential backoff for Colab Pro RL API resilience.

Features:
- Circuit states (CLOSED, OPEN, HALF_OPEN)
- Exponential backoff (30s → 60s → 120s → 300s max)
- Failure threshold tracking
- Comprehensive monitoring and diagnostics

Version: 3.1 Enhanced
"""

import asyncio
import time
from typing import Optional, Dict, Callable, Any
import logging

# Handle both module and script imports
try:
    from core.layer3_types import CircuitState, CircuitBreakerOpenException
    from core import layer3_metrics as metrics
except ImportError:
    from ..core.layer3_types import CircuitState, CircuitBreakerOpenException
    from ..core import layer3_metrics as metrics

logger = logging.getLogger(__name__)


class ColabProCircuitBreaker:
    """
    Circuit breaker for Colab Pro RL API with exponential backoff.
    
    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Too many failures, block requests for timeout period
    - HALF_OPEN: Testing recovery, allow single request
    
    Thresholds:
    - 5 failures within 60s → OPEN
    - Timeout: 30s initial, exponential backoff up to 5 minutes
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        initial_timeout_sec: int = 30,
        max_timeout_sec: int = 300,  # 5 minutes
        backoff_multiplier: float = 2.0,
        enable_metrics: bool = True
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures to trigger OPEN
            initial_timeout_sec: Initial timeout before retry
            max_timeout_sec: Maximum timeout cap
            backoff_multiplier: Exponential backoff multiplier
        """
        self.failure_threshold = failure_threshold
        self.initial_timeout_sec = initial_timeout_sec
        self.max_timeout_sec = max_timeout_sec
        self.backoff_multiplier = backoff_multiplier
        
        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.consecutive_failures = 0
        self.current_timeout = initial_timeout_sec
        
        # Monitoring
        self.total_requests = 0
        self.total_failures = 0
        self.total_fallbacks = 0
        self.enable_metrics = enable_metrics
        
        # Record initial state
        self._record_state_metric()
    
    async def call(self, operation: Callable, *args, **kwargs) -> Any:
        """
        Execute operation with circuit breaker protection.
        
        Args:
            operation: Async callable (e.g., Colab Pro API call)
            *args, **kwargs: Arguments to pass to operation
            
        Returns:
            Result from operation
            
        Raises:
            CircuitBreakerOpenException: If circuit is open
            Other exceptions from operation
        """
        self.total_requests += 1
        
        if self.state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time > self.current_timeout:
                self.state = CircuitState.HALF_OPEN
                self.failure_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                # Circuit still open: reject immediately
                self.total_fallbacks += 1
                raise CircuitBreakerOpenException(
                    f"Circuit breaker OPEN, retry in {self._time_until_retry():.1f}s"
                )
        
        try:
            result = await operation(*args, **kwargs)
            self._on_success()
            return result
        
        except asyncio.TimeoutError as e:
            self._on_failure("timeout")
            raise
        
        except Exception as e:
            self._on_failure(f"error: {type(e).__name__}")
            raise
    
    def _on_success(self):
        """Handle successful request."""
        if self.state == CircuitState.HALF_OPEN:
            # Recovery successful: close circuit
            self.state = CircuitState.CLOSED
            self.current_timeout = self.initial_timeout_sec
            logger.info("Circuit breaker recovered: HALF_OPEN -> CLOSED")
            self._record_state_metric()
        
        # Reset failure counters
        self.failure_count = 0
        self.consecutive_failures = 0
    
    def _on_failure(self, reason: str):
        """Handle failed request."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Record failure metric
        if self.enable_metrics:
            try:
                metrics.layer3_circuit_breaker_failures.inc()
            except:
                pass
        
        if self.state == CircuitState.HALF_OPEN:
            # Half-open test failed: reopen circuit
            self.state = CircuitState.OPEN
            self._increase_timeout()
            self._record_state_metric()
            logger.warning(
                f"Circuit breaker test failed: HALF_OPEN -> OPEN. "
                f"Reason: {reason}. Timeout: {self.current_timeout}s"
            )
        
        elif self.failure_count >= self.failure_threshold:
            # Threshold exceeded: open circuit
            self.state = CircuitState.OPEN
            self._record_state_metric()
            logger.warning(
                f"Circuit breaker OPEN after {self.failure_count} failures. "
                f"Last reason: {reason}. Timeout: {self.current_timeout}s"
            )
    
    def _increase_timeout(self):
        """Exponential backoff for timeout."""
        self.current_timeout = min(
            self.current_timeout * self.backoff_multiplier,
            self.max_timeout_sec
        )
    
    def _time_until_retry(self) -> float:
        """Calculate seconds until circuit can retry."""
        elapsed = time.time() - self.last_failure_time
        return max(0, self.current_timeout - elapsed)
    
    def get_state(self) -> Dict:
        """Get circuit breaker diagnostics."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'consecutive_failures': self.consecutive_failures,
            'current_timeout_sec': self.current_timeout,
            'time_until_retry_sec': self._time_until_retry() if self.state == CircuitState.OPEN else 0,
            'total_requests': self.total_requests,
            'total_failures': self.total_failures,
            'total_fallbacks': self.total_fallbacks,
            'success_rate': (self.total_requests - self.total_failures) / self.total_requests if self.total_requests > 0 else 1.0
        }
    
    def reset(self):
        """Reset circuit breaker to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.consecutive_failures = 0
        self.current_timeout = self.initial_timeout_sec
        self._record_state_metric()
        logger.info("Circuit breaker reset")
    
    def _record_state_metric(self):
        """Record circuit breaker state to Prometheus."""
        if not self.enable_metrics:
            return
        
        try:
            # State: 0=closed, 1=half_open, 2=open
            state_value = {
                CircuitState.CLOSED: 0,
                CircuitState.HALF_OPEN: 1,
                CircuitState.OPEN: 2
            }.get(self.state, 0)
            
            metrics.layer3_circuit_breaker_state.set(state_value)
            metrics.layer3_circuit_breaker_timeout_sec.set(self.current_timeout)
            metrics.layer3_circuit_breaker_success_rate.set(
                (self.total_requests - self.total_failures) / self.total_requests
                if self.total_requests > 0 else 1.0
            )
        except Exception as e:
            logger.debug(f"Metrics recording skipped: {e}")
    
    def record_fallback(self):
        """Record a fallback event."""
        self.total_fallbacks += 1
        if self.enable_metrics:
            try:
                metrics.layer3_circuit_breaker_fallbacks.inc()
            except:
                pass
