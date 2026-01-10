"""
Unit tests for Circuit Breaker
"""

import pytest
import asyncio
import sys
sys.path.insert(0, 'src')

from risk.circuit_breaker import ColabProCircuitBreaker
from core.layer3_types import CircuitState, CircuitBreakerOpenException


class TestColabProCircuitBreaker:
    """Test suite for Circuit Breaker."""
    
    def test_initialization(self):
        """Test circuit breaker initialization."""
        cb = ColabProCircuitBreaker(
            failure_threshold=5,
            initial_timeout_sec=30,
            max_timeout_sec=300
        )
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_threshold == 5
        assert cb.initial_timeout_sec == 30
        assert cb.max_timeout_sec == 300
    
    def test_successful_call(self):
        """Test successful operation keeps circuit closed."""
        cb = ColabProCircuitBreaker()
        
        async def success_operation():
            return "success"
        
        result = asyncio.get_event_loop().run_until_complete(cb.call(success_operation))
        
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_failure_increments_count(self):
        """Test failure increments failure count."""
        cb = ColabProCircuitBreaker(failure_threshold=5)
        
        async def fail_operation():
            raise ValueError("Test failure")
        
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(cb.call(fail_operation))
        
        assert cb.failure_count == 1
        assert cb.total_failures == 1
        assert cb.state == CircuitState.CLOSED
    
    def test_threshold_opens_circuit(self):
        """Test that reaching threshold opens circuit."""
        cb = ColabProCircuitBreaker(failure_threshold=3)
        
        async def fail_operation():
            raise ValueError("Test failure")
        
        for _ in range(3):
            with pytest.raises(ValueError):
                asyncio.get_event_loop().run_until_complete(cb.call(fail_operation))
        
        assert cb.state == CircuitState.OPEN
    
    def test_get_state(self):
        """Test state retrieval."""
        cb = ColabProCircuitBreaker()
        
        state = cb.get_state()
        
        assert state['state'] == 'closed'
        assert state['failure_count'] == 0
        assert state['success_rate'] == 1.0
    
    def test_reset(self):
        """Test circuit breaker reset."""
        cb = ColabProCircuitBreaker()
        cb.failure_count = 3
        cb.state = CircuitState.OPEN
        
        cb.reset()
        
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_exponential_backoff(self):
        """Test timeout increases exponentially."""
        cb = ColabProCircuitBreaker(
            initial_timeout_sec=30,
            max_timeout_sec=300,
            backoff_multiplier=2.0,
            enable_metrics=False
        )
        
        assert cb.current_timeout == 30
        
        cb._increase_timeout()
        assert cb.current_timeout == 60
        
        cb._increase_timeout()
        assert cb.current_timeout == 120
        
        cb._increase_timeout()
        assert cb.current_timeout == 240
        
        cb._increase_timeout()
        assert cb.current_timeout == 300  # Capped at max
    
    def test_half_open_success_closes_circuit(self):
        """Test HALF_OPEN -> CLOSED transition on success (Issue #6)."""
        cb = ColabProCircuitBreaker(failure_threshold=2, enable_metrics=False)
        
        # Force into HALF_OPEN state
        cb.state = CircuitState.HALF_OPEN
        
        async def success_operation():
            return "success"
        
        result = asyncio.get_event_loop().run_until_complete(cb.call(success_operation))
        
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_half_open_failure_reopens_circuit(self):
        """Test HALF_OPEN -> OPEN transition on failure."""
        cb = ColabProCircuitBreaker(
            failure_threshold=2, 
            initial_timeout_sec=1,
            enable_metrics=False
        )
        
        # Force into HALF_OPEN state
        cb.state = CircuitState.HALF_OPEN
        
        async def fail_operation():
            raise ValueError("Test failure")
        
        with pytest.raises(ValueError):
            asyncio.get_event_loop().run_until_complete(cb.call(fail_operation))
        
        assert cb.state == CircuitState.OPEN
    
    def test_circuit_breaker_open_exception(self):
        """Test CircuitBreakerOpenException is raised when circuit is OPEN (Issue #7)."""
        cb = ColabProCircuitBreaker(
            failure_threshold=2,
            initial_timeout_sec=60,  # Long timeout
            enable_metrics=False
        )
        
        async def fail_operation():
            raise ValueError("Test failure")
        
        # Trigger failures to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                asyncio.get_event_loop().run_until_complete(cb.call(fail_operation))
        
        assert cb.state == CircuitState.OPEN
        
        # Next call should raise CircuitBreakerOpenException
        async def any_operation():
            return "should not reach"
        
        with pytest.raises(CircuitBreakerOpenException):
            asyncio.get_event_loop().run_until_complete(cb.call(any_operation))
    
    def test_record_fallback(self):
        """Test fallback recording method."""
        cb = ColabProCircuitBreaker(enable_metrics=False)
        
        assert cb.total_fallbacks == 0
        
        cb.record_fallback()
        cb.record_fallback()
        
        assert cb.total_fallbacks == 2


class TestCircuitBreakerAsync:
    """Async tests for circuit breaker (Issue #4)."""
    
    def test_async_successful_call(self):
        """Test async successful operation."""
        cb = ColabProCircuitBreaker(enable_metrics=False)
        
        async def async_success():
            await asyncio.sleep(0.01)
            return "async success"
        
        result = asyncio.run(cb.call(async_success))
        
        assert result == "async success"
        assert cb.state == CircuitState.CLOSED
    
    def test_async_timeout_handling(self):
        """Test async timeout is handled."""
        cb = ColabProCircuitBreaker(failure_threshold=2, enable_metrics=False)
        
        async def slow_operation():
            await asyncio.sleep(10)  # Very slow
            return "never reached"
        
        # Wrap with timeout
        async def timed_operation():
            try:
                return await asyncio.wait_for(slow_operation(), timeout=0.05)
            except asyncio.TimeoutError:
                raise
        
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(cb.call(timed_operation))
        
        assert cb.failure_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

