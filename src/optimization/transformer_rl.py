"""
HIMARI OPUS V2 - Transformer-RL Client
========================================

Circuit breaker-protected client for optional Transformer-RL predictions
from Colab Pro endpoint.

Features:
- Circuit breaker protection
- Configurable timeout
- Automatic fallback to None
- Prediction confidence thresholds
- Prometheus metrics

Version: 3.1 Phase 3
"""

import asyncio
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
import aiohttp

try:
    from core.layer3_types import TacticalSignal, CircuitBreakerOpenException
    from risk.circuit_breaker import ColabProCircuitBreaker
    from core import layer3_metrics as metrics
except ImportError:
    from ..core.layer3_types import TacticalSignal, CircuitBreakerOpenException
    from ..risk.circuit_breaker import ColabProCircuitBreaker
    from ..core import layer3_metrics as metrics

logger = logging.getLogger(__name__)


@dataclass
class RLPrediction:
    """Transformer-RL prediction result."""
    optimal_position_pct: float
    confidence: float
    predicted_return: float
    predicted_volatility: float
    model_version: str
    latency_ms: float
    timestamp_ns: int


class TransformerRLClient:
    """
    Circuit breaker-protected client for Transformer-RL predictions.
    
    Connects to Colab Pro endpoint for advanced RL-based position sizing.
    Automatically falls back when endpoint is unavailable.
    
    Usage:
        client = TransformerRLClient(endpoint_url="...")
        prediction = await client.predict(signal)
        if prediction is None:
            # Fallback to Phase 1/2
    """
    
    def __init__(
        self,
        endpoint_url: str = "http://localhost:8888/predict",
        timeout_sec: float = 5.0,
        min_confidence: float = 0.5,
        circuit_breaker: Optional[ColabProCircuitBreaker] = None,
        enable_metrics: bool = True
    ):
        """
        Initialize Transformer-RL client.
        
        Args:
            endpoint_url: RL prediction endpoint URL
            timeout_sec: Request timeout in seconds
            min_confidence: Minimum confidence to accept prediction
            circuit_breaker: Custom circuit breaker (creates default if None)
            enable_metrics: Enable Prometheus metrics
        """
        self.endpoint_url = endpoint_url
        self.timeout_sec = timeout_sec
        self.min_confidence = min_confidence
        self.enable_metrics = enable_metrics
        
        # Circuit breaker
        self.circuit_breaker = circuit_breaker or ColabProCircuitBreaker(
            failure_threshold=3,
            initial_timeout_sec=30,
            max_timeout_sec=300
        )
        
        # Statistics
        self.total_predictions = 0
        self.successful_predictions = 0
        self.fallback_predictions = 0
        self.low_confidence_rejections = 0
        
        logger.info(
            f"TransformerRLClient initialized: endpoint={endpoint_url}, "
            f"timeout={timeout_sec}s"
        )
    
    async def predict(
        self,
        signal: TacticalSignal,
        market_data: Optional[Dict] = None
    ) -> Optional[RLPrediction]:
        """
        Get RL prediction for a trading signal.
        
        Args:
            signal: TacticalSignal from Layer 2
            market_data: Optional additional market context
            
        Returns:
            RLPrediction if successful, None if fallback needed
        """
        self.total_predictions += 1
        start_time = time.time()
        
        try:
            # Execute with circuit breaker protection
            result = await self.circuit_breaker.call(
                self._make_prediction_request,
                signal,
                market_data
            )
            
            if result is None:
                self.fallback_predictions += 1
                return None
            
            # Check confidence threshold
            if result.confidence < self.min_confidence:
                self.low_confidence_rejections += 1
                logger.info(
                    f"RL prediction rejected: confidence {result.confidence:.2f} < "
                    f"threshold {self.min_confidence}"
                )
                return None
            
            self.successful_predictions += 1
            
            # Record metrics
            if self.enable_metrics:
                try:
                    metrics.layer3_circuit_breaker_success_rate.set(
                        self.circuit_breaker.get_state()['success_rate']
                    )
                except:
                    pass
            
            return result
            
        except CircuitBreakerOpenException as e:
            self.fallback_predictions += 1
            logger.warning(f"Circuit breaker open: {e}")
            return None
            
        except asyncio.TimeoutError:
            self.fallback_predictions += 1
            logger.warning("RL prediction timeout")
            return None
            
        except Exception as e:
            self.fallback_predictions += 1
            logger.error(f"RL prediction error: {e}")
            return None
    
    async def _make_prediction_request(
        self,
        signal: TacticalSignal,
        market_data: Optional[Dict]
    ) -> Optional[RLPrediction]:
        """Make actual HTTP request to RL endpoint."""
        start_time = time.time()
        
        payload = {
            "symbol": signal.symbol,
            "action": signal.action.value,
            "confidence": signal.confidence,
            "risk_score": signal.risk_score,
            "regime": signal.regime.value,
            "expected_return": signal.expected_return,
            "predicted_volatility": signal.predicted_volatility,
            "market_data": market_data or {}
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.endpoint_url,
                json=payload
            ) as response:
                
                if response.status != 200:
                    logger.warning(f"RL endpoint returned status {response.status}")
                    return None
                
                data = await response.json()
                
                latency_ms = (time.time() - start_time) * 1000
                
                return RLPrediction(
                    optimal_position_pct=data.get("position_pct", 0.0),
                    confidence=data.get("confidence", 0.0),
                    predicted_return=data.get("predicted_return", 0.0),
                    predicted_volatility=data.get("predicted_volatility", 0.01),
                    model_version=data.get("model_version", "unknown"),
                    latency_ms=latency_ms,
                    timestamp_ns=time.time_ns()
                )
    
    def predict_sync(
        self,
        signal: TacticalSignal,
        market_data: Optional[Dict] = None
    ) -> Optional[RLPrediction]:
        """
        Synchronous wrapper for predict().
        
        For use in non-async contexts.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - return None as fallback
                self.fallback_predictions += 1
                return None
            return loop.run_until_complete(self.predict(signal, market_data))
        except RuntimeError:
            # No event loop - create one
            return asyncio.run(self.predict(signal, market_data))
    
    def get_state(self) -> Dict:
        """Get client state and statistics."""
        return {
            "endpoint_url": self.endpoint_url,
            "timeout_sec": self.timeout_sec,
            "min_confidence": self.min_confidence,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "fallback_predictions": self.fallback_predictions,
            "low_confidence_rejections": self.low_confidence_rejections,
            "success_rate": (
                self.successful_predictions / self.total_predictions
                if self.total_predictions > 0 else 0.0
            ),
            "circuit_breaker": self.circuit_breaker.get_state()
        }


class MockTransformerRLClient(TransformerRLClient):
    """
    Mock client for testing without actual RL endpoint.
    
    Returns simulated predictions based on signal properties.
    """
    
    def __init__(self, success_rate: float = 0.9, **kwargs):
        # Don't call super().__init__ to avoid endpoint URL requirement
        self.endpoint_url = "mock://localhost"
        self.timeout_sec = 1.0
        self.min_confidence = 0.5
        self.enable_metrics = False
        self.success_rate = success_rate
        
        self.circuit_breaker = ColabProCircuitBreaker(
            failure_threshold=3,
            initial_timeout_sec=5
        )
        
        self.total_predictions = 0
        self.successful_predictions = 0
        self.fallback_predictions = 0
        self.low_confidence_rejections = 0
        
        import random
        self._random = random
    
    async def predict(
        self,
        signal: TacticalSignal,
        market_data: Optional[Dict] = None
    ) -> Optional[RLPrediction]:
        """Generate mock prediction."""
        self.total_predictions += 1
        
        # Simulate occasional failures
        if self._random.random() > self.success_rate:
            self.fallback_predictions += 1
            return None
        
        # Generate mock prediction based on signal
        base_pct = signal.confidence * 0.15  # Max 15% position
        
        prediction = RLPrediction(
            optimal_position_pct=base_pct * (1 + self._random.gauss(0, 0.1)),
            confidence=signal.confidence * (0.8 + self._random.random() * 0.2),
            predicted_return=signal.expected_return or 0.05,
            predicted_volatility=signal.predicted_volatility or 0.03,
            model_version="mock-v1.0",
            latency_ms=self._random.uniform(10, 100),
            timestamp_ns=time.time_ns()
        )
        
        self.successful_predictions += 1
        return prediction
