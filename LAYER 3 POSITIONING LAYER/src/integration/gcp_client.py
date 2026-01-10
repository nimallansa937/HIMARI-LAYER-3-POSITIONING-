"""
HIMARI Layer 3 - GCP RL Client
Client for calling Cloud Run RL API with fallback to local model.
"""

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GCPRLConfig:
    """Configuration for GCP RL Client."""
    api_endpoint: str = ""  # e.g., "https://himari-rl-api-xxx.run.app/predict"
    timeout_ms: int = 150  # API timeout in ms
    enable_fallback: bool = True  # Fallback to local model on timeout
    max_retries: int = 1
    

class GCPRLClient:
    """
    Client for calling HIMARI RL inference API on Cloud Run.
    
    Features:
    - Async HTTP calls with configurable timeout
    - Fallback to Bayesian Kelly on timeout
    - Connection pooling for low latency
    - Automatic retry on transient failures
    """
    
    def __init__(self, config: GCPRLConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._fallback_multiplier = 1.0
        
        if not HTTPX_AVAILABLE:
            logger.warning("httpx not installed, GCP RL client disabled")
            return
            
        if not config.api_endpoint:
            logger.warning("No API endpoint configured, GCP RL client disabled")
            return
            
        self._init_client()
        
    def _init_client(self):
        """Initialize async HTTP client with connection pooling."""
        self._client = httpx.AsyncClient(
            timeout=self.config.timeout_ms / 1000.0,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10
            ),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
        logger.info(f"GCP RL client initialized: {self.config.api_endpoint}")
    
    async def _predict_async(self, state: np.ndarray) -> Tuple[float, float, bool]:
        """
        Make async prediction call to Cloud Run API.
        
        Args:
            state: 16-dimensional state vector
            
        Returns:
            Tuple of (multiplier, latency_ms, used_fallback)
        """
        if self._client is None:
            return self._fallback_multiplier, 0.0, True
            
        import time
        start_time = time.perf_counter()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.post(
                    self.config.api_endpoint,
                    json={"state": state.tolist()}
                )
                response.raise_for_status()
                
                data = response.json()
                multiplier = data.get("position_multiplier", 1.0)
                api_latency = data.get("latency_ms", 0)
                
                total_latency = (time.perf_counter() - start_time) * 1000
                logger.debug(
                    f"GCP RL response: multiplier={multiplier:.3f}, "
                    f"api_latency={api_latency:.2f}ms, total={total_latency:.2f}ms"
                )
                
                return multiplier, total_latency, False
                
            except httpx.TimeoutException:
                logger.warning(
                    f"GCP RL API timeout ({self.config.timeout_ms}ms), "
                    f"attempt {attempt + 1}/{self.config.max_retries + 1}"
                )
                if attempt == self.config.max_retries:
                    break
                    
            except httpx.HTTPError as e:
                logger.warning(f"GCP RL API HTTP error: {e}")
                if attempt == self.config.max_retries:
                    break
                    
            except Exception as e:
                logger.error(f"GCP RL API unexpected error: {e}")
                break
        
        # Use fallback
        total_latency = (time.perf_counter() - start_time) * 1000
        return self._calculate_fallback(state), total_latency, True
    
    def _calculate_fallback(self, state: np.ndarray) -> float:
        """
        Calculate fallback multiplier using Bayesian Kelly.
        
        Simple heuristic based on state features:
        - High confidence + low volatility = higher multiplier
        - Low confidence + high volatility = lower multiplier
        """
        if len(state) < 16:
            return 1.0
            
        confidence = state[0]  # signal_confidence
        volatility = abs(state[13]) if len(state) > 13 else 0.02  # volatility
        win_rate = state[14] if len(state) > 14 else 0.5  # recent_win_rate
        
        # Bayesian Kelly-inspired formula
        # f* = (p*b - q) / b where p=win_rate, q=1-p, b=1 (even odds)
        # Simplified: f* = 2*win_rate - 1 (adjusted for confidence and vol)
        
        kelly_fraction = max(0, 2 * win_rate - 1)
        vol_adjustment = max(0.5, 1.0 - volatility * 5)
        
        multiplier = 0.5 + 1.0 * confidence * kelly_fraction * vol_adjustment
        
        return max(0.0, min(2.0, multiplier))
    
    def predict(self, state: np.ndarray) -> Tuple[float, float, bool]:
        """
        Synchronous wrapper for prediction.
        
        Args:
            state: 16-dimensional state vector
            
        Returns:
            Tuple of (multiplier, latency_ms, used_fallback)
        """
        try:
            return asyncio.run(self._predict_async(state))
        except Exception as e:
            logger.error(f"GCP RL prediction failed: {e}")
            return self._calculate_fallback(state), 0.0, True
    
    async def health_check(self) -> bool:
        """Check if the API is healthy."""
        if self._client is None:
            return False
            
        try:
            # Derive health endpoint from predict endpoint
            health_url = self.config.api_endpoint.rsplit("/", 1)[0] + "/health"
            response = await self._client.get(health_url)
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            

# Singleton client for global access
_gcp_client: Optional[GCPRLClient] = None


def init_gcp_client(config: GCPRLConfig) -> GCPRLClient:
    """Initialize the global GCP RL client."""
    global _gcp_client
    _gcp_client = GCPRLClient(config)
    return _gcp_client


def get_gcp_client() -> Optional[GCPRLClient]:
    """Get the global GCP RL client."""
    return _gcp_client


def get_rl_multiplier(state: np.ndarray) -> Tuple[float, float, bool]:
    """
    Get RL multiplier from GCP API.
    
    Args:
        state: 16-dimensional state vector
        
    Returns:
        Tuple of (multiplier, latency_ms, used_fallback)
    """
    if _gcp_client is None:
        # No client initialized, use fallback
        return 1.0, 0.0, True
        
    return _gcp_client.predict(state)
