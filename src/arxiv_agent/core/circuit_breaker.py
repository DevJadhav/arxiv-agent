"""Circuit breaker pattern for resilient external API calls.

Implements a state machine to prevent cascading failures when
external services (arXiv, Semantic Scholar, LLMs) are unavailable.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Service is down, fail fast
- HALF_OPEN: Testing if service recovered
"""

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional
import asyncio

from loguru import logger


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit is open."""
    def __init__(self, name: str, time_remaining: float):
        self.name = name
        self.time_remaining = time_remaining
        super().__init__(f"Circuit breaker '{name}' is open. Retry in {time_remaining:.1f}s")


@dataclass
class CircuitBreaker:
    """Circuit breaker for protecting external service calls.
    
    Usage:
        cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
        
        try:
            cb.check_state()
            result = external_api_call()
            cb.record_success()
        except CircuitBreakerOpen:
            return cached_result_or_error()
        except Exception as e:
            cb.record_failure()
            raise
    """
    
    name: str = "default"
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    
    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    
    # Stats
    _total_requests: int = field(default=0, init=False)
    _total_successes: int = field(default=0, init=False)
    _total_failures: int = field(default=0, init=False)
    
    @property
    def state(self) -> str:
        """Get current state as string."""
        with self._lock:
            self._check_state_transition()
            return self._state.value
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED.value
    
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN.value
    
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN.value
    
    def should_allow_request(self) -> bool:
        """Check if a request should be allowed through.
        
        Returns:
            True if request can proceed, False otherwise.
        """
        with self._lock:
            self._check_state_transition()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                return self._half_open_calls < self.half_open_max_calls
    
    def check_state(self) -> None:
        """Check state and raise if circuit is open.
        
        Raises:
            CircuitBreakerOpen: If circuit is open and not accepting requests.
        """
        with self._lock:
            self._check_state_transition()
            self._total_requests += 1
            
            if self._state == CircuitState.OPEN:
                time_remaining = self._time_until_half_open()
                raise CircuitBreakerOpen(self.name, time_remaining)
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(self.name, 0)
                self._half_open_calls += 1
    
    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            # Check for state transitions first
            self._check_state_transition()
            
            self._success_count += 1
            self._total_successes += 1
            self._total_requests += 1
            
            if self._state == CircuitState.HALF_OPEN:
                # Recovery successful, close circuit
                self._transition_to(CircuitState.CLOSED)
            else:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._total_requests += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Recovery failed, reopen circuit
                self._transition_to(CircuitState.OPEN)
            elif self._failure_count >= self.failure_threshold:
                # Threshold reached, open circuit
                self._transition_to(CircuitState.OPEN)

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics.
        
        Returns:
            Dictionary with request/success/failure counts.
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "total_requests": self._total_requests,
                "successes": self._total_successes,
                "failures": self._total_failures,
                "current_failure_streak": self._failure_count,
                "failure_threshold": self.failure_threshold,
            }
    
    def _check_state_transition(self) -> None:
        """Check if state should transition based on time."""
        if self._state == CircuitState.OPEN:
            if self._time_since_last_failure() >= self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        if old_state != new_state:
            logger.warning(
                f"Circuit breaker '{self.name}' transitioned: {old_state.value} -> {new_state.value}"
            )
    
    def _time_since_last_failure(self) -> float:
        """Get time since last failure in seconds."""
        if self._last_failure_time == 0:
            return float('inf')
        return time.time() - self._last_failure_time
    
    def _time_until_half_open(self) -> float:
        """Get time until circuit transitions to half-open."""
        elapsed = self._time_since_last_failure()
        remaining = self.recovery_timeout - elapsed
        return max(0, remaining)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.
    
    Provides a centralized way to get/create circuit breakers
    for different external services.
    """
    
    _instance: Optional["CircuitBreakerRegistry"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "CircuitBreakerRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._breakers = {}
                cls._instance._breaker_lock = threading.RLock()
            return cls._instance
    
    def get(self, name: str, **kwargs) -> CircuitBreaker:
        """Get or create a circuit breaker by name.
        
        Args:
            name: Service name (e.g., "arxiv", "semantic_scholar")
            **kwargs: Optional CircuitBreaker configuration
            
        Returns:
            CircuitBreaker instance for the service.
        """
        with self._breaker_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name, **kwargs)
            return self._breakers[name]
    
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        with self._breaker_lock:
            return {name: cb.get_stats() for name, cb in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers to closed state."""
        with self._breaker_lock:
            for cb in self._breakers.values():
                cb.reset()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    """Get a circuit breaker by name from the global registry.
    
    Args:
        name: Service name
        **kwargs: Optional CircuitBreaker configuration
        
    Returns:
        CircuitBreaker instance.
    """
    return CircuitBreakerRegistry().get(name, **kwargs)


def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> Callable:
    """Decorator to protect a function with a circuit breaker.
    
    Args:
        name: Circuit breaker name/service identifier
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before testing recovery
        
    Returns:
        Decorated function.
        
    Example:
        @circuit_breaker(name="arxiv", failure_threshold=3)
        def fetch_from_arxiv(query):
            return requests.get(f"https://arxiv.org/api/{query}")
    """
    cb = get_circuit_breaker(
        name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                cb.check_state()
                try:
                    result = await func(*args, **kwargs)
                    cb.record_success()
                    return result
                except CircuitBreakerOpen:
                    raise
                except Exception as e:
                    cb.record_failure()
                    raise
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                cb.check_state()
                try:
                    result = func(*args, **kwargs)
                    cb.record_success()
                    return result
                except CircuitBreakerOpen:
                    raise
                except Exception as e:
                    cb.record_failure()
                    raise
            return sync_wrapper
    return decorator
