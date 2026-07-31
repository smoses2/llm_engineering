"""Retry executor implementing D004 with injectable sleeper/RNG/clock and async semaphore."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rag_evals.logging import get_logger

logger = get_logger("rag_evals.retry")

MAX_ATTEMPTS = 3
BASE_DELAY = 1.0
MAX_DELAY = 30.0
RETRYABLE_STATUS_CODES = {429, 503, 504}


@dataclass
class AttemptRecord:
    """Record of a single attempt."""

    attempt: int
    success: bool
    error_type: str | None = None
    error_message: str | None = None
    retryable: bool = False
    delay_seconds: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class RetryResult:
    """Result of a retry sequence."""

    success: bool
    attempts: list[AttemptRecord] = field(default_factory=list)
    response: Any = None
    total_elapsed: float = 0.0


def _compute_delay(attempt: int, rng: random.Random) -> float:
    """Compute exponential backoff with full jitter: min(30, 1 * 2^(attempt-1)) * random()."""
    base = min(MAX_DELAY, BASE_DELAY * (2 ** (attempt - 1)))
    result: float = base * rng.random()
    return result


def _is_retryable_error(exc: Exception) -> bool:
    """Check if an error is retryable per D004."""
    # Check for BdApiAdapterError or FakeAdapterError
    retryable = getattr(exc, "retryable", False)
    status_code = getattr(exc, "status_code", None)
    error_type = getattr(exc, "error_type", "unknown")

    if retryable:
        return True
    if status_code in RETRYABLE_STATUS_CODES:
        return True
    return error_type in ("transport", "timeout", "connection")


async def execute_with_retries(
    func: Callable[[], Any],
    max_attempts: int = MAX_ATTEMPTS,
    rng: random.Random | None = None,
    sleeper: Callable[[float], Any] | None = None,
    attempt_semaphore: asyncio.Semaphore | None = None,
) -> RetryResult:
    """Execute a coroutine with retry logic per D004.

    - Retries only explicit retryable=true, transport failures, and 429/503/504.
    - Exponential delay: min(30, 1 * 2^(attempt-1)) with full jitter.
    - Never retries schema/400 errors.
    - Persists each attempt record.
    """
    if rng is None:
        rng = random.Random()
    if sleeper is None:

        async def _default_sleep(delay: float) -> None:
            await asyncio.sleep(delay)

        sleeper = _default_sleep

    result = RetryResult(success=False)
    total_start = asyncio.get_event_loop().time()

    for attempt in range(1, max_attempts + 1):
        attempt_start = asyncio.get_event_loop().time()
        try:
            if attempt_semaphore is None:
                response = await func()
            else:
                async with attempt_semaphore:
                    response = await func()
            elapsed = asyncio.get_event_loop().time() - attempt_start
            result.attempts.append(
                AttemptRecord(attempt=attempt, success=True, elapsed_seconds=elapsed)
            )
            result.success = True
            result.response = response
            result.total_elapsed = asyncio.get_event_loop().time() - total_start
            return result
        except Exception as exc:
            elapsed = asyncio.get_event_loop().time() - attempt_start
            retryable = _is_retryable_error(exc)
            error_type = getattr(exc, "error_type", type(exc).__name__)
            result.attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    success=False,
                    error_type=error_type,
                    error_message=str(exc),
                    retryable=retryable,
                    elapsed_seconds=elapsed,
                )
            )

            if not retryable or attempt >= max_attempts:
                result.total_elapsed = asyncio.get_event_loop().time() - total_start
                return result

            delay = _compute_delay(attempt, rng)
            result.attempts[-1].delay_seconds = delay
            logger.info(
                "Attempt %d failed (retryable), retrying in %.2fs: %s",
                attempt,
                delay,
                exc,
            )
            await sleeper(delay)

    result.total_elapsed = asyncio.get_event_loop().time() - total_start
    return result


class ConcurrencyExecutor:
    """Async semaphore-based executor with configurable concurrency.

    Default concurrency is 5 per D005.
    """

    def __init__(self, concurrency: int = 5) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self._concurrency = concurrency

    @property
    def concurrency(self) -> int:
        return self._concurrency

    async def execute(
        self,
        func: Callable[[], Any],
    ) -> Any:
        """Execute a coroutine under the semaphore."""
        async with self._semaphore:
            return await func()
