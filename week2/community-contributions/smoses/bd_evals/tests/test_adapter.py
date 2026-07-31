"""Tests for the API adapter: fake adapter, retry executor, concurrency."""

import asyncio
import random

import pytest

from rag_evals.api.fake_adapter import FakeAdapter, FakeAdapterError
from rag_evals.api.protocol import (
    AdapterError,
    AskRequest,
    RetrieveRequest,
)
from rag_evals.api.retry import (
    ConcurrencyExecutor,
    execute_with_retries,
)

# ---------------------------------------------------------------------------
# Fake adapter tests
# ---------------------------------------------------------------------------


class TestFakeAdapterRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_success(self) -> None:
        adapter = FakeAdapter()
        req = RetrieveRequest(
            question="What is X?",
            knowledge_base_id="KB1",
            retrieval_mode="standard",
            candidate_count=20,
            result_count=5,
        )
        resp = await adapter.retrieve(req)
        assert resp.knowledge_base_id == "KB1"
        assert "context" in resp.prepared_prompt_input.get("promptVariables", {})
        assert len(resp.normalized_items) == 5
        assert resp.elapsed_seconds >= 0
        assert "retrieval_token_usage_unavailable" in resp.limitations

    @pytest.mark.asyncio
    async def test_retrieve_records_calls(self) -> None:
        adapter = FakeAdapter()
        await adapter.retrieve(RetrieveRequest(question="Q1", knowledge_base_id="KB1"))
        await adapter.retrieve(RetrieveRequest(question="Q2", knowledge_base_id="KB2"))
        assert len(adapter.calls) == 2
        assert adapter.calls[0].method == "retrieve"
        assert adapter.retrieve_call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_deterministic(self) -> None:
        adapter1 = FakeAdapter()
        adapter2 = FakeAdapter()
        req = RetrieveRequest(question="Same Q", knowledge_base_id="KB1")
        r1 = await adapter1.retrieve(req)
        r2 = await adapter2.retrieve(req)
        assert r1.raw_json == r2.raw_json

    @pytest.mark.asyncio
    async def test_retrieve_fails_then_succeeds(self) -> None:
        error = AdapterError(
            error_type="transport",
            message="Connection failed",
            retryable=True,
        )
        adapter = FakeAdapter(fail_retrieve_first_n=2, retrieve_error=error)
        req = RetrieveRequest(question="Q", knowledge_base_id="KB1")

        with pytest.raises(FakeAdapterError):
            await adapter.retrieve(req)
        with pytest.raises(FakeAdapterError):
            await adapter.retrieve(req)
        resp = await adapter.retrieve(req)
        assert resp.knowledge_base_id == "KB1"
        assert adapter.retrieve_call_count == 3


class TestFakeAdapterAsk:
    @pytest.mark.asyncio
    async def test_ask_success(self) -> None:
        adapter = FakeAdapter()
        req = AskRequest(
            model_id="model1",
            system_instructions="You are a judge.",
            user_message="Evaluate this.",
        )
        resp = await adapter.ask(req)
        assert resp.text.startswith("Answer to:")
        assert resp.usage["totalTokens"] == 150
        assert resp.metrics["latencyMs"] == 500
        assert resp.incidental_retrieval["used_as_frozen_generation_context"] is False

    @pytest.mark.asyncio
    async def test_ask_records_calls(self) -> None:
        adapter = FakeAdapter()
        await adapter.ask(AskRequest(model_id="m1", system_instructions="s", user_message="u"))
        assert len(adapter.calls) == 1
        assert adapter.ask_call_count == 1

    @pytest.mark.asyncio
    async def test_ask_fails_then_succeeds(self) -> None:
        error = AdapterError(error_type="http", message="503", status_code=503, retryable=True)
        adapter = FakeAdapter(fail_ask_first_n=1, ask_error=error)
        with pytest.raises(FakeAdapterError):
            await adapter.ask(AskRequest(model_id="m", system_instructions="s", user_message="u"))
        resp = await adapter.ask(
            AskRequest(model_id="m", system_instructions="s", user_message="u")
        )
        assert resp.text != ""


# ---------------------------------------------------------------------------
# Retry executor tests
# ---------------------------------------------------------------------------


class TestRetryExecutor:
    @pytest.mark.asyncio
    async def test_success_first_try(self) -> None:
        async def func() -> str:
            return "ok"

        result = await execute_with_retries(func, rng=random.Random(42))
        assert result.success
        assert len(result.attempts) == 1
        assert result.response == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_retryable(self) -> None:
        call_count = 0

        async def func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise FakeAdapterError(
                    AdapterError(error_type="transport", message="fail", retryable=True)
                )
            return "ok"

        async def no_sleep(delay: float) -> None:
            pass

        result = await execute_with_retries(func, rng=random.Random(42), sleeper=no_sleep)
        assert result.success
        assert len(result.attempts) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable(self) -> None:
        async def func() -> str:
            raise FakeAdapterError(
                AdapterError(error_type="schema", message="bad request", retryable=False)
            )

        async def no_sleep(delay: float) -> None:
            pass

        result = await execute_with_retries(func, rng=random.Random(42), sleeper=no_sleep)
        assert not result.success
        assert len(result.attempts) == 1

    @pytest.mark.asyncio
    async def test_max_attempts_exhausted(self) -> None:
        async def func() -> str:
            raise FakeAdapterError(
                AdapterError(error_type="transport", message="always fails", retryable=True)
            )

        async def no_sleep(delay: float) -> None:
            pass

        result = await execute_with_retries(
            func, max_attempts=3, rng=random.Random(42), sleeper=no_sleep
        )
        assert not result.success
        assert len(result.attempts) == 3

    @pytest.mark.asyncio
    async def test_retry_on_503(self) -> None:
        call_count = 0

        async def func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FakeAdapterError(
                    AdapterError(error_type="http", message="503", status_code=503, retryable=True)
                )
            return "ok"

        async def no_sleep(delay: float) -> None:
            pass

        result = await execute_with_retries(func, rng=random.Random(42), sleeper=no_sleep)
        assert result.success
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self) -> None:
        async def func() -> str:
            raise FakeAdapterError(
                AdapterError(
                    error_type="http", message="400 Bad Request", status_code=400, retryable=False
                )
            )

        async def no_sleep(delay: float) -> None:
            pass

        result = await execute_with_retries(func, rng=random.Random(42), sleeper=no_sleep)
        assert not result.success
        assert len(result.attempts) == 1

    @pytest.mark.asyncio
    async def test_delay_jitter(self) -> None:
        from rag_evals.api.retry import _compute_delay

        rng = random.Random(42)
        delays = [_compute_delay(1, rng) for _ in range(10)]
        assert all(0 <= d <= 30.0 for d in delays)
        # With full jitter, delays should vary
        assert len(set(delays)) > 1


# ---------------------------------------------------------------------------
# Concurrency executor tests
# ---------------------------------------------------------------------------


class TestConcurrencyExecutor:
    @pytest.mark.asyncio
    async def test_executes_under_semaphore(self) -> None:
        executor = ConcurrencyExecutor(concurrency=2)
        current = 0
        max_concurrent = 0

        async def task() -> str:
            nonlocal current, max_concurrent
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.01)
            current -= 1
            return "ok"

        tasks = [executor.execute(task) for _ in range(10)]
        results = await asyncio.gather(*tasks)
        assert all(r == "ok" for r in results)
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_default_concurrency_5(self) -> None:
        executor = ConcurrencyExecutor()
        assert executor.concurrency == 5

    @pytest.mark.asyncio
    async def test_concurrency_1_serial(self) -> None:
        executor = ConcurrencyExecutor(concurrency=1)
        order: list[int] = []

        async def task(i: int) -> int:
            order.append(i)
            await asyncio.sleep(0.001)
            return i

        tasks = [executor.execute(lambda i=i: task(i)) for i in range(5)]
        await asyncio.gather(*tasks)
        assert order == [0, 1, 2, 3, 4]
