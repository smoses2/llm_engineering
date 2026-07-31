"""Deterministic fake adapter with configurable failures and call recording."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from rag_evals.api.protocol import (
    AdapterError,
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
)


@dataclass
class FakeCall:
    """Recorded call to the fake adapter."""

    method: str
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class RetrieveFailureConfig:
    """Configurable failure for retrieve calls."""

    fail_first_n: int = 0
    error_type: str = "transport"
    message: str = "Simulated failure"
    status_code: int | None = None
    retryable: bool = True


@dataclass
class FakeAdapter:
    """Deterministic fake adapter for testing.

    - Records all calls.
    - Can simulate failures for the first N calls.
    - Generates deterministic responses based on request content.
    """

    fail_retrieve_first_n: int = 0
    fail_ask_first_n: int = 0
    retrieve_error: AdapterError | None = None
    ask_error: AdapterError | None = None
    calls: list[FakeCall] = field(default_factory=list)
    _retrieve_call_count: int = 0
    _ask_call_count: int = 0
    _sleeper: Callable[[float], Awaitable[None]] | None = None

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Generate a deterministic retrieve response."""
        self._retrieve_call_count += 1
        start = time.monotonic()

        call = FakeCall(method="retrieve", request=_request_to_dict_retrieve(request))
        self.calls.append(call)

        if self._retrieve_call_count <= self.fail_retrieve_first_n and self.retrieve_error:
            call.error = self.retrieve_error.message
            call.elapsed = time.monotonic() - start
            raise _make_error(self.retrieve_error)

        # Generate deterministic response
        context = (
            f"Context for: {request.question}\n"
            f"KB: {request.knowledge_base_id}\n"
            f"Mode: {request.retrieval_mode}"
        )
        context_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()

        response = RetrieveResponse(
            knowledge_base_id=request.knowledge_base_id,
            retrieval_metadata={
                "requested_mode": request.retrieval_mode,
                "effective_mode": request.retrieval_mode,
                "fallback_used": False,
                "candidate_count": request.candidate_count or 10,
                "result_count": request.result_count or 5,
            },
            raw_response={"question": request.question},
            normalized_items=[
                {"content": f"Chunk {i}", "score": 0.9 - i * 0.1}
                for i in range(request.result_count or 5)
            ],
            prepared_prompt_input={"promptVariables": {"context": context}},
            cost={"retrieval_estimate": 0.001, "reranking_estimate": 0.0},
            raw_json={"question": request.question, "context_hash": context_hash},
            elapsed_seconds=time.monotonic() - start,
            limitations=[
                "retrieval_token_usage_unavailable",
                "estimated_tokens_only",
            ],
        )

        call.response = {"knowledge_base_id": response.knowledge_base_id}
        call.elapsed = response.elapsed_seconds
        return response

    async def ask(self, request: AskRequest) -> AskResponse:
        """Generate a deterministic ask response."""
        self._ask_call_count += 1
        start = time.monotonic()

        call = FakeCall(method="ask", request=_request_to_dict_ask(request))
        self.calls.append(call)

        if self._ask_call_count <= self.fail_ask_first_n and self.ask_error:
            call.error = self.ask_error.message
            call.elapsed = time.monotonic() - start
            raise _make_error(self.ask_error)

        # Generate deterministic response
        text = f"Answer to: {request.user_message[:50]}..."

        response = AskResponse(
            text=text,
            sources=[{"uri": "s3://bucket/doc1", "title": "Doc 1"}],
            cost={"inference": 0.002, "total": 0.002},
            usage={"inputTokens": 100, "outputTokens": 50, "totalTokens": 150},
            metrics={"latencyMs": 500},
            stop_reason="stop",
            persisted=False,
            answer_id=None,
            retrieval_metadata={},
            elapsed_seconds=time.monotonic() - start,
            limitations=["raw_sse_unavailable", "start_metadata_unavailable"],
            incidental_retrieval={"used_as_frozen_generation_context": False},
        )

        call.response = {"text": response.text}
        call.elapsed = response.elapsed_seconds
        return response

    @property
    def retrieve_call_count(self) -> int:
        return self._retrieve_call_count

    @property
    def ask_call_count(self) -> int:
        return self._ask_call_count

    def reset(self) -> None:
        """Reset call counts and records."""
        self.calls.clear()
        self._retrieve_call_count = 0
        self._ask_call_count = 0


class FakeAdapterError(Exception):
    """Error raised by the fake adapter."""

    def __init__(self, error: AdapterError) -> None:
        super().__init__(error.message)
        self.adapter_error = error
        self.error_type = error.error_type
        self.status_code = error.status_code
        self.retryable = error.retryable


def _make_error(error: AdapterError) -> FakeAdapterError:
    """Create a fake adapter error."""
    return FakeAdapterError(error)


def _request_to_dict_retrieve(request: RetrieveRequest) -> dict[str, Any]:
    return {
        "question": request.question,
        "knowledge_base_id": request.knowledge_base_id,
        "retrieval_mode": request.retrieval_mode,
        "candidate_count": request.candidate_count,
        "result_count": request.result_count,
    }


def _request_to_dict_ask(request: AskRequest) -> dict[str, Any]:
    return {
        "model_id": request.model_id,
        "system_instructions": request.system_instructions[:100],
        "user_message_length": len(request.user_message),
        "inference_config": request.inference_config,
    }
