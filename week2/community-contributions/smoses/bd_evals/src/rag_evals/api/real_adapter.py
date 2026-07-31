"""Real bd_api adapter: imports public bd_api classes, maps outputs, measures latency."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from importlib.util import find_spec
from typing import Any

from rag_evals.api.protocol import (
    AskRequest,
    AskResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from rag_evals.logging import get_logger

logger = get_logger("rag_evals.adapter")


class BdApiAdapterError(Exception):
    """Error from the bd_api adapter with classification."""

    def __init__(
        self,
        message: str,
        error_type: str = "unknown",
        status_code: int | None = None,
        retryable: bool = False,
        raw_body: str | None = None,
        parsed_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code
        self.retryable = retryable
        self.raw_body = raw_body
        self.parsed_code = parsed_code


def preflight_bd_api() -> None:
    """Fail before dataset creation when the external sibling wrapper is unavailable."""
    if find_spec("bd_api") is None:
        raise RuntimeError(
            "bd_api is not importable. Add the directory containing the external bd_api package "
            "to PYTHONPATH or install it in the active environment; bd_evals does not vendor it."
        )


def _parse_error_body(raw_body: str | None) -> tuple[str | None, bool]:
    """Parse BdApiRequestError.response_body for code and retryable."""
    if not raw_body:
        return None, False
    try:
        data = json.loads(raw_body)
        code = data.get("code")
        retryable = data.get("retryable", False)
        if isinstance(retryable, str):
            retryable = retryable.lower() == "true"
        return code, bool(retryable)
    except (json.JSONDecodeError, TypeError):
        return None, False


def _classify_http_status(status: int | None) -> bool:
    """Classify HTTP status as retryable per D004."""
    if status is None:
        return False
    return status in (429, 503, 504)


class RealBdApiAdapter:
    """Real adapter that imports and uses AsyncBdApiClient.

    This adapter:
    - Converts nested retrieval controls exactly as verified.
    - Records monotonic end-to-end latency around each call.
    - Converts dataclasses to JSON-safe mappings explicitly.
    - Parses BdApiRequestError.response_body as JSON for code and retryable.
    - Records unavailable metrics as absent plus a limitation marker.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self._base_url = base_url
        self._api_key = api_key

    def _create_client(self) -> Any:
        """Create an AsyncBdApiClient. Import is deferred to avoid requiring bd_api for tests."""
        from bd_api import AsyncBdApiClient

        return AsyncBdApiClient(self._base_url, api_key=self._api_key)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        """Call /retrieve through bd_api and map to internal response."""
        from bd_api.errors import BdApiRequestError
        from bd_api.types import RetrievalConfig

        retrieval_config = RetrievalConfig(
            knowledge_base_id=request.knowledge_base_id,
            retrieval_mode=request.retrieval_mode,
            candidate_count=request.candidate_count,
            result_count=request.result_count,
        )

        client = self._create_client()
        start = time.monotonic()

        try:
            async with client:
                result = await client.retrieve.create(
                    question=request.question,
                    retrieval=retrieval_config,
                )
        except BdApiRequestError as exc:
            elapsed = time.monotonic() - start
            code, parsed_retryable = _parse_error_body(exc.response_body)
            retryable = parsed_retryable or _classify_http_status(exc.status_code)
            raise BdApiAdapterError(
                f"Retrieve failed: {exc}",
                error_type="http",
                status_code=exc.status_code,
                retryable=retryable,
                raw_body=exc.response_body,
                parsed_code=code,
            ) from exc
        except Exception as exc:
            if not _known_transport_error(exc):
                raise
            raise BdApiAdapterError(
                f"Retrieve transport error: {exc}",
                error_type="transport",
                retryable=True,
            ) from exc

        elapsed = time.monotonic() - start

        # Convert dataclass to dict
        raw = _dataclass_to_dict(result)

        return RetrieveResponse(
            knowledge_base_id=raw.get("knowledge_base_id", request.knowledge_base_id),
            retrieval_metadata=raw.get("retrieval", {}),
            raw_response=raw.get("raw_retrieval_response", {}),
            normalized_items=raw.get("retrieved_items", []),
            prepared_prompt_input=raw.get("prepared_prompt_input", {}),
            cost=raw.get("cost", {}),
            raw_json=raw.get("raw_json", {}),
            elapsed_seconds=elapsed,
            limitations=[
                "retrieval_token_usage_unavailable",
                "estimated_tokens_only",
            ],
        )

    async def ask(self, request: AskRequest) -> AskResponse:
        """Call /ask through bd_api and map to internal response."""
        if not request.question.strip():
            raise ValueError("AskRequest.question must be non-empty")
        from bd_api.errors import BdApiRequestError

        client = self._create_client()
        start = time.monotonic()

        inference_config = request.inference_config
        retrieval = request.retrieval

        try:
            async with client:
                result = await client.ask.create(
                    question=request.question,
                    model_id=request.model_id,
                    system_instructions=request.system_instructions,
                    user_message=request.user_message,
                    inference_config=inference_config,
                    retrieval=retrieval,
                )
        except BdApiRequestError as exc:
            elapsed = time.monotonic() - start
            code, parsed_retryable = _parse_error_body(exc.response_body)
            retryable = parsed_retryable or _classify_http_status(exc.status_code)
            raise BdApiAdapterError(
                f"Ask failed: {exc}",
                error_type="http",
                status_code=exc.status_code,
                retryable=retryable,
                raw_body=exc.response_body,
                parsed_code=code,
            ) from exc
        except Exception as exc:
            if not _known_transport_error(exc):
                raise
            raise BdApiAdapterError(
                f"Ask transport error: {exc}",
                error_type="transport",
                retryable=True,
            ) from exc

        elapsed = time.monotonic() - start

        raw = _dataclass_to_dict(result)

        return AskResponse(
            text=raw.get("text", ""),
            sources=raw.get("sources", []),
            cost=raw.get("cost", {}),
            usage=raw.get("usage", {}),
            metrics=raw.get("metrics", {}),
            stop_reason=raw.get("stop_reason"),
            persisted=raw.get("persisted_to_s3", False),
            answer_id=raw.get("answer_id"),
            retrieval_metadata=raw.get("retrieval", {}) or {},
            elapsed_seconds=elapsed,
            limitations=[
                "raw_sse_unavailable",
                "start_metadata_unavailable",
                "streaming_timings_unavailable",
            ],
            incidental_retrieval={
                "sources": raw.get("sources", []),
                "retrieval_metadata": raw.get("retrieval", {}) or {},
                "cost": raw.get("cost", {}),
                "used_as_frozen_generation_context": False,
            },
        )


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to a JSON-safe dict."""
    if is_dataclass(obj) and not isinstance(obj, type):
        value = asdict(obj)
        return {str(key): _convert_value(item) for key, item in value.items()}
    if isinstance(obj, dict):
        return {k: _convert_value(v) for k, v in obj.items()}
    raise TypeError(f"Unsupported bd_api response type: {type(obj).__name__}")


def _known_transport_error(exc: Exception) -> bool:
    """Recognize only network failures documented as retryable by D004."""
    return type(exc).__module__.startswith("httpx") and type(exc).__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }


def _convert_value(value: Any) -> Any:
    """Recursively convert values to JSON-safe types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _convert_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_value(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _dataclass_to_dict(value)
    raise TypeError(f"Unsupported nested bd_api value: {type(value).__name__}")
