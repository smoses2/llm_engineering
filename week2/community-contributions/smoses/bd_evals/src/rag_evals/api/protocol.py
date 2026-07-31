"""API adapter protocol and internal response models independent of bd_api dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrieveRequest:
    """Request for /retrieve via the adapter."""

    question: str
    knowledge_base_id: str
    retrieval_mode: str = "standard"
    candidate_count: int | None = None
    result_count: int | None = None


@dataclass(frozen=True)
class AskRequest:
    """Request for /ask via the adapter."""

    model_id: str
    system_instructions: str
    user_message: str
    question: str = ""
    inference_config: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None


@dataclass
class RetrieveResponse:
    """Internal response from /retrieve, independent of bd_api dataclasses."""

    knowledge_base_id: str
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    normalized_items: list[dict[str, Any]] = field(default_factory=list)
    prepared_prompt_input: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    raw_json: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    limitations: list[str] = field(default_factory=list)


@dataclass
class AskResponse:
    """Internal response from /ask, independent of bd_api dataclasses."""

    text: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    cost: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    persisted: bool = False
    answer_id: str | None = None
    retrieval_metadata: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    limitations: list[str] = field(default_factory=list)
    incidental_retrieval: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterError:
    """Classified error from the adapter."""

    error_type: str
    message: str
    status_code: int | None = None
    retryable: bool = False
    raw_body: str | None = None
    parsed_code: str | None = None


@runtime_checkable
class ApiAdapter(Protocol):
    """Narrow protocol for API access. Real and fake adapters implement this."""

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse: ...

    async def ask(self, request: AskRequest) -> AskResponse: ...
