"""Strict Pydantic schemas for catalogs, config blocks, and evaluation definitions."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants and patterns
# ---------------------------------------------------------------------------

DATASET_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,62}$"
_CATALOG_ID_PATTERN = r"^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,63}$"
_PLACEHOLDER_PREFIXES = ("KB_REPLACE_ME", "MODEL_REPLACE_ME")

_RETRIEVAL_MODES = ("standard", "rerank")
_CHUNKING_STRATEGIES = ("fixed", "semantic", "none")


def _is_placeholder(value: str) -> bool:
    return any(value.startswith(p) for p in _PLACEHOLDER_PREFIXES)


# ---------------------------------------------------------------------------
# Strict base model
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class _ExtensibleModel(BaseModel):
    """Base model that allows documented extension fields."""

    model_config = ConfigDict(extra="allow", strict=True)


# ---------------------------------------------------------------------------
# Source reference
# ---------------------------------------------------------------------------


class SourceRef(_StrictModel):
    """Reference to an external YAML/Markdown file."""

    source: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Rubric (extensible, no scores)
# ---------------------------------------------------------------------------


class Rubric(_ExtensibleModel):
    """Optional, extensible question rubric. No score fields are predefined."""

    expected_answer: str | None = None
    required_facts: list[str] | None = None
    expected_sources: list[str] | None = None
    unacceptable_claims: list[str] | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Question catalog
# ---------------------------------------------------------------------------


class Question(_StrictModel):
    """A single evaluation question with optional rubric."""

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    question: str = Field(min_length=1)
    rubric: Rubric | None = None


class QuestionCatalog(_StrictModel):
    """Catalog of evaluation questions loaded from YAML."""

    schema_version: Literal[1] = 1
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> QuestionCatalog:
        ids = [q.id for q in self.questions]
        dupes = _find_duplicates(ids)
        if dupes:
            raise ValueError(f"Duplicate question IDs: {sorted(dupes)}")
        return self


# ---------------------------------------------------------------------------
# Knowledge Base catalog
# ---------------------------------------------------------------------------


class ChunkingConfig(_StrictModel):
    """Chunking strategy metadata for a Knowledge Base.

    These values are descriptive provenance recorded with every artifact. They are
    never sent to the API, which receives only the Knowledge Base ID, retrieval
    mode, and counts. ``overlap`` is a percentage for fixed-size chunking.
    """

    strategy: str = Field(pattern=f"^({'|'.join(_CHUNKING_STRATEGIES)})$")
    size: int | None = None
    overlap: int | None = None
    similarity_percentile_threshold: int | None = None

    @model_validator(mode="after")
    def _validate_strategy_fields(self) -> ChunkingConfig:
        if self.strategy == "fixed" and self.size is None:
            raise ValueError("Fixed chunking requires 'size'")
        if self.size is not None and self.size < 1:
            raise ValueError("Chunking 'size' must be >= 1")
        if self.overlap is not None and self.overlap < 0:
            raise ValueError("Chunking 'overlap' must be >= 0")
        if self.similarity_percentile_threshold is not None:
            if self.strategy != "semantic":
                raise ValueError(
                    "'similarity_percentile_threshold' applies only to semantic chunking"
                )
            if not 1 <= self.similarity_percentile_threshold <= 99:
                raise ValueError("'similarity_percentile_threshold' must be between 1 and 99")
        return self


class KnowledgeBase(_StrictModel):
    """A single Knowledge Base entry with metadata."""

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    knowledge_base_id: str = Field(min_length=1)
    description: str | None = None
    chunking: ChunkingConfig
    supports_merge_eval: bool = True

    @property
    def is_placeholder(self) -> bool:
        return _is_placeholder(self.knowledge_base_id)


class KnowledgeBaseCatalog(_StrictModel):
    """Catalog of Knowledge Bases loaded from YAML."""

    schema_version: Literal[1] = 1
    knowledge_bases: list[KnowledgeBase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> KnowledgeBaseCatalog:
        ids = [kb.id for kb in self.knowledge_bases]
        dupes = _find_duplicates(ids)
        if dupes:
            raise ValueError(f"Duplicate knowledge base IDs: {sorted(dupes)}")
        actual = [kb.knowledge_base_id for kb in self.knowledge_bases if not kb.is_placeholder]
        actual_dupes = _find_duplicates(actual)
        if actual_dupes:
            raise ValueError(
                "Duplicate knowledge_base_id values make comparison meaningless: "
                f"{sorted(actual_dupes)}"
            )
        return self


# ---------------------------------------------------------------------------
# Model catalogs (answer and judge)
# ---------------------------------------------------------------------------


class InferenceConfig(_ExtensibleModel):
    """Inference configuration passed to /ask. Allows model-specific parameters."""

    maxTokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ModelEntry(_StrictModel):
    """A single model entry in an answer or judge catalog."""

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    model_id: str = Field(min_length=1)
    description: str | None = None
    default_inference_config: InferenceConfig | None = None

    @property
    def is_placeholder(self) -> bool:
        return _is_placeholder(self.model_id)


class ModelCatalog(_StrictModel):
    """Catalog of models (answer or judge) loaded from YAML."""

    schema_version: Literal[1] = 1
    models: list[ModelEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> ModelCatalog:
        ids = [m.id for m in self.models]
        dupes = _find_duplicates(ids)
        if dupes:
            raise ValueError(f"Duplicate model IDs: {sorted(dupes)}")
        return self


# ---------------------------------------------------------------------------
# API configuration (env var names only, no values)
# ---------------------------------------------------------------------------


class ApiConfig(_StrictModel):
    """API connection configuration. Stores env var names, never values."""

    base_url_env: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)

    @field_validator("base_url_env", "api_key_env")
    @classmethod
    def _validate_env_name(cls, v: str) -> str:
        if not re.match(r"^[A-Z][A-Z0-9_]*$", v):
            raise ValueError(
                f"Environment variable name must be uppercase letters/digits/underscores: '{v}'"
            )
        return v


# ---------------------------------------------------------------------------
# Deployment metadata
# ---------------------------------------------------------------------------


class MergeConfig(_StrictModel):
    """Declared merge state (not requested from the API)."""

    enabled: bool
    maximum_chunks_per_document: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_merge_fields(self) -> MergeConfig:
        if self.enabled and self.maximum_chunks_per_document is None:
            raise ValueError("Merge enabled requires 'maximum_chunks_per_document'")
        if not self.enabled and self.maximum_chunks_per_document is not None:
            raise ValueError("'maximum_chunks_per_document' should be null when merge is disabled")
        return self


class DeploymentConfig(_StrictModel):
    """Deployment metadata including merge state."""

    label: str = Field(min_length=1)
    merge: MergeConfig


# ---------------------------------------------------------------------------
# Execution, retry, output
# ---------------------------------------------------------------------------


class RetryConfig(_StrictModel):
    """Retry policy per D004."""

    maximum_attempts: int = Field(default=3, ge=1, le=10)


class ExecutionConfig(_StrictModel):
    """Execution controls for live runs."""

    concurrency: int = Field(default=5, ge=1)
    continue_on_error: bool = True
    retries: RetryConfig = Field(default_factory=RetryConfig)
    allow_mode_fallback: bool = False


class OutputConfig(_StrictModel):
    """Output dataset configuration."""

    dataset_id: str = Field(pattern=DATASET_ID_PATTERN)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class CatalogSelection(_StrictModel):
    """Include/exclude selection for a single catalog type."""

    include: list[str] | None = None
    exclude: list[str] | None = None

    @model_validator(mode="after")
    def _validate_not_both(self) -> CatalogSelection:
        if self.include is not None and self.exclude is not None:
            raise ValueError("Cannot specify both 'include' and 'exclude' in the same selection")
        if self.include is None and self.exclude is None:
            raise ValueError("Selection must specify either 'include' or 'exclude'")
        return self


class Selection(_StrictModel):
    """Selection rules for catalogs within a definition."""

    questions: CatalogSelection | None = None
    knowledge_bases: CatalogSelection | None = None
    answer_models: CatalogSelection | None = None
    judge_models: CatalogSelection | None = None


# ---------------------------------------------------------------------------
# Retrieval configuration (within retrieval definition)
# ---------------------------------------------------------------------------


class RetrievalTuning(_StrictModel):
    """Retrieval mode, candidate, and result count configuration."""

    modes: list[str] = Field(min_length=1)
    candidate_counts: list[int] = Field(min_length=1)
    result_counts: list[int] = Field(min_length=1)

    @field_validator("modes")
    @classmethod
    def _validate_modes(cls, v: list[str]) -> list[str]:
        for m in v:
            if m not in _RETRIEVAL_MODES:
                raise ValueError(f"Invalid retrieval mode '{m}'. Must be one of {_RETRIEVAL_MODES}")
        return v

    @field_validator("candidate_counts", "result_counts")
    @classmethod
    def _validate_positive(cls, v: list[int]) -> list[int]:
        for n in v:
            if n < 1:
                raise ValueError("Counts must be >= 1")
        return v

    @model_validator(mode="after")
    def _validate_result_le_candidate(self) -> RetrievalTuning:
        max_result = max(self.result_counts)
        max_candidate = max(self.candidate_counts)
        if max_result > max_candidate:
            raise ValueError(
                f"max result_count ({max_result}) cannot exceed "
                f"max candidate_count ({max_candidate})"
            )
        return self


# ---------------------------------------------------------------------------
# Template and variant configuration
# ---------------------------------------------------------------------------


class TemplateRef(_StrictModel):
    """Reference to system_instructions and user_message template files."""

    system_instructions: str = Field(min_length=1)
    user_message: str = Field(min_length=1)


class PromptVariant(_StrictModel):
    """A named prompt variant."""

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    template: TemplateRef


class InferenceVariant(_StrictModel):
    """A named inference configuration variant."""

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    inference_config: InferenceConfig


# ---------------------------------------------------------------------------
# Dataset reference (for downstream definitions)
# ---------------------------------------------------------------------------


class DatasetRef(_StrictModel):
    """Reference to a source dataset."""

    type: Literal["retrieval", "retrieval_judgments", "answers", "answer_judgments"]
    id: str = Field(pattern=DATASET_ID_PATTERN)


# ---------------------------------------------------------------------------
# Test block (common to all definitions)
# ---------------------------------------------------------------------------


class TestBlock(_StrictModel):
    """Test identity block within a definition."""

    __test__ = False

    id: str = Field(pattern=_CATALOG_ID_PATTERN)
    type: str


# ---------------------------------------------------------------------------
# Definition types
# ---------------------------------------------------------------------------


class RetrievalDefinition(_StrictModel):
    """Definition for a retrieval evaluation run."""

    schema_version: Literal[1] = 1
    test: TestBlock
    questions: SourceRef
    knowledge_bases: SourceRef
    selection: Selection
    retrieval: RetrievalTuning
    deployment: DeploymentConfig
    api: ApiConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("test")
    @classmethod
    def _validate_test_type(cls, v: TestBlock) -> TestBlock:
        if v.type != "retrieval":
            raise ValueError(f"Test type must be 'retrieval', got '{v.type}'")
        return v


class RetrievalJudgmentDefinition(_StrictModel):
    """Definition for retrieval judgment using /ask."""

    schema_version: Literal[1] = 1
    test: TestBlock
    source_dataset: DatasetRef
    judge_models: SourceRef
    selection: Selection
    prompt_variants: list[PromptVariant] = Field(min_length=1)
    inference_variants: list[InferenceVariant] = Field(min_length=1)
    api: ApiConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("test")
    @classmethod
    def _validate_test_type(cls, v: TestBlock) -> TestBlock:
        if v.type != "retrieval_judgment":
            raise ValueError(f"Test type must be 'retrieval_judgment', got '{v.type}'")
        return v

    @model_validator(mode="after")
    def _validate_unique_variant_ids(self) -> RetrievalJudgmentDefinition:
        _check_unique_ids([v.id for v in self.prompt_variants], "prompt variant")
        _check_unique_ids([v.id for v in self.inference_variants], "inference variant")
        return self


class AnswerDefinition(_StrictModel):
    """Definition for frozen-context answer generation using /ask."""

    schema_version: Literal[1] = 1
    test: TestBlock
    source_dataset: DatasetRef
    answer_models: SourceRef
    selection: Selection
    prompt_variants: list[PromptVariant] = Field(min_length=1)
    inference_variants: list[InferenceVariant] = Field(min_length=1)
    api: ApiConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("test")
    @classmethod
    def _validate_test_type(cls, v: TestBlock) -> TestBlock:
        if v.type != "answer":
            raise ValueError(f"Test type must be 'answer', got '{v.type}'")
        return v

    @model_validator(mode="after")
    def _validate_unique_variant_ids(self) -> AnswerDefinition:
        _check_unique_ids([v.id for v in self.prompt_variants], "prompt variant")
        _check_unique_ids([v.id for v in self.inference_variants], "inference variant")
        return self


class AnswerJudgmentDefinition(_StrictModel):
    """Definition for answer judgment using /ask."""

    schema_version: Literal[1] = 1
    test: TestBlock
    source_dataset: DatasetRef
    judge_models: SourceRef
    selection: Selection
    prompt_variants: list[PromptVariant] = Field(min_length=1)
    inference_variants: list[InferenceVariant] = Field(min_length=1)
    api: ApiConfig
    output: OutputConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    @field_validator("test")
    @classmethod
    def _validate_test_type(cls, v: TestBlock) -> TestBlock:
        if v.type != "answer_judgment":
            raise ValueError(f"Test type must be 'answer_judgment', got '{v.type}'")
        return v

    @model_validator(mode="after")
    def _validate_unique_variant_ids(self) -> AnswerJudgmentDefinition:
        _check_unique_ids([v.id for v in self.prompt_variants], "prompt variant")
        _check_unique_ids([v.id for v in self.inference_variants], "inference variant")
        return self


class CompareEntry(_StrictModel):
    """A single dataset entry in a compare definition."""

    label: str = Field(min_length=1)
    dataset: DatasetRef


class CompareDefinition(_StrictModel):
    """Definition for comparing multiple datasets."""

    schema_version: Literal[1] = 1
    test: TestBlock
    entries: list[CompareEntry] = Field(min_length=2)

    @field_validator("test")
    @classmethod
    def _validate_test_type(cls, v: TestBlock) -> TestBlock:
        if v.type != "compare":
            raise ValueError(f"Test type must be 'compare', got '{v.type}'")
        return v

    @model_validator(mode="after")
    def _validate_unique_labels(self) -> CompareDefinition:
        labels = [e.label for e in self.entries]
        dupes = _find_duplicates(labels)
        if dupes:
            raise ValueError(f"Duplicate compare entry labels: {sorted(dupes)}")
        return self


# ---------------------------------------------------------------------------
# Definition union
# ---------------------------------------------------------------------------

Definition = (
    RetrievalDefinition
    | RetrievalJudgmentDefinition
    | AnswerDefinition
    | AnswerJudgmentDefinition
    | CompareDefinition
)

DEFINITION_TYPE_MAP: dict[str, type[BaseModel]] = {
    "retrieval": RetrievalDefinition,
    "retrieval_judgment": RetrievalJudgmentDefinition,
    "answer": AnswerDefinition,
    "answer_judgment": AnswerJudgmentDefinition,
    "compare": CompareDefinition,
}


def parse_definition(data: dict[str, Any]) -> BaseModel:
    """Parse a definition dict by dispatching on test.type."""
    from rag_evals.errors import ConfigError

    if not isinstance(data, dict):
        raise ConfigError("Definition must be a mapping")
    test = data.get("test")
    if not isinstance(test, dict):
        raise ConfigError("Definition must contain a 'test' block")
    test_type = test.get("type")
    if test_type not in DEFINITION_TYPE_MAP:
        raise ConfigError(
            f"Unknown definition type '{test_type}'. Must be one of {list(DEFINITION_TYPE_MAP)}"
        )
    try:
        return DEFINITION_TYPE_MAP[test_type].model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Definition validation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_duplicates(items: list[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for item in items:
        if item in seen:
            dupes.add(item)
        seen.add(item)
    return dupes


def _check_unique_ids(ids: list[str], label: str) -> None:
    dupes = _find_duplicates(ids)
    if dupes:
        raise ValueError(f"Duplicate {label} IDs: {sorted(dupes)}")
