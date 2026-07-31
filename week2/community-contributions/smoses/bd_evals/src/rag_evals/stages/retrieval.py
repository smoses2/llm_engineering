"""Retrieval artifact schema, runner, and metrics.

Implements P6-T01 (schema), P6-T02 (runner), and P6-T03 (CLI integration).
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_evals.api.protocol import (
    ApiAdapter,
    RetrieveRequest,
    RetrieveResponse,
)
from rag_evals.api.retry import execute_with_retries
from rag_evals.artifacts.store import (
    DatasetPath,
    Lifecycle,
    append_failure,
    create_dataset,
    get_missing_artifacts,
    transition_lifecycle,
    update_status,
    write_artifact,
)
from rag_evals.config.execution import ExecutionBundle
from rag_evals.config.schemas import (
    KnowledgeBase,
    KnowledgeBaseCatalog,
    Question,
    QuestionCatalog,
    RetrievalDefinition,
)
from rag_evals.errors import ExecutionError
from rag_evals.logging import get_logger
from rag_evals.planning.identity import canonical_hash
from rag_evals.planning.matrices import RetrievalRow, generate_retrieval_matrix

logger = get_logger("rag_evals.stages.retrieval")

RETRIEVAL_SCHEMA_VERSION = "retrieval-artifact-v1"


def estimate_tokens(context: str) -> int:
    """Estimate token count using D008: ceil(character_count / 4)."""
    return math.ceil(len(context) / 4)


@dataclass
class RetrievalArtifact:
    """Complete retrieval artifact with identity, request, response, and metrics."""

    artifact_id: str
    identity: dict[str, Any]
    question_id: str
    question_text: str
    rubric: dict[str, Any] | None
    kb_catalog_id: str
    kb_actual_id: str
    kb_metadata: dict[str, Any]
    deployment_label: str
    merge_enabled: bool
    mode: str
    candidate_count: int
    result_count: int
    request: dict[str, Any]
    response: dict[str, Any]
    raw_json: dict[str, Any]
    normalized_items: list[dict[str, Any]]
    prepared_context: str
    retrieval_metadata: dict[str, Any]
    cost: dict[str, Any]
    elapsed_seconds: float
    limitations: list[str]
    objective_metrics: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "artifact_id": self.artifact_id,
            "schema_version": RETRIEVAL_SCHEMA_VERSION,
            "identity": self.identity,
            "question_id": self.question_id,
            "question_text": self.question_text,
            "rubric": self.rubric,
            "kb_catalog_id": self.kb_catalog_id,
            "kb_actual_id": self.kb_actual_id,
            "kb_metadata": self.kb_metadata,
            "deployment_label": self.deployment_label,
            "merge_enabled": self.merge_enabled,
            "mode": self.mode,
            "candidate_count": self.candidate_count,
            "result_count": self.result_count,
            "request": self.request,
            "response": self.response,
            "raw_json": self.raw_json,
            "normalized_items": self.normalized_items,
            "prepared_context": self.prepared_context,
            "retrieval_metadata": self.retrieval_metadata,
            "cost": self.cost,
            "elapsed_seconds": self.elapsed_seconds,
            "limitations": self.limitations,
            "objective_metrics": self.objective_metrics,
            "estimated_context_tokens": estimate_tokens(self.prepared_context),
            "token_estimate_method": "ceil(context_character_count / 4)",
            "token_estimate_label": "estimated",
            "attempts": self.attempts,
            "content_hash": self.content_hash or canonical_hash(self.identity),
        }


def detect_mode_fallback(requested_mode: str, metadata: dict[str, Any]) -> str | None:
    """Return a reason when retrieval did not run in the requested mode.

    A silently degraded mode invalidates any comparison between modes, so it is
    treated as a row failure rather than recorded and averaged. Returns None when
    the API reports no mode information, because an absent field cannot be checked.
    """
    effective = str(metadata.get("effective_mode") or "")
    reported = str(metadata.get("requested_mode") or "") or requested_mode
    reason = metadata.get("fallback_reason")
    suffix = f" (reason: {reason})" if reason else ""
    if effective and reported and effective != reported:
        return (
            f"Retrieval mode fallback: requested '{reported}' but effective "
            f"mode was '{effective}'{suffix}"
        )
    if metadata.get("fallback_used"):
        return (
            f"Retrieval reported fallback_used with requested '{reported}' and "
            f"effective '{effective or 'unknown'}'{suffix}"
        )
    return None


def build_retrieval_artifact(
    row: RetrievalRow,
    response: RetrieveResponse,
    rubric: dict[str, Any] | None,
    kb: KnowledgeBase,
    attempts: list[dict[str, Any]] | None = None,
) -> RetrievalArtifact:
    """Build a retrieval artifact from a matrix row and adapter response."""
    identity = dict(row.identity)

    context = response.prepared_prompt_input.get("promptVariables", {}).get("context", "")

    objective_metrics: dict[str, Any] = {
        "item_count": len(response.normalized_items),
        "has_fallback": response.retrieval_metadata.get("fallback_used", False),
    }

    if response.normalized_items:
        scores = [item.get("score", 0) for item in response.normalized_items if "score" in item]
        if scores:
            objective_metrics["max_score"] = max(scores)
            objective_metrics["min_score"] = min(scores)
            objective_metrics["mean_score"] = sum(scores) / len(scores)

    return RetrievalArtifact(
        artifact_id=row.artifact_id,
        identity=identity,
        question_id=row.question_id,
        question_text=row.question_text,
        rubric=rubric,
        kb_catalog_id=row.kb_catalog_id,
        kb_actual_id=row.kb_actual_id,
        kb_metadata={
            "id": kb.id,
            "chunking": kb.chunking.model_dump() if hasattr(kb.chunking, "model_dump") else {},
            "supports_merge_eval": kb.supports_merge_eval,
        },
        deployment_label=row.deployment_label,
        merge_enabled=row.merge_enabled,
        mode=row.mode,
        candidate_count=row.candidate_count,
        result_count=row.result_count,
        request={
            "question": row.question_text,
            "knowledge_base_id": row.kb_actual_id,
            "retrieval_mode": row.mode,
            "candidate_count": row.candidate_count,
            "result_count": row.result_count,
        },
        response=response.raw_response,
        raw_json=response.raw_json,
        normalized_items=response.normalized_items,
        prepared_context=context,
        retrieval_metadata=response.retrieval_metadata,
        cost=response.cost,
        elapsed_seconds=response.elapsed_seconds,
        limitations=response.limitations,
        objective_metrics=objective_metrics,
        attempts=attempts or [],
    )


async def run_retrieval_stage(
    project_root: Path,
    definition: RetrievalDefinition,
    questions: QuestionCatalog,
    knowledge_bases: KnowledgeBaseCatalog,
    adapter: ApiAdapter,
    dry_run: bool = False,
    only_missing: bool = False,
    question_id_filter: str | None = None,
    execution_bundle: ExecutionBundle | None = None,
) -> DatasetPath:
    """Run the retrieval stage: create dataset, execute calls, write artifacts.

    Uses the fake adapter for tests; real adapter for live runs.
    """
    logger.info("Starting retrieval stage")

    # Generate matrix
    matrix = generate_retrieval_matrix(
        questions=questions,
        knowledge_bases=knowledge_bases,
        selection=definition.selection,
        retrieval=definition.retrieval,
        deployment=definition.deployment,
    )

    logger.info("Matrix: %d rows, %d excluded", len(matrix.rows), len(matrix.excluded))

    # Filter rows
    rows = matrix.rows
    if question_id_filter:
        rows = [r for r in rows if r.question_id == question_id_filter]

    # Create or resume dataset
    plan_rows = [{"artifact_id": r.artifact_id} for r in matrix.rows]
    resolved_def = execution_bundle.snapshot() if execution_bundle else definition.model_dump()

    paths = create_dataset(
        project_root=project_root,
        dataset_type="retrieval",
        dataset_id=definition.output.dataset_id,
        resolved_definition=resolved_def,
        plan_rows=plan_rows,
        config_inputs=execution_bundle.snapshot_inputs() if execution_bundle else None,
    )

    if dry_run:
        logger.info("Dry run: %d planned artifacts, no API calls", len(plan_rows))
        return paths

    # Determine which rows to execute
    missing_ids = set(get_missing_artifacts(paths))
    rows = [r for r in rows if r.artifact_id in missing_ids]
    if only_missing:
        logger.info("Resuming: %d missing artifacts", len(rows))

    if not rows:
        logger.info("No artifacts to execute")
        from rag_evals.artifacts.store import _read_status

        current = _read_status(paths).get("lifecycle", Lifecycle.BUILDING.value)
        if current == Lifecycle.BUILDING.value:
            update_status(paths)
            if not get_missing_artifacts(paths):
                transition_lifecycle(paths, Lifecycle.COMPLETE)
        return paths

    # Check for placeholder KB IDs
    for row in rows:
        if row.kb_actual_id.startswith("KB_REPLACE_ME"):
            raise ExecutionError(
                f"Cannot execute live retrieval with placeholder KB ID: "
                f"{row.kb_actual_id} (catalog ID: {row.kb_catalog_id}). "
                f"Replace with a real Knowledge Base ID."
            )

    # Build question and KB lookups
    question_map = {q.id: q for q in questions.questions}
    kb_map = {kb.id: kb for kb in knowledge_bases.knowledge_bases}

    # Execute with concurrency
    semaphore = asyncio.Semaphore(definition.execution.concurrency)

    async def execute_row(row: RetrievalRow) -> None:
        await _execute_retrieval_row(
            paths, row, question_map, kb_map, adapter, definition, semaphore
        )

    if definition.execution.continue_on_error:
        results = await asyncio.gather(*(execute_row(row) for row in rows), return_exceptions=True)
    else:
        results = []
        for row in rows:
            try:
                await execute_row(row)
                results.append(None)
            except Exception as exc:
                results.append(exc)
                break

    # Log errors
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Row %d failed: %s", i, result)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors and not definition.execution.continue_on_error:
        raise ExecutionError(f"Retrieval stopped after failure: {errors[0]}")

    # Update status
    update_status(paths)

    # Check if complete (no missing AND no failures, or already complete)
    from rag_evals.artifacts.store import _read_status

    current_lifecycle = _read_status(paths).get("lifecycle", Lifecycle.BUILDING.value)
    if current_lifecycle == Lifecycle.BUILDING.value:
        missing = get_missing_artifacts(paths)
        if not missing:
            transition_lifecycle(paths, Lifecycle.COMPLETE)
            logger.info("Retrieval dataset complete")
        else:
            logger.info("Retrieval dataset still building: %d missing", len(missing))
    elif current_lifecycle == Lifecycle.COMPLETE.value:
        missing = get_missing_artifacts(paths)
        if missing:
            # Dataset was complete but artifacts are now missing
            logger.info("Retrieval dataset has %d missing artifacts", len(missing))
        else:
            logger.info("Retrieval dataset already complete")

    return paths


async def _execute_retrieval_row(
    paths: DatasetPath,
    row: RetrievalRow,
    question_map: dict[str, Question],
    kb_map: dict[str, KnowledgeBase],
    adapter: ApiAdapter,
    definition: RetrievalDefinition,
    attempt_semaphore: asyncio.Semaphore,
) -> None:
    """Execute a single retrieval row with retry."""
    q = question_map[row.question_id]
    kb = kb_map[row.kb_catalog_id]

    rubric = None
    if q.rubric:
        rubric = q.rubric.model_dump() if hasattr(q.rubric, "model_dump") else dict(q.rubric)

    request = RetrieveRequest(
        question=q.question,
        knowledge_base_id=row.kb_actual_id,
        retrieval_mode=row.mode,
        candidate_count=row.candidate_count,
        result_count=row.result_count,
    )

    max_attempts = definition.execution.retries.maximum_attempts

    async def make_call() -> RetrieveResponse:
        return await adapter.retrieve(request)

    result = await execute_with_retries(
        make_call, max_attempts=max_attempts, attempt_semaphore=attempt_semaphore
    )

    attempt_records = [
        {
            "attempt": a.attempt,
            "success": a.success,
            "error_type": a.error_type,
            "error_message": a.error_message,
            "retryable": a.retryable,
            "delay_seconds": a.delay_seconds,
            "elapsed_seconds": a.elapsed_seconds,
        }
        for a in result.attempts
    ]

    if result.success:
        fallback_reason = (
            None
            if definition.execution.allow_mode_fallback
            else detect_mode_fallback(row.mode, result.response.retrieval_metadata)
        )
        if fallback_reason is not None:
            append_failure(
                paths,
                row.artifact_id,
                len(attempt_records),
                {
                    "error_type": "retrieval_mode_fallback",
                    "error_message": fallback_reason,
                    "retryable": False,
                    "attempts": attempt_records,
                    "retrieval_metadata": result.response.retrieval_metadata,
                    "cost": result.response.cost,
                },
            )
            if definition.execution.continue_on_error:
                logger.warning(
                    "Row rejected for mode fallback: %s/%s/%s: %s",
                    row.question_id,
                    row.kb_catalog_id,
                    row.mode,
                    fallback_reason,
                )
                return
            raise ExecutionError(
                f"Row rejected: {row.question_id}/{row.kb_catalog_id}/{row.mode}: {fallback_reason}"
            )
        artifact = build_retrieval_artifact(
            row=row,
            response=result.response,
            rubric=rubric,
            kb=kb,
            attempts=attempt_records,
        )
        write_artifact(paths, artifact.artifact_id, artifact.to_dict())
        logger.info("Wrote artifact for %s/%s/%s", row.question_id, row.kb_catalog_id, row.mode)
    else:
        last_attempt = result.attempts[-1]
        append_failure(
            paths,
            row.artifact_id,
            last_attempt.attempt,
            {
                "error_type": last_attempt.error_type,
                "error_message": last_attempt.error_message,
                "retryable": last_attempt.retryable,
                "attempts": attempt_records,
            },
        )
        if definition.execution.continue_on_error:
            logger.warning(
                "Row failed after %d attempts: %s/%s/%s",
                last_attempt.attempt,
                row.question_id,
                row.kb_catalog_id,
                row.mode,
            )
        else:
            raise ExecutionError(
                f"Row failed: {row.question_id}/{row.kb_catalog_id}/{row.mode}: "
                f"{last_attempt.error_message}"
            )
