"""Frozen answer generation stage (P8).

Verifies retrieval hash, extracts context, renders user message, calls /ask,
stores answer with exact frozen lineage and incidental retrieval separation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_evals.api.protocol import ApiAdapter, AskRequest, AskResponse
from rag_evals.api.retry import execute_with_retries
from rag_evals.artifacts.lineage import LineageRef, load_source_dataset, verify_lineage
from rag_evals.artifacts.store import (
    DatasetPath,
    Lifecycle,
    _read_status,
    append_failure,
    create_dataset,
    get_missing_artifacts,
    transition_lifecycle,
    update_status,
    write_artifact,
)
from rag_evals.config.execution import ExecutionBundle
from rag_evals.config.schemas import (
    AnswerDefinition,
    ModelCatalog,
    PromptVariant,
    QuestionCatalog,
)
from rag_evals.errors import ExecutionError
from rag_evals.logging import get_logger
from rag_evals.planning.identity import canonical_hash
from rag_evals.planning.matrices import AskRow, generate_ask_matrix
from rag_evals.templates import Stage, render_template, validate_template

logger = get_logger("rag_evals.stages.answer")

ANSWER_SCHEMA_VERSION = "answer-artifact-v1"


@dataclass
class AnswerArtifact:
    """Answer artifact with frozen context lineage and exact API usage."""

    artifact_id: str
    identity: dict[str, Any]
    question_id: str
    model_catalog_id: str
    model_actual_id: str
    prompt_variant_id: str
    inference_variant_id: str
    source_dataset_type: str
    source_dataset_id: str
    source_artifact_id: str
    source_content_hash: str
    frozen_context_hash: str
    lineage: dict[str, Any]
    system_instructions: str
    user_message: str
    answer_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    elapsed_seconds: float = 0.0
    limitations: list[str] = field(default_factory=list)
    incidental_retrieval: dict[str, Any] = field(default_factory=dict)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "schema_version": ANSWER_SCHEMA_VERSION,
            "identity": self.identity,
            "question_id": self.question_id,
            "model_catalog_id": self.model_catalog_id,
            "model_actual_id": self.model_actual_id,
            "prompt_variant_id": self.prompt_variant_id,
            "inference_variant_id": self.inference_variant_id,
            "source_dataset_type": self.source_dataset_type,
            "source_dataset_id": self.source_dataset_id,
            "source_artifact_id": self.source_artifact_id,
            "source_content_hash": self.source_content_hash,
            "frozen_context_hash": self.frozen_context_hash,
            "lineage": self.lineage,
            "system_instructions": self.system_instructions,
            "user_message": self.user_message,
            "answer_text": self.answer_text,
            "usage": self.usage,
            "metrics": self.metrics,
            "cost": self.cost,
            "stop_reason": self.stop_reason,
            "elapsed_seconds": self.elapsed_seconds,
            "limitations": self.limitations,
            "incidental_retrieval": self.incidental_retrieval,
            "attempts": self.attempts,
            "content_hash": self.content_hash or canonical_hash(self.identity),
        }


async def run_answer_stage(
    project_root: Path,
    definition: AnswerDefinition,
    questions: QuestionCatalog,
    models: ModelCatalog,
    adapter: ApiAdapter,
    dry_run: bool = False,
    only_missing: bool = False,
    execution_bundle: ExecutionBundle | None = None,
    config_root: Path | None = None,
) -> DatasetPath:
    """Run the answer generation stage."""
    logger.info("Starting answer generation stage")
    template_root = config_root or project_root

    source = definition.source_dataset
    source_paths = DatasetPath(project_root, source.type, source.id)
    source_artifacts = load_source_dataset(project_root, source.type, source.id)

    matrix = generate_ask_matrix(
        questions=questions,
        models=models,
        selection=definition.selection,
        prompt_variants=definition.prompt_variants,
        inference_variants=definition.inference_variants,
        source_dataset_type=source.type,
        source_dataset_id=source.id,
        stage_label="answer",
        source_artifacts=source_artifacts,
    )

    logger.info("Matrix: %d rows", matrix.total_calls)

    plan_rows = [{"artifact_id": r.artifact_id} for r in matrix.rows]
    resolved_def = execution_bundle.snapshot() if execution_bundle else definition.model_dump()

    paths = create_dataset(
        project_root=project_root,
        dataset_type="answers",
        dataset_id=definition.output.dataset_id,
        resolved_definition=resolved_def,
        plan_rows=plan_rows,
        config_inputs=execution_bundle.snapshot_inputs() if execution_bundle else None,
    )

    if dry_run:
        logger.info("Dry run: %d planned artifacts", len(plan_rows))
        return paths

    # Check for placeholder model IDs
    for model in models.models:
        if model.is_placeholder:
            raise ExecutionError(f"Cannot execute with placeholder model ID: {model.model_id}")

    rows = matrix.rows
    missing_ids = set(get_missing_artifacts(paths))
    rows = [r for r in rows if r.artifact_id in missing_ids]

    if not rows:
        logger.info("No artifacts to execute")
        current = _read_status(paths).get("lifecycle", Lifecycle.BUILDING.value)
        if current == Lifecycle.BUILDING.value:
            update_status(paths)
            if not get_missing_artifacts(paths):
                transition_lifecycle(paths, Lifecycle.COMPLETE)
        return paths

    question_map = {q.id: q for q in questions.questions}
    model_map = {m.id: m for m in models.models}
    prompt_map = {pv.id: pv for pv in definition.prompt_variants}
    inference_map = {iv.id: iv for iv in definition.inference_variants}

    semaphore = asyncio.Semaphore(definition.execution.concurrency)

    async def execute_row(row: AskRow) -> None:
        try:
            await _execute_answer_row(
                paths,
                row,
                question_map,
                model_map,
                prompt_map,
                inference_map,
                source_paths,
                adapter,
                definition,
                project_root,
                template_root,
                semaphore,
            )
        except Exception as exc:
            append_failure(
                paths,
                row.artifact_id,
                0,
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "source_failure": True,
                },
            )
            raise

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

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("Row %d failed: %s", i, result)
    errors = [result for result in results if isinstance(result, Exception)]
    if errors and not definition.execution.continue_on_error:
        raise ExecutionError(f"Answer generation stopped after failure: {errors[0]}")

    update_status(paths)
    current = _read_status(paths).get("lifecycle", Lifecycle.BUILDING.value)
    if current == Lifecycle.BUILDING.value and not get_missing_artifacts(paths):
        transition_lifecycle(paths, Lifecycle.COMPLETE)

    return paths


async def _execute_answer_row(
    paths: DatasetPath,
    row: AskRow,
    question_map: dict[str, Any],
    model_map: dict[str, Any],
    prompt_map: dict[str, PromptVariant],
    inference_map: dict[str, Any],
    source_paths: DatasetPath,
    adapter: ApiAdapter,
    definition: AnswerDefinition,
    project_root: Path,
    template_root: Path,
    attempt_semaphore: asyncio.Semaphore,
) -> None:
    """Execute a single answer generation row."""
    from rag_evals.stages.judgment import _load_template_file

    q = question_map[row.question_id]
    m = model_map[row.model_catalog_id]
    pv = prompt_map[row.prompt_variant_id]
    iv = inference_map[row.inference_variant_id]

    # Load source retrieval artifact
    source_data = verify_lineage(
        project_root,
        LineageRef(
            source_paths.dataset_type,
            source_paths.dataset_id,
            row.source_artifact_id,
            row.source_relative_path,
            row.source_content_hash,
        ),
    )
    source_content_hash = row.source_content_hash
    context = source_data.get("prepared_context", "")
    frozen_context_hash = canonical_hash({"context": context})

    # Build template variables
    rubric: dict[str, Any] = {}
    if hasattr(q, "rubric") and q.rubric:
        rubric = q.rubric.model_dump() if hasattr(q.rubric, "model_dump") else {}

    template_vars = {
        "question": q.question,
        "question_id": q.id,
        "rubric": rubric,
        "retrieval_context": context,
    }

    # Load and render templates
    system_text = _load_template_file(pv.template.system_instructions, template_root)
    user_template = _load_template_file(pv.template.user_message, template_root)
    for text in (system_text, user_template):
        validation = validate_template(text, Stage.ANSWER)
        if not validation.valid:
            raise ExecutionError("Invalid answer template: " + "; ".join(validation.errors))
    rendered_system = render_template(system_text, template_vars)
    user_message = render_template(user_template, template_vars)

    inference_config = None
    if hasattr(iv, "inference_config") and iv.inference_config:
        inference_config = (
            iv.inference_config.model_dump() if hasattr(iv.inference_config, "model_dump") else {}
        )

    request = AskRequest(
        question=q.question,
        model_id=m.model_id,
        system_instructions=rendered_system,
        user_message=user_message,
        inference_config=inference_config,
    )

    max_attempts = definition.execution.retries.maximum_attempts

    async def make_call() -> AskResponse:
        return await adapter.ask(request)

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
        resp = result.response
        source_aid = source_data.get("artifact_id", "")

        identity = dict(row.identity)

        artifact = AnswerArtifact(
            artifact_id=row.artifact_id,
            identity=identity,
            question_id=row.question_id,
            model_catalog_id=row.model_catalog_id,
            model_actual_id=row.model_actual_id,
            prompt_variant_id=row.prompt_variant_id,
            inference_variant_id=row.inference_variant_id,
            source_dataset_type=source_paths.dataset_type,
            source_dataset_id=source_paths.dataset_id,
            source_artifact_id=source_aid,
            source_content_hash=source_content_hash,
            frozen_context_hash=frozen_context_hash,
            lineage={
                "source_dataset_type": source_paths.dataset_type,
                "source_dataset_id": source_paths.dataset_id,
                "source_artifact_id": source_aid,
                "relative_path": row.source_relative_path,
                "expected_content_hash": source_content_hash,
            },
            system_instructions=rendered_system,
            user_message=user_message,
            answer_text=resp.text,
            usage=resp.usage,
            metrics=resp.metrics,
            cost=resp.cost,
            stop_reason=resp.stop_reason,
            elapsed_seconds=resp.elapsed_seconds,
            limitations=resp.limitations,
            incidental_retrieval=resp.incidental_retrieval,
            attempts=attempt_records,
        )
        write_artifact(paths, artifact.artifact_id, artifact.to_dict())
        logger.info("Wrote answer artifact for %s/%s", row.question_id, row.model_catalog_id)
    else:
        last = result.attempts[-1]
        append_failure(
            paths,
            row.artifact_id,
            last.attempt,
            {
                "error_type": last.error_type,
                "error_message": last.error_message,
                "attempts": attempt_records,
            },
        )
        if definition.execution.continue_on_error:
            logger.warning("Row failed: %s/%s", row.question_id, row.model_catalog_id)
        else:
            raise ExecutionError(f"Row failed: {row.question_id}/{row.model_catalog_id}")
