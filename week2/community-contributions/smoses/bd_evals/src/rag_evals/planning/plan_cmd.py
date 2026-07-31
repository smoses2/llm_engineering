"""Plan command: print dimensions, IDs, call counts, and approval contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from rag_evals.config.resolver import SourceResolver, safe_load_yaml
from rag_evals.config.schemas import (
    AnswerDefinition,
    AnswerJudgmentDefinition,
    KnowledgeBaseCatalog,
    ModelCatalog,
    QuestionCatalog,
    RetrievalDefinition,
    RetrievalJudgmentDefinition,
    parse_definition,
)
from rag_evals.config.validate_cmd import _find_project_root
from rag_evals.errors import ConfigError, PlanError
from rag_evals.planning.matrices import (
    plan_retrieval_definition,
)


def plan_definition(
    definition_path: Path,
    project_root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Plan a definition and return a summary with dimensions, IDs, and call counts.

    Ask-stage definitions carry no question catalog: their questions come from the
    source dataset, so ``results_root`` is required to plan them.
    """
    if not definition_path.exists():
        raise ConfigError(f"Definition file not found: {definition_path}")

    root = project_root or _find_project_root(definition_path)

    raw_data = safe_load_yaml(definition_path)
    definition = parse_definition(raw_data)

    resolver = SourceResolver(root)
    resolved = resolver.resolve_definition(definition_path)
    resolved_data = resolved.definition_data

    if isinstance(definition, RetrievalDefinition):
        return _plan_retrieval(definition, resolved_data)
    elif isinstance(
        definition,
        (RetrievalJudgmentDefinition, AnswerDefinition, AnswerJudgmentDefinition),
    ):
        return _plan_ask(definition, definition_path, root, results_root)
    else:
        test_type = raw_data.get("test", {}).get("type", "unknown")
        raise PlanError(f"Planning not supported for definition type: {test_type}")


def _plan_retrieval(
    definition: RetrievalDefinition,
    resolved_data: dict[str, Any],
) -> dict[str, Any]:
    """Plan a retrieval definition."""
    questions = QuestionCatalog.model_validate(resolved_data["questions"])
    kbs = KnowledgeBaseCatalog.model_validate(resolved_data["knowledge_bases"])

    matrix = plan_retrieval_definition(definition, questions, kbs)

    return {
        "definition_type": "retrieval",
        "test_id": definition.test.id,
        "dimensions": {
            "questions": len(matrix.selected_question_ids),
            "knowledge_bases": len(matrix.selected_kb_ids),
            "modes": sorted(definition.retrieval.modes),
            "candidate_counts": sorted(definition.retrieval.candidate_counts),
            "result_counts": sorted(definition.retrieval.result_counts),
        },
        "selected_ids": {
            "questions": matrix.selected_question_ids,
            "knowledge_bases": matrix.selected_kb_ids,
        },
        "exclusions": [
            {
                "question_id": r.question_id,
                "kb_catalog_id": r.kb_catalog_id,
                "reason": "merge_enabled_no_chunk",
            }
            for r in matrix.excluded
        ],
        "total_calls": matrix.total_calls,
        "rows": [
            {
                "experiment_id": r.experiment_id,
                "artifact_id": r.artifact_id,
                "question_id": r.question_id,
                "kb_catalog_id": r.kb_catalog_id,
                "mode": r.mode,
                "candidate_count": r.candidate_count,
                "result_count": r.result_count,
            }
            for r in matrix.rows
        ],
        "required_approval": matrix.total_calls,
    }


def _plan_ask(
    definition: RetrievalJudgmentDefinition | AnswerDefinition | AnswerJudgmentDefinition,
    definition_path: Path,
    config_root: Path,
    results_root: Path | None,
) -> dict[str, Any]:
    """Plan an ask-stage definition against its source dataset.

    Questions and rubrics are rebuilt from the source dataset artifacts, and prompt
    contents are resolved, so counts and artifact IDs match live execution.
    """
    from rag_evals.artifacts.lineage import load_source_dataset
    from rag_evals.config.execution import load_execution_bundle
    from rag_evals.config.schemas import Question, Rubric
    from rag_evals.planning.matrices import generate_ask_matrix

    if results_root is None:
        raise PlanError(
            "Planning an ask-stage definition requires the workspace results directory, "
            "because its questions come from the source dataset."
        )

    source = definition.source_dataset
    try:
        descriptors = load_source_dataset(results_root, source.type, source.id)
    except Exception as exc:
        raise PlanError(
            f"Cannot plan against source dataset {source.type}/{source.id}: {exc}"
        ) from exc
    if not descriptors:
        raise PlanError(f"Source dataset has no artifacts: {source.type}/{source.id}")

    bundle = load_execution_bundle(definition_path, config_root)
    model_key = "answer_models" if isinstance(definition, AnswerDefinition) else "judge_models"
    models = ModelCatalog.model_validate(bundle.resolved_definition[model_key])

    by_question: dict[str, Question] = {}
    for item in descriptors:
        by_question.setdefault(
            item.question_id,
            Question(
                id=item.question_id,
                question=item.question_text,
                rubric=Rubric.model_validate(item.rubric) if item.rubric else None,
            ),
        )
    questions = QuestionCatalog(questions=list(by_question.values()))

    matrix = generate_ask_matrix(
        questions,
        models,
        definition.selection,
        definition.prompt_variants,
        definition.inference_variants,
        source.type,
        source.id,
        model_selection_attr=model_key,
        stage_label=definition.test.type,
        source_artifacts=descriptors,
        prompt_contents=dict(bundle.prompt_contents),
    )

    return {
        "definition_type": definition.test.type,
        "test_id": definition.test.id,
        "dimensions": {
            "source_artifacts": len(descriptors),
            "questions": len(matrix.selected_question_ids),
            "models": len(matrix.selected_model_ids),
            "prompt_variants": len(matrix.prompt_variant_ids),
            "inference_variants": len(matrix.inference_variant_ids),
        },
        "selected_ids": {
            "questions": matrix.selected_question_ids,
            "models": matrix.selected_model_ids,
            "prompt_variants": matrix.prompt_variant_ids,
            "inference_variants": matrix.inference_variant_ids,
        },
        "source_dataset": {
            "type": matrix.source_dataset_type,
            "id": matrix.source_dataset_id,
        },
        "total_calls": matrix.total_calls,
        "rows": [
            {
                "experiment_id": r.experiment_id,
                "artifact_id": r.artifact_id,
                "question_id": r.question_id,
                "model_catalog_id": r.model_catalog_id,
                "prompt_variant_id": r.prompt_variant_id,
                "inference_variant_id": r.inference_variant_id,
            }
            for r in matrix.rows
        ],
        "required_approval": matrix.total_calls,
    }


def plan_and_report(
    definition_path: Path,
    output_format: str = "text",
    project_root: Path | None = None,
    results_root: Path | None = None,
) -> None:
    """Plan a definition and print results."""
    try:
        result = plan_definition(definition_path, project_root, results_root)
    except (ConfigError, PlanError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc

    if output_format == "json":
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        _print_text(result)


def _print_text(result: dict[str, Any]) -> None:
    typer.echo(f"Definition type: {result['definition_type']}")
    typer.echo(f"Test ID: {result['test_id']}")
    typer.echo(f"Dimensions: {result['dimensions']}")
    typer.echo(f"Selected IDs: {result['selected_ids']}")
    if "exclusions" in result and result["exclusions"]:
        typer.echo(f"Exclusions ({len(result['exclusions'])}):")
        for exc in result["exclusions"]:
            typer.echo(f"  {exc}")
    if "source_dataset" in result:
        typer.echo(f"Source dataset: {result['source_dataset']}")
    typer.echo(f"Total calls: {result['total_calls']}")
    typer.echo(f"Required approval: --approve-call-count {result['required_approval']}")
