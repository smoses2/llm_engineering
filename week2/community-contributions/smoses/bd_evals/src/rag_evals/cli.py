"""CLI entry point for rag-evals."""

from pathlib import Path
from typing import Annotated

import typer

from rag_evals.logging import setup_logging
from rag_evals.workspace import Workspace

app = typer.Typer(
    name="rag-evals",
    help="Bedrock RAG evaluation framework.",
    no_args_is_help=True,
)


@app.callback()
def main(
    ctx: typer.Context,
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace",
            envvar="RAG_EVALS_WORKSPACE",
            show_envvar=True,
            help="Evaluation workspace containing config/ and results/.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose (DEBUG) logging."),
    ] = False,
) -> None:
    """Bedrock RAG evaluation framework."""
    setup_logging("DEBUG" if verbose else "INFO")
    ctx.obj = workspace


def _require_workspace(ctx: typer.Context) -> Workspace:
    """Return the selected workspace or exit with an actionable configuration error."""
    from rag_evals.env import load_workspace_env
    from rag_evals.errors import ConfigError

    if not isinstance(ctx.obj, Path):
        typer.echo(
            "Error: No evaluation workspace configured. "
            "Pass --workspace PATH or set RAG_EVALS_WORKSPACE.",
            err=True,
        )
        raise typer.Exit(code=ConfigError.exit_code)
    try:
        workspace = Workspace.from_path(ctx.obj)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc
    load_workspace_env(workspace)
    return workspace


def _resolve_definition(workspace: Workspace, definition: Path) -> Path:
    """Resolve a CLI definition argument or exit through the config error path."""
    from rag_evals.errors import ConfigError

    try:
        return workspace.resolve_definition(definition)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc


DefinitionArg = Annotated[
    Path,
    typer.Argument(help="Path to evaluation definition YAML.", show_default=False),
]

DatasetTypeArg = Annotated[
    str,
    typer.Argument(
        help="Dataset type (retrieval, retrieval_judgments, answers, answer_judgments).",
    ),
]

DatasetIdArg = Annotated[
    str,
    typer.Argument(help="Dataset ID."),
]


def _run_ask_command(
    workspace: Workspace,
    definition_path: Path,
    expected_stage: str,
    dry_run: bool,
    only_missing: bool,
    approve_call_count: int | None,
) -> None:
    """Shared resolved loader, call planner, preflight, and ask-stage dispatcher."""
    import asyncio
    import os

    from rag_evals.api.real_adapter import RealBdApiAdapter, preflight_bd_api
    from rag_evals.artifacts.lineage import load_source_dataset
    from rag_evals.artifacts.store import DatasetPath, get_missing_artifacts
    from rag_evals.config.execution import load_execution_bundle
    from rag_evals.config.schemas import (
        AnswerDefinition,
        AnswerJudgmentDefinition,
        ModelCatalog,
        Question,
        QuestionCatalog,
        RetrievalJudgmentDefinition,
        Rubric,
    )
    from rag_evals.errors import ConfigError, ExecutionError
    from rag_evals.planning.matrices import generate_ask_matrix
    from rag_evals.stages.answer import run_answer_stage
    from rag_evals.stages.judgment import run_judgment_stage
    from rag_evals.templates import Stage

    try:
        bundle = load_execution_bundle(definition_path, workspace.config_root)
        definition = bundle.definition
        if definition.test.type != expected_stage:
            raise ConfigError(f"Expected {expected_stage} definition")
        source = definition.source_dataset  # type: ignore[union-attr]
        descriptors = load_source_dataset(workspace.results_root, source.type, source.id)
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
        model_key = "answer_models" if expected_stage == "answer" else "judge_models"
        models = ModelCatalog.model_validate(bundle.resolved_definition[model_key])
        matrix = generate_ask_matrix(
            questions,
            models,
            definition.selection,  # type: ignore[union-attr]
            definition.prompt_variants,  # type: ignore[union-attr]
            definition.inference_variants,  # type: ignore[union-attr]
            source.type,
            source.id,
            model_selection_attr=model_key,
            stage_label=expected_stage,
            source_artifacts=descriptors,
            prompt_contents=dict(bundle.prompt_contents),
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc

    output = definition.output.dataset_id  # type: ignore[union-attr]
    selected = matrix.rows
    output_paths = DatasetPath(workspace.results_root, _dataset_type(expected_stage), output)
    if only_missing and output_paths.base.exists():
        missing = set(get_missing_artifacts(output_paths))
        selected = [row for row in selected if row.artifact_id in missing]
    if dry_run:
        typer.echo(f"Planned calls: {len(selected)}")
        typer.echo("Dry run complete. No API calls made.")
        return
    if approve_call_count != len(selected):
        typer.echo(f"Error: execution requires --approve-call-count {len(selected)}", err=True)
        raise typer.Exit(code=ExecutionError.exit_code)
    preflight_bd_api()
    base_url = os.environ.get(definition.api.base_url_env, "")  # type: ignore[union-attr]
    if not base_url:
        raise typer.BadParameter(f"{definition.api.base_url_env} is not set")  # type: ignore[union-attr]
    adapter = RealBdApiAdapter(
        base_url,
        os.environ.get(definition.api.api_key_env),  # type: ignore[union-attr]
    )
    if isinstance(definition, AnswerDefinition):
        asyncio.run(
            run_answer_stage(
                workspace.results_root,
                definition,
                questions,
                models,
                adapter,
                only_missing=only_missing,
                execution_bundle=bundle,
                config_root=workspace.config_root,
            )
        )
    elif isinstance(definition, (RetrievalJudgmentDefinition, AnswerJudgmentDefinition)):
        stage = (
            Stage.RETRIEVAL_JUDGE
            if isinstance(definition, RetrievalJudgmentDefinition)
            else Stage.ANSWER_JUDGE
        )
        asyncio.run(
            run_judgment_stage(
                workspace.results_root,
                definition,
                questions,
                models,
                adapter,
                source.type,
                source.id,
                stage,
                only_missing=only_missing,
                execution_bundle=bundle,
                config_root=workspace.config_root,
            )
        )


def _dataset_type(stage: str) -> str:
    return {
        "retrieval_judgment": "retrieval_judgments",
        "answer": "answers",
        "answer_judgment": "answer_judgments",
    }[stage]


@app.command()
def validate(
    ctx: typer.Context,
    definition: DefinitionArg,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Resolve and validate configuration and templates."""
    from rag_evals.config.validate_cmd import validate_and_report

    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    validate_and_report(
        definition_path,
        output_format=format,
        project_root=workspace.config_root,
    )


@app.command()
def plan(
    ctx: typer.Context,
    definition: DefinitionArg,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Show matrix, exclusions, IDs, and call counts."""
    from rag_evals.planning.plan_cmd import plan_and_report

    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    plan_and_report(
        definition_path,
        output_format=format,
        project_root=workspace.config_root,
        results_root=workspace.results_root,
    )


@app.command(name="run-retrieval")
def run_retrieval(
    ctx: typer.Context,
    definition: DefinitionArg,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Plan only, no API calls.")] = False,
    only_missing: Annotated[
        bool, typer.Option("--only-missing", help="Resume only missing artifacts.")
    ] = False,
    question_id: Annotated[
        str | None, typer.Option("--question-id", help="Filter to a specific question.")
    ] = None,
    approve_call_count: Annotated[
        int | None, typer.Option("--approve-call-count", help="Required for live execution.")
    ] = None,
) -> None:
    """Build or resume a retrieval dataset."""
    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    _run_retrieval_command(
        workspace,
        definition_path,
        dry_run=dry_run,
        only_missing=only_missing,
        question_id=question_id,
        approve_call_count=approve_call_count,
    )


def _run_retrieval_command(
    workspace: Workspace,
    definition: Path,
    *,
    dry_run: bool,
    only_missing: bool,
    question_id: str | None,
    approve_call_count: int | None,
) -> None:
    """Execute retrieval command logic for direct runs and resume."""
    import asyncio

    from rag_evals.config.execution import load_execution_bundle
    from rag_evals.config.schemas import (
        KnowledgeBaseCatalog,
        QuestionCatalog,
        RetrievalDefinition,
    )
    from rag_evals.errors import ConfigError, ExecutionError
    from rag_evals.stages.retrieval import run_retrieval_stage

    try:
        bundle = load_execution_bundle(definition, workspace.config_root)
        defn = bundle.definition
        if not isinstance(defn, RetrievalDefinition):
            test_type = defn.test.type
            typer.echo(
                f"Error: Expected retrieval definition, got type '{test_type}'.",
                err=True,
            )
            raise typer.Exit(code=ConfigError.exit_code)
        resolved_data = bundle.resolved_definition
        questions = QuestionCatalog.model_validate(resolved_data["questions"])
        kbs = KnowledgeBaseCatalog.model_validate(resolved_data["knowledge_bases"])
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc

    if dry_run:
        from rag_evals.planning.matrices import generate_retrieval_matrix

        matrix = generate_retrieval_matrix(
            questions=questions,
            knowledge_bases=kbs,
            selection=defn.selection,
            retrieval=defn.retrieval,
            deployment=defn.deployment,
        )
        typer.echo(f"Planned artifacts: {matrix.total_calls}")
        typer.echo(f"Excluded: {len(matrix.excluded)}")
        typer.echo("Dry run complete. No API calls made.")
        return

    # Live execution requires call count approval
    from rag_evals.planning.matrices import generate_retrieval_matrix

    matrix = generate_retrieval_matrix(
        questions=questions,
        knowledge_bases=kbs,
        selection=defn.selection,
        retrieval=defn.retrieval,
        deployment=defn.deployment,
    )
    selected_rows = matrix.rows
    if question_id:
        selected_rows = [row for row in selected_rows if row.question_id == question_id]
    if only_missing:
        from rag_evals.artifacts.store import DatasetPath, get_missing_artifacts

        existing_paths = DatasetPath(
            workspace.results_root,
            "retrieval",
            defn.output.dataset_id,
        )
        if existing_paths.base.exists():
            missing = set(get_missing_artifacts(existing_paths))
            selected_rows = [row for row in selected_rows if row.artifact_id in missing]
    actual_call_count = len(selected_rows)

    if approve_call_count is None:
        typer.echo(
            f"Error: Live execution requires --approve-call-count {actual_call_count}",
            err=True,
        )
        raise typer.Exit(code=ExecutionError.exit_code)

    if approve_call_count != actual_call_count:
        typer.echo(
            f"Error: Call count mismatch. Plan requires {actual_call_count}, "
            f"but --approve-call-count is {approve_call_count}.",
            err=True,
        )
        raise typer.Exit(code=ExecutionError.exit_code)

    # Check for placeholder KB IDs
    for kb in kbs.knowledge_bases:
        if kb.is_placeholder:
            typer.echo(
                f"Error: Knowledge base '{kb.id}' has placeholder ID '{kb.knowledge_base_id}'. "
                f"Replace with a real ID before live execution.",
                err=True,
            )
            raise typer.Exit(code=ExecutionError.exit_code)

    # Create real adapter
    import os

    from rag_evals.api.real_adapter import RealBdApiAdapter, preflight_bd_api

    preflight_bd_api()

    base_url = os.environ.get(defn.api.base_url_env, "")
    api_key = os.environ.get(defn.api.api_key_env)

    if not base_url:
        typer.echo(f"Error: {defn.api.base_url_env} environment variable not set.", err=True)
        raise typer.Exit(code=ConfigError.exit_code)

    adapter = RealBdApiAdapter(base_url=base_url, api_key=api_key)

    try:
        asyncio.run(
            run_retrieval_stage(
                project_root=workspace.results_root,
                definition=defn,
                questions=questions,
                knowledge_bases=kbs,
                adapter=adapter,
                only_missing=only_missing,
                question_id_filter=question_id,
                execution_bundle=bundle,
            )
        )
    except Exception as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ExecutionError.exit_code) from exc


@app.command(name="judge-retrieval")
def judge_retrieval(
    ctx: typer.Context,
    definition: DefinitionArg,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    only_missing: Annotated[bool, typer.Option("--only-missing")] = False,
    approve_call_count: Annotated[int | None, typer.Option("--approve-call-count")] = None,
) -> None:
    """Judge saved retrieval artifacts."""
    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    _run_ask_command(
        workspace,
        definition_path,
        "retrieval_judgment",
        dry_run,
        only_missing,
        approve_call_count,
    )


@app.command(name="generate-answers")
def generate_answers(
    ctx: typer.Context,
    definition: DefinitionArg,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    only_missing: Annotated[bool, typer.Option("--only-missing")] = False,
    approve_call_count: Annotated[int | None, typer.Option("--approve-call-count")] = None,
) -> None:
    """Generate answers from frozen retrieval context."""
    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    _run_ask_command(
        workspace,
        definition_path,
        "answer",
        dry_run,
        only_missing,
        approve_call_count,
    )


@app.command(name="judge-answers")
def judge_answers(
    ctx: typer.Context,
    definition: DefinitionArg,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    only_missing: Annotated[bool, typer.Option("--only-missing")] = False,
    approve_call_count: Annotated[int | None, typer.Option("--approve-call-count")] = None,
) -> None:
    """Judge saved answer artifacts."""
    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    _run_ask_command(
        workspace,
        definition_path,
        "answer_judgment",
        dry_run,
        only_missing,
        approve_call_count,
    )


@app.command()
def report(
    ctx: typer.Context,
    definition: DefinitionArg,
    format: Annotated[
        str,
        typer.Option("--format", help="Output format: text or json."),
    ] = "text",
) -> None:
    """Produce CSV and Markdown reports."""
    from rag_evals.config.resolver import safe_load_yaml
    from rag_evals.errors import ConfigError
    from rag_evals.reporting.reports import generate_report
    from rag_evals.reporting.scoring import load_scoring_specs

    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    data = safe_load_yaml(definition_path)
    if format not in {"csv", "markdown", "both", "text", "json"}:
        raise typer.BadParameter("format must be csv, markdown, both, text, or json")
    datasets = data.get("datasets", {})
    try:
        scoring = load_scoring_specs(data.get("scoring"))
        output_dir = workspace.resolve_report_output(Path(data.get("output_dir", "latest")))
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc
    result = generate_report(
        workspace.results_root,
        output_dir,
        datasets.get("retrieval"),
        datasets.get("retrieval_judgments"),
        datasets.get("answers"),
        datasets.get("answer_judgments"),
        output_format=format,
        scoring=scoring,
    )
    typer.echo(f"Report generated: {result}")


@app.command()
def compare(
    ctx: typer.Context,
    definition: DefinitionArg,
    format: Annotated[str, typer.Option("--format")] = "csv",
) -> None:
    """Compare selected datasets."""
    from rag_evals.config.resolver import safe_load_yaml
    from rag_evals.config.schemas import CompareDefinition, parse_definition
    from rag_evals.reporting.reports import join_lineage, render_csv, render_markdown

    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    parsed = parse_definition(safe_load_yaml(definition_path))
    if not isinstance(parsed, CompareDefinition):
        raise typer.BadParameter("compare requires a compare definition")
    rows = []
    for entry in parsed.entries:
        if entry.dataset.type == "retrieval":
            rows.extend(join_lineage(workspace.results_root, retrieval_dataset_id=entry.dataset.id))
        elif entry.dataset.type == "retrieval_judgments":
            rows.extend(
                join_lineage(
                    workspace.results_root,
                    retrieval_judgment_dataset_id=entry.dataset.id,
                )
            )
        elif entry.dataset.type == "answers":
            rows.extend(join_lineage(workspace.results_root, answer_dataset_id=entry.dataset.id))
        elif entry.dataset.type == "answer_judgments":
            rows.extend(
                join_lineage(
                    workspace.results_root,
                    answer_judgment_dataset_id=entry.dataset.id,
                )
            )
    typer.echo(render_markdown(rows) if format == "markdown" else render_csv(rows))


@app.command()
def status(
    ctx: typer.Context,
    dataset_type: DatasetTypeArg,
    dataset_id: DatasetIdArg,
) -> None:
    """Inspect dataset lifecycle and work counts."""
    from rag_evals.artifacts.store import DatasetPath, get_status

    workspace = _require_workspace(ctx)
    _validate_dataset_reference(dataset_type, dataset_id)
    paths = DatasetPath(workspace.results_root, dataset_type, dataset_id)

    if not paths.base.exists():
        typer.echo(f"Dataset not found: {dataset_type}/{dataset_id}", err=True)
        raise typer.Exit(code=3)

    status = get_status(paths)
    typer.echo(f"Dataset: {dataset_type}/{dataset_id}")
    typer.echo(f"Lifecycle: {status['lifecycle']}")
    typer.echo(f"Planned: {status['planned_count']}")
    typer.echo(f"Successful: {status['successful_count']}")
    typer.echo(f"Failed: {status['failed_count']}")
    typer.echo(f"Missing: {status['missing_count']}")
    typer.echo(f"Unplanned: {len(status['unplanned_artifacts'])}")
    typer.echo(f"Corrupt: {len(status['corrupt_artifacts'])}")
    for error in status["invariant_errors"]:
        typer.echo(f"Invariant error: {error}")


@app.command()
def resume(
    ctx: typer.Context,
    definition: DefinitionArg,
    approve_call_count: Annotated[int | None, typer.Option("--approve-call-count")] = None,
) -> None:
    """Resume eligible missing or failed work."""
    from rag_evals.config.resolver import safe_load_yaml

    workspace = _require_workspace(ctx)
    definition_path = _resolve_definition(workspace, definition)
    stage = safe_load_yaml(definition_path).get("test", {}).get("type")
    if stage == "retrieval":
        _run_retrieval_command(
            workspace,
            definition_path,
            dry_run=False,
            only_missing=True,
            question_id=None,
            approve_call_count=approve_call_count,
        )
    elif stage in {"retrieval_judgment", "answer", "answer_judgment"}:
        _run_ask_command(
            workspace,
            definition_path,
            stage,
            False,
            True,
            approve_call_count,
        )
    else:
        raise typer.BadParameter(f"Unsupported resumable stage: {stage}")


@app.command()
def seal(
    ctx: typer.Context,
    dataset_type: DatasetTypeArg,
    dataset_id: DatasetIdArg,
) -> None:
    """Make a complete dataset immutable."""
    from rag_evals.artifacts.store import DatasetPath, Lifecycle, transition_lifecycle
    from rag_evals.errors import ArtifactError

    workspace = _require_workspace(ctx)
    _validate_dataset_reference(dataset_type, dataset_id)
    paths = DatasetPath(workspace.results_root, dataset_type, dataset_id)

    if not paths.base.exists():
        typer.echo(f"Dataset not found: {dataset_type}/{dataset_id}", err=True)
        raise typer.Exit(code=3)

    try:
        status = transition_lifecycle(paths, Lifecycle.SEALED)
        typer.echo(f"Sealed: {dataset_type}/{dataset_id}")
        typer.echo(f"Lifecycle: {status['lifecycle']}")
    except ArtifactError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=5) from exc


@app.command(name="clean-dataset")
def clean_dataset(
    ctx: typer.Context,
    dataset_type: DatasetTypeArg,
    dataset_id: DatasetIdArg,
    confirm_dataset_id: Annotated[
        str | None,
        typer.Option("--confirm-dataset-id", help="Exact dataset ID to confirm cleanup."),
    ] = None,
) -> None:
    """Plan or trash an eligible dataset."""
    from rag_evals.artifacts.cleanup import execute_cleanup, plan_cleanup
    from rag_evals.errors import ArtifactError

    workspace = _require_workspace(ctx)
    _validate_dataset_reference(dataset_type, dataset_id)

    if confirm_dataset_id is None:
        plan = plan_cleanup(workspace.results_root, dataset_type, dataset_id)
        typer.echo(f"Dataset: {plan.dataset_type}/{plan.dataset_id}")
        typer.echo(f"Exists: {plan.exists}")
        if plan.exists:
            typer.echo(f"Lifecycle: {plan.lifecycle}")
            typer.echo(f"Files: {plan.file_count}")
            typer.echo(f"Size: {plan.total_size_bytes} bytes")
            typer.echo(f"Can clean: {plan.can_clean}")
            if plan.reason:
                typer.echo(f"Reason: {plan.reason}")
            if plan.downstream_references:
                typer.echo(f"Downstream references: {len(plan.downstream_references)}")
            if plan.report_references:
                typer.echo(f"Report references: {len(plan.report_references)}")
            typer.echo(
                f"To execute: rag-evals clean-dataset {dataset_type} {dataset_id} "
                f"--confirm-dataset-id {dataset_id}"
            )
        return

    try:
        trash_path = execute_cleanup(
            workspace.results_root,
            dataset_type,
            dataset_id,
            confirm_dataset_id,
        )
        typer.echo(f"Moved to trash: {trash_path}")
    except ArtifactError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=5) from exc


def _validate_dataset_reference(dataset_type: str, dataset_id: str) -> None:
    """Reject malformed lifecycle command arguments before constructing paths."""
    import re

    from rag_evals.artifacts.store import DATASET_TYPES
    from rag_evals.config.schemas import DATASET_ID_PATTERN
    from rag_evals.errors import ConfigError

    if dataset_type not in DATASET_TYPES:
        typer.echo(f"Error: Invalid dataset type: {dataset_type}", err=True)
        raise typer.Exit(code=ConfigError.exit_code)
    if re.fullmatch(DATASET_ID_PATTERN, dataset_id) is None:
        typer.echo(f"Error: Invalid dataset ID: {dataset_id}", err=True)
        raise typer.Exit(code=ConfigError.exit_code)
