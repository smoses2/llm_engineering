"""Validate command implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from rag_evals.config.resolver import SourceResolver, canonical_hash, safe_load_yaml
from rag_evals.config.schemas import (
    KnowledgeBaseCatalog,
    ModelCatalog,
    QuestionCatalog,
    parse_definition,
)
from rag_evals.errors import ConfigError
from rag_evals.logging import get_logger


def _find_project_root(start: Path) -> Path:
    """Find the project root by looking for pyproject.toml."""
    current = start.resolve().parent if start.is_file() else start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return start.resolve().parent


def validate_definition(
    definition_path: Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate a definition file.

    Returns a dict with definition data, resolved inputs, and validation status.
    Raises ConfigError on any validation failure.
    """
    logger = get_logger("rag_evals.validate")

    if not definition_path.exists():
        raise ConfigError(f"Definition file not found: {definition_path}")

    root = project_root or _find_project_root(definition_path)

    # Step 1: Load raw definition (without resolving source refs)
    raw_data = safe_load_yaml(definition_path)

    # Step 2: Validate definition schema (with SourceRef fields)
    parse_definition(raw_data)
    test_type: str = raw_data["test"]["type"]
    test_id: str = raw_data["test"]["id"]

    logger.info("Validated definition: type=%s, test_id=%s", test_type, test_id)

    # Step 3: Resolve all source references
    resolver = SourceResolver(root)
    resolved = resolver.resolve_definition(definition_path)

    def_hash = canonical_hash(raw_data)

    # Step 4: Validate loaded catalogs and selections
    warnings: list[str] = []

    _validate_catalogs_and_selections(raw_data, resolved.definition_data, warnings)

    return {
        "definition_path": definition_path,
        "definition_hash": def_hash,
        "definition_data": raw_data,
        "resolved_data": resolved.definition_data,
        "inputs": resolved.inputs,
        "test_type": test_type,
        "test_id": test_id,
        "warnings": warnings,
        "status": "valid",
    }


def _validate_catalogs_and_selections(
    raw_data: dict[str, Any],
    resolved_data: dict[str, Any],
    warnings: list[str],
) -> None:
    """Validate loaded catalogs and apply selection rules."""
    from rag_evals.config.schemas import Selection

    selection: Selection | None = None
    if "selection" in raw_data:
        selection = Selection.model_validate(raw_data["selection"])

    if "questions" in resolved_data and isinstance(resolved_data["questions"], dict):
        q_cat = QuestionCatalog.model_validate(resolved_data["questions"])
        if selection and selection.questions:
            from rag_evals.config.selection import select_questions

            select_questions(q_cat, selection.questions)

    if "knowledge_bases" in resolved_data and isinstance(resolved_data["knowledge_bases"], dict):
        kb_cat = KnowledgeBaseCatalog.model_validate(resolved_data["knowledge_bases"])
        for kb in kb_cat.knowledge_bases:
            if kb.is_placeholder:
                warnings.append(
                    f"Knowledge base '{kb.id}' has placeholder ID: {kb.knowledge_base_id}"
                )
        if selection and selection.knowledge_bases:
            from rag_evals.config.selection import select_knowledge_bases

            select_knowledge_bases(kb_cat, selection.knowledge_bases)

    for key, label in [("answer_models", "answer"), ("judge_models", "judge")]:
        if key in resolved_data and isinstance(resolved_data[key], dict):
            m_cat = ModelCatalog.model_validate(resolved_data[key])
            for m in m_cat.models:
                if m.is_placeholder:
                    warnings.append(f"{label} model '{m.id}' has placeholder ID: {m.model_id}")
            if selection:
                sel = getattr(selection, key, None)
                if sel:
                    from rag_evals.config.selection import select_models

                    select_models(m_cat, sel, f"{label} model")


def validate_and_report(
    definition_path: Path,
    output_format: str = "text",
    project_root: Path | None = None,
) -> None:
    """Validate a definition and print results."""
    try:
        result = validate_definition(definition_path, project_root)
    except ConfigError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=ConfigError.exit_code) from exc

    if output_format == "json":
        _print_json(result)
    else:
        _print_text(result)


def _print_text(result: dict[str, Any]) -> None:
    typer.echo(f"Definition: {result['definition_path']}")
    typer.echo(f"Type: {result['test_type']}")
    typer.echo(f"Test ID: {result['test_id']}")
    typer.echo(f"Hash: {result['definition_hash'][:16]}...")
    inputs = result["inputs"]
    typer.echo(f"Inputs ({len(inputs)}):")
    for inp in inputs:
        typer.echo(f"  {inp.source_path} -> {inp.resolved_path} ({inp.content_hash[:16]}...)")
    for w in result["warnings"]:
        typer.echo(f"Warning: {w}")
    typer.echo("Status: valid")


def _print_json(result: dict[str, Any]) -> None:
    data: dict[str, Any] = {
        "definition_path": str(result["definition_path"]),
        "definition_hash": result["definition_hash"],
        "test_type": result["test_type"],
        "test_id": result["test_id"],
        "inputs": [
            {
                "source_path": str(inp.source_path),
                "resolved_path": str(inp.resolved_path),
                "content_hash": inp.content_hash,
            }
            for inp in result["inputs"]
        ],
        "warnings": result["warnings"],
        "status": result["status"],
    }
    typer.echo(json.dumps(data, indent=2, default=str))
