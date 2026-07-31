"""Strict Jinja2 template validation and rendering per D012.

Sandboxed with StrictUndefined. Rejects statements, calls, attribute methods,
imports, and unknown variables during validation. Only variable interpolation
and safe built-in serialization filters are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import yaml
from jinja2 import StrictUndefined, nodes
from jinja2.exceptions import SecurityError, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from rag_evals.errors import ConfigError


class Stage(StrEnum):
    """Template variable allowlist stages."""

    RETRIEVAL_JUDGE = "retrieval_judge"
    ANSWER = "answer"
    ANSWER_JUDGE = "answer_judge"


ALLOWED_VARIABLES: dict[Stage, set[str]] = {
    Stage.RETRIEVAL_JUDGE: {
        "question",
        "question_id",
        "rubric",
        "retrieval_context",
        "retrieved_items",
        "retrieval_metadata",
    },
    Stage.ANSWER: {
        "question",
        "question_id",
        "rubric",
        "retrieval_context",
    },
    Stage.ANSWER_JUDGE: {
        "question",
        "question_id",
        "rubric",
        "retrieval_context",
        "retrieved_items",
        "retrieval_metadata",
        "candidate_answer",
    },
}

ALLOWED_FILTERS: frozenset[str] = frozenset(
    {
        "to_yaml",
        "to_json",
        "length",
        "default",
    }
)


@dataclass
class TemplateValidationResult:
    """Result of validating a template."""

    template_text: str
    variables_used: set[str] = field(default_factory=set)
    filters_used: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return len(self.errors) == 0


def _safe_to_yaml(value: Any) -> str:
    """Safe YAML serialization filter."""
    result: str = yaml.safe_dump(
        value, default_flow_style=False, sort_keys=True, allow_unicode=True
    )
    return result


def _safe_to_json(value: Any) -> str:
    """Safe JSON serialization filter."""
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def create_sandbox_env() -> SandboxedEnvironment:
    """Create a sandboxed Jinja2 environment with StrictUndefined."""
    env = SandboxedEnvironment(
        undefined=StrictUndefined,
        autoescape=False,
    )
    env.filters["to_yaml"] = _safe_to_yaml
    env.filters["to_json"] = _safe_to_json
    return env


def extract_template_variables(template_text: str) -> set[str]:
    """Extract all variable names referenced in a Jinja2 template."""
    env = create_sandbox_env()
    try:
        ast_node = env.parse(template_text)
    except TemplateSyntaxError as exc:
        raise ConfigError(f"Template syntax error: {exc}") from exc

    variables: set[str] = set()
    for node in ast_node.find_all(nodes.Name):
        if node.ctx == "load":
            variables.add(node.name)
    return variables


def extract_template_filters(template_text: str) -> set[str]:
    """Extract all filter names used in a Jinja2 template."""
    env = create_sandbox_env()
    try:
        ast_node = env.parse(template_text)
    except TemplateSyntaxError as exc:
        raise ConfigError(f"Template syntax error: {exc}") from exc

    filters: set[str] = set()
    for node in ast_node.find_all(nodes.Filter):
        filters.add(node.name)
    return filters


def validate_template(template_text: str, stage: Stage) -> TemplateValidationResult:
    """Validate a template against stage-specific rules.

    - Checks syntax is valid Jinja2.
    - Rejects statements, calls, imports, and unsafe constructs.
    - Checks all variables are in the stage allowlist.
    - Checks all filters are in the allowed set.
    """
    result = TemplateValidationResult(template_text=template_text)

    env = create_sandbox_env()

    try:
        ast_node = env.parse(template_text)
    except TemplateSyntaxError as exc:
        result.errors.append(f"Template syntax error: {exc}")
        return result

    # Check for disallowed node types
    disallowed_checks: list[tuple[type, str]] = [
        (nodes.Call, "Function calls are not allowed in templates"),
        (nodes.Import, "Imports are not allowed in templates"),
        (nodes.FromImport, "Imports are not allowed in templates"),
        (nodes.Include, "Includes are not allowed in templates"),
        (nodes.Macro, "Macro definitions are not allowed in templates"),
        (nodes.Block, "Block definitions are not allowed in templates"),
        (nodes.Extends, "Template inheritance is not allowed"),
    ]

    for node_type, error_msg in disallowed_checks:
        if list(ast_node.find_all(node_type)):
            result.errors.append(error_msg)

    # Check for statement blocks ({% ... %})
    try:
        if "{%" in template_text:
            result.errors.append("Statement blocks ({% %}) are not allowed")
    except Exception:
        pass

    # Extract and check variables
    result.variables_used = extract_template_variables(template_text)
    allowed = ALLOWED_VARIABLES[stage]
    unknown_vars = result.variables_used - allowed
    if unknown_vars:
        result.errors.append(
            f"Unknown variables for stage {stage.value}: {sorted(unknown_vars)}. "
            f"Allowed: {sorted(allowed)}"
        )

    # Extract and check filters
    result.filters_used = extract_template_filters(template_text)
    unknown_filters = result.filters_used - ALLOWED_FILTERS
    if unknown_filters:
        result.errors.append(
            f"Unknown filters: {sorted(unknown_filters)}. Allowed: {sorted(ALLOWED_FILTERS)}"
        )

    return result


def render_template(template_text: str, variables: dict[str, Any]) -> str:
    """Render a template with the given variables. Fails on undefined vars."""
    env = create_sandbox_env()
    try:
        template = env.from_string(template_text)
        return template.render(**variables)
    except UndefinedError as exc:
        raise ConfigError(f"Template rendering error - undefined variable: {exc}") from exc
    except SecurityError as exc:
        raise ConfigError(f"Template security error: {exc}") from exc
    except Exception as exc:
        raise ConfigError(f"Template rendering error: {exc}") from exc


def load_template_file(path_str: str, containing_dir: Any = None) -> str:
    """Load a template from a YAML file with a 'template' key, or raw text."""
    from pathlib import Path

    path = Path(path_str)
    if containing_dir is not None:
        path = (Path(containing_dir) / path).resolve()

    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
        if isinstance(data, dict) and "template" in data:
            return str(data["template"])
        raise ConfigError(f"Template YAML file must contain a 'template' key: {path}")

    return text
