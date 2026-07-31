"""Fully resolved, hash-addressed execution configuration."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from rag_evals.config.resolver import (
    SourceInput,
    SourceResolver,
    canonical_hash,
    read_text,
    safe_load_yaml,
)
from rag_evals.config.schemas import Definition, parse_definition
from rag_evals.errors import ConfigError


@dataclass(frozen=True)
class ExecutionBundle:
    """Validated definition plus every resolved byte used to plan or execute it."""

    definition_path: Path
    definition: Definition
    resolved_definition: Mapping[str, Any]
    resolved_definition_hash: str
    inputs: tuple[SourceInput, ...]
    input_hashes: Mapping[str, str]
    prompt_contents: Mapping[str, str]
    prompt_hashes: Mapping[str, str]

    def snapshot(self) -> dict[str, Any]:
        """Return the canonical, secret-free resolved snapshot persisted in datasets."""
        return {
            "bundle_schema_version": 1,
            "definition": dict(self.resolved_definition),
            "resolved_definition_hash": self.resolved_definition_hash,
            "input_hashes": dict(self.input_hashes),
            "prompt_hashes": dict(self.prompt_hashes),
        }

    def snapshot_inputs(self) -> list[dict[str, Any]]:
        """Return source and prompt bytes suitable for store snapshotting."""
        values = [
            {
                "source_path": str(item.source_path),
                "content": item.content,
                "content_hash": item.content_hash,
            }
            for item in self.inputs
        ]
        values.extend(
            {
                "source_path": path,
                "content": content,
                "content_hash": self.prompt_hashes[path],
            }
            for path, content in self.prompt_contents.items()
        )
        return values


def load_execution_bundle(definition_path: Path, project_root: Path) -> ExecutionBundle:
    """Resolve and validate all configuration and prompt inputs before side effects."""
    raw = safe_load_yaml(definition_path)
    parsed = parse_definition(raw)
    resolver = SourceResolver(project_root)
    resolved = resolver.resolve_definition(definition_path)
    prompts = _load_prompts(parsed, project_root)
    input_hashes = {
        str(item.resolved_path.relative_to(project_root)): item.content_hash
        for item in resolved.inputs
    }
    prompt_hashes = {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in prompts.items()
    }
    snapshot_payload = {
        "definition": resolved.definition_data,
        "input_hashes": input_hashes,
        "prompt_hashes": prompt_hashes,
    }
    return ExecutionBundle(
        definition_path=definition_path.resolve(),
        definition=parsed,  # type: ignore[arg-type]
        resolved_definition=MappingProxyType(resolved.definition_data),
        resolved_definition_hash=canonical_hash(snapshot_payload),
        inputs=tuple(resolved.inputs),
        input_hashes=MappingProxyType(input_hashes),
        prompt_contents=MappingProxyType(prompts),
        prompt_hashes=MappingProxyType(prompt_hashes),
    )


def _load_prompts(definition: BaseModel, project_root: Path) -> dict[str, str]:
    variants = getattr(definition, "prompt_variants", ())
    result: dict[str, str] = {}
    for variant in variants:
        for value in (variant.template.system_instructions, variant.template.user_message):
            candidate = (project_root / value).resolve()
            try:
                relative = candidate.relative_to(project_root)
            except ValueError:
                raise ConfigError(f"Prompt path resolves outside project root: {value}") from None
            if candidate.exists():
                result[str(relative)] = read_text(candidate)
            else:
                result[f"inline:{canonical_hash(value)}"] = value
    return dict(sorted(result.items()))
