"""YAML source resolver: safe loading, relative resolution, boundary checks, cycles, hashes."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rag_evals.errors import ConfigError

_SENTINEL = object()


@dataclass(frozen=True)
class SourceInput:
    """A resolved source file with its content and hash."""

    source_path: Path
    resolved_path: Path
    content: str
    content_hash: str


@dataclass
class ResolvedConfig:
    """Result of resolving a definition and all its source references."""

    definition_path: Path
    definition_data: dict[str, Any]
    definition_hash: str
    inputs: list[SourceInput] = field(default_factory=list)
    input_map: dict[Path, SourceInput] = field(default_factory=dict)

    @property
    def project_root(self) -> Path:
        return self.definition_path.parent


def canonical_hash(data: Any) -> str:
    """Compute SHA-256 of canonical UTF-8 JSON: sorted keys, compact separators."""
    import json

    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def safe_load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file safely and validate it's a mapping."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"File not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        raise ConfigError(f"Empty YAML file: {path}")
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping, got {type(data).__name__} in {path}")
    return data


def read_text(path: Path) -> str:
    """Read a text file, raising ConfigError on failure."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"File not found: {path}") from None
    except OSError as exc:
        raise ConfigError(f"Cannot read file {path}: {exc}") from exc


class SourceResolver:
    """Resolves {source: relative/path} references from containing files.

    - Resolves paths relative to the containing file's directory.
    - Enforces all resolved paths stay below the project root (bd_evals).
    - Detects cycles in the source graph.
    - Snapshots each input once.
    - Computes content hashes.
    """

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root.resolve()
        self._stack: list[Path] = []
        self._visited: dict[Path, SourceInput] = {}
        self._resolved_cache: dict[Path, Any] = {}

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def inputs(self) -> list[SourceInput]:
        return list(self._visited.values())

    def resolve_definition(self, definition_path: Path) -> ResolvedConfig:
        """Load a definition file and resolve all its source references."""
        resolved_def = self._resolve_path(definition_path)
        data = safe_load_yaml(resolved_def)

        def_hash = canonical_hash(data)

        self._stack = [resolved_def]
        self._visited = {}
        self._resolved_cache = {}

        resolved_data = self._resolve_source_refs(data, resolved_def)

        return ResolvedConfig(
            definition_path=definition_path,
            definition_data=resolved_data,
            definition_hash=def_hash,
            inputs=list(self._visited.values()),
            input_map=dict(self._visited),
        )

    def _resolve_path(self, path: Path) -> Path:
        """Resolve and validate a path stays within project root."""
        resolved = path.resolve()
        try:
            resolved.relative_to(self._project_root)
        except ValueError:
            raise ConfigError(
                f"Path '{path}' resolves outside project root '{self._project_root}'"
            ) from None
        return resolved

    def _resolve_relative(self, source: str, containing_dir: Path) -> Path:
        """Resolve a source path relative to the containing file's directory."""
        raw = Path(source)
        if raw.is_absolute():
            raise ConfigError(f"Absolute paths not allowed in source references: '{source}'")
        resolved = (containing_dir / raw).resolve()
        self._check_boundary(resolved)
        return resolved

    def _check_boundary(self, path: Path) -> None:
        """Ensure path stays within project root."""
        try:
            path.relative_to(self._project_root)
        except ValueError:
            raise ConfigError(
                f"Resolved path '{path}' is outside project root '{self._project_root}'"
            ) from None

    def _resolve_source_refs(self, data: Any, containing_file: Path) -> Any:
        """Recursively resolve {source: ...} references in a data structure."""
        if isinstance(data, dict):
            if "source" in data and len(data) == 1:
                return self._load_source(data["source"], containing_file)
            return {k: self._resolve_source_refs(v, containing_file) for k, v in data.items()}
        if isinstance(data, list):
            return [self._resolve_source_refs(item, containing_file) for item in data]
        return data

    def _load_source(self, source: str, containing_file: Path) -> Any:
        """Load a source file, detect cycles, and snapshot."""
        resolved = self._resolve_relative(source, containing_file.parent)

        if resolved in self._stack:
            cycle = " -> ".join(str(p) for p in self._stack + [resolved])
            raise ConfigError(f"Circular source reference detected: {cycle}")

        if resolved in self._resolved_cache:
            return deepcopy(self._resolved_cache[resolved])

        if not resolved.exists():
            raise ConfigError(
                f"Source file not found: {resolved} (referenced from {containing_file})"
            )

        text = read_text(resolved)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_input = SourceInput(
            source_path=Path(source),
            resolved_path=resolved,
            content=text,
            content_hash=content_hash,
        )
        self._visited[resolved] = source_input

        self._stack.append(resolved)
        try:
            if resolved.suffix in (".yaml", ".yml"):
                loaded = safe_load_yaml(resolved)
                resolved_data = self._resolve_source_refs(loaded, resolved)
            else:
                resolved_data = text
        finally:
            self._stack.pop()

        self._resolved_cache[resolved] = deepcopy(resolved_data)
        return deepcopy(resolved_data)


def atomic_write(path: Path, data: bytes) -> None:
    """Write data atomically using a temp file, fsync, and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.replace(str(tmp), str(path))
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
