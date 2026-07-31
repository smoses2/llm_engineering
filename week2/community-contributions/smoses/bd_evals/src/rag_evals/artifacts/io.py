"""Atomic IO: JSON, YAML, and text writes with fsync and os.replace."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from rag_evals.errors import ArtifactError
from rag_evals.planning.identity import canonical_hash


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically: temp file, flush, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, data: Any) -> str:
    """Write JSON atomically with canonical formatting. Returns content hash."""
    text = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2)
    atomic_write_text(path, text)
    return canonical_hash(data)


def atomic_write_yaml(path: Path, data: Any) -> str:
    """Write YAML atomically. Returns content hash."""
    text = yaml.dump(data, default_flow_style=False, sort_keys=True, allow_unicode=True, width=120)
    atomic_write_text(path, text)
    return canonical_hash(data)


def read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ArtifactError(f"File not found: {path}") from None
    result: dict[str, Any] = json.loads(text)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ArtifactError(f"File not found: {path}") from None
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ArtifactError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def verify_file_hash(path: Path, expected_hash: str) -> bool:
    """Verify a file's content matches the expected SHA-256 hash."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    actual = canonical_hash(data)
    return actual == expected_hash


def exclusive_create(path: Path, data: Any) -> bool:
    """Create a file exclusively. Returns True if created, False if exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return True
        except FileExistsError:
            return False
    finally:
        temporary.unlink(missing_ok=True)
