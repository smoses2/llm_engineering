"""Lineage references and loaders with hash verification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_evals.artifacts.io import read_json, read_yaml
from rag_evals.artifacts.store import DatasetPath, Lifecycle, _read_status, get_status
from rag_evals.errors import ArtifactError
from rag_evals.planning.identity import canonical_hash


@dataclass(frozen=True)
class LineageRef:
    """Typed reference to a source artifact."""

    source_dataset_type: str
    source_dataset_id: str
    source_artifact_id: str
    relative_path: str
    expected_content_hash: str


@dataclass(frozen=True)
class SourceArtifact:
    """Verified source descriptor used as the first downstream matrix dimension."""

    artifact_id: str
    content_hash: str
    relative_path: str
    question_id: str
    question_text: str
    rubric: dict[str, Any] | None


def _digest(artifact_id: str) -> str:
    """Extract the hex digest from a 'sha256:...' artifact ID."""
    return artifact_id[7:] if artifact_id.startswith("sha256:") else artifact_id


def create_lineage_ref(
    project_root: Path,
    source_dataset_type: str,
    source_dataset_id: str,
    source_artifact_id: str,
    content_hash: str,
) -> LineageRef:
    """Create a lineage reference to a source artifact."""
    paths = DatasetPath(project_root, source_dataset_type, source_dataset_id)
    artifact_path = paths.artifacts_dir / f"{_digest(source_artifact_id)}.json"
    rel = str(artifact_path.relative_to(project_root))

    return LineageRef(
        source_dataset_type=source_dataset_type,
        source_dataset_id=source_dataset_id,
        source_artifact_id=source_artifact_id,
        relative_path=rel,
        expected_content_hash=content_hash,
    )


def verify_lineage(project_root: Path, ref: LineageRef) -> dict[str, Any]:
    """Verify a lineage reference and return the source artifact data.

    Checks:
    - Source dataset exists
    - Source artifact exists
    - Content hash matches
    - Source dataset is not building (must be complete or sealed)
    """
    paths = DatasetPath(project_root, ref.source_dataset_type, ref.source_dataset_id)

    if not paths.base.exists():
        raise ArtifactError(
            f"Source dataset not found: {ref.source_dataset_type}/{ref.source_dataset_id}"
        )

    status = _read_status(paths)
    lifecycle = status.get("lifecycle", Lifecycle.BUILDING.value)
    if lifecycle not in (Lifecycle.COMPLETE.value, Lifecycle.SEALED.value):
        raise ArtifactError(
            f"Source dataset {ref.source_dataset_type}/{ref.source_dataset_id} "
            f"is still building; cannot use as lineage source"
        )

    digest = _digest(ref.source_artifact_id)
    manifest = read_yaml(paths.manifest)
    if ref.source_artifact_id not in manifest.get("planned_artifact_ids", []):
        raise ArtifactError("Source artifact is not a member of its dataset manifest")
    artifact_path = paths.artifacts_dir / f"{digest}.json"
    expected_path = (project_root / ref.relative_path).resolve()
    if expected_path != artifact_path.resolve():
        raise ArtifactError("Lineage relative path does not match the source artifact ID")

    if not artifact_path.exists():
        raise ArtifactError(
            f"Source artifact not found: {digest[:16]}... in "
            f"{ref.source_dataset_type}/{ref.source_dataset_id}"
        )

    data = read_json(artifact_path)
    actual_hash = canonical_hash(data)

    if actual_hash != ref.expected_content_hash:
        raise ArtifactError(
            f"Source artifact hash mismatch: expected {ref.expected_content_hash[:16]}..., "
            f"got {actual_hash[:16]}..."
        )

    return data


def load_source_dataset(
    project_root: Path, dataset_type: str, dataset_id: str
) -> list[SourceArtifact]:
    """Load every planned source artifact after lifecycle, membership, and ID verification."""
    paths = DatasetPath(project_root, dataset_type, dataset_id)
    status = _read_status(paths)
    lifecycle = status.get("lifecycle")
    if lifecycle not in (Lifecycle.COMPLETE.value, Lifecycle.SEALED.value):
        raise ArtifactError(f"Source dataset must be complete or sealed, got {lifecycle}")
    reconciled = get_status(paths)
    if reconciled["missing_count"] or reconciled["invariant_errors"]:
        raise ArtifactError(f"Source dataset violates invariants: {reconciled['invariant_errors']}")
    manifest = read_yaml(paths.manifest)
    descriptors: list[SourceArtifact] = []
    for artifact_id in manifest.get("planned_artifact_ids", []):
        digest = _digest(artifact_id)
        path = paths.artifacts_dir / f"{digest}.json"
        data = read_json(path)
        if data.get("artifact_id") != artifact_id:
            raise ArtifactError(f"Source artifact identity mismatch: {path}")
        descriptors.append(
            SourceArtifact(
                artifact_id=artifact_id,
                content_hash=canonical_hash(data),
                relative_path=str(path.relative_to(project_root)),
                question_id=str(data.get("question_id", "")),
                question_text=str(data.get("question_text", "")),
                rubric=data.get("rubric"),
            )
        )
    return descriptors


def lineage_ref_to_dict(ref: LineageRef) -> dict[str, str]:
    """Convert a LineageRef to a serializable dict."""
    return {
        "source_dataset_type": ref.source_dataset_type,
        "source_dataset_id": ref.source_dataset_id,
        "source_artifact_id": ref.source_artifact_id,
        "relative_path": ref.relative_path,
        "expected_content_hash": ref.expected_content_hash,
    }


def lineage_ref_from_dict(data: dict[str, Any]) -> LineageRef:
    """Create a LineageRef from a dict."""
    return LineageRef(
        source_dataset_type=data["source_dataset_type"],
        source_dataset_id=data["source_dataset_id"],
        source_artifact_id=data["source_artifact_id"],
        relative_path=data["relative_path"],
        expected_content_hash=data["expected_content_hash"],
    )
