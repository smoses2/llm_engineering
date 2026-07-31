"""Dataset store: manifests, snapshots, status, lifecycle, and lineage."""

from __future__ import annotations

import errno
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from rag_evals.artifacts.io import (
    atomic_write_yaml,
    exclusive_create,
    read_json,
    read_yaml,
)
from rag_evals.errors import ArtifactError
from rag_evals.logging import get_logger
from rag_evals.planning.identity import canonical_hash

logger = get_logger("rag_evals.artifacts")

DATASET_TYPES = ("retrieval", "retrieval_judgments", "answers", "answer_judgments")
TRASH_DIR = ".trash"
MANIFEST_SCHEMA_VERSION = 2
ARTIFACT_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class Lifecycle(StrEnum):
    __test__ = False

    BUILDING = "building"
    COMPLETE = "complete"
    SEALED = "sealed"
    FAILED = "failed"


def utc_now() -> str:
    """Return current UTC timestamp in RFC 3339 format ending in Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class DatasetPath:
    """Resolved paths for a dataset."""

    root: Path
    dataset_type: str
    dataset_id: str

    @property
    def base(self) -> Path:
        return self.root / "datasets" / self.dataset_type / self.dataset_id

    @property
    def manifest(self) -> Path:
        return self.base / "manifest.yaml"

    @property
    def resolved_definition(self) -> Path:
        return self.base / "resolved_definition.yaml"

    @property
    def status_file(self) -> Path:
        return self.base / "status.yaml"

    @property
    def snapshots_dir(self) -> Path:
        return self.base / "snapshots"

    @property
    def snapshots_index(self) -> Path:
        return self.snapshots_dir / "index.yaml"

    @property
    def artifacts_dir(self) -> Path:
        return self.base / "artifacts"

    @property
    def failures_dir(self) -> Path:
        return self.base / "failures"

    @property
    def summaries_dir(self) -> Path:
        return self.base / "summaries"

    @property
    def trash_path(self) -> Path:
        return self.root / "datasets" / TRASH_DIR / self.dataset_type


def create_dataset(
    project_root: Path,
    dataset_type: str,
    dataset_id: str,
    resolved_definition: dict[str, Any],
    plan_rows: list[dict[str, Any]],
    config_inputs: list[dict[str, Any]] | None = None,
) -> DatasetPath:
    """Create a new dataset with manifest, resolved definition, and status.

    Raises ArtifactError if the dataset already exists with a different definition hash
    or if the dataset is sealed.
    """
    if dataset_type not in DATASET_TYPES:
        raise ArtifactError(f"Invalid dataset type: {dataset_type}")

    paths = DatasetPath(project_root, dataset_type, dataset_id)
    if paths.base.exists():
        existing_status = _read_status(paths)
        if existing_status.get("lifecycle") == Lifecycle.SEALED.value:
            raise ArtifactError(f"Dataset {dataset_type}/{dataset_id} is sealed")

        # Check manifest schema version compatibility
        if paths.manifest.exists():
            existing_manifest = read_yaml(paths.manifest)
            existing_schema = existing_manifest.get("schema_version", 1)
            if existing_schema != MANIFEST_SCHEMA_VERSION:
                raise ArtifactError(
                    f"Dataset {dataset_type}/{dataset_id} has incompatible manifest "
                    f"schema version {existing_schema}; this framework requires "
                    f"version {MANIFEST_SCHEMA_VERSION}. Clean the old dataset "
                    f"and recreate it."
                )

    planned_ids = [r.get("artifact_id") for r in plan_rows]
    if any(
        not isinstance(value, str) or not ARTIFACT_ID_RE.fullmatch(value) for value in planned_ids
    ):
        raise ArtifactError("Every planned artifact ID must be sha256:<64 lowercase hex>")
    if len(set(planned_ids)) != len(planned_ids):
        raise ArtifactError("Plan contains duplicate artifact IDs")
    resolved_hash = canonical_hash(resolved_definition)
    plan_hash = canonical_hash(planned_ids)

    if paths.base.exists():
        # Check definition hash matches
        existing_def = read_yaml(paths.resolved_definition)
        existing_hash = canonical_hash(existing_def)
        existing_manifest = read_yaml(paths.manifest)
        if (
            existing_hash != resolved_hash
            or existing_manifest.get("resolved_definition_hash") != resolved_hash
        ):
            raise ArtifactError(
                f"Dataset {dataset_type}/{dataset_id} exists with a different "
                f"resolved-definition hash. Use a different dataset ID."
            )
        if existing_manifest.get("plan_hash") != plan_hash:
            raise ArtifactError(
                f"Dataset {dataset_type}/{dataset_id} exists with a different plan hash. "
                "Use a different dataset ID."
            )
        existing_lifecycle = existing_status.get("lifecycle")
        if existing_lifecycle in (Lifecycle.COMPLETE.value, Lifecycle.SEALED.value):
            reconciled = get_status(paths)
            if reconciled["invariant_errors"]:
                raise ArtifactError(
                    f"Immutable dataset {dataset_type}/{dataset_id} violates invariants: "
                    f"{reconciled['invariant_errors']}"
                )
        return paths

    parent = paths.base.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.initializing-", dir=parent))
    temp_paths = DatasetPath(project_root, dataset_type, temporary.name)

    # Write manifest
    manifest = {
        "dataset_type": dataset_type,
        "dataset_id": dataset_id,
        "created_utc": utc_now(),
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "resolved_definition_hash": resolved_hash,
        "plan_hash": plan_hash,
        "package_version": "0.1.0",
        "planned_artifact_ids": planned_ids,
        "planned_count": len(plan_rows),
    }
    try:
        temp_paths.artifacts_dir.mkdir()
        temp_paths.failures_dir.mkdir()
        temp_paths.summaries_dir.mkdir()
        temp_paths.snapshots_dir.mkdir()
        atomic_write_yaml(temp_paths.manifest, manifest)
        atomic_write_yaml(temp_paths.resolved_definition, resolved_definition)
        atomic_write_yaml(
            temp_paths.status_file,
            {
                "lifecycle": Lifecycle.BUILDING.value,
                "planned_count": len(plan_rows),
                "successful_count": 0,
                "failed_count": 0,
                "missing_count": len(plan_rows),
                "updated_utc": utc_now(),
            },
        )
        if config_inputs:
            _write_snapshots(temp_paths, config_inputs)
        try:
            os.rename(temporary, paths.base)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            shutil.rmtree(temporary, ignore_errors=True)
            return create_dataset(
                project_root,
                dataset_type,
                dataset_id,
                resolved_definition,
                plan_rows,
                config_inputs,
            )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    logger.info(
        "Created dataset %s/%s with %d planned artifacts",
        dataset_type,
        dataset_id,
        len(plan_rows),
    )
    return paths


def _write_snapshots(paths: DatasetPath, inputs: list[dict[str, Any]]) -> None:
    """Write config input snapshots with collision-safe names."""
    index: list[dict[str, Any]] = []
    inputs_dir = paths.snapshots_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)

    for i, inp in enumerate(inputs):
        source_path = inp.get("source_path", f"input_{i}")
        content = inp.get("content", "")
        content_hash = inp.get("content_hash", canonical_hash(content))

        safe_name = f"input_{i:04d}.yaml"
        snap_path = inputs_dir / safe_name
        atomic_write_text(snap_path, content)

        index.append(
            {
                "source_path": str(source_path),
                "snapshot_path": str(snap_path.relative_to(paths.base)),
                "content_hash": content_hash,
            }
        )

    atomic_write_yaml(paths.snapshots_index, {"inputs": index})


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically (local import to avoid circular dependency)."""
    from rag_evals.artifacts.io import atomic_write_bytes

    atomic_write_bytes(path, text.encode("utf-8"))


def _read_status(paths: DatasetPath) -> dict[str, Any]:
    """Read the status file."""
    if not paths.status_file.exists():
        return {"lifecycle": Lifecycle.BUILDING.value}
    return read_yaml(paths.status_file)


def get_status(paths: DatasetPath) -> dict[str, Any]:
    """Get current dataset status, reconciled from actual artifacts."""
    status = _read_status(paths)
    lifecycle = status.get("lifecycle", Lifecycle.BUILDING.value)

    # Reconcile counts from actual files
    artifacts = list(paths.artifacts_dir.glob("*.json"))
    failures = list(paths.failures_dir.glob("*"))
    manifest = read_yaml(paths.manifest) if paths.manifest.exists() else {}
    planned = manifest.get("planned_count", 0)

    successful_ids: set[str] = set()
    corrupt_files: list[str] = []
    for file in artifacts:
        try:
            data = read_json(file)
            expected = f"sha256:{file.stem}"
            if data.get("artifact_id") != expected:
                corrupt_files.append(file.name)
            else:
                successful_ids.add(file.stem)
        except Exception:
            corrupt_files.append(file.name)
    planned_ids = manifest.get("planned_artifact_ids", [])
    # Normalize planned IDs to digest-only for comparison
    planned_digests = {aid[7:] if aid.startswith("sha256:") else aid for aid in planned_ids}
    missing_count = len(planned_digests - successful_ids)
    unplanned = sorted(successful_ids - planned_digests)
    valid_successes = successful_ids & planned_digests
    invariant_errors = []
    if unplanned:
        invariant_errors.append(f"unplanned artifacts: {unplanned}")
    if corrupt_files:
        invariant_errors.append(f"corrupt artifacts: {sorted(corrupt_files)}")
    if lifecycle in (Lifecycle.COMPLETE.value, Lifecycle.SEALED.value) and missing_count:
        invariant_errors.append(f"immutable dataset has {missing_count} missing artifacts")

    return {
        "lifecycle": lifecycle,
        "planned_count": planned,
        "successful_count": len(valid_successes),
        "failed_count": len(failures),
        "missing_count": missing_count,
        "updated_utc": status.get("updated_utc", utc_now()),
        "unplanned_artifacts": unplanned,
        "corrupt_artifacts": sorted(corrupt_files),
        "invariant_errors": invariant_errors,
    }


def update_status(paths: DatasetPath) -> dict[str, Any]:
    """Reconcile and update status from actual artifacts. Returns new status."""
    existing = _read_status(paths)
    if existing.get("lifecycle") != Lifecycle.BUILDING.value:
        raise ArtifactError(
            f"Dataset {paths.dataset_type}/{paths.dataset_id} is "
            f"{existing.get('lifecycle')}; status is immutable"
        )
    status = get_status(paths)

    status["updated_utc"] = utc_now()
    atomic_write_yaml(paths.status_file, status)
    return status


def transition_lifecycle(paths: DatasetPath, target: Lifecycle) -> dict[str, Any]:
    """Transition dataset lifecycle. Enforces transition rules per D006."""
    status = _read_status(paths)
    current = status.get("lifecycle", Lifecycle.BUILDING.value)

    if current == Lifecycle.SEALED.value:
        raise ArtifactError(
            f"Dataset {paths.dataset_type}/{paths.dataset_id} is sealed; no mutations allowed"
        )

    if target == Lifecycle.COMPLETE:
        if current != Lifecycle.BUILDING.value:
            raise ArtifactError(f"Cannot transition from {current} to {target.value}")
        status = update_status(paths)
        if status["missing_count"] > 0 or status["invariant_errors"]:
            raise ArtifactError(
                f"Cannot complete: {status['missing_count']} artifacts missing; "
                f"invariants={status['invariant_errors']}"
            )

    elif target == Lifecycle.SEALED:
        if current != Lifecycle.COMPLETE.value:
            raise ArtifactError(f"Cannot seal from {current}; must be complete first")

    elif target == Lifecycle.FAILED:
        if current != Lifecycle.BUILDING.value:
            raise ArtifactError(f"Cannot fail from {current}")

    status["lifecycle"] = target.value
    status["updated_utc"] = utc_now()
    atomic_write_yaml(paths.status_file, status)
    logger.info("Transitioned %s/%s to %s", paths.dataset_type, paths.dataset_id, target.value)
    return status


def write_artifact(paths: DatasetPath, artifact_id: str, data: dict[str, Any]) -> bool:
    """Write an artifact exclusively. Returns True if newly created.

    - If file exists with same content hash, returns False (no overwrite).
    - If file exists with different content, raises ArtifactError.
    - Rejects if dataset is sealed.
    """
    status = _read_status(paths)
    lifecycle = status.get("lifecycle")
    if lifecycle != Lifecycle.BUILDING.value:
        raise ArtifactError(f"Dataset is {lifecycle}; artifacts may be written only while building")
    _validate_planned_id(paths, artifact_id)

    digest = artifact_id[7:] if artifact_id.startswith("sha256:") else artifact_id
    artifact_path = paths.artifacts_dir / f"{digest}.json"

    content = dict(data)
    content["artifact_id"] = artifact_id
    if artifact_path.exists():
        existing = read_json(artifact_path)
        existing_hash = canonical_hash({k: v for k, v in existing.items() if k != "created_utc"})
        new_hash = canonical_hash(content)
        if existing_hash == new_hash:
            return False
        raise ArtifactError(
            f"Artifact {digest[:16]} already exists with different content; will not overwrite"
        )

    content["created_utc"] = utc_now()
    if exclusive_create(artifact_path, content):
        return True
    existing = read_json(artifact_path)
    if canonical_hash({k: v for k, v in existing.items() if k != "created_utc"}) == canonical_hash(
        {k: v for k, v in content.items() if k != "created_utc"}
    ):
        return False
    raise ArtifactError(f"Artifact {digest[:16]} was concurrently created with different content")


def append_failure(
    paths: DatasetPath, artifact_id: str, attempt: int, error_data: dict[str, Any]
) -> None:
    """Append a failure attempt (append-only, never overwritten)."""
    status = _read_status(paths)
    lifecycle = status.get("lifecycle")
    if lifecycle != Lifecycle.BUILDING.value:
        raise ArtifactError(f"Dataset is {lifecycle}; failures may be written only while building")
    _validate_planned_id(paths, artifact_id)

    digest = artifact_id[7:] if artifact_id.startswith("sha256:") else artifact_id
    failure_dir = paths.failures_dir / digest
    failure_dir.mkdir(parents=True, exist_ok=True)
    failure_path = failure_dir / f"attempt_{attempt:03d}.json"

    content = dict(error_data)
    content["artifact_id"] = artifact_id
    content["attempt"] = attempt
    content["timestamp_utc"] = utc_now()
    if not exclusive_create(failure_path, content):
        existing = read_json(failure_path)
        existing_logical = {k: v for k, v in existing.items() if k != "timestamp_utc"}
        content_logical = {k: v for k, v in content.items() if k != "timestamp_utc"}
        if canonical_hash(existing_logical) != canonical_hash(content_logical):
            raise ArtifactError(f"Failure attempt {attempt} already has different content")


def _validate_planned_id(paths: DatasetPath, artifact_id: str) -> None:
    if not ARTIFACT_ID_RE.fullmatch(artifact_id):
        raise ArtifactError("Artifact ID must be sha256:<64 lowercase hex>")
    manifest = read_yaml(paths.manifest)
    if artifact_id not in manifest.get("planned_artifact_ids", []):
        raise ArtifactError(f"Artifact {artifact_id} is not present in the immutable plan")


def get_missing_artifacts(paths: DatasetPath) -> list[str]:
    """Return list of planned artifact IDs that don't have successful artifacts."""
    manifest = read_yaml(paths.manifest) if paths.manifest.exists() else {}
    planned_ids = manifest.get("planned_artifact_ids", [])
    existing = {f.stem for f in paths.artifacts_dir.glob("*.json")}
    result: list[str] = []
    for aid in planned_ids:
        digest = aid[7:] if aid.startswith("sha256:") else aid
        if digest not in existing:
            result.append(aid)
    return result


def get_failed_artifacts(paths: DatasetPath) -> list[str]:
    """Return list of artifact digests that have failure records."""
    failures = set()
    for d in paths.failures_dir.iterdir():
        if d.is_dir():
            failures.add(d.name)
    return list(failures)
