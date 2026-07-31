"""Safe dataset cleanup: exact-ID selection, dry-run, atomic move to trash."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_evals.artifacts.io import read_yaml
from rag_evals.artifacts.store import (
    TRASH_DIR,
    DatasetPath,
    Lifecycle,
    _read_status,
    utc_now,
)
from rag_evals.errors import ArtifactError
from rag_evals.logging import get_logger

logger = get_logger("rag_evals.cleanup")

DATASET_TYPES = ("retrieval", "retrieval_judgments", "answers", "answer_judgments")


@dataclass
class CleanupPlan:
    """Dry-run result for a cleanup operation."""

    dataset_type: str
    dataset_id: str
    exists: bool
    lifecycle: str
    file_count: int
    total_size_bytes: int
    downstream_references: list[dict[str, str]] = field(default_factory=list)
    report_references: list[dict[str, str]] = field(default_factory=list)
    can_clean: bool = False
    reason: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "dataset_type": self.dataset_type,
            "dataset_id": self.dataset_id,
            "exists": self.exists,
            "lifecycle": self.lifecycle,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "downstream_references": self.downstream_references,
            "report_references": self.report_references,
            "can_clean": self.can_clean,
            "reason": self.reason,
        }


def plan_cleanup(
    project_root: Path,
    dataset_type: str,
    dataset_id: str,
) -> CleanupPlan:
    """Plan a cleanup operation. Does not modify anything."""
    if dataset_type not in DATASET_TYPES:
        raise ArtifactError(f"Invalid dataset type: {dataset_type}")

    paths = DatasetPath(project_root, dataset_type, dataset_id)

    if not paths.base.exists():
        return CleanupPlan(
            dataset_type=dataset_type,
            dataset_id=dataset_id,
            exists=False,
            lifecycle="",
            file_count=0,
            total_size_bytes=0,
            can_clean=False,
            reason="Dataset does not exist",
        )

    status = _read_status(paths)
    lifecycle = status.get("lifecycle", Lifecycle.BUILDING.value)

    # Count files and size
    file_count = 0
    total_size = 0
    for f in paths.base.rglob("*"):
        if f.is_file():
            file_count += 1
            total_size += f.stat().st_size

    # Check downstream references
    downstream = _find_downstream_references(project_root, dataset_type, dataset_id)
    report_refs = _find_report_references(project_root, dataset_type, dataset_id)

    can_clean = True
    reason = ""

    if lifecycle == Lifecycle.SEALED.value:
        can_clean = False
        reason = "Dataset is sealed"
    elif downstream:
        can_clean = False
        reason = f"Dataset is referenced by {len(downstream)} downstream dataset(s)"

    return CleanupPlan(
        dataset_type=dataset_type,
        dataset_id=dataset_id,
        exists=True,
        lifecycle=lifecycle,
        file_count=file_count,
        total_size_bytes=total_size,
        downstream_references=downstream,
        report_references=report_refs,
        can_clean=can_clean,
        reason=reason,
    )


def execute_cleanup(
    project_root: Path,
    dataset_type: str,
    dataset_id: str,
    confirm_dataset_id: str,
) -> Path:
    """Execute cleanup: atomically move dataset to trash.

    Requires exact confirmation of dataset ID.
    """
    if confirm_dataset_id != dataset_id:
        raise ArtifactError(
            f"Confirmation ID '{confirm_dataset_id}' does not match dataset ID '{dataset_id}'"
        )

    plan = plan_cleanup(project_root, dataset_type, dataset_id)

    if not plan.exists:
        raise ArtifactError(f"Dataset not found: {dataset_type}/{dataset_id}")

    if not plan.can_clean:
        raise ArtifactError(f"Cannot clean: {plan.reason}")

    final_plan = plan_cleanup(project_root, dataset_type, dataset_id)
    if not final_plan.can_clean:
        raise ArtifactError(f"Cannot clean after dependency recheck: {final_plan.reason}")

    paths = DatasetPath(project_root, dataset_type, dataset_id)
    trash_base = paths.trash_path
    trash_base.mkdir(parents=True, exist_ok=True)

    timestamp = utc_now().replace(":", "").replace("-", "")
    trash_name = f"{dataset_id}-{timestamp}-{uuid.uuid4().hex}"
    trash_path = trash_base / trash_name

    # Atomic move on same filesystem
    os.rename(str(paths.base), str(trash_path))

    logger.info("Moved %s/%s to %s", dataset_type, dataset_id, trash_path)
    return trash_path


def restore_dataset(project_root: Path, trash_path: Path) -> DatasetPath:
    """Restore one trashed dataset only when its identity and destination are valid."""
    resolved = trash_path.resolve()
    trash_root = (project_root / "datasets" / TRASH_DIR).resolve()
    try:
        relative = resolved.relative_to(trash_root)
    except ValueError:
        raise ArtifactError("Restore source must be inside datasets/.trash") from None
    if len(relative.parts) != 2 or relative.parts[0] not in DATASET_TYPES:
        raise ArtifactError("Invalid trash dataset path")
    manifest = read_yaml(resolved / "manifest.yaml")
    dataset_type = str(manifest.get("dataset_type", ""))
    dataset_id = str(manifest.get("dataset_id", ""))
    if dataset_type != relative.parts[0] or not dataset_id:
        raise ArtifactError("Trash manifest identity does not match its path")
    destination = DatasetPath(project_root, dataset_type, dataset_id)
    if destination.base.exists():
        raise ArtifactError(f"Cannot restore: destination already exists: {destination.base}")
    os.rename(resolved, destination.base)
    return destination


def _find_downstream_references(
    project_root: Path,
    dataset_type: str,
    dataset_id: str,
) -> list[dict[str, str]]:
    """Find datasets that reference this dataset as a source."""
    refs: list[dict[str, str]] = []
    datasets_dir = project_root / "datasets"

    if not datasets_dir.exists():
        return refs

    for type_dir in datasets_dir.iterdir():
        if not type_dir.is_dir() or type_dir.name == TRASH_DIR:
            continue
        for ds_dir in type_dir.iterdir():
            if not ds_dir.is_dir():
                continue
            resolved_def = ds_dir / "resolved_definition.yaml"
            if not resolved_def.exists():
                continue
            try:
                data = read_yaml(resolved_def)
                source = data.get("source_dataset", {})
                if (
                    isinstance(source, dict)
                    and source.get("type") == dataset_type
                    and source.get("id") == dataset_id
                ):
                    refs.append(
                        {
                            "dataset_type": type_dir.name,
                            "dataset_id": ds_dir.name,
                        }
                    )
            except Exception as exc:
                raise ArtifactError(
                    f"Cannot prove cleanup safety: unreadable metadata {resolved_def}: {exc}"
                ) from exc

    return refs


def _find_report_references(
    project_root: Path,
    dataset_type: str,
    dataset_id: str,
) -> list[dict[str, str]]:
    """Find reports that reference this dataset."""
    refs: list[dict[str, str]] = []
    reports_dir = project_root / "reports"

    if not reports_dir.exists():
        return refs

    for f in reports_dir.rglob("*.json"):
        try:
            data = read_yaml(f)
            if dataset_id in str(data):
                refs.append({"path": str(f.relative_to(project_root))})
        except Exception:
            continue

    return refs
