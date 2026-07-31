"""Regression tests for schema-version guardrails (R0-T02).

Demonstrates that pre-remediation datasets with manifest schema_version=1
must be rejected when attempting to create or resume a v2 dataset.
"""

from pathlib import Path

import pytest
import yaml

from rag_evals.artifacts.store import DatasetPath, create_dataset
from rag_evals.errors import ArtifactError


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


def _write_old_manifest(paths: DatasetPath) -> None:
    """Write a manifest with schema_version=1 (pre-remediation format)."""
    manifest = {
        "dataset_type": paths.dataset_type,
        "dataset_id": paths.dataset_id,
        "created_utc": "2026-01-01T00:00:00Z",
        "schema_version": 1,
        "planned_artifact_ids": ["sha256:old"],
        "planned_count": 1,
    }
    from rag_evals.artifacts.io import atomic_write_yaml

    atomic_write_yaml(paths.manifest, manifest)


class TestSchemaGuardrail:
    def test_old_manifest_rejected_on_create(self, project_root: Path) -> None:
        """Creating a v2 dataset where a v1 manifest exists must fail clearly."""
        paths = DatasetPath(project_root, "retrieval", "test-ds")
        paths.base.mkdir(parents=True)
        paths.artifacts_dir.mkdir()
        paths.failures_dir.mkdir()
        paths.summaries_dir.mkdir()
        paths.snapshots_dir.mkdir()
        _write_old_manifest(paths)

        from rag_evals.artifacts.io import atomic_write_yaml

        atomic_write_yaml(paths.resolved_definition, {"test": "data"})
        atomic_write_yaml(
            paths.status_file,
            {"lifecycle": "building", "planned_count": 1},
        )

        with pytest.raises(ArtifactError, match="incompatible.*schema"):
            create_dataset(
                project_root=project_root,
                dataset_type="retrieval",
                dataset_id="test-ds",
                resolved_definition={"test": "data"},
                plan_rows=[{"artifact_id": "sha256:new"}],
            )

    def test_new_manifest_has_schema_2(self, project_root: Path) -> None:
        """Newly created datasets must have manifest schema_version=2."""
        paths = create_dataset(
            project_root=project_root,
            dataset_type="retrieval",
            dataset_id="fresh-ds",
            resolved_definition={"test": "data"},
            plan_rows=[{"artifact_id": f"sha256:{'a' * 64}"}],
        )
        manifest = yaml.safe_load(paths.manifest.read_text())
        assert manifest["schema_version"] == 2
