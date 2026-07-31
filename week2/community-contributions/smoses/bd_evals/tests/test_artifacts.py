"""Tests for atomic IO and artifact store."""

from pathlib import Path

import pytest

from rag_evals.artifacts.io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
    exclusive_create,
    read_json,
    read_yaml,
    verify_file_hash,
)
from rag_evals.artifacts.store import (
    DatasetPath,
    Lifecycle,
    append_failure,
    create_dataset,
    get_failed_artifacts,
    get_missing_artifacts,
    get_status,
    transition_lifecycle,
    update_status,
    write_artifact,
)
from rag_evals.errors import ArtifactError


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


class TestAtomicIO:
    def test_write_and_read_json(self, project_root: Path) -> None:
        path = project_root / "test.json"
        data = {"b": 2, "a": 1}
        h = atomic_write_json(path, data)
        assert path.exists()
        result = read_json(path)
        assert result == data
        assert len(h) == 64

    def test_write_and_read_yaml(self, project_root: Path) -> None:
        path = project_root / "test.yaml"
        data = {"key": "value", "list": [1, 2, 3]}
        h = atomic_write_yaml(path, data)
        assert path.exists()
        result = read_yaml(path)
        assert result == data
        assert len(h) == 64

    def test_write_text(self, project_root: Path) -> None:
        path = project_root / "text.txt"
        atomic_write_text(path, "hello world")
        assert path.read_text() == "hello world"

    def test_write_bytes(self, project_root: Path) -> None:
        path = project_root / "data.bin"
        atomic_write_bytes(path, b"\x00\x01\x02")
        assert path.read_bytes() == b"\x00\x01\x02"

    def test_creates_parent_dirs(self, project_root: Path) -> None:
        path = project_root / "sub" / "dir" / "test.json"
        atomic_write_json(path, {"x": 1})
        assert path.exists()

    def test_no_temp_file_remains(self, project_root: Path) -> None:
        path = project_root / "test.json"
        atomic_write_json(path, {"x": 1})
        assert not (project_root / "test.json.tmp").exists()

    def test_verify_file_hash(self, project_root: Path) -> None:
        path = project_root / "test.json"
        data = {"x": 1}
        h = atomic_write_json(path, data)
        assert verify_file_hash(path, h)
        assert not verify_file_hash(path, "0" * 64)

    def test_exclusive_create_new(self, project_root: Path) -> None:
        path = project_root / "new.json"
        assert exclusive_create(path, {"x": 1}) is True
        assert exclusive_create(path, {"x": 1}) is False

    def test_read_json_missing(self, project_root: Path) -> None:
        with pytest.raises(ArtifactError, match="not found"):
            read_json(project_root / "missing.json")


class TestDatasetCreation:
    def _plan_rows(self, n: int = 3) -> list[dict]:
        return [{"artifact_id": f"sha256:{i:064d}"} for i in range(n)]

    def test_create_dataset(self, project_root: Path) -> None:
        rows = self._plan_rows()
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        assert paths.base.exists()
        assert paths.manifest.exists()
        assert paths.resolved_definition.exists()
        assert paths.status_file.exists()
        assert paths.artifacts_dir.exists()
        assert paths.failures_dir.exists()

    def test_status_building(self, project_root: Path) -> None:
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=self._plan_rows(),
        )
        status = get_status(paths)
        assert status["lifecycle"] == "building"
        assert status["planned_count"] == 3
        assert status["successful_count"] == 0
        assert status["missing_count"] == 3

    def test_idempotent_same_definition(self, project_root: Path) -> None:
        rows = self._plan_rows()
        defn = {"test": "data"}
        paths1 = create_dataset(project_root, "retrieval", "test-ds", defn, rows)
        paths2 = create_dataset(project_root, "retrieval", "test-ds", defn, rows)
        assert paths1.base == paths2.base

    def test_reject_different_definition(self, project_root: Path) -> None:
        rows = self._plan_rows()
        create_dataset(project_root, "retrieval", "test-ds", {"v": 1}, rows)
        with pytest.raises(ArtifactError, match="different"):
            create_dataset(project_root, "retrieval", "test-ds", {"v": 2}, rows)

    def test_invalid_dataset_type(self, project_root: Path) -> None:
        with pytest.raises(ArtifactError, match="Invalid"):
            create_dataset(project_root, "bad_type", "test-ds", {}, [])


class TestArtifactWriting:
    AID_0 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    def _create(self, project_root: Path, n: int = 3) -> DatasetPath:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(n)]
        return create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )

    def test_write_new_artifact(self, project_root: Path) -> None:
        paths = self._create(project_root)
        result = write_artifact(paths, self.AID_0, {"data": "test"})
        assert result is True
        status = get_status(paths)
        assert status["successful_count"] == 1

    def test_write_same_artifact_no_overwrite(self, project_root: Path) -> None:
        paths = self._create(project_root)
        aid = self.AID_0
        data = {"data": "test"}
        write_artifact(paths, aid, data)
        result = write_artifact(paths, aid, data)
        assert result is False

    def test_write_different_content_rejected(self, project_root: Path) -> None:
        paths = self._create(project_root)
        aid = self.AID_0
        write_artifact(paths, aid, {"data": "v1"})
        with pytest.raises(ArtifactError, match="will not overwrite"):
            write_artifact(paths, aid, {"data": "v2"})

    def test_append_failure(self, project_root: Path) -> None:
        paths = self._create(project_root)
        aid = self.AID_0
        append_failure(paths, aid, 1, {"error": "timeout"})
        append_failure(paths, aid, 2, {"error": "timeout again"})
        failures = get_failed_artifacts(paths)
        assert len(failures) == 1
        assert failures[0] == aid[7:]

    def test_missing_artifacts(self, project_root: Path) -> None:
        paths = self._create(project_root, n=3)
        missing = get_missing_artifacts(paths)
        assert len(missing) == 3
        write_artifact(paths, self.AID_0, {"data": "test"})
        missing = get_missing_artifacts(paths)
        assert len(missing) == 2


class TestLifecycle:
    AID_0 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    def _create_complete(self, project_root: Path) -> DatasetPath:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(2)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        for i in range(2):
            aid = f"sha256:{i:064d}"
            write_artifact(paths, aid, {"index": i})
        return paths

    def test_complete_requires_all_artifacts(self, project_root: Path) -> None:
        paths = self._create_complete(project_root)
        status = transition_lifecycle(paths, Lifecycle.COMPLETE)
        assert status["lifecycle"] == "complete"

    def test_complete_rejects_missing(self, project_root: Path) -> None:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(3)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        write_artifact(paths, self.AID_0, {"index": 0})
        with pytest.raises(ArtifactError, match="missing"):
            transition_lifecycle(paths, Lifecycle.COMPLETE)

    def test_seal_requires_complete(self, project_root: Path) -> None:
        paths = self._create_complete(project_root)
        transition_lifecycle(paths, Lifecycle.COMPLETE)
        status = transition_lifecycle(paths, Lifecycle.SEALED)
        assert status["lifecycle"] == "sealed"

    def test_seal_rejects_building(self, project_root: Path) -> None:
        paths = self._create_complete(project_root)
        with pytest.raises(ArtifactError, match="Cannot seal"):
            transition_lifecycle(paths, Lifecycle.SEALED)

    def test_sealed_rejects_writes(self, project_root: Path) -> None:
        paths = self._create_complete(project_root)
        transition_lifecycle(paths, Lifecycle.COMPLETE)
        transition_lifecycle(paths, Lifecycle.SEALED)
        with pytest.raises(ArtifactError, match="sealed"):
            write_artifact(paths, "sha256:new", {"data": "x"})

    def test_sealed_rejects_failures(self, project_root: Path) -> None:
        paths = self._create_complete(project_root)
        transition_lifecycle(paths, Lifecycle.COMPLETE)
        transition_lifecycle(paths, Lifecycle.SEALED)
        with pytest.raises(ArtifactError, match="sealed"):
            append_failure(paths, "sha256:x", 1, {"error": "test"})


class TestResume:
    AID_0 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

    def test_resume_finds_missing(self, project_root: Path) -> None:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(5)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        # Write 2 of 5
        for i in range(2):
            write_artifact(paths, f"sha256:{i:064d}", {"index": i})

        missing = get_missing_artifacts(paths)
        assert len(missing) == 3

        # Write remaining
        for i in range(2, 5):
            write_artifact(paths, f"sha256:{i:064d}", {"index": i})

        missing = get_missing_artifacts(paths)
        assert len(missing) == 0
        status = update_status(paths)
        assert status["successful_count"] == 5

    def test_resume_preserves_existing(self, project_root: Path) -> None:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(3)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        write_artifact(paths, self.AID_0, {"index": 0})

        # Simulate resume: try to write the same artifact
        result = write_artifact(paths, self.AID_0, {"index": 0})
        assert result is False  # already exists, not overwritten


class TestCleanup:
    def _create_ds(self, project_root: Path, n: int = 2) -> DatasetPath:
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(n)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        for i in range(n):
            write_artifact(paths, f"sha256:{i:064d}", {"index": i})
        transition_lifecycle(paths, Lifecycle.COMPLETE)
        return paths

    def test_plan_cleanup_dry_run(self, project_root: Path) -> None:
        self._create_ds(project_root)
        from rag_evals.artifacts.cleanup import plan_cleanup

        plan = plan_cleanup(project_root, "retrieval", "test-ds")
        assert plan.exists
        assert plan.can_clean
        assert plan.file_count > 0

    def test_plan_cleanup_missing(self, project_root: Path) -> None:
        from rag_evals.artifacts.cleanup import plan_cleanup

        plan = plan_cleanup(project_root, "retrieval", "missing-ds")
        assert not plan.exists
        assert not plan.can_clean

    def test_plan_cleanup_sealed_rejected(self, project_root: Path) -> None:
        paths = self._create_ds(project_root)
        transition_lifecycle(paths, Lifecycle.SEALED)
        from rag_evals.artifacts.cleanup import plan_cleanup

        plan = plan_cleanup(project_root, "retrieval", "test-ds")
        assert not plan.can_clean
        assert "sealed" in plan.reason.lower()

    def test_execute_cleanup_moves_to_trash(self, project_root: Path) -> None:
        self._create_ds(project_root)
        from rag_evals.artifacts.cleanup import execute_cleanup

        trash_path = execute_cleanup(project_root, "retrieval", "test-ds", "test-ds")
        assert trash_path.exists()
        assert not (project_root / "datasets" / "retrieval" / "test-ds").exists()

    def test_execute_cleanup_wrong_confirmation(self, project_root: Path) -> None:
        self._create_ds(project_root)
        from rag_evals.artifacts.cleanup import execute_cleanup

        with pytest.raises(ArtifactError, match="Confirmation"):
            execute_cleanup(project_root, "retrieval", "test-ds", "wrong-id")

    def test_id_reuse_after_cleanup(self, project_root: Path) -> None:
        self._create_ds(project_root)
        from rag_evals.artifacts.cleanup import execute_cleanup

        execute_cleanup(project_root, "retrieval", "test-ds", "test-ds")
        # Should be able to create again
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(2)]
        paths = create_dataset(
            project_root,
            "retrieval",
            "test-ds",
            resolved_definition={"test": "data"},
            plan_rows=rows,
        )
        assert paths.base.exists()

    def test_downstream_reference_blocks_cleanup(self, project_root: Path) -> None:
        # Create source dataset
        self._create_ds(project_root)

        # Create downstream dataset referencing source
        rows = [{"artifact_id": f"sha256:{i:064d}"} for i in range(1)]
        create_dataset(
            project_root,
            "answers",
            "ans-ds",
            resolved_definition={
                "source_dataset": {"type": "retrieval", "id": "test-ds"},
            },
            plan_rows=rows,
        )

        from rag_evals.artifacts.cleanup import plan_cleanup

        plan = plan_cleanup(project_root, "retrieval", "test-ds")
        assert not plan.can_clean
        assert "referenced" in plan.reason.lower()

    def test_path_traversal_rejected(self, project_root: Path) -> None:
        from rag_evals.artifacts.cleanup import plan_cleanup

        with pytest.raises(ArtifactError, match="Invalid"):
            plan_cleanup(project_root, "../../../etc", "passwd")
