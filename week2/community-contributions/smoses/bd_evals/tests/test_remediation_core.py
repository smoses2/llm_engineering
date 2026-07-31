"""Cross-phase regression tests for the remediation contracts."""

from __future__ import annotations

import asyncio
import multiprocessing
import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag_evals.api import real_adapter
from rag_evals.api.real_adapter import (
    _dataclass_to_dict,
    _known_transport_error,
    preflight_bd_api,
)
from rag_evals.api.retry import execute_with_retries
from rag_evals.artifacts.cleanup import execute_cleanup, restore_dataset
from rag_evals.artifacts.io import read_yaml
from rag_evals.artifacts.lineage import SourceArtifact
from rag_evals.artifacts.store import create_dataset, get_status, write_artifact
from rag_evals.config.schemas import (
    InferenceVariant,
    ModelCatalog,
    ModelEntry,
    PromptVariant,
    Question,
    QuestionCatalog,
    Selection,
    TemplateRef,
)
from rag_evals.errors import ArtifactError
from rag_evals.planning.matrices import generate_ask_matrix
from rag_evals.reporting.reports import cost_dollars


def _artifact_id(character: str) -> str:
    return f"sha256:{character * 64}"


def _create_worker(root: str, queue: multiprocessing.Queue[object]) -> None:
    try:
        paths = create_dataset(
            Path(root),
            "retrieval",
            "race",
            {"resolved": True},
            [{"artifact_id": _artifact_id("a")}],
        )
        queue.put(read_yaml(paths.manifest)["plan_hash"])
    except Exception as exc:  # pragma: no cover - reported in parent
        queue.put(exc)


def _write_worker(root: str, queue: multiprocessing.Queue[object]) -> None:
    try:
        paths = create_dataset(
            Path(root),
            "retrieval",
            "artifact-race",
            {"resolved": True},
            [{"artifact_id": _artifact_id("a")}],
        )
        queue.put(write_artifact(paths, _artifact_id("a"), {"value": "same"}))
    except Exception as exc:  # pragma: no cover - reported in parent
        queue.put(exc)


def test_f003_f023_manifest_rejects_changed_plan(tmp_path: Path) -> None:
    create_dataset(
        tmp_path,
        "retrieval",
        "plan",
        {"resolved": True},
        [{"artifact_id": _artifact_id("a")}],
    )
    with pytest.raises(ArtifactError, match="plan hash"):
        create_dataset(
            tmp_path,
            "retrieval",
            "plan",
            {"resolved": True},
            [{"artifact_id": _artifact_id("b")}],
        )


def test_f004_f025_multiprocess_dataset_initialization_is_transactional(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[object] = context.Queue()
    processes = [
        context.Process(target=_create_worker, args=(str(tmp_path), queue)) for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    values = [queue.get(timeout=2) for _ in processes]
    assert not any(isinstance(value, Exception) for value in values)
    assert len(set(values)) == 1
    assert not list((tmp_path / "datasets/retrieval").glob("*.initializing-*"))


def test_f004_multiprocess_artifact_creation_never_overwrites(tmp_path: Path) -> None:
    create_dataset(
        tmp_path,
        "retrieval",
        "artifact-race",
        {"resolved": True},
        [{"artifact_id": _artifact_id("a")}],
    )
    context = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue[object] = context.Queue()
    processes = [
        context.Process(target=_write_worker, args=(str(tmp_path), queue)) for _ in range(3)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(10)
        assert process.exitcode == 0
    values = [queue.get(timeout=2) for _ in processes]
    assert sorted(values) == [False, False, True]


def test_f001_f007_exact_source_is_matrix_dimension() -> None:
    questions = QuestionCatalog(questions=[Question(id="q1", question="Question?")])
    models = ModelCatalog(models=[ModelEntry(id="m1", model_id="actual")])
    prompts = [
        PromptVariant(
            id="p1",
            template=TemplateRef(system_instructions="system", user_message="{{ question }}"),
        )
    ]
    inference = [InferenceVariant(id="i1", inference_config={"temperature": 0.0})]
    sources = [
        SourceArtifact(_artifact_id("a"), "hash-a", "a.json", "q1", "Question?", None),
        SourceArtifact(_artifact_id("b"), "hash-b", "b.json", "q1", "Question?", None),
    ]
    matrix = generate_ask_matrix(
        questions,
        models,
        Selection(),
        prompts,
        inference,
        "retrieval",
        "source",
        source_artifacts=sources,
    )
    assert len(matrix.rows) == 2
    assert {row.source_artifact_id for row in matrix.rows} == {_artifact_id("a"), _artifact_id("b")}
    assert len({row.artifact_id for row in matrix.rows}) == 2


@pytest.mark.asyncio
async def test_f016_retry_records_selected_delay_before_sleep() -> None:
    sleeps: list[float] = []
    calls = 0

    class RetryableError(Exception):
        retryable = True
        error_type = "http"

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("retry")
        return "ok"

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    result = await execute_with_retries(call, rng=random.Random(1), sleeper=sleep)
    assert result.success
    assert result.attempts[0].delay_seconds == sleeps[0]
    assert result.attempts[0].delay_seconds > 0


@pytest.mark.asyncio
async def test_f017_backoff_releases_attempt_concurrency_slot() -> None:
    semaphore = asyncio.Semaphore(1)
    sleeping = asyncio.Event()
    release = asyncio.Event()
    attempts = 0

    class RetryableError(Exception):
        retryable = True

    async def first() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableError("retry")
        return "first"

    async def sleep(_: float) -> None:
        sleeping.set()
        await release.wait()

    first_task = asyncio.create_task(
        execute_with_retries(first, sleeper=sleep, attempt_semaphore=semaphore)
    )
    await sleeping.wait()
    second = await execute_with_retries(
        lambda: asyncio.sleep(0, result="second"), attempt_semaphore=semaphore
    )
    assert second.response == "second"
    release.set()
    assert (await first_task).response == "first"


def test_f011_f015_f022_complete_is_immutable_and_write_is_logically_idempotent(
    tmp_path: Path,
) -> None:
    artifact_id = _artifact_id("a")
    paths = create_dataset(
        tmp_path, "retrieval", "logical", {"v": 1}, [{"artifact_id": artifact_id}]
    )
    assert write_artifact(paths, artifact_id, {"value": 1})
    assert not write_artifact(paths, artifact_id, {"value": 1})
    with pytest.raises(ArtifactError, match="immutable plan"):
        write_artifact(paths, _artifact_id("b"), {"value": 2})


def test_f012_f027_verified_wrapper_fields_are_serialized() -> None:
    @dataclass
    class Retrieval:
        fallback_used: bool = False

    @dataclass
    class Wrapper:
        retrieval: Retrieval = field(default_factory=Retrieval)
        raw_retrieval_response: dict[str, object] = field(default_factory=lambda: {"raw": True})
        retrieved_items: list[dict[str, object]] = field(default_factory=lambda: [{"id": "1"}])
        prepared_prompt_input: dict[str, object] = field(default_factory=lambda: {"context": "x"})
        persisted_to_s3: bool = True

    serialized = _dataclass_to_dict(Wrapper())
    assert serialized["raw_retrieval_response"] == {"raw": True}
    assert serialized["retrieved_items"] == [{"id": "1"}]
    assert serialized["persisted_to_s3"] is True
    with pytest.raises(TypeError, match="Unsupported"):
        _dataclass_to_dict(object())


def test_f013_dependency_preflight_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real_adapter, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="PYTHONPATH"):
        preflight_bd_api()


def test_f026_programming_errors_are_not_retryable_transport_failures() -> None:
    assert not _known_transport_error(TypeError("programming error"))


def test_f019_nested_cost_normalization_keeps_exact_units() -> None:
    assert cost_dollars({"inference": {"microdollars": 250_000}}) == 0.25
    assert cost_dollars({"retrieval": {"dollars": 1.5}, "rerank": {"dollars": 0.5}}) == 2.0


def test_f023_status_surfaces_unplanned_and_corrupt_files(tmp_path: Path) -> None:
    artifact_id = _artifact_id("a")
    paths = create_dataset(
        tmp_path, "retrieval", "status", {"v": 1}, [{"artifact_id": artifact_id}]
    )
    unplanned = paths.artifacts_dir / f"{'b' * 64}.json"
    unplanned.write_text('{"artifact_id":"sha256:' + "b" * 64 + '"}')
    status = get_status(paths)
    assert status["unplanned_artifacts"] == ["b" * 64]
    assert status["invariant_errors"]


def test_f024_cleanup_uses_collision_safe_trash_and_validated_restore(tmp_path: Path) -> None:
    create_dataset(
        tmp_path,
        "retrieval",
        "cleanup",
        {"v": 1},
        [{"artifact_id": _artifact_id("a")}],
    )
    trash = execute_cleanup(tmp_path, "retrieval", "cleanup", "cleanup")
    assert len(trash.name.rsplit("-", 1)[-1]) == 32
    restored = restore_dataset(tmp_path, trash)
    assert restored.manifest.exists()
    with pytest.raises(ArtifactError, match="inside"):
        restore_dataset(tmp_path, tmp_path / "other")
