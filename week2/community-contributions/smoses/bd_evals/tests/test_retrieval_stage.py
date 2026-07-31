"""Integration tests for the retrieval stage with fake adapter."""

import json
from pathlib import Path

import pytest

from rag_evals.api.fake_adapter import FakeAdapter
from rag_evals.api.protocol import AdapterError, RetrieveRequest, RetrieveResponse
from rag_evals.artifacts.store import get_status
from rag_evals.config.schemas import (
    ApiConfig,
    CatalogSelection,
    DeploymentConfig,
    ExecutionConfig,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    OutputConfig,
    Question,
    QuestionCatalog,
    RetrievalDefinition,
    RetrievalTuning,
    Selection,
    SourceRef,
    TestBlock,
)
from rag_evals.errors import ArtifactError
from rag_evals.stages.retrieval import (
    detect_mode_fallback,
    estimate_tokens,
    run_retrieval_stage,
)


def _make_definition(dataset_id: str = "test-retrieval") -> RetrievalDefinition:
    return RetrievalDefinition(
        schema_version=1,
        test=TestBlock(id="test", type="retrieval"),
        questions=SourceRef(source="questions.yaml"),
        knowledge_bases=SourceRef(source="kbs.yaml"),
        selection=Selection(knowledge_bases=CatalogSelection(include=["kb1", "kb2"])),
        retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[10], result_counts=[3]),
        deployment=DeploymentConfig(
            label="test",
            merge={"enabled": False, "maximum_chunks_per_document": None},
        ),
        api=ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY"),
        output=OutputConfig(dataset_id=dataset_id),
        execution=ExecutionConfig(concurrency=2, continue_on_error=True),
    )


def _make_questions() -> QuestionCatalog:
    return QuestionCatalog(
        questions=[
            Question(id="q1", question="What is condition A?"),
            Question(
                id="q2",
                question="What is condition B?",
                rubric={"expected_answer": "B is X"},
            ),
        ]
    )


def _make_kbs() -> KnowledgeBaseCatalog:
    return KnowledgeBaseCatalog(
        knowledge_bases=[
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB_REAL_1",
                chunking={"strategy": "semantic"},
                supports_merge_eval=True,
            ),
            KnowledgeBase(
                id="kb2",
                knowledge_base_id="KB_REAL_2",
                chunking={"strategy": "fixed", "size": 500, "overlap": 100},
                supports_merge_eval=True,
            ),
        ]
    )


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


class TestEstimateTokens:
    def test_empty(self) -> None:
        assert estimate_tokens("") == 0

    def test_short(self) -> None:
        assert estimate_tokens("hello") == 2  # ceil(5/4) = 2

    def test_exact_multiple(self) -> None:
        assert estimate_tokens("abcdefgh") == 2  # ceil(8/4) = 2

    def test_long(self) -> None:
        text = "x" * 100
        assert estimate_tokens(text) == 25  # ceil(100/4) = 25


class FallbackAdapter(FakeAdapter):
    """Fake adapter whose retrieval silently degrades to standard mode."""

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        response = await super().retrieve(request)
        response.retrieval_metadata["effective_mode"] = "standard"
        response.retrieval_metadata["fallback_used"] = True
        response.retrieval_metadata["fallback_reason"] = "rerank model unavailable"
        return response


class TestDetectModeFallback:
    def test_matching_modes_pass(self) -> None:
        metadata = {"requested_mode": "rerank", "effective_mode": "rerank", "fallback_used": False}
        assert detect_mode_fallback("rerank", metadata) is None

    def test_differing_modes_detected(self) -> None:
        metadata = {"requested_mode": "rerank", "effective_mode": "standard"}
        reason = detect_mode_fallback("rerank", metadata)
        assert reason is not None
        assert "requested 'rerank'" in reason and "'standard'" in reason

    def test_fallback_reason_included(self) -> None:
        metadata = {
            "requested_mode": "rerank",
            "effective_mode": "standard",
            "fallback_reason": "rerank model unavailable",
        }
        reason = detect_mode_fallback("rerank", metadata)
        assert reason is not None and "rerank model unavailable" in reason

    def test_fallback_flag_alone_detected(self) -> None:
        metadata = {"requested_mode": "rerank", "effective_mode": "rerank", "fallback_used": True}
        assert detect_mode_fallback("rerank", metadata) is not None

    def test_absent_metadata_cannot_be_checked(self) -> None:
        assert detect_mode_fallback("rerank", {}) is None

    def test_requested_mode_falls_back_to_row_mode(self) -> None:
        assert detect_mode_fallback("rerank", {"effective_mode": "standard"}) is not None


def _rerank_definition(dataset_id: str = "test-rerank", allow: bool = False) -> RetrievalDefinition:
    definition = _make_definition(dataset_id)
    return definition.model_copy(
        update={
            "retrieval": RetrievalTuning(
                modes=["rerank"], candidate_counts=[10], result_counts=[3]
            ),
            "execution": definition.execution.model_copy(update={"allow_mode_fallback": allow}),
        }
    )


class TestRetrievalModeFallbackRejection:
    @pytest.mark.asyncio
    async def test_fallback_rows_are_failed_not_stored(self, project_root: Path) -> None:
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_rerank_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=FallbackAdapter(),
        )
        status = get_status(paths)
        assert status["successful_count"] == 0
        assert status["failed_count"] == 4
        assert status["lifecycle"] != "complete"

    @pytest.mark.asyncio
    async def test_failure_record_explains_and_keeps_cost(self, project_root: Path) -> None:
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_rerank_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=FallbackAdapter(),
        )
        record = json.loads(sorted(paths.failures_dir.rglob("*.json"))[0].read_text())
        assert record["error_type"] == "retrieval_mode_fallback"
        assert record["retryable"] is False
        assert "rerank model unavailable" in record["error_message"]
        assert record["retrieval_metadata"]["effective_mode"] == "standard"
        assert "cost" in record

    @pytest.mark.asyncio
    async def test_opt_in_allows_fallback(self, project_root: Path) -> None:
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_rerank_definition(allow=True),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=FallbackAdapter(),
        )
        status = get_status(paths)
        assert status["successful_count"] == 4
        assert status["lifecycle"] == "complete"

    @pytest.mark.asyncio
    async def test_honest_adapter_is_unaffected(self, project_root: Path) -> None:
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_rerank_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=FakeAdapter(),
        )
        status = get_status(paths)
        assert status["successful_count"] == 4
        assert status["failed_count"] == 0


class TestRetrievalStageSuccess:
    @pytest.mark.asyncio
    async def test_full_success(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        status = get_status(paths)
        assert status["lifecycle"] == "complete"
        assert status["successful_count"] == 4  # 2 questions * 2 KBs * 1 mode
        assert status["missing_count"] == 0
        assert adapter.retrieve_call_count == 4

    @pytest.mark.asyncio
    async def test_dry_run(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
            dry_run=True,
        )
        assert adapter.retrieve_call_count == 0
        status = get_status(paths)
        assert status["lifecycle"] == "building"
        assert status["planned_count"] == 4

    @pytest.mark.asyncio
    async def test_artifacts_written(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        artifacts = list(paths.artifacts_dir.glob("*.json"))
        assert len(artifacts) == 4

    @pytest.mark.asyncio
    async def test_artifacts_contain_context(self, project_root: Path) -> None:
        import json

        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        for f in paths.artifacts_dir.glob("*.json"):
            data = json.loads(f.read_text())
            assert "prepared_context" in data
            assert len(data["prepared_context"]) > 0
            assert "estimated_context_tokens" in data
            assert data["token_estimate_label"] == "estimated"

    @pytest.mark.asyncio
    async def test_objective_metrics(self, project_root: Path) -> None:
        import json

        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        for f in paths.artifacts_dir.glob("*.json"):
            data = json.loads(f.read_text())
            metrics = data["objective_metrics"]
            assert "item_count" in metrics
            assert metrics["item_count"] == 3


class TestRetrievalStageFailure:
    @pytest.mark.asyncio
    async def test_partial_failure_continue_on_error(self, project_root: Path) -> None:
        error = AdapterError(
            error_type="schema",
            message="Bad request",
            retryable=False,
        )
        adapter = FakeAdapter(fail_retrieve_first_n=1, retrieve_error=error)
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        status = get_status(paths)
        # 3 succeeded, 1 failed - not all planned artifacts have successful results
        assert status["successful_count"] == 3
        assert status["failed_count"] >= 1

    @pytest.mark.asyncio
    async def test_retry_then_success(self, project_root: Path) -> None:
        error = AdapterError(
            error_type="transport",
            message="Timeout",
            retryable=True,
        )
        adapter = FakeAdapter(fail_retrieve_first_n=1, retrieve_error=error)
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        status = get_status(paths)
        assert status["successful_count"] == 4
        assert adapter.retrieve_call_count == 5  # 1 retry + 4 original

    @pytest.mark.asyncio
    async def test_placeholder_kb_rejected(self, project_root: Path) -> None:
        from rag_evals.errors import ExecutionError

        adapter = FakeAdapter()
        kbs = KnowledgeBaseCatalog(
            knowledge_bases=[
                KnowledgeBase(
                    id="kb1",
                    knowledge_base_id="KB_REPLACE_ME_1",
                    chunking={"strategy": "semantic"},
                ),
            ]
        )
        defn = RetrievalDefinition(
            schema_version=1,
            test=TestBlock(id="test", type="retrieval"),
            questions=SourceRef(source="questions.yaml"),
            knowledge_bases=SourceRef(source="kbs.yaml"),
            selection=Selection(knowledge_bases=CatalogSelection(include=["kb1"])),
            retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[10], result_counts=[3]),
            deployment=DeploymentConfig(
                label="test",
                merge={"enabled": False, "maximum_chunks_per_document": None},
            ),
            api=ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY"),
            output=OutputConfig(dataset_id="test-placeholder"),
            execution=ExecutionConfig(concurrency=1),
        )
        with pytest.raises(ExecutionError, match="placeholder"):
            await run_retrieval_stage(
                project_root=project_root,
                definition=defn,
                questions=_make_questions(),
                knowledge_bases=kbs,
                adapter=adapter,
            )


class TestRetrievalStageResume:
    @pytest.mark.asyncio
    async def test_resume_missing(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )
        # Delete one artifact to simulate interruption
        artifacts = list(paths.artifacts_dir.glob("*.json"))
        artifacts[0].unlink()

        status = get_status(paths)
        assert status["missing_count"] == 1

        # A tampered complete dataset is immutable and cannot be treated as an interruption.
        adapter2 = FakeAdapter()
        with pytest.raises(ArtifactError, match="violates invariants"):
            await run_retrieval_stage(
                project_root=project_root,
                definition=_make_definition(),
                questions=_make_questions(),
                knowledge_bases=_make_kbs(),
                adapter=adapter2,
                only_missing=True,
            )
        status = get_status(paths)
        assert status["lifecycle"] == "complete"
        assert status["invariant_errors"]
        assert adapter2.retrieve_call_count == 0

    @pytest.mark.asyncio
    async def test_resume_preserves_existing(self, project_root: Path) -> None:

        adapter = FakeAdapter()
        paths = await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter,
        )

        # Record existing artifact content
        artifacts = list(paths.artifacts_dir.glob("*.json"))
        original_content = artifacts[0].read_text()

        # Resume (should not overwrite)
        adapter2 = FakeAdapter()
        await run_retrieval_stage(
            project_root=project_root,
            definition=_make_definition(),
            questions=_make_questions(),
            knowledge_bases=_make_kbs(),
            adapter=adapter2,
            only_missing=True,
        )

        # Check existing artifact unchanged
        assert artifacts[0].read_text() == original_content
