"""Tests for judgment, answer, and reporting stages with fake adapter."""

import json
from pathlib import Path

import pytest

from rag_evals.api.fake_adapter import FakeAdapter
from rag_evals.artifacts.store import get_status
from rag_evals.config.schemas import (
    AnswerDefinition,
    ApiConfig,
    CatalogSelection,
    DatasetRef,
    DeploymentConfig,
    ExecutionConfig,
    InferenceVariant,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    ModelCatalog,
    ModelEntry,
    OutputConfig,
    PromptVariant,
    Question,
    QuestionCatalog,
    RetrievalDefinition,
    RetrievalJudgmentDefinition,
    RetrievalTuning,
    Selection,
    SourceRef,
    TemplateRef,
    TestBlock,
)
from rag_evals.reporting.reports import (
    generate_report,
    join_lineage,
    render_csv,
    render_markdown,
)
from rag_evals.stages.answer import run_answer_stage
from rag_evals.stages.judgment import run_judgment_stage
from rag_evals.stages.retrieval import run_retrieval_stage
from rag_evals.templates import Stage


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


def _make_questions() -> QuestionCatalog:
    return QuestionCatalog(
        questions=[
            Question(id="q1", question="What is X?"),
            Question(id="q2", question="What is Y?"),
        ]
    )


def _make_kbs() -> KnowledgeBaseCatalog:
    return KnowledgeBaseCatalog(
        knowledge_bases=[
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB1",
                chunking={"strategy": "semantic"},
            ),
        ]
    )


def _make_models(model_id: str = "m1") -> ModelCatalog:
    return ModelCatalog(
        models=[
            ModelEntry(id=model_id, model_id="MODEL1"),
        ]
    )


def _retrieval_def(ds_id: str = "ret") -> RetrievalDefinition:
    return RetrievalDefinition(
        schema_version=1,
        test=TestBlock(id="test", type="retrieval"),
        questions=SourceRef(source="questions.yaml"),
        knowledge_bases=SourceRef(source="kbs.yaml"),
        selection=Selection(knowledge_bases=CatalogSelection(include=["kb1"])),
        retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[10], result_counts=[3]),
        deployment=DeploymentConfig(
            label="test", merge={"enabled": False, "maximum_chunks_per_document": None}
        ),
        api=ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY"),
        output=OutputConfig(dataset_id=ds_id),
        execution=ExecutionConfig(concurrency=2),
    )


def _judgment_def(
    ds_id: str, source_id: str, source_type: str = "retrieval"
) -> RetrievalJudgmentDefinition:
    return RetrievalJudgmentDefinition(
        schema_version=1,
        test=TestBlock(id="test", type="retrieval_judgment"),
        source_dataset=DatasetRef(type=source_type, id=source_id),
        judge_models=SourceRef(source="models.yaml"),
        selection=Selection(judge_models=CatalogSelection(include=["j1"])),
        prompt_variants=[
            PromptVariant(
                id="default",
                template=TemplateRef(
                    system_instructions="You are a judge.",
                    user_message="Question: {{ question }}\nContext: {{ retrieval_context }}",
                ),
            )
        ],
        inference_variants=[
            InferenceVariant(id="default", inference_config={"maxTokens": 2048, "temperature": 0})
        ],
        api=ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY"),
        output=OutputConfig(dataset_id=ds_id),
        execution=ExecutionConfig(concurrency=2),
    )


def _answer_def(ds_id: str, source_id: str) -> AnswerDefinition:
    return AnswerDefinition(
        schema_version=1,
        test=TestBlock(id="test", type="answer"),
        source_dataset=DatasetRef(type="retrieval", id=source_id),
        answer_models=SourceRef(source="models.yaml"),
        selection=Selection(answer_models=CatalogSelection(include=["a1"])),
        prompt_variants=[
            PromptVariant(
                id="default",
                template=TemplateRef(
                    system_instructions="Answer the question.",
                    user_message="Q: {{ question }}\nContext: {{ retrieval_context }}",
                ),
            )
        ],
        inference_variants=[
            InferenceVariant(id="default", inference_config={"maxTokens": 2048, "temperature": 0})
        ],
        api=ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY"),
        output=OutputConfig(dataset_id=ds_id),
        execution=ExecutionConfig(concurrency=2),
    )


class TestRetrievalJudgmentStage:
    @pytest.mark.asyncio
    async def test_judgment_success(self, project_root: Path) -> None:
        # First create a retrieval dataset
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        # Now run judgment
        models = _make_models("j1")
        defn = _judgment_def("rj", "ret")
        paths = await run_judgment_stage(
            project_root,
            defn,
            _make_questions(),
            models,
            adapter,
            "retrieval",
            "ret",
            Stage.RETRIEVAL_JUDGE,
        )
        status = get_status(paths)
        assert status["successful_count"] == 2
        assert adapter.ask_call_count == 2

    @pytest.mark.asyncio
    async def test_judgment_stores_raw_text(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )
        models = _make_models("j1")
        defn = _judgment_def("rj", "ret")
        paths = await run_judgment_stage(
            project_root,
            defn,
            _make_questions(),
            models,
            adapter,
            "retrieval",
            "ret",
            Stage.RETRIEVAL_JUDGE,
        )
        for f in paths.artifacts_dir.glob("*.json"):
            data = json.loads(f.read_text())
            assert "raw_judge_text" in data
            assert len(data["raw_judge_text"]) > 0
            assert data["parser_mode"] == "raw_text"

    @pytest.mark.asyncio
    async def test_judgment_dry_run(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )
        models = _make_models("j1")
        defn = _judgment_def("rj", "ret")
        await run_judgment_stage(
            project_root,
            defn,
            _make_questions(),
            models,
            adapter,
            "retrieval",
            "ret",
            Stage.RETRIEVAL_JUDGE,
            dry_run=True,
        )
        assert adapter.ask_call_count == 0


class TestAnswerStage:
    @pytest.mark.asyncio
    async def test_answer_success(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        models = _make_models("a1")
        defn = _answer_def("ans", "ret")
        paths = await run_answer_stage(project_root, defn, _make_questions(), models, adapter)
        status = get_status(paths)
        assert status["successful_count"] == 2
        assert adapter.ask_call_count == 2

    @pytest.mark.asyncio
    async def test_answer_stores_usage(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        models = _make_models("a1")
        defn = _answer_def("ans", "ret")
        paths = await run_answer_stage(project_root, defn, _make_questions(), models, adapter)
        for f in paths.artifacts_dir.glob("*.json"):
            data = json.loads(f.read_text())
            assert data["usage"]["totalTokens"] == 150
            assert data["incidental_retrieval"]["used_as_frozen_generation_context"] is False
            assert "frozen_context_hash" in data

    @pytest.mark.asyncio
    async def test_answer_no_retrieval_call(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )
        retrieve_count_before = adapter.retrieve_call_count

        models = _make_models("a1")
        defn = _answer_def("ans", "ret")
        await run_answer_stage(project_root, defn, _make_questions(), models, adapter)
        assert adapter.retrieve_call_count == retrieve_count_before


class TestReporting:
    @pytest.mark.asyncio
    async def test_report_generation(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        result = generate_report(
            project_root,
            project_root / "reports",
            retrieval_dataset_id="ret",
        )
        assert "csv" in result
        assert "markdown" in result
        csv_path = Path(result["csv"])
        md_path = Path(result["markdown"])
        assert csv_path.exists()
        assert md_path.exists()
        csv_content = csv_path.read_text()
        assert "question_id" in csv_content
        assert "q1" in csv_content

    @pytest.mark.asyncio
    async def test_report_joins_judgments(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )
        models = _make_models("j1")
        defn = _judgment_def("rj", "ret")
        await run_judgment_stage(
            project_root,
            defn,
            _make_questions(),
            models,
            adapter,
            "retrieval",
            "ret",
            Stage.RETRIEVAL_JUDGE,
        )

        rows = join_lineage(
            project_root,
            retrieval_dataset_id="ret",
            retrieval_judgment_dataset_id="rj",
        )
        assert len(rows) == 2
        assert rows[0].retrieval_judgment_text != ""

    @pytest.mark.asyncio
    async def test_csv_deterministic(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        rows1 = join_lineage(project_root, retrieval_dataset_id="ret")
        rows2 = join_lineage(project_root, retrieval_dataset_id="ret")
        assert render_csv(rows1) == render_csv(rows2)

    @pytest.mark.asyncio
    async def test_markdown_contains_cost_summary(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )

        rows = join_lineage(project_root, retrieval_dataset_id="ret")
        md = render_markdown(rows)
        assert "Cost Summary" in md
        assert "tested-system" in md.lower() or "tested system" in md.lower()

    @pytest.mark.asyncio
    async def test_report_zero_adapter_calls(self, project_root: Path) -> None:
        adapter = FakeAdapter()
        await run_retrieval_stage(
            project_root, _retrieval_def(), _make_questions(), _make_kbs(), adapter
        )
        ask_before = adapter.ask_call_count
        retrieve_before = adapter.retrieve_call_count

        generate_report(
            project_root,
            project_root / "reports",
            retrieval_dataset_id="ret",
        )
        assert adapter.ask_call_count == ask_before
        assert adapter.retrieve_call_count == retrieve_before
