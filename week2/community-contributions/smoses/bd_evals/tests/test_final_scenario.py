"""Mandatory offline multi-variant end-to-end remediation scenario."""

from pathlib import Path

import pytest

from rag_evals.api.fake_adapter import FakeAdapter
from rag_evals.api.protocol import AdapterError
from rag_evals.config.schemas import (
    AnswerJudgmentDefinition,
    CatalogSelection,
    DatasetRef,
    ExecutionConfig,
    InferenceVariant,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    ModelCatalog,
    ModelEntry,
    OutputConfig,
    PromptVariant,
    Selection,
    SourceRef,
    TemplateRef,
    TestBlock,
)
from rag_evals.errors import ArtifactError, ExecutionError
from rag_evals.reporting.reports import join_lineage
from rag_evals.stages.answer import run_answer_stage
from rag_evals.stages.judgment import run_judgment_stage
from rag_evals.stages.retrieval import run_retrieval_stage
from rag_evals.templates import Stage
from tests.test_stages_integration import (
    _answer_def,
    _judgment_def,
    _make_questions,
    _retrieval_def,
)


@pytest.mark.asyncio
async def test_f001_f002_f008_multi_variant_pipeline_preserves_every_row(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    retrieval_definition = _retrieval_def("ret").model_copy(
        update={
            "retrieval": _retrieval_def().retrieval.model_copy(
                update={"modes": ["standard", "rerank"]}
            ),
            "selection": Selection(),
        }
    )
    knowledge_bases = KnowledgeBaseCatalog(
        knowledge_bases=[
            KnowledgeBase(id="kb1", knowledge_base_id="KB1", chunking={"strategy": "semantic"}),
            KnowledgeBase(id="kb2", knowledge_base_id="KB2", chunking={"strategy": "semantic"}),
        ]
    )
    await run_retrieval_stage(
        tmp_path, retrieval_definition, _make_questions(), knowledge_bases, adapter
    )
    assert adapter.retrieve_call_count == 8

    judge_models = ModelCatalog(models=[ModelEntry(id="j1", model_id="JUDGE")])
    retrieval_judgment = _judgment_def("rj", "ret").model_copy(
        update={"selection": Selection(judge_models=CatalogSelection(include=["j1"]))}
    )
    await run_judgment_stage(
        tmp_path,
        retrieval_judgment,
        _make_questions(),
        judge_models,
        adapter,
        "retrieval",
        "ret",
        Stage.RETRIEVAL_JUDGE,
    )

    answer_models = ModelCatalog(
        models=[
            ModelEntry(id="a1", model_id="ANSWER1"),
            ModelEntry(id="a2", model_id="ANSWER2"),
        ]
    )
    prompts = [
        PromptVariant(
            id="p1",
            template=TemplateRef(
                system_instructions="Answer {{ question_id }}",
                user_message="{{ question }} {{ retrieval_context }}",
            ),
        ),
        PromptVariant(
            id="p2",
            template=TemplateRef(
                system_instructions="Respond to {{ question_id }}",
                user_message="Context: {{ retrieval_context }} Question: {{ question }}",
            ),
        ),
    ]
    answer_definition = _answer_def("ans", "ret").model_copy(
        update={
            "selection": Selection(),
            "prompt_variants": prompts,
            "inference_variants": [
                InferenceVariant(id="cold", inference_config={"temperature": 0.0})
            ],
        }
    )
    await run_answer_stage(tmp_path, answer_definition, _make_questions(), answer_models, adapter)

    answer_judgment = AnswerJudgmentDefinition(
        test=TestBlock(id="answer-judge", type="answer_judgment"),
        source_dataset=DatasetRef(type="answers", id="ans"),
        judge_models=SourceRef(source="models.yaml"),
        selection=Selection(),
        prompt_variants=[
            PromptVariant(
                id="judge",
                template=TemplateRef(
                    system_instructions="Judge {{ question_id }}",
                    user_message="{{ candidate_answer }}",
                ),
            )
        ],
        inference_variants=[InferenceVariant(id="judge", inference_config={})],
        api=answer_definition.api,
        output=OutputConfig(dataset_id="aj"),
    )
    await run_judgment_stage(
        tmp_path,
        answer_judgment,
        _make_questions(),
        judge_models,
        adapter,
        "answers",
        "ans",
        Stage.ANSWER_JUDGE,
    )

    rows = join_lineage(tmp_path, "ret", "rj", "ans", "aj")
    assert len(rows) == 32
    assert all(row.status == "complete" for row in rows)
    assert len({row.retrieval_artifact_id for row in rows}) == 8
    assert len({row.answer_artifact_id for row in rows}) == 32


@pytest.mark.asyncio
async def test_f010_fail_fast_stops_scheduling_and_leaves_resumable_failure(
    tmp_path: Path,
) -> None:
    definition = _retrieval_def("fail-fast").model_copy(
        update={
            "execution": ExecutionConfig(
                concurrency=2,
                continue_on_error=False,
                retries={"maximum_attempts": 1},
            )
        }
    )
    adapter = FakeAdapter(
        fail_retrieve_first_n=1,
        retrieve_error=AdapterError("schema", "bad request", retryable=False),
    )
    with pytest.raises(ExecutionError, match="stopped after failure"):
        await run_retrieval_stage(
            tmp_path,
            definition,
            _make_questions(),
            KnowledgeBaseCatalog(
                knowledge_bases=[
                    KnowledgeBase(
                        id="kb1",
                        knowledge_base_id="KB1",
                        chunking={"strategy": "semantic"},
                    )
                ]
            ),
            adapter,
        )
    assert adapter.retrieve_call_count == 1
    failure_files = list((tmp_path / "datasets/retrieval/fail-fast/failures").rglob("*.json"))
    assert len(failure_files) == 1


@pytest.mark.asyncio
async def test_f014_source_verification_failure_is_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = FakeAdapter()
    await run_retrieval_stage(
        tmp_path,
        _retrieval_def("ret"),
        _make_questions(),
        KnowledgeBaseCatalog(
            knowledge_bases=[
                KnowledgeBase(
                    id="kb1",
                    knowledge_base_id="KB1",
                    chunking={"strategy": "semantic"},
                )
            ]
        ),
        adapter,
    )

    def fail_verification(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ArtifactError("source became corrupt")

    monkeypatch.setattr("rag_evals.stages.answer.verify_lineage", fail_verification)
    definition = _answer_def("failed-answers", "ret")
    await run_answer_stage(
        tmp_path,
        definition,
        _make_questions(),
        ModelCatalog(models=[ModelEntry(id="a1", model_id="ANSWER")]),
        adapter,
    )
    failures = list(
        (tmp_path / "datasets/answers/failed-answers/failures").rglob("attempt_000.json")
    )
    assert len(failures) == 2
