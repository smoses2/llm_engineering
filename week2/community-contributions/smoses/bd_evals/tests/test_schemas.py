"""Tests for core Pydantic schemas."""

import pytest
from pydantic import ValidationError

from rag_evals.config.schemas import (
    AnswerDefinition,
    AnswerJudgmentDefinition,
    ApiConfig,
    CatalogSelection,
    CompareDefinition,
    DeploymentConfig,
    ExecutionConfig,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    ModelCatalog,
    ModelEntry,
    OutputConfig,
    Question,
    QuestionCatalog,
    RetrievalDefinition,
    RetrievalJudgmentDefinition,
    RetrievalTuning,
    parse_definition,
)
from rag_evals.errors import ConfigError

# ---------------------------------------------------------------------------
# Question catalog
# ---------------------------------------------------------------------------


class TestQuestionCatalog:
    def test_valid_question(self) -> None:
        q = Question(id="q001", question="What is X?")
        assert q.id == "q001"
        assert q.rubric is None

    def test_question_with_rubric(self) -> None:
        q = Question(
            id="q001",
            question="What is X?",
            rubric={"expected_answer": "X is Y", "required_facts": ["fact1", "fact2"]},
        )
        assert q.rubric is not None
        assert q.rubric.expected_answer == "X is Y"
        assert q.rubric.required_facts == ["fact1", "fact2"]

    def test_rubric_extension_fields_allowed(self) -> None:
        q = Question(
            id="q001",
            question="What is X?",
            rubric={"expected_answer": "Y", "custom_field": "custom_value"},
        )
        assert q.rubric is not None
        assert q.rubric.model_extra is not None
        assert "custom_field" in q.rubric.model_extra

    def test_question_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Question(id="q001", question="")

    def test_question_invalid_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Question(id="", question="What?")

    def test_catalog_valid(self) -> None:
        cat = QuestionCatalog(
            questions=[
                {"id": "q001", "question": "What?"},
                {"id": "q002", "question": "Why?"},
            ]
        )
        assert len(cat.questions) == 2

    def test_catalog_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            QuestionCatalog(
                questions=[
                    {"id": "q001", "question": "What?"},
                    {"id": "q001", "question": "Why?"},
                ]
            )

    def test_catalog_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QuestionCatalog(questions=[])

    def test_catalog_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            QuestionCatalog(
                questions=[{"id": "q001", "question": "What?"}],
                unknown_field="bad",
            )


# ---------------------------------------------------------------------------
# Knowledge Base catalog
# ---------------------------------------------------------------------------


class TestKnowledgeBaseCatalog:
    def test_valid_kb_fixed(self) -> None:
        kb = KnowledgeBase(
            id="kb1",
            knowledge_base_id="KB123",
            chunking={"strategy": "fixed", "size": 500, "overlap": 100},
        )
        assert kb.chunking.strategy == "fixed"
        assert kb.supports_merge_eval is True

    def test_valid_kb_semantic(self) -> None:
        kb = KnowledgeBase(
            id="kb2",
            knowledge_base_id="KB456",
            chunking={"strategy": "semantic"},
        )
        assert kb.chunking.size is None

    def test_valid_kb_none(self) -> None:
        kb = KnowledgeBase(
            id="kb3",
            knowledge_base_id="KB789",
            chunking={"strategy": "none"},
            supports_merge_eval=False,
        )
        assert kb.supports_merge_eval is False

    def test_fixed_requires_size(self) -> None:
        with pytest.raises(ValidationError, match="size"):
            KnowledgeBase(id="kb1", knowledge_base_id="KB", chunking={"strategy": "fixed"})

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB",
                chunking={"strategy": "bad"},
            )

    def test_semantic_accepts_similarity_percentile_threshold(self) -> None:
        kb = KnowledgeBase(
            id="kb1",
            knowledge_base_id="KB",
            chunking={"strategy": "semantic", "similarity_percentile_threshold": 96},
        )
        assert kb.chunking.similarity_percentile_threshold == 96

    def test_similarity_percentile_threshold_rejected_for_fixed(self) -> None:
        with pytest.raises(ValidationError, match="only to semantic"):
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB",
                chunking={
                    "strategy": "fixed",
                    "size": 500,
                    "similarity_percentile_threshold": 96,
                },
            )

    @pytest.mark.parametrize("threshold", [0, 100])
    def test_similarity_percentile_threshold_out_of_range(self, threshold: int) -> None:
        with pytest.raises(ValidationError, match="between 1 and 99"):
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB",
                chunking={
                    "strategy": "semantic",
                    "similarity_percentile_threshold": threshold,
                },
            )

    def test_unknown_chunking_field_still_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KnowledgeBase(
                id="kb1",
                knowledge_base_id="KB",
                chunking={"strategy": "semantic", "similarity_percentil": 96},
            )

    def test_placeholder_detection(self) -> None:
        kb = KnowledgeBase(
            id="kb1",
            knowledge_base_id="KB_REPLACE_ME_1",
            chunking={"strategy": "semantic"},
        )
        assert kb.is_placeholder is True

    def test_non_placeholder(self) -> None:
        kb = KnowledgeBase(
            id="kb1",
            knowledge_base_id="ABC123",
            chunking={"strategy": "semantic"},
        )
        assert kb.is_placeholder is False

    def test_catalog_duplicate_ids(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            KnowledgeBaseCatalog(
                knowledge_bases=[
                    {"id": "kb1", "knowledge_base_id": "KB1", "chunking": {"strategy": "none"}},
                    {"id": "kb1", "knowledge_base_id": "KB2", "chunking": {"strategy": "none"}},
                ]
            )

    def test_catalog_duplicate_knowledge_base_ids(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate knowledge_base_id"):
            KnowledgeBaseCatalog(
                knowledge_bases=[
                    {"id": "kb1", "knowledge_base_id": "SAME", "chunking": {"strategy": "none"}},
                    {
                        "id": "kb2",
                        "knowledge_base_id": "SAME",
                        "chunking": {"strategy": "semantic"},
                    },
                ]
            )

    def test_catalog_allows_repeated_placeholders(self) -> None:
        catalog = KnowledgeBaseCatalog(
            knowledge_bases=[
                {
                    "id": "kb1",
                    "knowledge_base_id": "KB_REPLACE_ME_1",
                    "chunking": {"strategy": "none"},
                },
                {
                    "id": "kb2",
                    "knowledge_base_id": "KB_REPLACE_ME_1",
                    "chunking": {"strategy": "semantic"},
                },
            ]
        )
        assert len(catalog.knowledge_bases) == 2


# ---------------------------------------------------------------------------
# Model catalogs
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_valid_model(self) -> None:
        m = ModelEntry(id="m1", model_id="anthropic.claude-3")
        assert m.default_inference_config is None

    def test_model_with_inference_config(self) -> None:
        m = ModelEntry(
            id="m1",
            model_id="MODEL_REPLACE_ME_1",
            default_inference_config={"maxTokens": 2048, "temperature": 0},
        )
        assert m.default_inference_config is not None
        assert m.default_inference_config.maxTokens == 2048

    def test_inference_config_extension_fields(self) -> None:
        m = ModelEntry(
            id="m1",
            model_id="m1",
            default_inference_config={"maxTokens": 100, "topP": 0.9},
        )
        assert m.default_inference_config is not None
        assert m.default_inference_config.model_extra is not None

    def test_model_placeholder(self) -> None:
        m = ModelEntry(id="m1", model_id="MODEL_REPLACE_ME_1")
        assert m.is_placeholder is True

    def test_catalog_duplicate_ids(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            ModelCatalog(
                models=[
                    {"id": "m1", "model_id": "A"},
                    {"id": "m1", "model_id": "B"},
                ]
            )


# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------


class TestApiConfig:
    def test_valid(self) -> None:
        api = ApiConfig(base_url_env="BD_API_BASE_URL", api_key_env="BD_API_KEY")
        assert api.base_url_env == "BD_API_BASE_URL"

    def test_lowercase_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApiConfig(base_url_env="bd_api_base_url", api_key_env="BD_API_KEY")

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApiConfig(base_url_env="", api_key_env="BD_API_KEY")

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ApiConfig(base_url_env="BD_API", api_key_env="BD_API_KEY", secret="value")


# ---------------------------------------------------------------------------
# Deployment and merge
# ---------------------------------------------------------------------------


class TestDeployment:
    def test_merge_disabled_valid(self) -> None:
        d = DeploymentConfig(
            label="test",
            merge={"enabled": False, "maximum_chunks_per_document": None},
        )
        assert d.merge.enabled is False

    def test_merge_enabled_valid(self) -> None:
        d = DeploymentConfig(
            label="test",
            merge={"enabled": True, "maximum_chunks_per_document": 10},
        )
        assert d.merge.maximum_chunks_per_document == 10

    def test_merge_enabled_without_limit(self) -> None:
        with pytest.raises(ValidationError, match="maximum_chunks_per_document"):
            DeploymentConfig(label="test", merge={"enabled": True})

    def test_merge_disabled_with_limit(self) -> None:
        with pytest.raises(ValidationError, match="null"):
            DeploymentConfig(
                label="test",
                merge={"enabled": False, "maximum_chunks_per_document": 10},
            )


# ---------------------------------------------------------------------------
# Execution, retry, output
# ---------------------------------------------------------------------------


class TestExecutionOutput:
    def test_execution_defaults(self) -> None:
        e = ExecutionConfig()
        assert e.concurrency == 5
        assert e.continue_on_error is True
        assert e.retries.maximum_attempts == 3

    def test_output_valid_id(self) -> None:
        o = OutputConfig(dataset_id="my-dataset-001")
        assert o.dataset_id == "my-dataset-001"

    def test_output_invalid_id_uppercase(self) -> None:
        with pytest.raises(ValidationError):
            OutputConfig(dataset_id="MyDataset")

    def test_output_invalid_id_starts_with_dash(self) -> None:
        with pytest.raises(ValidationError):
            OutputConfig(dataset_id="-bad")

    def test_output_invalid_id_too_long(self) -> None:
        with pytest.raises(ValidationError):
            OutputConfig(dataset_id="a" * 64)

    def test_retry_max_attempts_bounds(self) -> None:
        from rag_evals.config.schemas import RetryConfig

        assert RetryConfig(maximum_attempts=1).maximum_attempts == 1
        with pytest.raises(ValidationError):
            RetryConfig(maximum_attempts=0)
        with pytest.raises(ValidationError):
            RetryConfig(maximum_attempts=11)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class TestSelection:
    def test_include_only(self) -> None:
        s = CatalogSelection(include=["a", "b"])
        assert s.include == ["a", "b"]
        assert s.exclude is None

    def test_exclude_only(self) -> None:
        s = CatalogSelection(exclude=["c"])
        assert s.exclude == ["c"]

    def test_both_rejected(self) -> None:
        with pytest.raises(ValidationError, match="both"):
            CatalogSelection(include=["a"], exclude=["b"])

    def test_neither_rejected(self) -> None:
        with pytest.raises(ValidationError, match="either"):
            CatalogSelection()


# ---------------------------------------------------------------------------
# Retrieval tuning
# ---------------------------------------------------------------------------


class TestRetrievalTuning:
    def test_valid(self) -> None:
        rt = RetrievalTuning(modes=["standard", "rerank"], candidate_counts=[20], result_counts=[5])
        assert rt.modes == ["standard", "rerank"]

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalTuning(modes=["bad"], candidate_counts=[20], result_counts=[5])

    def test_result_gt_candidate_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed"):
            RetrievalTuning(modes=["standard"], candidate_counts=[5], result_counts=[20])

    def test_result_eq_candidate_ok(self) -> None:
        rt = RetrievalTuning(modes=["standard"], candidate_counts=[5], result_counts=[5])
        assert rt.result_counts == [5]

    def test_multi_counts_valid(self) -> None:
        rt = RetrievalTuning(modes=["standard"], candidate_counts=[20, 50], result_counts=[5, 20])
        assert len(rt.candidate_counts) == 2

    def test_zero_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalTuning(modes=["standard"], candidate_counts=[0], result_counts=[0])

    def test_empty_lists_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalTuning(modes=[], candidate_counts=[20], result_counts=[5])


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def _retrieval_def_data() -> dict:
    return {
        "schema_version": 1,
        "test": {"id": "ret-example", "type": "retrieval"},
        "questions": {"source": "../../config/questions.yaml"},
        "knowledge_bases": {"source": "../../config/knowledge-bases.yaml"},
        "selection": {"knowledge_bases": {"include": ["kb1", "kb2"]}},
        "retrieval": {"modes": ["standard"], "candidate_counts": [20], "result_counts": [5]},
        "deployment": {
            "label": "test",
            "merge": {"enabled": False, "maximum_chunks_per_document": None},
        },
        "api": {"base_url_env": "BD_API_BASE_URL", "api_key_env": "BD_API_KEY"},
        "output": {"dataset_id": "ret-example"},
    }


def _ask_def_data(test_type: str) -> dict:
    return {
        "schema_version": 1,
        "test": {"id": f"{test_type}-example", "type": test_type},
        "source_dataset": {"type": "retrieval", "id": "ret-example"},
        "judge_models": {"source": "../../config/judge-models.yaml"},
        "selection": {"judge_models": {"include": ["j1"]}},
        "prompt_variants": [
            {
                "id": "default",
                "template": {"system_instructions": "sys.md", "user_message": "user.yaml"},
            }
        ],
        "inference_variants": [
            {"id": "default", "inference_config": {"maxTokens": 2048, "temperature": 0}},
        ],
        "api": {"base_url_env": "BD_API_BASE_URL", "api_key_env": "BD_API_KEY"},
        "output": {"dataset_id": f"{test_type.replace('_', '-')}-example"},
    }


class TestRetrievalDefinition:
    def test_valid(self) -> None:
        d = RetrievalDefinition.model_validate(_retrieval_def_data())
        assert d.test.type == "retrieval"
        assert d.execution.concurrency == 5

    def test_wrong_test_type(self) -> None:
        data = _retrieval_def_data()
        data["test"]["type"] = "answer"
        with pytest.raises(ValidationError, match="retrieval"):
            RetrievalDefinition.model_validate(data)

    def test_extra_field_rejected(self) -> None:
        data = _retrieval_def_data()
        data["extra"] = "bad"
        with pytest.raises(ValidationError):
            RetrievalDefinition.model_validate(data)


class TestAskDefinitions:
    @pytest.mark.parametrize(
        "model_class,test_type",
        [
            (RetrievalJudgmentDefinition, "retrieval_judgment"),
            (AnswerJudgmentDefinition, "answer_judgment"),
        ],
    )
    def test_valid(self, model_class: type, test_type: str) -> None:
        data = _ask_def_data(test_type)
        if test_type == "answer_judgment":
            data["source_dataset"] = {"type": "answers", "id": "ans-example"}
        d = model_class.model_validate(data)
        assert d.test.type == test_type

    def test_answer_definition_valid(self) -> None:
        data = _ask_def_data("answer")
        data["answer_models"] = data.pop("judge_models")
        data["selection"]["answer_models"] = data["selection"].pop("judge_models")
        data["source_dataset"] = {"type": "retrieval", "id": "ret-example"}
        d = AnswerDefinition.model_validate(data)
        assert d.test.type == "answer"

    def test_duplicate_prompt_variants(self) -> None:
        data = _ask_def_data("retrieval_judgment")
        data["prompt_variants"].append(data["prompt_variants"][0])
        with pytest.raises(ValidationError, match="Duplicate"):
            RetrievalJudgmentDefinition.model_validate(data)

    def test_duplicate_inference_variants(self) -> None:
        data = _ask_def_data("retrieval_judgment")
        data["inference_variants"].append(data["inference_variants"][0])
        with pytest.raises(ValidationError, match="Duplicate"):
            RetrievalJudgmentDefinition.model_validate(data)


class TestCompareDefinition:
    def test_valid(self) -> None:
        d = CompareDefinition.model_validate(
            {
                "schema_version": 1,
                "test": {"id": "cmp", "type": "compare"},
                "entries": [
                    {"label": "a", "dataset": {"type": "retrieval", "id": "ds1"}},
                    {"label": "b", "dataset": {"type": "retrieval", "id": "ds2"}},
                ],
            }
        )
        assert len(d.entries) == 2

    def test_single_entry_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompareDefinition.model_validate(
                {
                    "schema_version": 1,
                    "test": {"id": "cmp", "type": "compare"},
                    "entries": [{"label": "a", "dataset": {"type": "retrieval", "id": "ds1"}}],
                }
            )

    def test_duplicate_labels(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            CompareDefinition.model_validate(
                {
                    "schema_version": 1,
                    "test": {"id": "cmp", "type": "compare"},
                    "entries": [
                        {"label": "a", "dataset": {"type": "retrieval", "id": "ds1"}},
                        {"label": "a", "dataset": {"type": "retrieval", "id": "ds2"}},
                    ],
                }
            )


# ---------------------------------------------------------------------------
# parse_definition
# ---------------------------------------------------------------------------


class TestParseDefinition:
    def test_retrieval(self) -> None:
        d = parse_definition(_retrieval_def_data())
        assert isinstance(d, RetrievalDefinition)

    def test_retrieval_judgment(self) -> None:
        d = parse_definition(_ask_def_data("retrieval_judgment"))
        assert isinstance(d, RetrievalJudgmentDefinition)

    def test_answer(self) -> None:
        data = _ask_def_data("answer")
        data["answer_models"] = data.pop("judge_models")
        data["selection"]["answer_models"] = data["selection"].pop("judge_models")
        data["source_dataset"] = {"type": "retrieval", "id": "ret-example"}
        d = parse_definition(data)
        assert isinstance(d, AnswerDefinition)

    def test_answer_judgment(self) -> None:
        data = _ask_def_data("answer_judgment")
        data["source_dataset"] = {"type": "answers", "id": "ans-example"}
        d = parse_definition(data)
        assert isinstance(d, AnswerJudgmentDefinition)

    def test_compare(self) -> None:
        d = parse_definition(
            {
                "schema_version": 1,
                "test": {"id": "cmp", "type": "compare"},
                "entries": [
                    {"label": "a", "dataset": {"type": "retrieval", "id": "ds1"}},
                    {"label": "b", "dataset": {"type": "retrieval", "id": "ds2"}},
                ],
            }
        )
        assert isinstance(d, CompareDefinition)

    def test_unknown_type(self) -> None:
        with pytest.raises(ConfigError, match="Unknown"):
            parse_definition({"test": {"id": "x", "type": "bad"}})

    def test_no_test_block(self) -> None:
        with pytest.raises(ConfigError, match="test"):
            parse_definition({"schema_version": 1})

    def test_not_mapping(self) -> None:
        with pytest.raises(ConfigError, match="mapping"):
            parse_definition("not a dict")  # type: ignore[arg-type]
