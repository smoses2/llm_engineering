"""Tests for selection validation."""

import pytest

from rag_evals.config.schemas import (
    CatalogSelection,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    ModelCatalog,
    ModelEntry,
    Question,
    QuestionCatalog,
)
from rag_evals.config.selection import (
    select_knowledge_bases,
    select_models,
    select_questions,
)
from rag_evals.errors import ConfigError


def _question_catalog(ids: list[str]) -> QuestionCatalog:
    return QuestionCatalog(questions=[Question(id=i, question=f"Q for {i}?") for i in ids])


def _kb_catalog(ids: list[str]) -> KnowledgeBaseCatalog:
    return KnowledgeBaseCatalog(
        knowledge_bases=[
            KnowledgeBase(id=i, knowledge_base_id=f"KB_{i}", chunking={"strategy": "none"})
            for i in ids
        ]
    )


def _model_catalog(ids: list[str]) -> ModelCatalog:
    return ModelCatalog(models=[ModelEntry(id=i, model_id=f"M_{i}") for i in ids])


class TestSelectQuestions:
    def test_no_selection_returns_all(self) -> None:
        cat = _question_catalog(["q1", "q2", "q3"])
        result = select_questions(cat, None)
        assert result.selected_ids == ["q1", "q2", "q3"]
        assert result.excluded_ids == []

    def test_include(self) -> None:
        cat = _question_catalog(["q1", "q2", "q3"])
        result = select_questions(cat, CatalogSelection(include=["q1", "q3"]))
        assert result.selected_ids == ["q1", "q3"]
        assert result.excluded_ids == ["q2"]

    def test_exclude(self) -> None:
        cat = _question_catalog(["q1", "q2", "q3"])
        result = select_questions(cat, CatalogSelection(exclude=["q2"]))
        assert result.selected_ids == ["q1", "q3"]
        assert result.excluded_ids == ["q2"]

    def test_unknown_include(self) -> None:
        cat = _question_catalog(["q1", "q2"])
        with pytest.raises(ConfigError, match="Unknown"):
            select_questions(cat, CatalogSelection(include=["q1", "q9"]))

    def test_unknown_exclude(self) -> None:
        cat = _question_catalog(["q1", "q2"])
        with pytest.raises(ConfigError, match="Unknown"):
            select_questions(cat, CatalogSelection(exclude=["q9"]))

    def test_empty_result_rejected(self) -> None:
        cat = _question_catalog(["q1", "q2"])
        with pytest.raises(ConfigError, match="zero"):
            select_questions(cat, CatalogSelection(exclude=["q1", "q2"]))

    def test_preserves_catalog_order(self) -> None:
        cat = _question_catalog(["c", "a", "b"])
        result = select_questions(cat, CatalogSelection(include=["b", "a", "c"]))
        assert result.selected_ids == ["c", "a", "b"]


class TestSelectKnowledgeBases:
    def test_include(self) -> None:
        cat = _kb_catalog(["kb1", "kb2", "kb3"])
        result = select_knowledge_bases(cat, CatalogSelection(include=["kb1", "kb3"]))
        assert result.selected_ids == ["kb1", "kb3"]
        assert result.excluded_ids == ["kb2"]

    def test_unknown_id(self) -> None:
        cat = _kb_catalog(["kb1"])
        with pytest.raises(ConfigError, match="Unknown"):
            select_knowledge_bases(cat, CatalogSelection(include=["kb9"]))


class TestSelectModels:
    def test_include(self) -> None:
        cat = _model_catalog(["m1", "m2"])
        result = select_models(cat, CatalogSelection(include=["m1"]), "answer model")
        assert result.selected_ids == ["m1"]

    def test_unknown_id(self) -> None:
        cat = _model_catalog(["m1"])
        with pytest.raises(ConfigError, match="Unknown.*answer model"):
            select_models(cat, CatalogSelection(include=["m9"]), "answer model")

    def test_no_selection_returns_all(self) -> None:
        cat = _model_catalog(["m1", "m2"])
        result = select_models(cat, None, "model")
        assert result.selected_ids == ["m1", "m2"]
