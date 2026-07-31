"""Selection validation: apply include/exclude rules against catalogs."""

from __future__ import annotations

from dataclasses import dataclass

from rag_evals.config.schemas import (
    CatalogSelection,
    KnowledgeBaseCatalog,
    ModelCatalog,
    QuestionCatalog,
    Selection,
)
from rag_evals.errors import ConfigError


@dataclass(frozen=True)
class SelectedCatalog:
    """Result of applying selection rules to a catalog."""

    selected_ids: list[str]
    excluded_ids: list[str]


def select_questions(
    catalog: QuestionCatalog,
    selection: CatalogSelection | None,
) -> SelectedCatalog:
    """Apply selection to a question catalog."""
    return _apply_selection(
        [q.id for q in catalog.questions],
        selection,
        "question",
    )


def select_knowledge_bases(
    catalog: KnowledgeBaseCatalog,
    selection: CatalogSelection | None,
) -> SelectedCatalog:
    """Apply selection to a KB catalog."""
    return _apply_selection(
        [kb.id for kb in catalog.knowledge_bases],
        selection,
        "knowledge base",
    )


def select_models(
    catalog: ModelCatalog,
    selection: CatalogSelection | None,
    label: str = "model",
) -> SelectedCatalog:
    """Apply selection to a model catalog."""
    return _apply_selection(
        [m.id for m in catalog.models],
        selection,
        label,
    )


def _apply_selection(
    all_ids: list[str],
    selection: CatalogSelection | None,
    label: str,
) -> SelectedCatalog:
    """Apply include/exclude rules and return deterministic selected/excluded lists."""
    catalog_set = set(all_ids)
    if selection is None:
        return SelectedCatalog(selected_ids=list(all_ids), excluded_ids=[])

    if selection.include is not None:
        _check_unknown(selection.include, catalog_set, label)
        include_set = set(selection.include)
        selected = [id_ for id_ in all_ids if id_ in include_set]
        excluded = [id_ for id_ in all_ids if id_ not in include_set]
    else:
        assert selection.exclude is not None
        _check_unknown(selection.exclude, catalog_set, label)
        exclude_set = set(selection.exclude)
        selected = [id_ for id_ in all_ids if id_ not in exclude_set]
        excluded = [id_ for id_ in all_ids if id_ in exclude_set]

    if not selected:
        raise ConfigError(f"Selection resulted in zero {label}s")

    return SelectedCatalog(selected_ids=selected, excluded_ids=excluded)


def _check_unknown(requested: list[str], available: set[str], label: str) -> None:
    unknown = set(requested) - available
    if unknown:
        raise ConfigError(
            f"Unknown {label} IDs in selection: {sorted(unknown)}. Available: {sorted(available)}"
        )


def resolve_full_selection(
    selection: Selection | None,
    questions: QuestionCatalog | None,
    knowledge_bases: KnowledgeBaseCatalog | None,
    answer_models: ModelCatalog | None,
    judge_models: ModelCatalog | None,
) -> dict[str, SelectedCatalog]:
    """Apply all selection rules and return a dict of results."""
    results: dict[str, SelectedCatalog] = {}
    if selection is None:
        selection = Selection()

    if questions is not None:
        results["questions"] = select_questions(questions, selection.questions)
    if knowledge_bases is not None:
        results["knowledge_bases"] = select_knowledge_bases(
            knowledge_bases, selection.knowledge_bases
        )
    if answer_models is not None:
        results["answer_models"] = select_models(
            answer_models, selection.answer_models, "answer model"
        )
    if judge_models is not None:
        results["judge_models"] = select_models(judge_models, selection.judge_models, "judge model")

    return results
