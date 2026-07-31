"""Tests for retrieval matrix generation."""

from rag_evals.config.schemas import (
    CatalogSelection,
    DeploymentConfig,
    KnowledgeBase,
    KnowledgeBaseCatalog,
    Question,
    QuestionCatalog,
    RetrievalTuning,
    Selection,
)
from rag_evals.planning.matrices import generate_retrieval_matrix


def _questions(ids: list[str]) -> QuestionCatalog:
    return QuestionCatalog(questions=[Question(id=i, question=f"Question {i}?") for i in ids])


def _kbs(ids: list[str], merge_eval: bool = True) -> KnowledgeBaseCatalog:
    return KnowledgeBaseCatalog(
        knowledge_bases=[
            KnowledgeBase(
                id=i,
                knowledge_base_id=f"KB_{i}",
                chunking={"strategy": "semantic"},
                supports_merge_eval=merge_eval,
            )
            for i in ids
        ]
    )


def _selection(kb_ids: list[str]) -> Selection:
    return Selection(knowledge_bases=CatalogSelection(include=kb_ids))


class TestRetrievalMatrix:
    def test_single_combination(self) -> None:
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1"]),
            knowledge_bases=_kbs(["kb1"]),
            selection=_selection(["kb1"]),
            retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[20], result_counts=[5]),
            deployment=DeploymentConfig(
                label="test",
                merge={"enabled": False, "maximum_chunks_per_document": None},
            ),
        )
        assert len(matrix.rows) == 1
        assert matrix.total_calls == 1
        assert matrix.rows[0].question_id == "q1"
        assert matrix.rows[0].kb_catalog_id == "kb1"

    def test_cartesian_product(self) -> None:
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1", "q2"]),
            knowledge_bases=_kbs(["kb1", "kb2"]),
            selection=_selection(["kb1", "kb2"]),
            retrieval=RetrievalTuning(
                modes=["standard", "rerank"], candidate_counts=[20], result_counts=[5]
            ),
            deployment=DeploymentConfig(
                label="test",
                merge={"enabled": False, "maximum_chunks_per_document": None},
            ),
        )
        # 2 questions * 2 KBs * 2 modes = 8 rows
        assert len(matrix.rows) == 8
        assert matrix.total_calls == 8

    def test_deterministic_ordering(self) -> None:
        retrieval = RetrievalTuning(
            modes=["rerank", "standard"], candidate_counts=[20], result_counts=[5]
        )
        deployment = DeploymentConfig(
            label="test", merge={"enabled": False, "maximum_chunks_per_document": None}
        )
        m1 = generate_retrieval_matrix(
            _questions(["q1", "q2"]),
            _kbs(["kb1", "kb2"]),
            _selection(["kb1", "kb2"]),
            retrieval,
            deployment,
        )
        m2 = generate_retrieval_matrix(
            _questions(["q2", "q1"]),
            _kbs(["kb2", "kb1"]),
            _selection(["kb2", "kb1"]),
            retrieval,
            deployment,
        )
        # Same set of artifact IDs despite different input order
        assert sorted(r.artifact_id for r in m1.rows) == sorted(r.artifact_id for r in m2.rows)

    def test_merge_excludes_no_chunk_kb(self) -> None:
        kbs = KnowledgeBaseCatalog(
            knowledge_bases=[
                KnowledgeBase(
                    id="kb_merge",
                    knowledge_base_id="KB1",
                    chunking={"strategy": "semantic"},
                    supports_merge_eval=True,
                ),
                KnowledgeBase(
                    id="kb_no_merge",
                    knowledge_base_id="KB2",
                    chunking={"strategy": "none"},
                    supports_merge_eval=False,
                ),
            ]
        )
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1"]),
            knowledge_bases=kbs,
            selection=_selection(["kb_merge", "kb_no_merge"]),
            retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[20], result_counts=[5]),
            deployment=DeploymentConfig(
                label="merge-enabled",
                merge={"enabled": True, "maximum_chunks_per_document": 5},
            ),
        )
        assert len(matrix.rows) == 1  # only kb_merge
        assert len(matrix.excluded) == 1  # kb_no_merge excluded
        assert matrix.excluded[0].kb_catalog_id == "kb_no_merge"

    def test_merge_disabled_includes_all(self) -> None:
        kbs = KnowledgeBaseCatalog(
            knowledge_bases=[
                KnowledgeBase(
                    id="kb1",
                    knowledge_base_id="KB1",
                    chunking={"strategy": "semantic"},
                    supports_merge_eval=True,
                ),
                KnowledgeBase(
                    id="kb2",
                    knowledge_base_id="KB2",
                    chunking={"strategy": "none"},
                    supports_merge_eval=False,
                ),
            ]
        )
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1"]),
            knowledge_bases=kbs,
            selection=_selection(["kb1", "kb2"]),
            retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[20], result_counts=[5]),
            deployment=DeploymentConfig(
                label="merge-disabled",
                merge={"enabled": False, "maximum_chunks_per_document": None},
            ),
        )
        assert len(matrix.rows) == 2  # both included
        assert len(matrix.excluded) == 0

    def test_result_gt_candidate_skipped(self) -> None:
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1"]),
            knowledge_bases=_kbs(["kb1"]),
            selection=_selection(["kb1"]),
            retrieval=RetrievalTuning(
                modes=["standard"], candidate_counts=[10, 20], result_counts=[5, 15]
            ),
            deployment=DeploymentConfig(
                label="test", merge={"enabled": False, "maximum_chunks_per_document": None}
            ),
        )
        # (10, 5), (10, 15) -> 15 > 10 skip, (20, 5), (20, 15)
        # So 3 valid rows
        assert len(matrix.rows) == 3

    def test_artifact_ids_stable(self) -> None:
        retrieval = RetrievalTuning(modes=["standard"], candidate_counts=[20], result_counts=[5])
        deployment = DeploymentConfig(
            label="test", merge={"enabled": False, "maximum_chunks_per_document": None}
        )
        m1 = generate_retrieval_matrix(
            _questions(["q1"]), _kbs(["kb1"]), _selection(["kb1"]), retrieval, deployment
        )
        m2 = generate_retrieval_matrix(
            _questions(["q1"]), _kbs(["kb1"]), _selection(["kb1"]), retrieval, deployment
        )
        assert m1.rows[0].artifact_id == m2.rows[0].artifact_id

    def test_selected_ids_preserved(self) -> None:
        matrix = generate_retrieval_matrix(
            questions=_questions(["q1", "q2", "q3"]),
            knowledge_bases=_kbs(["kb1", "kb2"]),
            selection=_selection(["kb1"]),
            retrieval=RetrievalTuning(modes=["standard"], candidate_counts=[20], result_counts=[5]),
            deployment=DeploymentConfig(
                label="test", merge={"enabled": False, "maximum_chunks_per_document": None}
            ),
        )
        assert matrix.selected_question_ids == ["q1", "q2", "q3"]
        assert matrix.selected_kb_ids == ["kb1"]
