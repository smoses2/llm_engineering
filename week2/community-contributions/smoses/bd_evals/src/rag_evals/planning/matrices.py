"""Retrieval and ask-stage matrix generation with deterministic ordering."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_evals.artifacts.lineage import SourceArtifact
from rag_evals.config.schemas import (
    AnswerDefinition,
    AnswerJudgmentDefinition,
    DeploymentConfig,
    InferenceVariant,
    KnowledgeBaseCatalog,
    ModelCatalog,
    PromptVariant,
    QuestionCatalog,
    RetrievalDefinition,
    RetrievalJudgmentDefinition,
    RetrievalTuning,
    Selection,
)
from rag_evals.config.selection import (
    select_knowledge_bases,
    select_models,
    select_questions,
)
from rag_evals.errors import PlanError
from rag_evals.planning.identity import artifact_id, ask_identity, retrieval_identity
from rag_evals.planning.identity import canonical_hash as identity_hash


@dataclass(frozen=True)
class RetrievalRow:
    """A single row in the retrieval matrix."""

    question_id: str
    question_text: str
    kb_catalog_id: str
    kb_actual_id: str
    mode: str
    candidate_count: int
    result_count: int
    deployment_label: str
    merge_enabled: bool
    supports_merge_eval: bool
    experiment_id: str
    artifact_id: str
    identity: dict[str, object]

    @property
    def is_excluded_merge(self) -> bool:
        """True if this row is excluded because merge is enabled but KB doesn't support it."""
        return self.merge_enabled and not self.supports_merge_eval


@dataclass(frozen=True)
class AskRow:
    """A single row in an ask-stage matrix."""

    question_id: str
    model_catalog_id: str
    model_actual_id: str
    prompt_variant_id: str
    inference_variant_id: str
    source_dataset_type: str
    source_dataset_id: str
    source_artifact_id: str
    source_content_hash: str
    source_relative_path: str
    question_text: str
    rubric: dict[str, object] | None
    experiment_id: str
    artifact_id: str
    identity: dict[str, object]


@dataclass
class RetrievalMatrix:
    """Result of generating a retrieval matrix."""

    rows: list[RetrievalRow] = field(default_factory=list)
    excluded: list[RetrievalRow] = field(default_factory=list)
    selected_question_ids: list[str] = field(default_factory=list)
    selected_kb_ids: list[str] = field(default_factory=list)
    total_calls: int = 0


@dataclass
class AskMatrix:
    """Result of generating an ask-stage matrix."""

    rows: list[AskRow] = field(default_factory=list)
    selected_question_ids: list[str] = field(default_factory=list)
    selected_model_ids: list[str] = field(default_factory=list)
    prompt_variant_ids: list[str] = field(default_factory=list)
    inference_variant_ids: list[str] = field(default_factory=list)
    source_dataset_type: str = ""
    source_dataset_id: str = ""
    total_calls: int = 0


SCHEMA_VERSION = "retrieval-v1"


def generate_retrieval_matrix(
    questions: QuestionCatalog,
    knowledge_bases: KnowledgeBaseCatalog,
    selection: Selection,
    retrieval: RetrievalTuning,
    deployment: DeploymentConfig,
) -> RetrievalMatrix:
    """Generate a deterministic retrieval matrix.

    Rows are sorted by question ID, KB catalog ID, mode, candidate count, result count.
    Applies merge/no-chunk exclusion and makes exclusions visible.
    """
    sel_questions = select_questions(questions, selection.questions)
    sel_kbs = select_knowledge_bases(knowledge_bases, selection.knowledge_bases)

    matrix = RetrievalMatrix()
    matrix.selected_question_ids = sel_questions.selected_ids
    matrix.selected_kb_ids = sel_kbs.selected_ids

    for q_id in sel_questions.selected_ids:
        q = next(q for q in questions.questions if q.id == q_id)
        for kb_id in sel_kbs.selected_ids:
            kb = next(kb for kb in knowledge_bases.knowledge_bases if kb.id == kb_id)

            excluded_merge = deployment.merge.enabled and not kb.supports_merge_eval

            for mode in sorted(retrieval.modes):
                for cc in sorted(retrieval.candidate_counts):
                    for rc in sorted(retrieval.result_counts):
                        if rc > cc:
                            continue

                        exp_id = f"{q.id}-{kb.id}-{mode}-c{cc}-r{rc}-{deployment.label}"

                        identity = retrieval_identity(
                            question_id=q.id,
                            question_text=q.question,
                            kb_catalog_id=kb.id,
                            kb_actual_id=kb.knowledge_base_id,
                            mode=mode,
                            candidate_count=cc,
                            result_count=rc,
                            deployment_label=deployment.label,
                            merge_enabled=deployment.merge.enabled,
                        )

                        row = RetrievalRow(
                            question_id=q.id,
                            question_text=q.question,
                            kb_catalog_id=kb.id,
                            kb_actual_id=kb.knowledge_base_id,
                            mode=mode,
                            candidate_count=cc,
                            result_count=rc,
                            deployment_label=deployment.label,
                            merge_enabled=deployment.merge.enabled,
                            supports_merge_eval=kb.supports_merge_eval,
                            experiment_id=exp_id,
                            artifact_id=artifact_id(identity),
                            identity=identity,
                        )

                        if excluded_merge:
                            matrix.excluded.append(row)
                        else:
                            matrix.rows.append(row)

    matrix.total_calls = len(matrix.rows)
    return matrix


ASK_SCHEMA_VERSION = "ask-v1"


def generate_ask_matrix(
    questions: QuestionCatalog,
    models: ModelCatalog,
    selection: Selection,
    prompt_variants: list[PromptVariant],
    inference_variants: list[InferenceVariant],
    source_dataset_type: str,
    source_dataset_id: str,
    model_selection_attr: str = "judge_models",
    stage_label: str = "judge",
    source_artifacts: list[SourceArtifact] | None = None,
    prompt_contents: dict[str, str] | None = None,
) -> AskMatrix:
    """Generate a deterministic ask-stage matrix.

    Rows are sorted by question ID, model ID, prompt variant, inference variant.
    Requires named prompt and inference variants.
    """
    sel_questions = select_questions(questions, selection.questions)
    model_selection = getattr(selection, model_selection_attr, None)
    sel_models = select_models(models, model_selection, f"{stage_label} model")

    matrix = AskMatrix()
    matrix.selected_question_ids = sel_questions.selected_ids
    matrix.selected_model_ids = sel_models.selected_ids
    matrix.prompt_variant_ids = [pv.id for pv in prompt_variants]
    matrix.inference_variant_ids = [iv.id for iv in inference_variants]
    matrix.source_dataset_type = source_dataset_type
    matrix.source_dataset_id = source_dataset_id

    question_map = {q.id: q for q in questions.questions}
    sources = source_artifacts or [
        SourceArtifact("", "", "", q_id, question_map[q_id].question, None)
        for q_id in sel_questions.selected_ids
    ]
    selected = set(sel_questions.selected_ids)
    for source_artifact in sources:
        q_id = source_artifact.question_id
        if q_id not in selected:
            continue
        for m_id in sel_models.selected_ids:
            m = next(m for m in models.models if m.id == m_id)
            for pv in prompt_variants:
                for iv in inference_variants:
                    exp_id = f"{q_id}-{m_id}-{pv.id}-{iv.id}"

                    question = question_map[q_id]
                    rubric = question.rubric.model_dump() if question.rubric else None
                    contents = prompt_contents or {}
                    system = contents.get(
                        pv.template.system_instructions, pv.template.system_instructions
                    )
                    user = contents.get(pv.template.user_message, pv.template.user_message)
                    inference = iv.inference_config.model_dump(exclude_none=True)
                    identity = ask_identity(
                        stage=stage_label,
                        source_artifact_id=source_artifact.artifact_id,
                        source_content_hash=source_artifact.content_hash,
                        question_id=q_id,
                        question_text=question.question,
                        rubric_hash=identity_hash(rubric),
                        model_catalog_id=m.id,
                        model_actual_id=m.model_id,
                        prompt_variant_id=pv.id,
                        system_template_hash=identity_hash(system),
                        user_template_hash=identity_hash(user),
                        inference_variant_id=iv.id,
                        inference_config=inference,
                    )

                    row = AskRow(
                        question_id=q_id,
                        model_catalog_id=m.id,
                        model_actual_id=m.model_id,
                        prompt_variant_id=pv.id,
                        inference_variant_id=iv.id,
                        source_dataset_type=source_dataset_type,
                        source_dataset_id=source_dataset_id,
                        source_artifact_id=source_artifact.artifact_id,
                        source_content_hash=source_artifact.content_hash,
                        source_relative_path=source_artifact.relative_path,
                        question_text=question.question,
                        rubric=rubric,
                        experiment_id=exp_id,
                        artifact_id=artifact_id(identity),
                        identity=identity,
                    )
                    matrix.rows.append(row)

    matrix.total_calls = len(matrix.rows)
    return matrix


def plan_retrieval_definition(
    definition: RetrievalDefinition,
    questions: QuestionCatalog,
    knowledge_bases: KnowledgeBaseCatalog,
) -> RetrievalMatrix:
    """Generate a retrieval matrix from a validated definition and resolved catalogs."""
    return generate_retrieval_matrix(
        questions=questions,
        knowledge_bases=knowledge_bases,
        selection=definition.selection,
        retrieval=definition.retrieval,
        deployment=definition.deployment,
    )


def plan_ask_definition(
    definition: RetrievalJudgmentDefinition | AnswerDefinition | AnswerJudgmentDefinition,
    questions: QuestionCatalog,
    models: ModelCatalog,
) -> AskMatrix:
    """Generate an ask-stage matrix from a validated definition and resolved catalogs."""
    source = definition.source_dataset
    stage_label = definition.test.type

    if isinstance(definition, RetrievalJudgmentDefinition):
        model_attr = "judge_models"
    elif isinstance(definition, AnswerDefinition):
        model_attr = "answer_models"
    elif isinstance(definition, AnswerJudgmentDefinition):
        model_attr = "judge_models"
    else:
        raise PlanError(f"Unsupported definition type: {type(definition).__name__}")

    return generate_ask_matrix(
        questions=questions,
        models=models,
        selection=definition.selection,
        prompt_variants=definition.prompt_variants,
        inference_variants=definition.inference_variants,
        source_dataset_type=source.type,
        source_dataset_id=source.id,
        model_selection_attr=model_attr,
        stage_label=stage_label,
    )
