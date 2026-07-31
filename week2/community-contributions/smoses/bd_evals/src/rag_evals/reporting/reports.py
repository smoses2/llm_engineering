"""Lineage join model and CSV/Markdown report renderers (P10).

Joins question -> retrieval -> retrieval_judgment -> answer -> answer_judgment
by IDs and verified hashes. Produces deterministic CSV and Markdown reports.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rag_evals.artifacts.io import atomic_write_text, read_json
from rag_evals.artifacts.store import DatasetPath
from rag_evals.errors import ArtifactError
from rag_evals.logging import get_logger
from rag_evals.planning.identity import canonical_hash
from rag_evals.reporting.scoring import (
    PARSE_FAILED,
    PARSE_NOT_CONFIGURED,
    ScoringSpec,
    parse_judgment,
)

logger = get_logger("rag_evals.reporting")


@dataclass
class JoinedRow:
    """A single joined row in the lineage report."""

    question_id: str
    question_text: str = ""
    # Retrieval
    retrieval_artifact_id: str = ""
    kb_catalog_id: str = ""
    kb_actual_id: str = ""
    chunking_strategy: str = ""
    chunking_size: int | None = None
    chunking_overlap: int | None = None
    chunking_similarity_percentile_threshold: int | None = None
    deployment_label: str = ""
    merge_enabled: bool = False
    retrieval_mode: str = ""
    effective_mode: str = ""
    candidate_count: int | None = None
    result_count: int | None = None
    item_count: int | None = None
    duplicates_removed: int | None = None
    max_score: float | None = None
    mean_score: float | None = None
    min_score: float | None = None
    estimated_context_tokens: int | None = None
    retrieval_cost: dict[str, Any] = field(default_factory=dict)
    retrieval_fallback: bool = False
    # Retrieval judgment
    retrieval_judgment_text: str = ""
    retrieval_judge_model: str = ""
    retrieval_judgment_artifact_id: str = ""
    retrieval_judge_cost: dict[str, Any] = field(default_factory=dict)
    retrieval_scores: dict[str, int] = field(default_factory=dict)
    retrieval_score_total: int | None = None
    retrieval_judge_comment: str = ""
    retrieval_parse_status: str = PARSE_NOT_CONFIGURED
    retrieval_parse_error: str = ""
    judged_source_count: int | None = None
    judged_sources_relevant: int | None = None
    judged_sources_partial: int | None = None
    judged_sources_irrelevant: int | None = None
    judged_relevance_ratio: float | None = None
    judged_sources_status: str = PARSE_NOT_CONFIGURED
    judged_sources_error: str = ""
    # Answer
    answer_artifact_id: str = ""
    answer_text: str = ""
    answer_model: str = ""
    answer_input_tokens: int | None = None
    answer_output_tokens: int | None = None
    answer_total_tokens: int | None = None
    answer_latency_ms: int | None = None
    answer_cost: dict[str, Any] = field(default_factory=dict)
    answer_prompt_variant: str = ""
    answer_inference_variant: str = ""
    # Answer judgment
    answer_judgment_text: str = ""
    answer_judge_model: str = ""
    answer_judgment_artifact_id: str = ""
    answer_judge_cost: dict[str, Any] = field(default_factory=dict)
    answer_scores: dict[str, int] = field(default_factory=dict)
    answer_score_total: int | None = None
    answer_judge_comment: str = ""
    answer_parse_status: str = PARSE_NOT_CONFIGURED
    answer_parse_error: str = ""
    # Status
    status: str = "complete"


def load_dataset_artifacts(paths: DatasetPath) -> list[dict[str, Any]]:
    """Load all artifacts from a dataset."""
    artifacts = []
    for f in sorted(paths.artifacts_dir.glob("*.json")):
        try:
            artifacts.append(read_json(f))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", f, exc)
    return artifacts


def join_lineage(
    project_root: Path,
    retrieval_dataset_id: str | None = None,
    retrieval_judgment_dataset_id: str | None = None,
    answer_dataset_id: str | None = None,
    answer_judgment_dataset_id: str | None = None,
) -> list[JoinedRow]:
    """Join artifacts across datasets by question_id and verified hashes."""
    rows: list[JoinedRow] = []
    retrieval_sources: dict[str, dict[str, Any]] = {}
    answer_sources: dict[str, dict[str, Any]] = {}

    # Load retrieval artifacts
    if retrieval_dataset_id:
        paths = DatasetPath(project_root, "retrieval", retrieval_dataset_id)
        if paths.base.exists():
            for art in load_dataset_artifacts(paths):
                artifact_id = art.get("artifact_id", "")
                if not artifact_id:
                    continue
                retrieval_sources[artifact_id] = art
                metrics = art.get("objective_metrics", {})
                metadata = art.get("retrieval_metadata", {})
                chunking = art.get("kb_metadata", {}).get("chunking", {})
                dedup = metadata.get("deduplication") or {}
                rows.append(
                    JoinedRow(
                        question_id=art.get("question_id", ""),
                        question_text=art.get("question_text", ""),
                        retrieval_artifact_id=artifact_id,
                        kb_catalog_id=art.get("kb_catalog_id", ""),
                        kb_actual_id=art.get("kb_actual_id", ""),
                        chunking_strategy=chunking.get("strategy", ""),
                        chunking_size=chunking.get("size"),
                        chunking_overlap=chunking.get("overlap"),
                        chunking_similarity_percentile_threshold=chunking.get(
                            "similarity_percentile_threshold"
                        ),
                        deployment_label=art.get("deployment_label", ""),
                        merge_enabled=bool(art.get("merge_enabled", False)),
                        retrieval_mode=art.get("mode", ""),
                        effective_mode=metadata.get("effective_mode", ""),
                        candidate_count=art.get("candidate_count"),
                        result_count=art.get("result_count"),
                        item_count=metrics.get("item_count"),
                        duplicates_removed=dedup.get("duplicatesRemoved"),
                        max_score=metrics.get("max_score"),
                        mean_score=metrics.get("mean_score"),
                        min_score=metrics.get("min_score"),
                        estimated_context_tokens=art.get("estimated_context_tokens"),
                        retrieval_cost=art.get("cost", {}),
                        retrieval_fallback=bool(
                            metadata.get("fallback_used", metrics.get("has_fallback", False))
                        ),
                        status="retrieval_only",
                    )
                )

    # Load retrieval judgments
    if retrieval_judgment_dataset_id:
        paths = DatasetPath(project_root, "retrieval_judgments", retrieval_judgment_dataset_id)
        if paths.base.exists():
            judgments = _by_source(load_dataset_artifacts(paths), retrieval_sources)
            expanded: list[JoinedRow] = []
            for row in rows:
                matches = judgments.get(row.retrieval_artifact_id, [])
                expanded.extend(
                    [
                        replace(
                            row,
                            retrieval_judgment_artifact_id=art.get("artifact_id", ""),
                            retrieval_judgment_text=art.get("raw_judge_text", ""),
                            retrieval_judge_model=art.get("model_actual_id", ""),
                            retrieval_judge_cost=art.get("cost", {}),
                        )
                        for art in matches
                    ]
                    or [row]
                )
            rows = expanded

    # Load answers
    if answer_dataset_id:
        paths = DatasetPath(project_root, "answers", answer_dataset_id)
        if paths.base.exists():
            answer_artifacts = load_dataset_artifacts(paths)
            answers = _by_source(answer_artifacts, retrieval_sources)
            answer_sources = {
                artifact["artifact_id"]: artifact
                for artifact in answer_artifacts
                if artifact.get("artifact_id")
            }
            expanded = []
            for row in rows:
                matches = answers.get(row.retrieval_artifact_id, [])
                for art in matches:
                    usage = art.get("usage", {})
                    metrics = art.get("metrics", {})
                    expanded.append(
                        replace(
                            row,
                            answer_artifact_id=art.get("artifact_id", ""),
                            answer_text=art.get("answer_text", ""),
                            answer_model=art.get("model_actual_id", ""),
                            answer_prompt_variant=art.get("prompt_variant_id", ""),
                            answer_inference_variant=art.get("inference_variant_id", ""),
                            answer_input_tokens=usage.get("inputTokens"),
                            answer_output_tokens=usage.get("outputTokens"),
                            answer_total_tokens=usage.get("totalTokens"),
                            answer_latency_ms=metrics.get("latencyMs"),
                            answer_cost=art.get("cost", {}),
                            status="answer_missing_judgment",
                        )
                    )
                if not matches:
                    expanded.append(row)
            rows = expanded

    # Load answer judgments
    if answer_judgment_dataset_id:
        paths = DatasetPath(project_root, "answer_judgments", answer_judgment_dataset_id)
        if paths.base.exists():
            judgments = _by_source(load_dataset_artifacts(paths), answer_sources)
            expanded = []
            for row in rows:
                matches = judgments.get(row.answer_artifact_id, [])
                expanded.extend(
                    [
                        replace(
                            row,
                            answer_judgment_artifact_id=art.get("artifact_id", ""),
                            answer_judgment_text=art.get("raw_judge_text", ""),
                            answer_judge_model=art.get("model_actual_id", ""),
                            answer_judge_cost=art.get("cost", {}),
                            status="complete",
                        )
                        for art in matches
                    ]
                    or [row]
                )
            rows = expanded

    return sorted(
        rows,
        key=lambda row: (
            row.retrieval_artifact_id,
            row.retrieval_judgment_artifact_id,
            row.answer_artifact_id,
            row.answer_judgment_artifact_id,
        ),
    )


def _by_source(
    artifacts: list[dict[str, Any]], sources: dict[str, dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for artifact in artifacts:
        source_id = artifact.get("source_artifact_id")
        if source_id:
            source = sources.get(source_id)
            if source is None:
                raise ArtifactError(f"Report lineage source is missing: {source_id}")
            expected_hash = artifact.get("source_content_hash") or artifact.get("lineage", {}).get(
                "expected_content_hash"
            )
            if expected_hash != canonical_hash(source):
                raise ArtifactError(f"Report lineage hash mismatch for source: {source_id}")
            result.setdefault(source_id, []).append(artifact)
    return result


def apply_scoring(rows: list[JoinedRow], specs: dict[str, ScoringSpec]) -> None:
    """Derive structured scores from raw judge text, in place.

    Raw text is never modified. Rows without judge text keep the not-configured
    status so a missing judgment is distinguishable from a parse failure.
    """
    retrieval_spec = specs.get("retrieval_judgment")
    answer_spec = specs.get("answer_judgment")
    for row in rows:
        if retrieval_spec is not None and row.retrieval_judgment_artifact_id:
            parsed = parse_judgment(row.retrieval_judgment_text, retrieval_spec)
            row.retrieval_scores = parsed.scores
            row.retrieval_score_total = parsed.total
            row.retrieval_judge_comment = parsed.comment
            row.retrieval_parse_status = parsed.status
            row.retrieval_parse_error = parsed.error
            row.judged_source_count = parsed.sources.source_count
            row.judged_sources_relevant = parsed.sources.relevant
            row.judged_sources_partial = parsed.sources.partial
            row.judged_sources_irrelevant = parsed.sources.irrelevant
            row.judged_relevance_ratio = parsed.sources.ratio
            row.judged_sources_status = parsed.sources.status
            row.judged_sources_error = parsed.sources.error
        if answer_spec is not None and row.answer_judgment_artifact_id:
            parsed = parse_judgment(row.answer_judgment_text, answer_spec)
            row.answer_scores = parsed.scores
            row.answer_score_total = parsed.total
            row.answer_judge_comment = parsed.comment
            row.answer_parse_status = parsed.status
            row.answer_parse_error = parsed.error


def render_csv(rows: list[JoinedRow], specs: dict[str, ScoringSpec] | None = None) -> str:
    """Render joined rows as deterministic CSV.

    Text columns are not truncated: the CSV is the analysis source of truth and is
    intended for a spreadsheet pivot.
    """
    specs = specs or {}
    retrieval_criteria = (
        list(specs["retrieval_judgment"].criteria) if "retrieval_judgment" in specs else []
    )
    answer_criteria = list(specs["answer_judgment"].criteria) if "answer_judgment" in specs else []
    source_columns = bool(
        "retrieval_judgment" in specs and specs["retrieval_judgment"].source_assessment_field
    )
    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "question_id",
        "question_text",
        "kb_catalog_id",
        "kb_actual_id",
        "chunking_strategy",
        "chunking_size",
        "chunking_overlap",
        "chunking_similarity_percentile_threshold",
        "deployment_label",
        "merge_enabled",
        "retrieval_mode",
        "effective_mode",
        "retrieval_fallback",
        "candidate_count",
        "result_count",
        "item_count",
        "duplicates_removed",
        "max_score",
        "mean_score",
        "min_score",
        "estimated_context_tokens",
        "retrieval_judge_model",
        *[f"retrieval_score_{name}" for name in retrieval_criteria],
        *(["retrieval_score_total"] if retrieval_criteria else []),
        *(
            [
                "judged_source_count",
                "judged_sources_relevant",
                "judged_sources_partial",
                "judged_sources_irrelevant",
                "judged_relevance_ratio",
                "judged_sources_status",
                "judged_sources_error",
            ]
            if source_columns
            else []
        ),
        *(
            ["retrieval_judge_comment", "retrieval_parse_status", "retrieval_parse_error"]
            if retrieval_criteria
            else []
        ),
        "retrieval_judgment_text",
        "answer_model",
        "answer_text",
        "answer_input_tokens",
        "answer_output_tokens",
        "answer_total_tokens",
        "answer_latency_ms",
        "answer_judge_model",
        *[f"answer_score_{name}" for name in answer_criteria],
        *(["answer_score_total"] if answer_criteria else []),
        *(
            ["answer_judge_comment", "answer_parse_status", "answer_parse_error"]
            if answer_criteria
            else []
        ),
        "answer_judgment_text",
        "status",
    ]
    writer.writerow(headers)

    for row in rows:
        writer.writerow(
            [
                row.question_id,
                row.question_text,
                row.kb_catalog_id,
                row.kb_actual_id,
                row.chunking_strategy,
                row.chunking_size,
                row.chunking_overlap,
                row.chunking_similarity_percentile_threshold,
                row.deployment_label,
                row.merge_enabled,
                row.retrieval_mode,
                row.effective_mode,
                row.retrieval_fallback,
                row.candidate_count,
                row.result_count,
                row.item_count,
                row.duplicates_removed,
                row.max_score,
                row.mean_score,
                row.min_score,
                row.estimated_context_tokens,
                row.retrieval_judge_model,
                *[row.retrieval_scores.get(name, "") for name in retrieval_criteria],
                *(
                    [row.retrieval_score_total if row.retrieval_score_total is not None else ""]
                    if retrieval_criteria
                    else []
                ),
                *(
                    [
                        row.judged_source_count if row.judged_source_count is not None else "",
                        row.judged_sources_relevant
                        if row.judged_sources_relevant is not None
                        else "",
                        row.judged_sources_partial
                        if row.judged_sources_partial is not None
                        else "",
                        row.judged_sources_irrelevant
                        if row.judged_sources_irrelevant is not None
                        else "",
                        row.judged_relevance_ratio
                        if row.judged_relevance_ratio is not None
                        else "",
                        row.judged_sources_status,
                        row.judged_sources_error,
                    ]
                    if source_columns
                    else []
                ),
                *(
                    [
                        row.retrieval_judge_comment,
                        row.retrieval_parse_status,
                        row.retrieval_parse_error,
                    ]
                    if retrieval_criteria
                    else []
                ),
                row.retrieval_judgment_text,
                row.answer_model,
                row.answer_text,
                row.answer_input_tokens,
                row.answer_output_tokens,
                row.answer_total_tokens,
                row.answer_latency_ms,
                row.answer_judge_model,
                *[row.answer_scores.get(name, "") for name in answer_criteria],
                *(
                    [row.answer_score_total if row.answer_score_total is not None else ""]
                    if answer_criteria
                    else []
                ),
                *(
                    [row.answer_judge_comment, row.answer_parse_status, row.answer_parse_error]
                    if answer_criteria
                    else []
                ),
                row.answer_judgment_text,
                row.status,
            ]
        )

    return output.getvalue()


def render_markdown(rows: list[JoinedRow], specs: dict[str, ScoringSpec] | None = None) -> str:
    """Render joined rows as deterministic Markdown."""
    specs = specs or {}
    lines = [
        "# Evaluation Report",
        "",
        f"Total questions: {len(rows)}",
        "",
        "| Question | KB | Mode | Est. Tokens | Answer Model | "
        "Input Tokens | Output Tokens | Retrieval Judge | Answer Judge |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {_markdown_cell(row.question_id)} "
            f"| {_markdown_cell(row.kb_catalog_id)} "
            f"| {_markdown_cell(row.retrieval_mode)} "
            f"| {row.estimated_context_tokens or '-'} "
            f"| {_markdown_cell(row.answer_model or '-')} "
            f"| {row.answer_input_tokens or '-'} "
            f"| {row.answer_output_tokens or '-'} "
            f"| {_markdown_cell(row.retrieval_judge_model or '-')} "
            f"| {_markdown_cell(row.answer_judge_model or '-')} |"
        )

    lines.append("")
    lines.extend(_render_configuration_summary(rows, specs))
    lines.append("")
    lines.append("## Cost Summary")

    total_retrieval = sum(cost_dollars(r.retrieval_cost) for r in rows)
    total_answer = sum(cost_dollars(r.answer_cost) for r in rows)
    retrieval_judges = sum(cost_dollars(r.retrieval_judge_cost) for r in rows)
    answer_judges = sum(cost_dollars(r.answer_judge_cost) for r in rows)

    lines.append(f"- Retrieval cost (tested system): {total_retrieval:.4f}")
    lines.append(f"- Answer inference cost (tested system): {total_answer:.4f}")
    lines.append(f"- Total tested-system cost: {total_retrieval + total_answer:.4f}")
    lines.append(f"- Retrieval-judge cost (evaluation): {retrieval_judges:.4f}")
    lines.append(f"- Answer-judge cost (evaluation): {answer_judges:.4f}")
    lines.append("")
    lines.append("*Judge costs are evaluation infrastructure, not tested-system cost.*")

    return "\n".join(lines)


def _mean(values: list[float]) -> str:
    """Format a mean, or '-' when there is nothing to average."""
    return f"{sum(values) / len(values):.2f}" if values else "-"


def _render_configuration_summary(
    rows: list[JoinedRow], specs: dict[str, ScoringSpec]
) -> list[str]:
    """Aggregate per knowledge base and retrieval mode.

    Shows trade-offs only. It deliberately does not rank configurations or name a
    winner: that decision belongs to the human reading the evidence.
    """
    criteria = list(specs["retrieval_judgment"].criteria) if "retrieval_judgment" in specs else []
    groups: dict[tuple[str, str, str, bool], list[JoinedRow]] = {}
    for row in rows:
        if not row.kb_catalog_id:
            continue
        key = (row.kb_catalog_id, row.chunking_strategy, row.retrieval_mode, row.merge_enabled)
        groups.setdefault(key, []).append(row)
    if not groups:
        return []

    header = (
        ["KB", "Chunking", "Mode", "Merge", "Rows"]
        + [f"Mean {name}" for name in criteria]
        + (["Mean total"] if criteria else [])
        + ["Mean relevant-source ratio", "Mean est. tokens", "Mean rel. score"]
        + ["Fallbacks", "Parse failures"]
    )
    lines = [
        "## Configuration Summary",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for key in sorted(groups):
        kb, chunking, mode, merge = key
        group = groups[key]
        cells = [
            _markdown_cell(kb),
            _markdown_cell(chunking or "-"),
            _markdown_cell(mode or "-"),
            "yes" if merge else "no",
            str(len(group)),
        ]
        for name in criteria:
            cells.append(
                _mean(
                    [float(r.retrieval_scores[name]) for r in group if name in r.retrieval_scores]
                )
            )
        if criteria:
            cells.append(
                _mean(
                    [
                        float(r.retrieval_score_total)
                        for r in group
                        if r.retrieval_score_total is not None
                    ]
                )
            )
        cells.append(
            _mean([r.judged_relevance_ratio for r in group if r.judged_relevance_ratio is not None])
        )
        cells.append(
            _mean(
                [
                    float(r.estimated_context_tokens)
                    for r in group
                    if r.estimated_context_tokens is not None
                ]
            )
        )
        cells.append(_mean([float(r.mean_score) for r in group if r.mean_score is not None]))
        cells.append(str(sum(1 for r in group if r.retrieval_fallback)))
        cells.append(str(sum(1 for r in group if r.retrieval_parse_status == PARSE_FAILED)))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "*Estimated context tokens are informational and must not be treated as a quality score. "
        "API relevance scores are not calibrated across differently chunked knowledge bases, and "
        "are not comparable across retrieval modes because standard and rerank scores come from "
        "different models. The relevant-source ratio counts partial sources as half and ignores "
        "rank order, so it under-credits reranking.*"
    )
    return lines


def cost_dollars(value: Any) -> float:
    """Normalize nested API cost sections without combining their categories."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return 0.0
    if isinstance(value.get("dollars"), (int, float)):
        return float(value["dollars"])
    if isinstance(value.get("microdollars"), (int, float)):
        return float(value["microdollars"]) / 1_000_000
    if isinstance(value.get("total"), (int, float)):
        return float(value["total"])
    return sum(cost_dollars(item) for item in value.values())


def _markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def generate_report(
    project_root: Path,
    output_dir: Path,
    retrieval_dataset_id: str | None = None,
    retrieval_judgment_dataset_id: str | None = None,
    answer_dataset_id: str | None = None,
    answer_judgment_dataset_id: str | None = None,
    output_format: str = "both",
    scoring: dict[str, ScoringSpec] | None = None,
) -> dict[str, str]:
    """Generate CSV and Markdown reports. Returns dict of file paths."""
    rows = join_lineage(
        project_root,
        retrieval_dataset_id,
        retrieval_judgment_dataset_id,
        answer_dataset_id,
        answer_judgment_dataset_id,
    )
    specs = scoring or {}
    apply_scoring(rows, specs)

    csv_path = output_dir / "report.csv"
    md_path = output_dir / "report.md"

    result = {"rows": str(len(rows))}
    if specs:
        failures = sum(
            1
            for row in rows
            if PARSE_FAILED in (row.retrieval_parse_status, row.answer_parse_status)
        )
        result["parse_failures"] = str(failures)
    if output_format in {"both", "csv", "text", "json"}:
        atomic_write_text(csv_path, render_csv(rows, specs))
        result["csv"] = str(csv_path)
    if output_format in {"both", "markdown", "text", "json"}:
        atomic_write_text(md_path, render_markdown(rows, specs))
        result["markdown"] = str(md_path)
    return result
