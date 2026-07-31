"""Tests for report-time judge score parsing and enriched report columns."""

import csv
import io

import pytest

from rag_evals.errors import ConfigError
from rag_evals.reporting.reports import JoinedRow, apply_scoring, render_csv, render_markdown
from rag_evals.reporting.scoring import (
    PARSE_FAILED,
    PARSE_MISSING_TEXT,
    PARSE_NOT_CONFIGURED,
    PARSE_OK,
    ScoringSpec,
    extract_json_object,
    load_scoring_specs,
    parse_judgment,
)

SPEC = ScoringSpec(criteria=["safety", "relevance"], comment_field="comments")


class TestScoringSpec:
    def test_rejects_unknown_parser(self) -> None:
        with pytest.raises(ValueError, match="Unsupported scoring parser"):
            ScoringSpec(parser="yaml", criteria=["a"])

    def test_rejects_duplicate_criteria(self) -> None:
        with pytest.raises(ValueError, match="Duplicate scoring criteria"):
            ScoringSpec(criteria=["a", "a"])

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="must be less than"):
            ScoringSpec(criteria=["a"], minimum=5, maximum=1)

    def test_rejects_comment_field_colliding_with_criterion(self) -> None:
        with pytest.raises(ValueError, match="must not also be"):
            ScoringSpec(criteria=["a"], comment_field="a")

    def test_load_specs_rejects_unknown_stage(self) -> None:
        with pytest.raises(ConfigError, match="Unknown scoring stages"):
            load_scoring_specs({"retrieval": {"criteria": ["a"]}})

    def test_load_specs_returns_empty_when_absent(self) -> None:
        assert load_scoring_specs(None) == {}

    def test_load_specs_builds_stage_specs(self) -> None:
        specs = load_scoring_specs({"retrieval_judgment": {"criteria": ["safety"]}})
        assert specs["retrieval_judgment"].criteria == ["safety"]


class TestExtractJsonObject:
    def test_plain_object(self) -> None:
        assert extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_fenced_block_with_prose(self) -> None:
        text = 'Here is my assessment.\n```json\n{"a": 1}\n```\nDone.'
        assert extract_json_object(text) == '{"a": 1}'

    def test_nested_object(self) -> None:
        assert extract_json_object('x {"a": {"b": 2}} y') == '{"a": {"b": 2}}'

    def test_ignores_braces_inside_strings(self) -> None:
        assert extract_json_object('{"a": "not } an end"}') == '{"a": "not } an end"}'

    def test_no_object(self) -> None:
        assert extract_json_object("no json here") is None


class TestParseJudgment:
    def test_parses_scores_and_computes_total(self) -> None:
        result = parse_judgment('{"safety": 5, "relevance": 3, "comments": "ok"}', SPEC)
        assert result.status == PARSE_OK
        assert result.scores == {"safety": 5, "relevance": 3}
        assert result.total == 8
        assert result.comment == "ok"

    def test_ignores_judge_reported_total(self) -> None:
        result = parse_judgment('{"safety": 5, "relevance": 3, "total": 99}', SPEC)
        assert result.status == PARSE_OK
        assert result.total == 8
        assert result.extra_keys == ["total"]

    def test_missing_criterion_fails_without_inventing(self) -> None:
        result = parse_judgment('{"safety": 5}', SPEC)
        assert result.status == PARSE_FAILED
        assert "missing criterion 'relevance'" in result.error
        assert result.scores == {}
        assert result.total is None

    def test_out_of_range_fails(self) -> None:
        result = parse_judgment('{"safety": 9, "relevance": 3}', SPEC)
        assert result.status == PARSE_FAILED
        assert "out of range 1-5" in result.error

    def test_non_integer_fails(self) -> None:
        result = parse_judgment('{"safety": "high", "relevance": 3}', SPEC)
        assert result.status == PARSE_FAILED
        assert "not an integer" in result.error

    def test_boolean_is_not_an_integer(self) -> None:
        result = parse_judgment('{"safety": true, "relevance": 3}', SPEC)
        assert result.status == PARSE_FAILED

    def test_integral_float_accepted(self) -> None:
        result = parse_judgment('{"safety": 4.0, "relevance": 3}', SPEC)
        assert result.status == PARSE_OK
        assert result.scores["safety"] == 4

    def test_prose_only_fails(self) -> None:
        result = parse_judgment("The context was quite good.", SPEC)
        assert result.status == PARSE_FAILED
        assert "No JSON object" in result.error

    def test_malformed_json_fails(self) -> None:
        result = parse_judgment('{"safety": 5, "relevance": }', SPEC)
        assert result.status == PARSE_FAILED
        assert "Invalid JSON" in result.error

    def test_empty_text(self) -> None:
        assert parse_judgment("   ", SPEC).status == PARSE_MISSING_TEXT


def make_row(judge_text: str = "", judged: bool = True) -> JoinedRow:
    return JoinedRow(
        question_id="q001",
        question_text="Q?",
        retrieval_artifact_id="sha256:aaa",
        kb_catalog_id="kb_fixed_500",
        kb_actual_id="AV0IEYJYTF",
        chunking_strategy="fixed",
        chunking_size=500,
        chunking_overlap=20,
        retrieval_mode="standard",
        effective_mode="standard",
        estimated_context_tokens=2206,
        mean_score=0.76,
        item_count=5,
        duplicates_removed=12,
        retrieval_judgment_artifact_id="sha256:bbb" if judged else "",
        retrieval_judgment_text=judge_text,
    )


class TestApplyScoring:
    def test_scores_applied_to_rows(self) -> None:
        rows = [make_row('{"safety": 5, "relevance": 4}')]
        apply_scoring(rows, {"retrieval_judgment": SPEC})
        assert rows[0].retrieval_parse_status == PARSE_OK
        assert rows[0].retrieval_score_total == 9

    def test_unjudged_row_stays_not_configured(self) -> None:
        rows = [make_row(judged=False)]
        apply_scoring(rows, {"retrieval_judgment": SPEC})
        assert rows[0].retrieval_parse_status == PARSE_NOT_CONFIGURED

    def test_no_spec_leaves_rows_untouched(self) -> None:
        rows = [make_row('{"safety": 5, "relevance": 4}')]
        apply_scoring(rows, {})
        assert rows[0].retrieval_parse_status == PARSE_NOT_CONFIGURED
        assert rows[0].retrieval_scores == {}

    def test_raw_text_is_never_modified(self) -> None:
        text = 'prose {"safety": 5, "relevance": 4} more'
        rows = [make_row(text)]
        apply_scoring(rows, {"retrieval_judgment": SPEC})
        assert rows[0].retrieval_judgment_text == text


class TestReportOutput:
    def test_csv_has_score_and_metadata_columns(self) -> None:
        rows = [make_row('{"safety": 5, "relevance": 4, "comments": "fine"}')]
        specs = {"retrieval_judgment": SPEC}
        apply_scoring(rows, specs)
        parsed = list(csv.reader(io.StringIO(render_csv(rows, specs))))
        header, data = parsed[0], parsed[1]
        record = dict(zip(header, data, strict=True))
        assert record["retrieval_score_safety"] == "5"
        assert record["retrieval_score_relevance"] == "4"
        assert record["retrieval_score_total"] == "9"
        assert record["retrieval_parse_status"] == PARSE_OK
        assert record["chunking_strategy"] == "fixed"
        assert record["chunking_size"] == "500"
        assert record["effective_mode"] == "standard"
        assert record["duplicates_removed"] == "12"
        assert record["mean_score"] == "0.76"

    def test_csv_omits_score_columns_without_spec(self) -> None:
        header = list(csv.reader(io.StringIO(render_csv([make_row()]))))[0]
        assert not [name for name in header if name.startswith("retrieval_score_")]

    def test_csv_does_not_truncate_text(self) -> None:
        long_text = "x" * 900
        row = make_row(long_text)
        row.question_text = long_text
        output = render_csv([row], {"retrieval_judgment": SPEC})
        record = dict(zip(*list(csv.reader(io.StringIO(output)))[:2], strict=True))
        assert record["question_text"] == long_text
        assert record["retrieval_judgment_text"] == long_text

    def test_markdown_summary_reports_means_and_failures(self) -> None:
        rows = [
            make_row('{"safety": 5, "relevance": 3}'),
            make_row('{"safety": 3, "relevance": 1}'),
            make_row("unparseable"),
        ]
        specs = {"retrieval_judgment": SPEC}
        apply_scoring(rows, specs)
        text = render_markdown(rows, specs)
        assert "## Configuration Summary" in text
        assert "Mean safety" in text
        assert "| kb_fixed_500 | fixed | standard | no | 3 | 4.00 | 2.00 | 6.00 |" in text

    def test_markdown_summary_absent_without_retrieval_rows(self) -> None:
        assert "## Configuration Summary" not in render_markdown([])


SOURCE_SPEC = ScoringSpec(
    criteria=["safety", "relevance"],
    comment_field="comments",
    source_assessment_field="source_assessments",
)


def judgment_json(assessments: str) -> str:
    return '{"safety": 4, "relevance": 4, "source_assessments": ' + assessments + "}"


class TestSourceAssessments:
    def test_spec_rejects_collision_with_criterion(self) -> None:
        with pytest.raises(ValueError, match="must not also be a scoring criterion"):
            ScoringSpec(criteria=["a"], source_assessment_field="a")

    def test_spec_rejects_collision_with_comment_field(self) -> None:
        with pytest.raises(ValueError, match="must differ from"):
            ScoringSpec(criteria=["a"], comment_field="c", source_assessment_field="c")

    def test_counts_and_ratio_computed_in_code(self) -> None:
        text = judgment_json(
            '[{"index": 1, "relevance": "relevant"},'
            ' {"index": 2, "relevance": "relevant"},'
            ' {"index": 3, "relevance": "partial"},'
            ' {"index": 4, "relevance": "irrelevant"},'
            ' {"index": 5, "relevance": "irrelevant"}]'
        )
        result = parse_judgment(text, SOURCE_SPEC)
        assert result.status == PARSE_OK
        s = result.sources
        assert s.status == PARSE_OK
        assert (s.source_count, s.relevant, s.partial, s.irrelevant) == (5, 2, 1, 2)
        # (2 * 1.0 + 1 * 0.5 + 2 * 0.0) / 5
        assert s.ratio == 0.5

    def test_relevance_values_are_case_insensitive(self) -> None:
        text = judgment_json('[{"index": 1, "relevance": "  RELEVANT "}]')
        assert parse_judgment(text, SOURCE_SPEC).sources.ratio == 1.0

    def test_missing_enumeration_does_not_fail_scores(self) -> None:
        result = parse_judgment('{"safety": 4, "relevance": 4}', SOURCE_SPEC)
        assert result.status == PARSE_OK
        assert result.scores == {"safety": 4, "relevance": 4}
        assert result.sources.status == PARSE_FAILED
        assert "missing 'source_assessments'" in result.sources.error
        assert result.sources.ratio is None

    def test_unknown_relevance_value_reported(self) -> None:
        text = judgment_json('[{"index": 1, "relevance": "sort of"}]')
        sources = parse_judgment(text, SOURCE_SPEC).sources
        assert sources.status == PARSE_FAILED
        assert "relevance must be one of" in sources.error
        assert sources.ratio is None

    def test_duplicate_index_reported(self) -> None:
        text = judgment_json(
            '[{"index": 1, "relevance": "relevant"}, {"index": 1, "relevance": "irrelevant"}]'
        )
        sources = parse_judgment(text, SOURCE_SPEC).sources
        assert sources.status == PARSE_FAILED
        assert "duplicate source index 1" in sources.error

    def test_empty_list_reported(self) -> None:
        sources = parse_judgment(judgment_json("[]"), SOURCE_SPEC).sources
        assert sources.status == PARSE_FAILED
        assert "non-empty list" in sources.error

    def test_enumeration_is_not_an_extra_key(self) -> None:
        text = judgment_json('[{"index": 1, "relevance": "relevant"}]')
        assert parse_judgment(text, SOURCE_SPEC).extra_keys == []

    def test_not_configured_when_field_absent_from_spec(self) -> None:
        result = parse_judgment('{"safety": 4, "relevance": 4}', SPEC)
        assert result.sources.status == PARSE_NOT_CONFIGURED
        assert result.sources.error == ""

    def test_csv_includes_computed_source_columns(self) -> None:
        text = judgment_json(
            '[{"index": 1, "relevance": "relevant"}, {"index": 2, "relevance": "irrelevant"}]'
        )
        rows = [make_row(text)]
        specs = {"retrieval_judgment": SOURCE_SPEC}
        apply_scoring(rows, specs)
        record = dict(zip(*list(csv.reader(io.StringIO(render_csv(rows, specs))))[:2], strict=True))
        assert record["judged_source_count"] == "2"
        assert record["judged_sources_relevant"] == "1"
        assert record["judged_sources_irrelevant"] == "1"
        assert record["judged_relevance_ratio"] == "0.5"
        assert record["judged_sources_status"] == PARSE_OK

    def test_csv_omits_source_columns_without_field(self) -> None:
        rows = [make_row('{"safety": 4, "relevance": 4}')]
        specs = {"retrieval_judgment": SPEC}
        apply_scoring(rows, specs)
        header = list(csv.reader(io.StringIO(render_csv(rows, specs))))[0]
        assert not [name for name in header if name.startswith("judged_")]

    def test_markdown_summary_includes_ratio(self) -> None:
        text = judgment_json('[{"index": 1, "relevance": "relevant"}]')
        rows = [make_row(text)]
        specs = {"retrieval_judgment": SOURCE_SPEC}
        apply_scoring(rows, specs)
        assert "Mean relevant-source ratio" in render_markdown(rows, specs)
