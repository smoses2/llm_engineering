"""Tests for strict template validation and rendering."""

import pytest

from rag_evals.errors import ConfigError
from rag_evals.templates import (
    Stage,
    extract_template_filters,
    extract_template_variables,
    render_template,
    validate_template,
)

# ---------------------------------------------------------------------------
# Variable extraction
# ---------------------------------------------------------------------------


class TestExtractVariables:
    def test_simple(self) -> None:
        vars_ = extract_template_variables("Hello {{ question }}")
        assert vars_ == {"question"}

    def test_multiple(self) -> None:
        vars_ = extract_template_variables("{{ question }} {{ rubric }}")
        assert vars_ == {"question", "rubric"}

    def test_nested_access(self) -> None:
        vars_ = extract_template_variables("{{ rubric.expected_answer }}")
        assert "rubric" in vars_

    def test_no_variables(self) -> None:
        vars_ = extract_template_variables("Just text.")
        assert vars_ == set()

    def test_syntax_error(self) -> None:
        with pytest.raises(ConfigError, match="syntax"):
            extract_template_variables("{{ question ")


class TestExtractFilters:
    def test_no_filters(self) -> None:
        assert extract_template_filters("{{ question }}") == set()

    def test_to_yaml(self) -> None:
        assert extract_template_filters("{{ rubric | to_yaml }}") == {"to_yaml"}

    def test_default(self) -> None:
        assert "default" in extract_template_filters("{{ rubric | default('') }}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateTemplate:
    def test_valid_retrieval_judge(self) -> None:
        result = validate_template(
            "Question: {{ question }}\nContext: {{ retrieval_context }}",
            Stage.RETRIEVAL_JUDGE,
        )
        assert result.valid
        assert "question" in result.variables_used
        assert "retrieval_context" in result.variables_used

    def test_valid_answer(self) -> None:
        result = validate_template(
            "Q: {{ question }}\nC: {{ retrieval_context }}",
            Stage.ANSWER,
        )
        assert result.valid

    def test_valid_answer_judge(self) -> None:
        result = validate_template(
            "Q: {{ question }}\nA: {{ candidate_answer }}",
            Stage.ANSWER_JUDGE,
        )
        assert result.valid

    def test_unknown_variable(self) -> None:
        result = validate_template("{{ unknown_var }}", Stage.ANSWER)
        assert not result.valid
        assert any("unknown_var" in e for e in result.errors)

    def test_answer_stage_rejects_retrieved_items(self) -> None:
        result = validate_template("{{ retrieved_items }}", Stage.ANSWER)
        assert not result.valid

    def test_unknown_filter(self) -> None:
        result = validate_template("{{ question | evil_filter }}", Stage.ANSWER)
        assert not result.valid
        assert any("evil_filter" in e for e in result.errors)

    def test_allowed_filter_to_yaml(self) -> None:
        result = validate_template("{{ rubric | to_yaml }}", Stage.ANSWER)
        assert result.valid

    def test_allowed_filter_default(self) -> None:
        result = validate_template("{{ rubric | default('') }}", Stage.ANSWER)
        assert result.valid

    def test_syntax_error(self) -> None:
        result = validate_template("{{ question ", Stage.ANSWER)
        assert not result.valid
        assert any("syntax" in e for e in result.errors)

    def test_statement_block_rejected(self) -> None:
        result = validate_template("{% if question %}{{ question }}{% endif %}", Stage.ANSWER)
        assert not result.valid


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_simple_render(self) -> None:
        result = render_template("Hello {{ name }}", {"name": "world"})
        assert result == "Hello world"

    def test_nested_access(self) -> None:
        result = render_template(
            "Answer: {{ rubric.expected_answer }}",
            {"rubric": {"expected_answer": "42"}},
        )
        assert result == "Answer: 42"

    def test_to_yaml_filter(self) -> None:
        result = render_template(
            "{{ rubric | to_yaml }}",
            {"rubric": {"key": "value"}},
        )
        assert "key: value" in result

    def test_to_json_filter(self) -> None:
        result = render_template(
            "{{ data | to_json }}",
            {"data": {"b": 1, "a": 2}},
        )
        assert '"a": 2' in result
        assert '"b": 1' in result

    def test_default_filter(self) -> None:
        result = render_template(
            "{{ rubric | default('none') }}",
            {"rubric": {}},
        )
        # rubric is defined (empty dict), so default doesn't apply
        assert result == "{}"

    def test_undefined_variable_fails(self) -> None:
        with pytest.raises(ConfigError, match="undefined"):
            render_template("{{ missing }}", {})

    def test_missing_rubric_field_fails(self) -> None:
        with pytest.raises(ConfigError, match="undefined"):
            render_template("{{ rubric.expected_answer }}", {"rubric": {}})

    def test_empty_rubric_provided(self) -> None:
        result = render_template(
            "{{ rubric | to_yaml }}",
            {"rubric": {}},
        )
        assert result.strip() == "{}"

    def test_multiline(self) -> None:
        result = render_template(
            "Q: {{ question }}\nA: {{ answer }}",
            {"question": "What?", "answer": "Because."},
        )
        assert "Q: What?" in result
        assert "A: Because." in result
