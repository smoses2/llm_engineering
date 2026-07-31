"""Regression coverage for F003 and F028 resolved execution snapshots."""

from pathlib import Path

from rag_evals.config.execution import load_execution_bundle


def test_bundle_hashes_resolved_catalog_and_prompt_contents(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "questions.yaml").write_text(
        "schema_version: 1\nquestions:\n  - id: q1\n    question: Original?\n"
    )
    (tmp_path / "models.yaml").write_text(
        "schema_version: 1\nmodels:\n  - id: m1\n    model_id: actual\n"
    )
    (tmp_path / "system.md").write_text("System {{ question_id }}")
    (tmp_path / "user.md").write_text("Question: {{ question }}")
    definition = tmp_path / "definition.yaml"
    definition.write_text(
        """schema_version: 1
test: {id: answer-test, type: answer}
source_dataset: {type: retrieval, id: source}
answer_models: {source: models.yaml}
selection: {}
prompt_variants:
  - id: p1
    template: {system_instructions: system.md, user_message: user.md}
inference_variants:
  - id: i1
    inference_config: {temperature: 0.0}
api: {base_url_env: API_URL, api_key_env: API_KEY}
output: {dataset_id: output}
"""
    )

    first = load_execution_bundle(definition, tmp_path)
    assert first.resolved_definition["answer_models"]["models"][0]["model_id"] == "actual"
    assert set(first.prompt_contents) == {"system.md", "user.md"}
    assert {item["source_path"] for item in first.snapshot_inputs()} == {
        "models.yaml",
        "system.md",
        "user.md",
    }

    (tmp_path / "user.md").write_text("Changed: {{ question }}")
    second = load_execution_bundle(definition, tmp_path)
    assert second.resolved_definition_hash != first.resolved_definition_hash
