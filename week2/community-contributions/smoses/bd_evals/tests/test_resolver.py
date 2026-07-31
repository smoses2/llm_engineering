"""Tests for YAML source resolver."""

from pathlib import Path

import pytest

from rag_evals.config.resolver import SourceResolver, canonical_hash, safe_load_yaml
from rag_evals.errors import ConfigError


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def resolver(project_root: Path) -> SourceResolver:
    return SourceResolver(project_root)


class TestSafeLoadYaml:
    def test_valid(self, project_root: Path) -> None:
        f = project_root / "test.yaml"
        f.write_text("key: value\n")
        data = safe_load_yaml(f)
        assert data == {"key": "value"}

    def test_missing_file(self, project_root: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            safe_load_yaml(project_root / "missing.yaml")

    def test_invalid_yaml(self, project_root: Path) -> None:
        f = project_root / "bad.yaml"
        f.write_text("key: [unclosed\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            safe_load_yaml(f)

    def test_empty_file(self, project_root: Path) -> None:
        f = project_root / "empty.yaml"
        f.write_text("")
        with pytest.raises(ConfigError, match="Empty"):
            safe_load_yaml(f)

    def test_non_mapping_root(self, project_root: Path) -> None:
        f = project_root / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="mapping"):
            safe_load_yaml(f)


class TestCanonicalHash:
    def test_deterministic(self) -> None:
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_different_data_different_hash(self) -> None:
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})

    def test_unicode_stable(self) -> None:
        h = canonical_hash({"text": "héllo wörld"})
        assert len(h) == 64

    def test_nested(self) -> None:
        h1 = canonical_hash({"a": {"b": [1, 2, 3]}})
        h2 = canonical_hash({"a": {"b": [1, 2, 3]}})
        assert h1 == h2


class TestSourceResolver:
    def test_resolve_simple_definition(self, resolver: SourceResolver, project_root: Path) -> None:
        # Create a definition with inline data
        def_file = project_root / "def.yaml"
        def_file.write_text("schema_version: 1\ntest:\n  id: t1\n  type: retrieval\n")

        result = resolver.resolve_definition(def_file)
        assert result.definition_data["schema_version"] == 1
        assert result.definition_hash

    def test_resolve_source_ref(self, resolver: SourceResolver, project_root: Path) -> None:
        # Create a catalog file
        catalog = project_root / "questions.yaml"
        catalog.write_text("schema_version: 1\nquestions:\n  - id: q1\n    question: What?\n")

        # Create a definition referencing it
        def_file = project_root / "def.yaml"
        def_file.write_text("schema_version: 1\nquestions:\n  source: questions.yaml\n")

        result = resolver.resolve_definition(def_file)
        assert "questions" in result.definition_data
        assert isinstance(result.definition_data["questions"], dict)
        assert result.definition_data["questions"]["questions"][0]["id"] == "q1"

    def test_resolve_nested_source_ref(self, resolver: SourceResolver, project_root: Path) -> None:
        # Create nested directory structure
        subdir = project_root / "config"
        subdir.mkdir()
        catalog = subdir / "kbs.yaml"
        catalog.write_text("schema_version: 1\nknowledge_bases: []\n")

        eval_dir = project_root / "eval_definitions"
        eval_dir.mkdir()
        def_file = eval_dir / "def.yaml"
        def_file.write_text("schema_version: 1\nknowledge_bases:\n  source: ../config/kbs.yaml\n")

        result = resolver.resolve_definition(def_file)
        assert result.definition_data["knowledge_bases"]["schema_version"] == 1

    def test_cycle_detection(self, resolver: SourceResolver, project_root: Path) -> None:
        f1 = project_root / "a.yaml"
        f2 = project_root / "b.yaml"
        f1.write_text("ref:\n  source: b.yaml\n")
        f2.write_text("ref:\n  source: a.yaml\n")

        with pytest.raises(ConfigError, match="Circular"):
            resolver.resolve_definition(f1)

    def test_self_reference(self, resolver: SourceResolver, project_root: Path) -> None:
        f = project_root / "self.yaml"
        f.write_text("ref:\n  source: self.yaml\n")

        with pytest.raises(ConfigError, match="Circular"):
            resolver.resolve_definition(f)

    def test_path_traversal_rejected(self, resolver: SourceResolver, project_root: Path) -> None:
        def_file = project_root / "def.yaml"
        def_file.write_text("data:\n  source: ../../etc/passwd\n")

        with pytest.raises(ConfigError, match="outside project root"):
            resolver.resolve_definition(def_file)

    def test_absolute_path_rejected(self, resolver: SourceResolver, project_root: Path) -> None:
        def_file = project_root / "def.yaml"
        def_file.write_text("data:\n  source: /etc/passwd\n")

        with pytest.raises(ConfigError, match="Absolute"):
            resolver.resolve_definition(def_file)

    def test_missing_source(self, resolver: SourceResolver, project_root: Path) -> None:
        def_file = project_root / "def.yaml"
        def_file.write_text("data:\n  source: missing.yaml\n")

        with pytest.raises(ConfigError, match="not found"):
            resolver.resolve_definition(def_file)

    def test_each_input_snapshotted_once(
        self, resolver: SourceResolver, project_root: Path
    ) -> None:
        catalog = project_root / "shared.yaml"
        catalog.write_text("schema_version: 1\nitems: []\n")

        def_file = project_root / "def.yaml"
        def_file.write_text("a:\n  source: shared.yaml\nb:\n  source: shared.yaml\n")

        result = resolver.resolve_definition(def_file)
        assert len(result.inputs) == 1

    def test_source_inputs_have_hashes(self, resolver: SourceResolver, project_root: Path) -> None:
        catalog = project_root / "data.yaml"
        catalog.write_text("key: value\n")

        def_file = project_root / "def.yaml"
        def_file.write_text("data:\n  source: data.yaml\n")

        result = resolver.resolve_definition(def_file)
        assert len(result.inputs) == 1
        assert len(result.inputs[0].content_hash) == 64

    def test_definition_hash_stable(self, resolver: SourceResolver, project_root: Path) -> None:
        def_file = project_root / "def.yaml"
        def_file.write_text("schema_version: 1\nkey: value\n")

        r1 = resolver.resolve_definition(def_file)
        r2 = resolver.resolve_definition(def_file)
        assert r1.definition_hash == r2.definition_hash

    def test_markdown_source_loaded_as_text(
        self, resolver: SourceResolver, project_root: Path
    ) -> None:
        md = project_root / "instructions.md"
        md.write_text("# Instructions\nDo the thing.\n")

        def_file = project_root / "def.yaml"
        def_file.write_text("template:\n  source: instructions.md\n")

        result = resolver.resolve_definition(def_file)
        assert isinstance(result.definition_data["template"], str)
        assert "Instructions" in result.definition_data["template"]

    def test_non_source_dict_not_treated_as_ref(
        self, resolver: SourceResolver, project_root: Path
    ) -> None:
        def_file = project_root / "def.yaml"
        def_file.write_text(
            "schema_version: 1\ntest:\n  id: t1\n  type: retrieval\n"
            "api:\n  base_url_env: BD_API_BASE_URL\n  api_key_env: BD_API_KEY\n"
        )

        result = resolver.resolve_definition(def_file)
        assert result.definition_data["api"]["base_url_env"] == "BD_API_BASE_URL"
