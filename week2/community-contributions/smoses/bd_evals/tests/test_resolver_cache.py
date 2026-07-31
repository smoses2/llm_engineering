"""Regression test for F009: resolver cache returns raw text instead of resolved value.

When a YAML source is referenced multiple times, the second access should return
the same fully-parsed and recursively-resolved value, not the raw text string.
"""

from pathlib import Path

import pytest

from rag_evals.config.resolver import SourceResolver


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path


class TestResolverCacheReturnsResolvedValue:
    def test_repeated_yaml_reference_returns_parsed_dict(self, project_root: Path) -> None:
        """A YAML source referenced twice must return a parsed dict both times,
        not raw text on the second access."""
        shared = project_root / "shared.yaml"
        shared.write_text("schema_version: 1\nitems: []\n")

        def_file = project_root / "def.yaml"
        def_file.write_text("a:\n  source: shared.yaml\nb:\n  source: shared.yaml\n")

        resolver = SourceResolver(project_root)
        result = resolver.resolve_definition(def_file)

        a = result.definition_data["a"]
        b = result.definition_data["b"]

        # Both should be parsed dicts, not raw text strings
        assert isinstance(a, dict), f"First reference returned {type(a).__name__} instead of dict"
        assert isinstance(b, dict), f"Second reference returned {type(b).__name__} instead of dict"
        assert a == b, "Repeated references should produce equal resolved values"
        assert a is not b
        a["items"].append("mutation")
        assert b["items"] == []
        assert "schema_version" in a
        assert "items" in a

    def test_repeated_nested_reference_resolves_correctly(self, project_root: Path) -> None:
        """A YAML file that itself has source references must return the
        fully-resolved value on repeated access."""
        inner = project_root / "inner.yaml"
        inner.write_text("value: 42\n")

        outer = project_root / "outer.yaml"
        outer.write_text("schema_version: 1\nnested:\n  source: inner.yaml\n")

        def_file = project_root / "def.yaml"
        def_file.write_text("first:\n  source: outer.yaml\nsecond:\n  source: outer.yaml\n")

        resolver = SourceResolver(project_root)
        result = resolver.resolve_definition(def_file)

        first = result.definition_data["first"]
        second = result.definition_data["second"]

        assert isinstance(first, dict), f"First returned {type(first).__name__}"
        assert isinstance(second, dict), f"Second returned {type(second).__name__}"
        assert isinstance(first.get("nested"), dict), (
            f"First nested returned {type(first.get('nested')).__name__}"
        )
        assert isinstance(second.get("nested"), dict), (
            f"Second nested returned {type(second.get('nested')).__name__} - "
            f"this is the F009 bug: cached raw text instead of resolved value"
        )
        assert first["nested"]["value"] == 42
        assert second["nested"]["value"] == 42
        first["nested"]["value"] = 99
        assert second["nested"]["value"] == 42
