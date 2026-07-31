"""Tests for deterministic identity utilities."""

import pytest

from rag_evals.planning.identity import (
    _reject_nan,
    artifact_id,
    artifact_path,
    canonical_hash,
    identity_hash,
    stable_short_id,
)


class TestCanonicalHash:
    def test_deterministic_ordering(self) -> None:
        h1 = canonical_hash({"b": 2, "a": 1})
        h2 = canonical_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_different_data(self) -> None:
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})

    def test_unicode(self) -> None:
        h = canonical_hash({"text": "héllo wörld 日本語"})
        assert len(h) == 64

    def test_nested_structures(self) -> None:
        h1 = canonical_hash({"a": {"b": [1, 2, {"c": 3}]}})
        h2 = canonical_hash({"a": {"b": [1, 2, {"c": 3}]}})
        assert h1 == h2

    def test_sets_serialized_sorted(self) -> None:
        h1 = canonical_hash({"s": {3, 1, 2}})
        h2 = canonical_hash({"s": {1, 2, 3}})
        assert h1 == h2

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_hash({"x": float("nan")})

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="Infinity"):
            canonical_hash({"x": float("inf")})

    def test_nested_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_hash({"a": [1, 2, {"b": float("nan")}]})

    def test_stability_across_calls(self) -> None:
        data = {"question": "What?", "kb": "kb1", "mode": "standard"}
        assert canonical_hash(data) == canonical_hash(data)

    def test_none_value(self) -> None:
        h = canonical_hash({"x": None})
        assert len(h) == 64

    def test_bool_value(self) -> None:
        h1 = canonical_hash({"x": True})
        h2 = canonical_hash({"x": False})
        assert h1 != h2

    def test_float_precision(self) -> None:
        h1 = canonical_hash({"x": 0.1})
        h2 = canonical_hash({"x": 0.1})
        assert h1 == h2

    def test_empty_containers(self) -> None:
        h1 = canonical_hash({})
        h2 = canonical_hash({"a": []})
        assert h1 != h2


class TestArtifactId:
    def test_format(self) -> None:
        aid = artifact_id({"a": 1})
        assert aid.startswith("sha256:")
        assert len(aid) == 71  # "sha256:" + 64 hex chars

    def test_deterministic(self) -> None:
        assert artifact_id({"a": 1}) == artifact_id({"a": 1})

    def test_different_identity(self) -> None:
        assert artifact_id({"a": 1}) != artifact_id({"a": 2})


class TestArtifactPath:
    def test_short_digest(self) -> None:
        aid = artifact_id({"a": 1})
        path = artifact_path(aid)
        assert len(path) == 16

    def test_with_slug(self) -> None:
        aid = artifact_id({"a": 1})
        path = artifact_path(aid, "my-experiment")
        assert len(path) > 16
        assert "my-experiment" in path

    def test_raw_digest(self) -> None:
        h = canonical_hash({"a": 1})
        path = artifact_path(h)
        assert len(path) == 16


class TestIdentityHash:
    def test_excludes_content_hash(self) -> None:
        h1 = identity_hash({"a": 1, "content_hash": "abc"})
        h2 = identity_hash({"a": 1, "content_hash": "xyz"})
        assert h1 == h2

    def test_includes_other_fields(self) -> None:
        assert identity_hash({"a": 1}) != identity_hash({"a": 2})


class TestStableShortId:
    def test_length(self) -> None:
        assert len(stable_short_id({"a": 1})) == 16

    def test_deterministic(self) -> None:
        assert stable_short_id({"a": 1}) == stable_short_id({"a": 1})


class TestRejectNan:
    def test_clean_data(self) -> None:
        _reject_nan({"a": 1, "b": [1, 2.0, "x"]})  # should not raise

    def test_nan_in_set(self) -> None:
        with pytest.raises(ValueError):
            _reject_nan({"s": {1.0, float("nan")}})
