"""Deterministic identity utilities: canonical JSON, SHA-256, artifact IDs."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any
from urllib.parse import quote


def _canonical_json(data: Any) -> str:
    """Serialize to canonical UTF-8 JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_default_serializer,
    )


def _default_serializer(obj: Any) -> Any:
    """Handle types JSON doesn't natively support."""
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, frozenset):
        return sorted(obj, key=str)
    raise TypeError(f"Cannot serialize {type(obj).__name__} to canonical JSON")


def canonical_hash(data: Any) -> str:
    """Compute SHA-256 of canonical JSON representation.

    - Sorted keys
    - Compact separators (no whitespace)
    - UTF-8 encoded
    - No NaN or Infinity (rejected)
    - Sets serialized as sorted lists
    """
    _reject_nan(data)
    canonical = _canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reject_nan(data: Any) -> None:
    """Recursively reject NaN and Infinity floats."""
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            raise ValueError("NaN and Infinity are not allowed in canonical JSON")
    elif isinstance(data, dict):
        for v in data.values():
            _reject_nan(v)
    elif isinstance(data, (list, tuple, set, frozenset)):
        for item in data:
            _reject_nan(item)


def artifact_id(identity: dict[str, Any]) -> str:
    """Generate a full artifact ID: 'sha256:' + lowercase hex digest."""
    digest = canonical_hash(identity)
    return f"sha256:{digest}"


def artifact_path(digest_or_id: str, slug: str = "") -> str:
    """Generate a short artifact path component: first 16 chars + slug.

    The full ID is stored separately; this is only for filesystem paths.
    """
    digest = digest_or_id[7:] if digest_or_id.startswith("sha256:") else digest_or_id
    short = digest[:16]
    if slug:
        safe_slug = quote(slug, safe="")[:48]
        return f"{short}-{safe_slug}"
    return short


def identity_hash(identity: dict[str, Any]) -> str:
    """Hash an identity object, excluding the content_hash field if present."""
    filtered = {k: v for k, v in identity.items() if k != "content_hash"}
    return canonical_hash(filtered)


def stable_short_id(identity: dict[str, Any]) -> str:
    """Generate a short stable ID (first 16 chars of the SHA-256 digest)."""
    return canonical_hash(identity)[:16]


IDENTITY_SCHEMA_VERSION = 2


def retrieval_identity(**values: Any) -> dict[str, Any]:
    """Build the sole versioned retrieval identity contract."""
    return {"identity_schema_version": IDENTITY_SCHEMA_VERSION, "stage": "retrieval", **values}


def ask_identity(
    *,
    stage: str,
    source_artifact_id: str,
    source_content_hash: str,
    question_id: str,
    question_text: str,
    rubric_hash: str,
    model_catalog_id: str,
    model_actual_id: str,
    prompt_variant_id: str,
    system_template_hash: str,
    user_template_hash: str,
    inference_variant_id: str,
    inference_config: dict[str, Any],
) -> dict[str, Any]:
    """Build the complete versioned identity for any ask-based artifact."""
    return {
        "identity_schema_version": IDENTITY_SCHEMA_VERSION,
        "stage": stage,
        "source_artifact_id": source_artifact_id,
        "source_content_hash": source_content_hash,
        "question_id": question_id,
        "question_text": question_text,
        "rubric_hash": rubric_hash,
        "model_catalog_id": model_catalog_id,
        "model_actual_id": model_actual_id,
        "prompt_variant_id": prompt_variant_id,
        "system_template_hash": system_template_hash,
        "user_template_hash": user_template_hash,
        "inference_variant_id": inference_variant_id,
        "inference_config": inference_config,
    }
