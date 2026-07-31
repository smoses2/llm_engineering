"""Report-time parsing of structured judge scores.

Raw judge text is the immutable source of truth. Scores are derived on every
report run, so criteria can be refined or a parser bug fixed without new API
calls. Nothing here invents a score: unparseable or out-of-range output is
reported as a failure and the raw text is preserved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_evals.errors import ConfigError

PARSE_NOT_CONFIGURED = "not_configured"
PARSE_OK = "parsed"
PARSE_FAILED = "failed"
PARSE_MISSING_TEXT = "no_judge_text"

RELEVANCE_WEIGHTS: dict[str, float] = {"relevant": 1.0, "partial": 0.5, "irrelevant": 0.0}

__all__ = [
    "PARSE_FAILED",
    "PARSE_MISSING_TEXT",
    "PARSE_NOT_CONFIGURED",
    "PARSE_OK",
    "RELEVANCE_WEIGHTS",
    "ParsedJudgment",
    "ScoringSpec",
    "extract_json_object",
    "load_scoring_specs",
    "parse_judgment",
]


class ScoringSpec(BaseModel):
    """Declared scoring criteria for one judgment stage.

    Criteria are declared rather than inferred so report columns stay stable and
    a judge that silently drops a criterion is detected instead of averaged away.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    parser: str = "json"
    minimum: int = 1
    maximum: int = 5
    criteria: list[str] = Field(min_length=1)
    comment_field: str | None = None
    source_assessment_field: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> ScoringSpec:
        if self.parser != "json":
            raise ValueError(
                f"Unsupported scoring parser: '{self.parser}'. Only 'json' is supported"
            )
        if self.minimum >= self.maximum:
            raise ValueError("Scoring 'minimum' must be less than 'maximum'")
        duplicates = {name for name in self.criteria if self.criteria.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate scoring criteria: {sorted(duplicates)}")
        if self.comment_field and self.comment_field in self.criteria:
            raise ValueError("'comment_field' must not also be a scoring criterion")
        if self.source_assessment_field:
            if self.source_assessment_field in self.criteria:
                raise ValueError("'source_assessment_field' must not also be a scoring criterion")
            if self.source_assessment_field == self.comment_field:
                raise ValueError("'source_assessment_field' must differ from 'comment_field'")
        return self


@dataclass
class SourceRelevance:
    """Counts derived arithmetically from judge-enumerated per-source verdicts.

    The judge observes and classifies; the ratio is computed here. Nothing is
    inferred when the enumeration is absent or malformed.
    """

    status: str = PARSE_NOT_CONFIGURED
    source_count: int | None = None
    relevant: int | None = None
    partial: int | None = None
    irrelevant: int | None = None
    ratio: float | None = None
    error: str = ""


@dataclass
class ParsedJudgment:
    """Outcome of parsing one judge response."""

    status: str
    scores: dict[str, int] = field(default_factory=dict)
    total: int | None = None
    comment: str = ""
    extra_keys: list[str] = field(default_factory=list)
    error: str = ""
    sources: SourceRelevance = field(default_factory=SourceRelevance)


def load_scoring_specs(data: Any) -> dict[str, ScoringSpec]:
    """Build stage-keyed scoring specs from a report definition's 'scoring' block."""
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("Report 'scoring' must be a mapping of stage name to scoring spec")
    allowed = {"retrieval_judgment", "answer_judgment"}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"Unknown scoring stages: {sorted(unknown)}. Allowed: {sorted(allowed)}")
    specs: dict[str, ScoringSpec] = {}
    for stage, raw in data.items():
        try:
            specs[stage] = ScoringSpec.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"Invalid scoring spec for '{stage}': {exc}") from exc
    return specs


def extract_json_object(text: str) -> str | None:
    """Return the first balanced JSON object in text, or None.

    Handles fenced blocks and surrounding prose by scanning for a balanced brace
    span while ignoring braces inside strings.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _coerce_score(value: Any) -> int | None:
    """Return an integer score, or None if the value is not an integer score."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def parse_source_assessments(payload: Any, spec: ScoringSpec) -> SourceRelevance:
    """Count per-source relevance verdicts and compute the relevance ratio.

    The judge supplies observations only. Counts and the ratio are computed here,
    so the judge is never asked to do arithmetic. Malformed or absent enumerations
    are reported without affecting the criterion scores.
    """
    if spec.source_assessment_field is None:
        return SourceRelevance()
    if not isinstance(payload, dict) or spec.source_assessment_field not in payload:
        return SourceRelevance(
            status=PARSE_FAILED,
            error=f"missing '{spec.source_assessment_field}' enumeration",
        )
    entries = payload[spec.source_assessment_field]
    if not isinstance(entries, list) or not entries:
        return SourceRelevance(
            status=PARSE_FAILED,
            error=f"'{spec.source_assessment_field}' must be a non-empty list",
        )

    counts = dict.fromkeys(RELEVANCE_WEIGHTS, 0)
    problems: list[str] = []
    seen: set[int] = set()
    for position, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            problems.append(f"entry {position} is not an object")
            continue
        index = _coerce_score(entry.get("index"))
        if index is None:
            problems.append(f"entry {position} has no integer 'index'")
        elif index in seen:
            problems.append(f"duplicate source index {index}")
        else:
            seen.add(index)
        verdict = entry.get("relevance")
        key = verdict.strip().lower() if isinstance(verdict, str) else None
        if key not in RELEVANCE_WEIGHTS:
            problems.append(
                f"entry {position} relevance must be one of "
                f"{sorted(RELEVANCE_WEIGHTS)}, got {verdict!r}"
            )
        else:
            counts[key] += 1

    if problems:
        return SourceRelevance(status=PARSE_FAILED, error="; ".join(problems))

    total = sum(counts.values())
    weighted = sum(counts[key] * weight for key, weight in RELEVANCE_WEIGHTS.items())
    return SourceRelevance(
        status=PARSE_OK,
        source_count=total,
        relevant=counts["relevant"],
        partial=counts["partial"],
        irrelevant=counts["irrelevant"],
        ratio=round(weighted / total, 4),
    )


def parse_judgment(text: str, spec: ScoringSpec) -> ParsedJudgment:
    """Parse one raw judge response against a scoring spec.

    The judge's own arithmetic is ignored; ``total`` is computed from the parsed
    criteria. Unknown keys are reported but do not fail the row.
    """
    if not text.strip():
        return ParsedJudgment(status=PARSE_MISSING_TEXT, error="Judge text is empty")

    candidate = extract_json_object(text)
    if candidate is None:
        return ParsedJudgment(status=PARSE_FAILED, error="No JSON object found in judge text")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParsedJudgment(status=PARSE_FAILED, error=f"Invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return ParsedJudgment(status=PARSE_FAILED, error="Judge JSON is not an object")

    scores: dict[str, int] = {}
    problems: list[str] = []
    for name in spec.criteria:
        if name not in payload:
            problems.append(f"missing criterion '{name}'")
            continue
        score = _coerce_score(payload[name])
        if score is None:
            problems.append(f"criterion '{name}' is not an integer: {payload[name]!r}")
        elif not spec.minimum <= score <= spec.maximum:
            problems.append(
                f"criterion '{name}' out of range {spec.minimum}-{spec.maximum}: {score}"
            )
        else:
            scores[name] = score

    comment = ""
    if spec.comment_field is not None:
        raw_comment = payload.get(spec.comment_field, "")
        comment = raw_comment if isinstance(raw_comment, str) else json.dumps(raw_comment)

    sources = parse_source_assessments(payload, spec)

    known = set(spec.criteria)
    for optional in (spec.comment_field, spec.source_assessment_field):
        if optional:
            known.add(optional)
    extra_keys = sorted(set(payload) - known)

    if problems:
        return ParsedJudgment(
            status=PARSE_FAILED,
            comment=comment,
            extra_keys=extra_keys,
            error="; ".join(problems),
            sources=sources,
        )

    return ParsedJudgment(
        status=PARSE_OK,
        scores=scores,
        total=sum(scores.values()),
        comment=comment,
        extra_keys=extra_keys,
        sources=sources,
    )
