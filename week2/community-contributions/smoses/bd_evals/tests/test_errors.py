"""Tests for domain error hierarchy and exit codes."""

import pytest

from rag_evals.errors import (
    AdapterError,
    ArtifactError,
    ConfigError,
    ExecutionError,
    NotImplementedCommandError,
    PlanError,
    RagEvalsError,
)

ALL_ERROR_CLASSES = [
    RagEvalsError,
    NotImplementedCommandError,
    ConfigError,
    PlanError,
    ArtifactError,
    AdapterError,
    ExecutionError,
]

EXPECTED_EXIT_CODES = {
    RagEvalsError: 1,
    NotImplementedCommandError: 2,
    ConfigError: 3,
    PlanError: 4,
    ArtifactError: 5,
    AdapterError: 6,
    ExecutionError: 7,
}


@pytest.mark.parametrize("error_class", ALL_ERROR_CLASSES)
def test_inherits_base(error_class: type[Exception]) -> None:
    assert issubclass(error_class, RagEvalsError)


@pytest.mark.parametrize("error_class", ALL_ERROR_CLASSES)
def test_exit_codes(error_class: type[RagEvalsError]) -> None:
    assert error_class.exit_code == EXPECTED_EXIT_CODES[error_class]


@pytest.mark.parametrize("error_class", ALL_ERROR_CLASSES)
def test_exit_code_positive(error_class: type[RagEvalsError]) -> None:
    assert error_class.exit_code > 0


def test_exit_codes_unique() -> None:
    codes = [cls.exit_code for cls in ALL_ERROR_CLASSES]
    assert len(codes) == len(set(codes))


def test_message_preserved() -> None:
    err = ConfigError("bad config")
    assert str(err) == "bad config"
