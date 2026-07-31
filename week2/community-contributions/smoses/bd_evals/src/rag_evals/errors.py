"""Domain errors with stable exit codes."""


class RagEvalsError(Exception):
    """Base error for all rag-evals domain errors."""

    exit_code: int = 1


class NotImplementedCommandError(RagEvalsError):
    """Command exists but is not yet implemented."""

    exit_code: int = 2


class ConfigError(RagEvalsError):
    """Configuration loading, validation, or resolution error."""

    exit_code: int = 3


class PlanError(RagEvalsError):
    """Planning or matrix expansion error."""

    exit_code: int = 4


class ArtifactError(RagEvalsError):
    """Artifact store, lifecycle, or lineage error."""

    exit_code: int = 5


class AdapterError(RagEvalsError):
    """API adapter or bd_api interaction error."""

    exit_code: int = 6


class ExecutionError(RagEvalsError):
    """Execution or pipeline stage error."""

    exit_code: int = 7
