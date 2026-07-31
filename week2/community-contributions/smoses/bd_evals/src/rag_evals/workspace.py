"""Evaluation workspace selection and boundary-safe path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_evals.errors import ConfigError


@dataclass(frozen=True)
class Workspace:
    """External configuration and results workspace."""

    root: Path

    @classmethod
    def from_path(cls, value: Path) -> Workspace:
        """Resolve and validate a workspace containing config/ and results/."""
        root = value.expanduser().resolve()
        if not root.exists():
            raise ConfigError(f"Workspace does not exist: {root}")
        if not root.is_dir():
            raise ConfigError(f"Workspace is not a directory: {root}")
        for name in ("config", "results"):
            required = root / name
            if not required.is_dir():
                raise ConfigError(f"Workspace must contain a '{name}' directory: {required}")
        return cls(root=root)

    @property
    def config_root(self) -> Path:
        return self.root / "config"

    @property
    def results_root(self) -> Path:
        return self.root / "results"

    @property
    def datasets_root(self) -> Path:
        return self.results_root / "datasets"

    @property
    def reports_root(self) -> Path:
        return self.results_root / "reports"

    def resolve_definition(self, argument: Path) -> Path:
        """Resolve a definition relative to config/ without allowing escape."""
        if argument.is_absolute():
            raise ConfigError("Definition paths must be relative to the workspace config directory")
        resolved = (self.config_root / argument).resolve()
        self._require_within(resolved, self.config_root, "Definition")
        if not resolved.is_file():
            raise ConfigError(f"Definition file not found: {resolved}")
        return resolved

    def resolve_report_output(self, argument: Path) -> Path:
        """Resolve a report output directory below results/reports/."""
        if argument.is_absolute():
            raise ConfigError(
                "Report output paths must be relative to the workspace reports directory"
            )
        resolved = (self.reports_root / argument).resolve()
        self._require_within(resolved, self.reports_root, "Report output")
        return resolved

    @staticmethod
    def _require_within(path: Path, boundary: Path, label: str) -> None:
        try:
            path.relative_to(boundary.resolve())
        except ValueError:
            raise ConfigError(f"{label} path escapes workspace boundary: {path}") from None
