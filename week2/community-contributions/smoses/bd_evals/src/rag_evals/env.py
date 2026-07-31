"""Environment file loading for the selected evaluation workspace.

Secret values live in ``.env`` files owned by the operator. This module only
copies them into the process environment; it never logs, snapshots, or returns
any value. Only file paths and variable names are observable.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from rag_evals.logging import get_logger
from rag_evals.workspace import Workspace

ENV_FILENAME = ".env"

logger = get_logger("rag_evals.env")


def env_file_candidates(workspace: Workspace) -> list[Path]:
    """Return ``.env`` search paths for a workspace, most specific first.

    The workspace's own file wins over one shared with sibling workspaces in the
    parent directory. No other location is searched.
    """
    candidates = [workspace.root / ENV_FILENAME]
    parent = workspace.root.parent
    if parent != workspace.root:
        candidates.append(parent / ENV_FILENAME)
    return candidates


def load_workspace_env(workspace: Workspace) -> list[Path]:
    """Load workspace ``.env`` files and return the files applied, in order.

    Variables already present in the process environment are never overridden,
    so an explicit shell export always wins. Earlier candidates win over later
    ones for the same reason.
    """
    loaded: list[Path] = []
    for candidate in env_file_candidates(workspace):
        if not candidate.is_file():
            continue
        load_dotenv(candidate, override=False)
        loaded.append(candidate)
        names = sorted(key for key in dotenv_values(candidate) if key)
        logger.debug("Loaded environment file %s defining %s", candidate, ", ".join(names))
    return loaded
