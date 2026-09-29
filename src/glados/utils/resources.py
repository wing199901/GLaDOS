from functools import lru_cache
import os
from pathlib import Path

from loguru import logger


def find_project_root(start: Path) -> Path | None:
    """Walk parents for the checkout that contains this repo.

    A source tree is ``.../GLaDOS/src/glados/...``. An installed copy is
    ``.../GLaDOS/.venv/.../site-packages/glados/...``. Counting a fixed number
    of parents lands in ``site-packages`` or ``Lib`` for the install, so
    model files miss while ``glados start`` still finds ``models/`` from
    the working directory.
    """
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "glados").is_dir():
            return candidate
    return None


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the project root directory (cached)."""
    here = Path(__file__).resolve()
    found = find_project_root(here)
    if found is None:
        found = find_project_root(Path.cwd())
    if found is not None:
        return found
    # Source layout fallback: utils -> glados -> src -> project_root.
    return here.parents[3]


def resource_path(relative_path: str) -> Path:
    """Return absolute path to a model file."""
    return get_package_root() / relative_path


def resolve_repo_path(path_value: str) -> Path:
    """Resolve a config path from the project root, like ``models/TTS/glados.onnx``.

    Relative paths are rooted at the checkout. Absolute paths are kept so an
    environment override can still point at a file outside the repo.
    """
    expanded = os.path.expandvars(os.path.expanduser(path_value.strip()))
    path = Path(expanded)
    if path.is_absolute():
        return path
    rooted = resource_path(expanded)
    if rooted.is_file():
        return rooted
    # `glados start` checks model files from the working directory. A relative
    # file that is missing at the package root can still be found there.
    from_cwd = Path.cwd() / expanded
    if from_cwd.is_file():
        logger.success(
            f"Resolved {expanded} from the working directory {from_cwd} "
            f"(not found at package root {rooted})."
        )
        return from_cwd
    return rooted
