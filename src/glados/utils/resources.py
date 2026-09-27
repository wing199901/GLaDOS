from functools import lru_cache
import os
from pathlib import Path


@lru_cache(maxsize=1)
def get_package_root() -> Path:
    """Get the absolute path to the project root directory (cached)."""
    # utils -> glados -> src -> project_root
    return Path(__file__).resolve().parents[3]


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
    return resource_path(expanded)
