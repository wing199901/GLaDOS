"""Locate Piper JSON sidecars for an ONNX voice model."""

from pathlib import Path


def piper_config_candidates(model_path: Path) -> tuple[Path, Path]:
    """Return the standard Piper sidecar and this project's classic sidecar.

    Piper exports ``model.onnx`` next to ``model.onnx.json``.
    The bundled GLaDOS voice uses ``model.json`` next to ``model.onnx``.
    """
    standard = Path(f"{model_path}.json")
    classic = model_path.with_suffix(".json")
    return standard, classic


def resolve_piper_config_path(model_path: Path) -> Path:
    """Pick the Piper config that exists beside ``model_path``.

    The standard ``*.onnx.json`` file wins when both are present. Otherwise
    the classic ``*.json`` path is returned, including when neither file
    exists, so missing-file errors still name the bundled GLaDOS layout.
    """
    standard, classic = piper_config_candidates(model_path)
    if standard.is_file():
        return standard
    return classic
