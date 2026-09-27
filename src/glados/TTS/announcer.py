"""Optional local Piper voice for the startup announcement and later notice lines.

The conversation voice stays on the configured synthesizer (GLaDOS Piper by
default). :func:`try_load_announcer_voice` never raises: a missing or unreadable
Announcer model falls back to that conversation voice. :func:`require_announcer_voice`
raises when ``glados say --announcer`` must not fall back.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from ..utils.resources import resolve_repo_path
from .piper_config import piper_config_candidates
from .tts_glados import SpeechSynthesizer

# Same layout as the bundled GLaDOS voice: models/TTS/glados.onnx beside glados.json.
# The Announcer files are a local drop-in and are not shipped with the repo.
DEFAULT_ANNOUNCER_MODEL = "models/TTS/announcer.onnx"


def resolve_announcer_model_path(model_path: str) -> Path:
    """Resolve an Announcer ONNX path the same way other repo models are resolved."""
    return resolve_repo_path(model_path)


class AnnouncerVoiceUnavailableError(RuntimeError):
    """The Announcer Piper model could not be loaded.

    ``glados say --announcer`` raises this instead of speaking with GLaDOS.
    Startup notices still use :func:`try_load_announcer_voice`, which returns
    None and lets the conversation voice speak.
    """


def try_load_announcer_voice(model_path: str | None, *, fallback: bool = True) -> SpeechSynthesizer | None:
    """Load a local Announcer Piper ONNX, or return None so callers can fall back.

    ``model_path`` is the ``.onnx`` file, usually ``models/TTS/announcer.onnx``.
    The matching config is ``<file>.onnx.json`` (standard Piper) or ``<stem>.json``.
    Inference settings such as ``length_scale`` come from that sidecar.
    Nothing in this function downloads or vendors voice weights.

    ``fallback`` controls the log text. Startup leaves it true. ``glados say
    --announcer`` passes false so the log does not promise a GLaDOS voice.
    """
    if model_path is None or not str(model_path).strip():
        if not fallback:
            _report_announcer_unavailable("Announcer voice path is empty.", fallback=False)
        return None

    onnx_path = resolve_announcer_model_path(model_path)
    if not onnx_path.is_file():
        _report_announcer_unavailable(f"Announcer voice model not found at {onnx_path}.", fallback=fallback)
        return None

    standard, classic = piper_config_candidates(onnx_path)
    if not standard.is_file() and not classic.is_file():
        _report_announcer_unavailable(
            f"Announcer Piper config not found beside {onnx_path} (looked for {standard} and {classic}).",
            fallback=fallback,
        )
        return None

    try:
        voice = SpeechSynthesizer(model_path=onnx_path, use_config_phoneme_map=True)
    except Exception:  # noqa: BLE001 - any ONNX/config failure must fall back, not crash startup
        logger.exception(f"Failed to load Announcer voice from {onnx_path}.")
        if fallback:
            logger.warning("Startup and notice lines will use the conversation voice.")
        return None

    logger.success(f"Announcer voice loaded from {onnx_path}")
    return voice


def require_announcer_voice(model_path: str | None) -> SpeechSynthesizer:
    """Load the Announcer voice for an explicit request such as ``glados say --announcer``.

    A missing or unreadable model raises :class:`AnnouncerVoiceUnavailableError`.
    This does not speak with the GLaDOS conversation voice.
    """
    voice = try_load_announcer_voice(model_path, fallback=False)
    if voice is None:
        raise AnnouncerVoiceUnavailableError(
            "Announcer voice is required for `glados say --announcer` but could not be loaded "
            f"from {model_path!r}. This command does not fall back to the GLaDOS voice. "
            "Set announcer_model_path or GLADOS_ANNOUNCER_MODEL to a Piper ONNX with its "
            "announcer.onnx.json sidecar."
        )
    return voice


def _report_announcer_unavailable(reason: str, *, fallback: bool) -> None:
    if fallback:
        logger.warning(f"{reason} Startup and notice lines will use the conversation voice.")
    else:
        logger.error(f"{reason} The GLaDOS voice will not be used.")
