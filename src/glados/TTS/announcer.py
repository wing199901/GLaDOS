"""Optional local Piper voice for the startup announcement and later notice lines.

The conversation voice stays on the configured synthesizer (GLaDOS Piper by
default). This loader never raises: a missing or unreadable Announcer model
falls back to that conversation voice.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from .piper_config import piper_config_candidates
from .tts_glados import SpeechSynthesizer


def try_load_announcer_voice(model_path: str | None) -> SpeechSynthesizer | None:
    """Load a local Announcer Piper ONNX, or return None so callers can fall back.

    ``model_path`` is the ``.onnx`` file. The matching config is
    ``<file>.onnx.json`` (standard Piper) or ``<stem>.json``. ``~`` and
    environment variables in the path are expanded. Nothing in this function
    downloads or vendors voice weights.
    """
    if model_path is None or not str(model_path).strip():
        return None

    expanded = os.path.expandvars(os.path.expanduser(str(model_path).strip()))
    onnx_path = Path(expanded)
    if not onnx_path.is_file():
        logger.warning(
            f"Announcer voice model not found at {onnx_path}; "
            "startup and notice lines will use the conversation voice."
        )
        return None

    standard, classic = piper_config_candidates(onnx_path)
    if not standard.is_file() and not classic.is_file():
        logger.warning(
            f"Announcer Piper config not found beside {onnx_path} "
            f"(looked for {standard} and {classic}); "
            "startup and notice lines will use the conversation voice."
        )
        return None

    try:
        voice = SpeechSynthesizer(model_path=onnx_path, use_config_phoneme_map=True)
    except Exception:  # noqa: BLE001 - any ONNX/config failure must fall back, not crash startup
        logger.exception(
            f"Failed to load Announcer voice from {onnx_path}; "
            "startup and notice lines will use the conversation voice."
        )
        return None

    logger.info(f"Announcer voice loaded from {onnx_path}")
    return voice
