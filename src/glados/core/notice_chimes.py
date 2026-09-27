"""Local PA chimes that bracket Announcer notice lines.

The wav files are a personal drop-in under ``models/SFX/``. They are not
shipped with the repo. A missing file is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import soundfile as sf

from ..utils.resources import resolve_repo_path

DEFAULT_NOTICE_CHIME_ON = "models/SFX/ding_on.wav"
DEFAULT_NOTICE_CHIME_OFF = "models/SFX/ding_off.wav"


@dataclass(frozen=True)
class NoticeChime:
    """One mono chime ready for the speech player."""

    audio: NDArray[np.float32]
    sample_rate: int


def load_notice_chime(path_value: str | None) -> NoticeChime | None:
    """Load a chime wav as mono float32, or return None if it cannot be played.

    Stereo files are averaged to one channel. Relative paths resolve from the
    repo root. Nothing here downloads or vendors audio.
    """
    if path_value is None or not str(path_value).strip():
        return None

    wav_path = resolve_repo_path(path_value)
    if not wav_path.is_file():
        logger.warning(f"Notice chime not found at {wav_path}; that chime will be skipped.")
        return None

    try:
        data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - a bad local wav must not stop startup
        logger.exception(f"Failed to read notice chime {wav_path}; that chime will be skipped.")
        return None

    if data.size == 0 or int(sample_rate) <= 0:
        logger.warning(f"Notice chime {wav_path} is empty; that chime will be skipped.")
        return None

    mono = np.mean(data, axis=1).astype(np.float32, copy=False)
    return NoticeChime(audio=mono, sample_rate=int(sample_rate))
