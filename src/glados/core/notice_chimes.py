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


# Silence around the ding so device start/stop does not eat a ~0.2s clip,
# and a beat of quiet before the announcement so the chime is not masked.
DEFAULT_CHIME_LEAD_S = 0.04
DEFAULT_CHIME_TAIL_S = 0.12
DEFAULT_CHIME_GAP_S = 0.10


@dataclass(frozen=True)
class NoticeChime:
    """One mono chime ready for the speech player."""

    audio: NDArray[np.float32]
    sample_rate: int
    source: str


def load_notice_chime(path_value: str | None) -> NoticeChime | None:
    """Load a chime wav as mono float32, or return None if it cannot be played.

    Stereo files are averaged to one channel. Relative paths resolve from the
    repo root. Nothing here downloads or vendors audio.
    """
    if path_value is None or not str(path_value).strip():
        logger.info("Notice chime skipped: path is empty.")
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
    duration_s = len(mono) / int(sample_rate)
    logger.success(
        f"Notice chime loaded from {wav_path}: {int(sample_rate)} Hz, {len(mono)} samples, {duration_s:.2f}s."
    )
    return NoticeChime(audio=mono, sample_rate=int(sample_rate), source=str(wav_path))


def describe_notice_chime(clip: NoticeChime | None) -> str:
    """One-line description for startup logs."""
    if clip is None:
        return "not loaded"
    seconds = len(clip.audio) / clip.sample_rate if clip.sample_rate else 0.0
    return (
        f"{clip.source} shape={tuple(clip.audio.shape)} sr={clip.sample_rate} "
        f"samples={len(clip.audio)} ({seconds:.2f}s)"
    )


def with_chime_edges(clip: NoticeChime, lead_s: float, tail_s: float) -> NoticeChime:
    """Pad a chime so the ding is not the first or last sample in the device buffer."""
    lead = max(0, round(clip.sample_rate * lead_s))
    tail = max(0, round(clip.sample_rate * tail_s))
    if lead == 0 and tail == 0:
        return clip
    audio = np.concatenate(
        (
            np.zeros(lead, dtype=np.float32),
            np.asarray(clip.audio, dtype=np.float32),
            np.zeros(tail, dtype=np.float32),
        )
    )
    return NoticeChime(audio=audio, sample_rate=clip.sample_rate, source=clip.source)
