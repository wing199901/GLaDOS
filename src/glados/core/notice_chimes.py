"""Local PA chimes that bracket Announcer notice lines.

The wav files are a personal drop-in under ``models/SFX/``. They are not
shipped with the repo. A missing file is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import soundfile as sf

from ..utils.resources import resolve_repo_path

DEFAULT_NOTICE_CHIME_ON = "models/SFX/ding_on.wav"
DEFAULT_NOTICE_CHIME_OFF = "models/SFX/ding_off.wav"


# Silence around the ding so device start/stop does not eat a ~0.2s clip.
# The tail stays inside the same output stream. The gap is after that stream
# closes and before speech opens another one. Opening the announcement
# immediately drops a ding Windows already accepted at 100%.
DEFAULT_CHIME_LEAD_S = 0.04
DEFAULT_CHIME_TAIL_S = 0.20
DEFAULT_CHIME_GAP_S = 0.35
# Portal dings sit under the announcement. Boost and clip so a full callback
# is still obvious next to TTS. 1.0 leaves the file unchanged.
DEFAULT_CHIME_GAIN = 1.8


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
        logger.error("Notice chime skipped: path is empty.")
        return None

    wav_path = resolve_repo_path(path_value)
    if not wav_path.is_file():
        logger.error(
            f"Notice chime not found at {wav_path} (cwd={Path.cwd()}); that chime will be skipped."
        )
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
        f"Notice chime loaded from {wav_path}: shape={tuple(mono.shape)} sr={int(sample_rate)} "
        f"samples={len(mono)} ({duration_s:.2f}s)."
    )
    return NoticeChime(audio=mono, sample_rate=int(sample_rate), source=str(wav_path))


def describe_configured_chime(label: str, configured: str | None, env_name: str) -> str:
    """Show the config value, env override, resolved file, and whether it exists."""
    env_value = os.environ.get(env_name)
    if configured is None or not str(configured).strip():
        return (
            f"{label}_config={configured!r} {label}_env={env_value!r} "
            f"{label}_resolved=None {label}_exists=False"
        )
    resolved = resolve_repo_path(configured)
    return (
        f"{label}_config={configured!r} {label}_env={env_value!r} "
        f"{label}_resolved={resolved} {label}_exists={resolved.is_file()}"
    )


def describe_notice_chime(clip: NoticeChime | None) -> str:
    """One-line description for startup logs."""
    if clip is None:
        return "not loaded"
    seconds = len(clip.audio) / clip.sample_rate if clip.sample_rate else 0.0
    return (
        f"{clip.source} shape={tuple(clip.audio.shape)} sr={clip.sample_rate} "
        f"samples={len(clip.audio)} ({seconds:.2f}s)"
    )


def chime_peak(clip: NoticeChime) -> float:
    """Peak absolute sample. A 100% callback of a near-silent buffer is still silent."""
    if clip.audio.size == 0:
        return 0.0
    return float(np.max(np.abs(clip.audio)))


def apply_chime_gain(clip: NoticeChime, gain: float) -> NoticeChime:
    """Scale a chime and clip to [-1, 1]. Silence padding is applied later."""
    if gain == 1.0 or clip.audio.size == 0:
        return clip
    boosted = np.clip(np.asarray(clip.audio, dtype=np.float32) * np.float32(gain), -1.0, 1.0)
    return NoticeChime(audio=boosted.astype(np.float32, copy=False), sample_rate=clip.sample_rate, source=clip.source)


def format_loaded_chimes(
    chime_on: NoticeChime | None,
    chime_off: NoticeChime | None,
    path_on: str | None,
    path_off: str | None,
) -> str:
    """One SUCCESS line for the load that used to be a temporary CHIME_LOAD_DEBUG print."""

    def _shape(clip: NoticeChime | None) -> str:
        if clip is None:
            return "None"
        return f"({clip.sample_rate}, {len(clip.audio)})"

    return (
        f"Notice chimes ready: on={_shape(chime_on)} off={_shape(chime_off)} "
        f"paths={path_on!r},{path_off!r}"
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
