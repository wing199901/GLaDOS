"""Core audio data structures for GLaDOS voice assistant.

This module defines message classes used for audio processing and communication
between different components of the voice assistant pipeline.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

SPEAKER_GLADOS = "GLaDOS"
SPEAKER_ANNOUNCER = "Announcer"


@dataclass
class AudioMessage:
    """Audio message container for TTS output.

    Args:
        audio: Generated audio samples as float32 array
        text: Associated text that was synthesized
        is_eos: Flag indicating end of speech stream
        sample_rate: Sample rate of this clip. Playback uses this when set,
            so a notice voice can differ from the conversation voice.
        speaker: Name shown in the TUI for this line. Notice lines that were
            actually synthesized with the Announcer model use ``Announcer``.
            A missing Announcer model keeps ``GLaDOS``.
        notice: True for startup and ``speak_notice`` lines. PA chimes follow
            this flag even when the voice falls back to GLaDOS.
    """

    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
    sample_rate: int | None = None
    speaker: str = SPEAKER_GLADOS
    notice: bool = False


def tts_dialog_role(meta: dict[str, Any] | None) -> str:
    """Speaker label for a TTS play event.

    Only an explicit Announcer speaker is relabeled. Missing metadata and
    GLaDOS-voice fallback both stay GLaDOS.
    """
    if meta and meta.get("speaker") == SPEAKER_ANNOUNCER:
        return SPEAKER_ANNOUNCER
    return SPEAKER_GLADOS


@dataclass
class AudioInputMessage:
    """Audio input message container for ASR processing.

    Args:
        audio_sample: Raw audio input samples as float32 array
        vad_confidence: Voice activity detection confidence flag
    """

    audio_sample: NDArray[np.float32]
    vad_confidence: bool = False
