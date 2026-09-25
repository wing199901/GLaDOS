"""Core audio data structures for GLaDOS voice assistant.

This module defines message classes used for audio processing and communication
between different components of the voice assistant pipeline.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMessage:
    """Audio message container for TTS output.

    Args:
        audio: Generated audio samples as float32 array
        text: Associated text that was synthesized
        is_eos: Flag indicating end of speech stream
        sample_rate: Sample rate of this clip. Playback uses this when set,
            so a notice voice can differ from the conversation voice.
    """

    audio: NDArray[np.float32]
    text: str
    is_eos: bool = False
    sample_rate: int | None = None


@dataclass
class AudioInputMessage:
    """Audio input message container for ASR processing.

    Args:
        audio_sample: Raw audio input samples as float32 array
        vad_confidence: Voice activity detection confidence flag
    """

    audio_sample: NDArray[np.float32]
    vad_confidence: bool = False
