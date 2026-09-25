"""TTS queue items that can select the notice voice."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpokenLine:
    """One line waiting for synthesis.

    ``notice`` is set for the startup announcement and other short system
    lines. Those lines use the optional Announcer Piper model. Every other
    line stays on the conversation voice.
    """

    text: str
    notice: bool = False


TtsQueueItem = str | SpokenLine
