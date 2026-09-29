"""TTS queue items that can select the notice voice."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpokenLine:
    """One line waiting for synthesis.

    ``notice`` is set for the startup announcement and other short system
    lines. Those lines use the optional Announcer Piper model. Every other
    line stays on the conversation voice.

    ``ends_startup`` is set on the last line of the startup sequence. The
    microphone stays closed until that line has finished. Later notices and
    conversation lines leave it false.

    ``playback_delay_s`` is silence before this line starts. The startup
    follow-up uses it so GLaDOS waits after the notice.
    """

    text: str
    notice: bool = False
    ends_startup: bool = False
    playback_delay_s: float = 0.0


TtsQueueItem = str | SpokenLine
