from __future__ import annotations

import queue
import threading
import time

from loguru import logger

from ..audio_io import AudioProtocol
from ..observability import ObservabilityBus, trim_message
from .audio_data import SPEAKER_ANNOUNCER, AudioMessage
from .conversation_store import ConversationStore
from .notice_chimes import (
    DEFAULT_CHIME_GAP_S,
    DEFAULT_CHIME_LEAD_S,
    DEFAULT_CHIME_TAIL_S,
    NoticeChime,
    describe_notice_chime,
    with_chime_edges,
)


class SpeechPlayer:
    """
    A thread that plays audio messages from a queue, handling interruptions and end-of-stream tokens.
    This class is designed to run in a separate thread, continuously checking for audio messages to play
    until a shutdown event is set. It manages conversation history and handles interruptions gracefully.
    """

    def __init__(
        self,
        audio_io: AudioProtocol,
        audio_output_queue: queue.Queue[AudioMessage],
        conversation_store: ConversationStore,
        tts_sample_rate: int,
        shutdown_event: threading.Event,
        currently_speaking_event: threading.Event,
        processing_active_event: threading.Event,
        pause_time: float,
        tts_muted_event: threading.Event | None = None,
        interaction_state: "InteractionState | None" = None,
        observability_bus: ObservabilityBus | None = None,
        chime_on: NoticeChime | None = None,
        chime_off: NoticeChime | None = None,
        chime_off_after_interrupt: bool = True,
        chime_hold_event: threading.Event | None = None,
        chime_lead_s: float = DEFAULT_CHIME_LEAD_S,
        chime_tail_s: float = DEFAULT_CHIME_TAIL_S,
        chime_gap_s: float = DEFAULT_CHIME_GAP_S,
    ) -> None:
        self.audio_io = audio_io
        self.audio_output_queue = audio_output_queue
        self._conversation_store = conversation_store
        self.tts_sample_rate = tts_sample_rate
        self.shutdown_event = shutdown_event
        self.currently_speaking_event = currently_speaking_event
        self.processing_active_event = processing_active_event
        self.pause_time = pause_time
        self._tts_muted_event = tts_muted_event
        self._interaction_state = interaction_state
        self._observability_bus = observability_bus
        self._chime_on = chime_on
        self._chime_off = chime_off
        self._chime_off_after_interrupt = chime_off_after_interrupt
        self._chime_hold_event = chime_hold_event
        self._chime_lead_s = chime_lead_s
        self._chime_tail_s = chime_tail_s
        self._chime_gap_s = chime_gap_s
        self._log_armed_chime("ding_on", chime_on)
        self._log_armed_chime("ding_off", chime_off)

    def run(self) -> None:
        """
        Starts the main loop for the AudioPlayer thread.
        This method continuously checks the audio output queue for messages to process.
        It plays audio messages, handles end-of-stream tokens, and manages the conversation history.
        """
        assistant_text_accumulator: list[str] = []

        logger.info("AudioPlayer thread started.")
        while not self.shutdown_event.is_set():
            try:
                audio_msg = self.audio_output_queue.get(timeout=self.pause_time)

                audio_len = len(audio_msg.audio) if audio_msg.audio is not None else 0
                tts_muted = bool(self._tts_muted_event and self._tts_muted_event.is_set())

                if audio_msg.is_eos:
                    logger.debug("AudioPlayer: Processing end of stream token.")
                    if assistant_text_accumulator:
                        self._conversation_store.append(
                            {"role": "assistant", "content": " ".join(assistant_text_accumulator)}
                        )
                    assistant_text_accumulator = []
                    self.currently_speaking_event.clear()
                    continue

                if tts_muted:
                    if self._uses_notice_chimes(audio_msg):
                        logger.info("Notice chimes skipped: TTS is muted.")
                    if audio_msg.text:
                        logger.info(f"Assistant: {audio_msg.text}")
                        if self._interaction_state:
                            self._interaction_state.mark_assistant()
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tts",
                                kind="play",
                                message=trim_message(audio_msg.text),
                                meta={"audio_samples": 0, "muted": True, "speaker": audio_msg.speaker},
                            )
                            self._observability_bus.emit(
                                source="tts",
                                kind="finish",
                                message=trim_message(audio_msg.text),
                                meta={"muted": True},
                            )
                        assistant_text_accumulator.append(audio_msg.text)
                    else:
                        logger.warning(f"AudioPlayer: Received empty audio message or no text: {audio_len, audio_msg}")
                    self.currently_speaking_event.clear()
                    continue

                playback_rate = audio_msg.sample_rate or self.tts_sample_rate

                if audio_len and audio_msg.text:  # Ensure there's audio and text
                    self.currently_speaking_event.set()  # We are about to speak
                    if self._interaction_state:
                        self._interaction_state.mark_assistant()
                    if self._observability_bus:
                        self._observability_bus.emit(
                            source="tts",
                            kind="play",
                            message=trim_message(audio_msg.text),
                            meta={"audio_samples": audio_len, "speaker": audio_msg.speaker},
                        )

                    # Notice lines only. The mic must not cut the chimes: echo often
                    # trips VAD, and a ~0.2s ding is gone if that aborts the stream.
                    # ding_off still plays after an interrupted notice unless config
                    # turns that off. Conversation lines never enter this branch.
                    if self._uses_notice_chimes(audio_msg):
                        logger.success(
                            "Notice playback starting: "
                            f"speaker={audio_msg.speaker} notice={audio_msg.notice} "
                            f"ding_on={describe_notice_chime(self._chime_on)} "
                            f"ding_off={describe_notice_chime(self._chime_off)}"
                        )
                        self._hold_chime()
                        try:
                            self._play_notice_chime(self._chime_on, "ding_on")
                            if self._chime_on is not None and self._chime_gap_s > 0:
                                time.sleep(self._chime_gap_s)
                        finally:
                            self._release_chime()

                    self.audio_io.start_speaking(audio_msg.audio, playback_rate)
                    logger.success(f"TTS text: {audio_msg.text}")
                    interrupted, percentage_played = self.audio_io.measure_percentage_spoken(
                        audio_len, playback_rate
                    )

                    if interrupted:
                        clipped_text = self.clip_interrupted_sentence(audio_msg.text, percentage_played)
                        logger.success(f"TTS interrupted at {percentage_played}%: {clipped_text}")
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tts",
                                kind="interrupt",
                                message=trim_message(clipped_text),
                                level="warning",
                                meta={"percentage": round(float(percentage_played), 2)},
                            )

                        assistant_text_accumulator.append(clipped_text)
                        # Atomically append both messages to avoid race conditions
                        self._conversation_store.append_multiple([
                            {"role": "assistant", "content": " ".join(assistant_text_accumulator)},
                            {
                                "role": "user",
                                "content": (
                                    "[SYSTEM: User interrupted mid-response! Full intended output: "
                                    f"'{audio_msg.text}']"
                                ),
                            },
                        ])
                        assistant_text_accumulator = []  # Reset accumulator
                        self._clear_audio_queue()

                    else:  # Playback completed normally
                        logger.success(f"AudioPlayer: Playback completed for: '{audio_msg.text}'")
                        assistant_text_accumulator.append(audio_msg.text)
                        if self._observability_bus:
                            self._observability_bus.emit(
                                source="tts",
                                kind="finish",
                                message=trim_message(audio_msg.text),
                            )
                    if self._uses_notice_chimes(audio_msg) and (not interrupted or self._chime_off_after_interrupt):
                        if interrupted:
                            logger.info("Notice speech was interrupted; still playing ding_off.")
                        self._hold_chime()
                        try:
                            self._play_notice_chime(self._chime_off, "ding_off")
                        finally:
                            self._release_chime()
                    elif self._uses_notice_chimes(audio_msg):
                        logger.info(
                            "Notice chime skipped (ding_off): speech was interrupted "
                            "and notice_chime_off_after_interrupt is off."
                        )

                    self.currently_speaking_event.clear()
    
                else:
                    logger.warning(f"AudioPlayer: Received empty audio message or no text: {audio_len, audio_msg}")

            except queue.Empty:
                pass  # No audio to play right now

            except Exception as e:
                logger.exception(f"AudioPlayer: Unexpected error in run loop: {e}")
                time.sleep(self.pause_time)  # small sleep here to prevent tight loop on persistent error
        logger.info("AudioPlayer thread finished.")

    @staticmethod
    def _log_armed_chime(label: str, clip: NoticeChime | None) -> None:
        detail = describe_notice_chime(clip)
        if clip is None:
            logger.info(f"Notice chime {label} not armed: {detail}.")
        else:
            logger.success(f"Notice chime {label} armed: {detail}.")

    def _play_notice_chime(self, clip: NoticeChime | None, label: str) -> None:
        """Play one notice chime through to the end.

        Missing clips are logged and skipped. Playback ignores mic barge-in so
        a speaker echo cannot erase the ding. A failure here does not drop the
        spoken line.
        """
        if clip is None:
            logger.error(f"Notice chime skipped ({label}): not loaded.")
            return
        if clip.audio.size == 0:
            logger.error(f"Notice chime skipped ({label}): empty clip at {clip.source}.")
            return

        playback = with_chime_edges(clip, self._chime_lead_s, self._chime_tail_s)
        seconds = len(clip.audio) / clip.sample_rate if clip.sample_rate else 0.0
        logger.success(
            f"PLAYING notice chime {label} from {clip.source}: "
            f"{clip.sample_rate} Hz, {len(clip.audio)} samples, {seconds:.2f}s"
        )
        try:
            interrupted, percentage = self._play_chime_audio(playback)
        except Exception:  # noqa: BLE001 - a bad chime must not drop the spoken line
            logger.exception(f"Notice chime failed ({label}) from {clip.source}; continuing without that chime.")
            return

        if percentage <= 0:
            logger.error(
                f"Notice chime produced no audio ({label}) from {clip.source}: "
                f"{percentage}% played, interrupted={interrupted}."
            )
        elif interrupted:
            logger.warning(
                f"Notice chime cut short ({label}) from {clip.source} at {percentage}%. Continuing the notice."
            )
        else:
            logger.success(f"PLAYED notice chime {label} from {clip.source} at {percentage}%")

    def _uses_notice_chimes(self, audio_msg: AudioMessage) -> bool:
        """Notice lines chime, including an Announcer line whose flag was dropped."""
        return bool(audio_msg.notice or audio_msg.speaker == SPEAKER_ANNOUNCER)

    def _play_chime_audio(self, playback: NoticeChime) -> tuple[bool, int]:
        """Play a chime. Local sounddevice pauses the mic around that call."""
        play_chime = getattr(self.audio_io, "play_notice_chime", None)
        if callable(play_chime):
            return play_chime(playback.audio, playback.sample_rate)
        self.audio_io.start_speaking(playback.audio, playback.sample_rate, interruptible=False)
        return self.audio_io.measure_percentage_spoken(len(playback.audio), playback.sample_rate)

    def _hold_chime(self) -> None:
        """Keep the mic from treating this chime as the user talking."""
        if self._chime_hold_event is not None:
            self._chime_hold_event.set()

    def _release_chime(self) -> None:
        if self._chime_hold_event is not None:
            self._chime_hold_event.clear()

    def _clear_audio_queue(self) -> None:
        """Clears the audio output queue and resets the speaking event.

        This is called when an interruption occurs to ensure no stale audio messages remain.
        """

        logger.debug("AudioPlayer: Clearing audio queue due to interruption.")
        self.currently_speaking_event.clear()
        # with self.audio_output_queue.mutex:
        #     self.audio_output_queue.queue.clear()
        try:
            while True:
                self.audio_output_queue.get_nowait()
        except queue.Empty:
            pass

    def clip_interrupted_sentence(self, generated_text: str, percentage_played: float) -> str:
        """
        Clips the generated text based on the percentage of audio played before interruption.
        Args:
            generated_text (str): The full text that was being spoken.
            percentage_played (float): The percentage of the audio that was played before interruption.
        Returns:
            str: The clipped text that corresponds to the percentage of audio played.
        """
        tokens = generated_text.split()
        percentage_played = max(0.0, min(100.0, float(percentage_played)))  # Ensure percentage_played is within 0-100
        words_to_print = round((percentage_played / 100) * len(tokens))
        text = " ".join(tokens[:words_to_print])
        return text
