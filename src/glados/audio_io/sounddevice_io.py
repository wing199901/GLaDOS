"""Local microphone and speaker backend implemented with sounddevice."""

import queue
import threading
import time

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import sounddevice as sd  # type: ignore

from . import VAD
from .base import AudioIO
from .resample import resample as resample_audio


def fill_output_buffer(
    audio: NDArray[np.float32],
    position: int,
    frames: int,
    *,
    stop_requested: bool,
    interruptible: bool,
) -> tuple[NDArray[np.float32], int, bool, bool]:
    """Fill one speaker block.

    Returns ``(block, new_position, stop_stream, interrupted)``.

    The block that finishes the clip does not stop the stream. The next
    block is silence and then stops. ``sounddevice.play()`` works the same
    way: it returns from the callback that wrote the last samples, and only
    the following callback stops the stream.

    Raising CallbackStop in the same callback that wrote a ~0.2s ding drops
    that buffer on Windows before the speakers play it. A multi-second
    announcement still sounds normal because only its tail buffer is lost.
    A direct ``sd.play`` of the same wav is audible because it does not stop
    in that callback.
    """
    block = np.zeros(frames, dtype=np.float32)
    if frames <= 0:
        return block, position, True, False
    if stop_requested and interruptible:
        return block, position, True, True
    remaining = len(audio) - position
    if remaining <= 0:
        return block, position, True, False
    chunk = min(frames, remaining)
    block[:chunk] = audio[position : position + chunk]
    return block, position + chunk, False, False


def capture_should_yield_to_chime(is_playing: bool, interruptible: bool) -> bool:
    """The mic callback should not run VAD while an uninterruptible chime is playing.

    That callback and the speaker callback share the audio thread. A cold VAD
    pass during startup is slow enough to starve a ~0.2s ding. The announcement
    is long enough that the same stall only clips its start.
    """
    return is_playing and not interruptible


def finalize_spoken_playback(
    position: int,
    total: int,
    interrupted: bool,
    *,
    stream_failed: bool,
    chime: bool,
) -> tuple[bool, int]:
    """Return ``(interrupted, percentage)`` after an output stream ends.

    A chime whose stream errors before any frame is written must not look like
    a finished play. That used to come back as ``interrupted=False`` and ``0``,
    and the speech player logged it as played.
    """
    if total <= 0:
        return False, 100
    percentage = min(int(position / total * 100), 100)
    if chime and stream_failed and position <= 0:
        return True, 0
    return interrupted, percentage


class SoundDeviceAudioIO(AudioIO):
    """Audio I/O implementation using sounddevice for both input and output.

    This class provides an implementation of the AudioIO interface using the
    sounddevice library to interact with system audio devices. It handles
    real-time audio capture with voice activity detection and audio playback.
    """

    SAMPLE_RATE: int = 16000  # Sample rate for input stream
    VAD_SIZE: int = 32  # Milliseconds of sample for Voice Activity Detection (VAD)
    VAD_THRESHOLD: float = 0.8  # Threshold for VAD detection

    def __init__(self, vad_threshold: float | None = None) -> None:
        """Initialize the sounddevice audio I/O.

        Args:
            vad_threshold: Threshold for VAD detection (default: 0.8)

        Raises:
            ImportError: If the sounddevice module is not available
            ValueError: If invalid parameters are provided
        """
        if vad_threshold is None:
            self.vad_threshold = self.VAD_THRESHOLD
        else:
            self.vad_threshold = vad_threshold

        if not 0 <= self.vad_threshold <= 1:
            raise ValueError("VAD threshold must be between 0 and 1")

        self._vad_model = VAD()

        self._sample_queue: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()
        self.input_stream: sd.InputStream | None = None
        self._is_playing = False
        self._playback_thread = None
        self._stop_event = threading.Event()
        self._pending_audio: NDArray[np.float32] | None = None
        self._pending_sample_rate: int = self.SAMPLE_RATE
        self._playback_interruptible = True
        # start_listening must not open the mic while an output stream is active.
        # play_announcement() is queued before run(), so the first ding can already
        # be in measure_percentage_spoken when the microphone opens.
        self._device_lock = threading.RLock()

    def start_listening(self) -> None:
        """Start capturing audio from the system microphone.

        Creates and starts a sounddevice InputStream that continuously captures
        audio from the default input device. Each audio chunk is processed with
        the VAD model and placed in the sample queue.

        Raises:
            RuntimeError: If the audio input stream cannot be started
            sd.PortAudioError: If there's an issue with the audio hardware
        """
        announced_wait = False
        waited = 0.0
        while True:
            if self._is_playing and waited < 8.0:
                if not announced_wait:
                    logger.success("Delaying microphone open until current playback finishes.")
                    announced_wait = True
                time.sleep(0.05)
                waited += 0.05
                continue
            if self._is_playing:
                logger.error("Opening the microphone while playback is still marked active.")
            with self._device_lock:
                if self._is_playing and waited < 8.0:
                    continue
                self._open_input_stream()
                return

    def _open_input_stream(self) -> None:
        if self.input_stream is not None:
            self.stop_listening()

        def audio_callback(
            indata: NDArray[np.float32],
            frames: int,
            time: sd.CallbackStop,
            status: sd.CallbackFlags,
        ) -> None:
            """Process incoming audio data and put it in the queue with VAD confidence.

            Parameters:
                indata: Input audio data from the sounddevice stream
                frames: Number of audio frames in the current chunk
                time: Timing information for the audio callback
                status: Status flags for the audio callback

            Notes:
                - Copies and squeezes the input data to ensure single-channel processing
                - Applies voice activity detection to determine speech presence
                - Puts processed audio samples and VAD confidence into a thread-safe queue
            """
            if status:
                # Log any errors for debugging
                logger.debug(f"Audio callback status: {status}")

            if capture_should_yield_to_chime(self._is_playing, self._playback_interruptible):
                return

            data = np.array(indata).copy().squeeze()  # Reduce to single channel if necessary
            vad_value = self._vad_model(np.expand_dims(data, 0))
            vad_confidence = vad_value > self.vad_threshold
            self._sample_queue.put((data, bool(vad_confidence)))

        try:
            self.input_stream = sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                callback=audio_callback,
                blocksize=int(self.SAMPLE_RATE * self.VAD_SIZE / 1000),
            )
            self.input_stream.start()
        except sd.PortAudioError as e:
            raise RuntimeError(f"Failed to start audio input stream: {e}") from e

    def stop_listening(self) -> None:
        """Stop capturing audio and clean up resources.

        Stops the input stream if it's active and releases associated resources.
        This method should be called when audio input is no longer needed or
        before application shutdown.
        """
        if self.input_stream is not None:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception as e:
                logger.error(f"Error stopping input stream: {e}")
            finally:
                self.input_stream = None

    def play_notice_chime(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
    ) -> tuple[bool, int]:
        """Play one notice chime with the calls that are audible while ASR is open.

        ``start_speaking`` then ``measure_percentage_spoken``, with the microphone
        input stream left running. A callback that finishes with no frames is
        retried with a blocking write. That 0% result used to come back as a
        clean completion.
        """
        logger.success(
            "PLAYING notice chime via start_speaking: "
            f"shape={getattr(audio_data, 'shape', None)} sr={sample_rate} "
            f"input_stream_open={self.input_stream is not None}"
        )
        # Hold the device lock across queue + playback. play_announcement() runs
        # before run() opens the mic, and opening an InputStream while this short
        # OutputStream is active aborts the ding. Speech still follows.
        with self._device_lock:
            self.start_speaking(audio_data, sample_rate, interruptible=False)
            prepared = self._pending_audio
            prepared_rate = self._pending_sample_rate
            interrupted, percentage = self.measure_percentage_spoken(len(audio_data), prepared_rate)
        if percentage <= 0:
            logger.success("Retrying notice chime after the output device settles.")
            time.sleep(0.3)
            with self._device_lock:
                self.start_speaking(audio_data, sample_rate, interruptible=False)
                prepared = self._pending_audio
                prepared_rate = self._pending_sample_rate
                interrupted, percentage = self.measure_percentage_spoken(len(audio_data), prepared_rate)
        if percentage <= 0:
            logger.error(
                "Notice chime callback finished with no audible frames "
                f"(interrupted={interrupted}, {percentage}%). Retrying with a blocking write."
            )
            if prepared is None or prepared.size == 0:
                return True, 0
            interrupted, percentage = self._blocking_chime_write(prepared, prepared_rate)
        if percentage <= 0:
            logger.error(
                f"Notice chime still produced no audio after retry ({percentage}%, interrupted={interrupted})."
            )
        else:
            logger.success(
                f"PLAYED notice chime via start_speaking: {percentage}% interrupted={interrupted} "
                f"input_stream_open={self.input_stream is not None}"
            )
        return interrupted, percentage

    def _blocking_chime_write(self, audio_data: NDArray[np.float32], sample_rate: int) -> tuple[bool, int]:
        """Write a chime without a callback. Used when the callback path plays nothing."""
        logger.success(
            f"PLAYING notice chime with blocking write: {len(audio_data)} samples, {sample_rate} Hz, "
            f"{len(audio_data) / sample_rate:.2f}s"
        )
        with self._device_lock:
            self._playback_interruptible = False
            self._is_playing = True
            try:
                with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32") as stream:
                    stream.write(np.asarray(audio_data, dtype=np.float32).reshape(-1, 1))
                    try:
                        latency = float(stream.latency)
                    except (TypeError, ValueError):
                        latency = 0.2
                    time.sleep(min(0.4, max(0.12, latency)))
            except (sd.PortAudioError, RuntimeError) as exc:
                logger.error(f"Notice chime blocking write failed: {exc}")
                return True, 0
            finally:
                self._is_playing = False
        return False, 100

    def start_speaking(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
        interruptible: bool = True,
    ) -> None:
        """Queue audio for playback through the system speakers.

        Stores audio data for playback via measure_percentage_spoken(), which
        uses a single OutputStream to both play and monitor progress. This avoids
        the race condition that occurs when sd.play() and a monitoring OutputStream
        run concurrently.

        Parameters:
            audio_data: The audio data to play as a numpy float32 array
            sample_rate: The sample rate of the audio data in Hz
            text: Optional text associated with the audio (not used by this implementation)

        Raises:
            ValueError: If audio_data is empty or not a valid numpy array
        """
        if not isinstance(audio_data, np.ndarray) or audio_data.size == 0:
            raise ValueError("Invalid audio data")

        if sample_rate is None:
            sample_rate = self.SAMPLE_RATE

        # A second start_speaking during a chime used to flip interruptible back
        # to True after stop_speaking returned, so the ding could be cut and the
        # spoken line still played. Leave the chime in place.
        if self._is_playing and not self._playback_interruptible:
            logger.success("Ignoring start_speaking during an uninterruptible notice chime.")
            return

        # Stop any existing playback and create a fresh stop event for this session
        self.stop_speaking()
        self._stop_event = threading.Event()
        self._playback_interruptible = interruptible
        # Mark playback before the device query. run() can open the microphone
        # while this clip is still being prepared, and that open aborts a ding.
        self._is_playing = True

        try:
            # Resample to the output device's native sample rate so PortAudio's
            # low-quality built-in sample-rate converter is never used. This avoids
            # the audible crackling/distortion that occurs when the TTS rate differs
            # from the device rate (e.g. 22050 Hz TTS out, 44100 Hz device).
            try:
                device_rate = int(sd.query_devices(kind="output")["default_samplerate"])
            except Exception as e:
                device_rate = 0
                logger.debug(f"Could not query output device sample rate: {e}")

            if device_rate > 0 and sample_rate != device_rate:
                logger.debug(f"Resampling audio {sample_rate} Hz -> {device_rate} Hz")
                audio_data = resample_audio(audio_data, sample_rate, device_rate)
                sample_rate = device_rate

            logger.debug(f"Playing audio with sample rate: {sample_rate} Hz, length: {len(audio_data)} samples")
            self._pending_audio = audio_data
            self._pending_sample_rate = sample_rate
        except Exception:
            self._is_playing = False
            self._pending_audio = None
            raise

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """Play queued audio. Holds the device lock so the mic cannot open mid-clip."""
        with self._device_lock:
            return self._measure_percentage_spoken(total_samples, sample_rate)

    def _measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        """
        Play queued audio and monitor playback progress with interrupt detection.

        Uses a single OutputStream to both play the audio stored by start_speaking()
        and track progress, avoiding the race condition from running sd.play() and a
        separate monitoring stream concurrently.

        Args:
            total_samples (int): Total number of samples in the audio data being played.
            sample_rate (int | None): Sample rate override; uses the value from start_speaking() if None.
        Returns:
            tuple[bool, int]: A tuple containing:
                - bool: True if playback was interrupted, False if completed normally
                - int: Percentage of audio played (0-100)
        """
        audio_data = self._pending_audio
        if audio_data is None:
            logger.error("measure_percentage_spoken: no audio was queued, so playback did not start.")
            return False, 0

        # Prefer the sample rate stored by start_speaking() -- that reflects any
        # device-rate resampling that happened, so the stream opens at the true
        # playback rate and the timeout arithmetic stays correct.
        if self._pending_sample_rate and self._pending_sample_rate > 0:
            sample_rate = self._pending_sample_rate
        elif sample_rate is None:
            sample_rate = self._pending_sample_rate

        if sample_rate is None or sample_rate <= 0:
            logger.error(f"Invalid sample rate {sample_rate}; playback did not start.")
            if self._pending_audio is audio_data:
                self._pending_audio = None
                self._is_playing = False
            return False, 0

        # Derive playback length from the actual buffer so a wrong caller-supplied
        # total_samples can't break the timeout or percentage math.
        effective_total = len(audio_data)
        if effective_total <= 0:
            logger.error("measure_percentage_spoken: queued audio is empty, so playback did not start.")
            if self._pending_audio is audio_data:
                self._pending_audio = None
                self._is_playing = False
            return False, 0

        position = 0
        interrupted = False
        stream_failed = False
        completion_event = threading.Event()
        # Capture current stop_event so a new start_speaking() call doesn't affect this session
        stop_event = self._stop_event

        def stream_callback(
            outdata: NDArray[np.float32], frames: int, time_info: object, status: sd.CallbackFlags
        ) -> None:
            """Fill the next output block and track completion or interruption."""
            nonlocal position, interrupted

            block, position, stop_stream, was_interrupted = fill_output_buffer(
                audio_data,
                position,
                frames,
                stop_requested=stop_event.is_set(),
                interruptible=self._playback_interruptible,
            )
            outdata[:, 0] = block
            if was_interrupted:
                interrupted = True
            if stop_stream:
                completion_event.set()
                raise sd.CallbackStop

        chime = not self._playback_interruptible
        try:
            logger.debug(f"Using sample rate: {sample_rate} Hz, total samples: {effective_total}")
            max_timeout = effective_total / sample_rate + 1
            playback_finished = threading.Event()
            if chime:
                logger.success(
                    f"PLAYING notice chime on output device: {effective_total} samples, "
                    f"{sample_rate} Hz, {effective_total / sample_rate:.2f}s, "
                    f"input_stream_open={self.input_stream is not None}"
                )

            def _playback_finished() -> None:
                playback_finished.set()

            with sd.OutputStream(
                callback=stream_callback,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                finished_callback=_playback_finished,
            ) as stream:
                completed = completion_event.wait(max_timeout)
                if not completed:
                    # Timeout: signal stop and mark as interrupted
                    stop_event.set()
                    interrupted = True
                    logger.debug("Audio playback timed out, forcing interruption")
                if chime:
                    # Keep the stream open for the device latency. finished_callback
                    # can fire when the callback stops, which is before a short ding
                    # has come out of the speakers. Closing then discards it.
                    try:
                        latency = float(stream.latency)
                    except (TypeError, ValueError):
                        latency = 0.2
                    hold_s = min(0.4, max(0.12, latency))
                    playback_finished.wait(hold_s)
                    time.sleep(hold_s)
                    if position > 0:
                        logger.success(
                            f"PLAYED notice chime on output device: {sample_rate} Hz, "
                            f"{min(int(position / effective_total * 100), 100)}%, interrupted={interrupted}, "
                            f"input_stream_open={self.input_stream is not None}"
                        )
                    else:
                        logger.error(
                            "Notice chime output stream ended before any frames were written "
                            f"(interrupted={interrupted}, input_stream_open={self.input_stream is not None})."
                        )

        except (sd.PortAudioError, RuntimeError) as exc:
            stream_failed = True
            if chime:
                logger.error(f"Notice chime output stream failed: {exc}")
            else:
                logger.debug(f"Audio stream already closed or invalid: {exc}")

        # Identity-checked teardown: only clear shared state if it still belongs to this
        # session, otherwise a new start_speaking() that ran concurrently could be wiped out.
        if self._pending_audio is audio_data:
            self._pending_audio = None
        if self._stop_event is stop_event:
            self._is_playing = False
        return finalize_spoken_playback(
            position,
            effective_total,
            interrupted,
            stream_failed=stream_failed,
            chime=chime,
        )

    def check_if_speaking(self) -> bool:
        """Check if audio is currently being played.

        Returns:
            bool: True if audio is currently playing, False otherwise
        """
        return self._is_playing

    def stop_speaking(self) -> None:
        """Stop audio playback and clean up resources.

        Signals the current playback session to stop by setting the stop event.
        The active OutputStream callback will detect this on its next invocation
        and raise CallbackStop to cleanly terminate the stream.
        """
        if self._is_playing and not self._playback_interruptible:
            logger.success("Ignoring stop_speaking during an uninterruptible notice chime.")
            return
        if self._is_playing:
            self._stop_event.set()
            self._is_playing = False

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        """Get the queue containing audio samples and VAD confidence.

        Returns:
            queue.Queue: A thread-safe queue containing tuples of
                        (audio_sample, vad_confidence)
        """
        return self._sample_queue

    def close(self) -> None:
        """Release local input and output resources."""
        self.stop_speaking()
        self.stop_listening()
