"""Startup announcement uses a local Announcer Piper voice when configured."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import queue
import threading
import time
from typing import Any

from loguru import logger
import numpy as np
from numpy.typing import NDArray
import pytest
import soundfile as sf

from glados.audio_io import sounddevice_io
from glados.audio_io.sounddevice_io import (
    SoundDeviceAudioIO,
    capture_should_yield_to_chime,
    fill_output_buffer,
    finalize_spoken_playback,
)
from glados.core.audio_data import SPEAKER_ANNOUNCER, SPEAKER_GLADOS, AudioMessage, tts_dialog_role
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados, GladosConfig
from glados.core.notice_chimes import (
    DEFAULT_NOTICE_CHIME_OFF,
    DEFAULT_NOTICE_CHIME_ON,
    NoticeChime,
    load_notice_chime,
    with_chime_edges,
)
from glados.core.speech_listener import SpeechListener
from glados.core.speech_player import SpeechPlayer
from glados.core.spoken_line import SpokenLine
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from glados.observability import ObservabilityBus
from glados.TTS.announcer import DEFAULT_ANNOUNCER_MODEL, resolve_announcer_model_path, try_load_announcer_voice
from glados.TTS.piper_config import piper_config_candidates, resolve_piper_config_path
from glados.TTS.tts_glados import SpeechSynthesizer
from glados.utils.resources import resource_path


class _FakeVoice:
    def __init__(self, name: str, sample_rate: int) -> None:
        self.name = name
        self.sample_rate = sample_rate
        self.calls: list[str] = []

    def generate_speech_audio(self, text: str) -> NDArray[np.float32]:
        self.calls.append(text)
        return np.ones(8, dtype=np.float32)


class _IdentityConverter:
    def text_to_spoken(self, text: str) -> str:
        return text


class _FakeAudio:
    def __init__(
        self,
        interrupts: list[bool] | None = None,
        hold: threading.Event | None = None,
        percentages: list[int] | None = None,
    ) -> None:
        self.played: list[tuple[int | None, str]] = []
        self.clips: list[tuple[int | None, int]] = []
        self.interruptible_flags: list[bool] = []
        self.hold_during_start: list[bool] = []
        self._interrupts = list(interrupts or [])
        self._percentages = list(percentages or [])
        self._hold = hold

    def start_speaking(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
        interruptible: bool = True,
    ) -> None:
        self.played.append((sample_rate, text))
        self.clips.append((sample_rate, len(audio_data)))
        self.interruptible_flags.append(interruptible)
        self.hold_during_start.append(bool(self._hold is not None and self._hold.is_set()))

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        interrupted = self._interrupts.pop(0) if self._interrupts else False
        if self._percentages:
            return interrupted, self._percentages.pop(0)
        return interrupted, 0 if interrupted else 100

    def stop_speaking(self) -> None:
        return None

    def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
        return queue.Queue()

    def stop_listening(self) -> None:
        return None


def _minimal_glados_yaml(announcer_line: str | None) -> str:
    announcer = "null" if announcer_line is None else json.dumps(announcer_line)
    return f"""
Glados:
  llm_model: "llama3.2"
  completion_url: "http://localhost:11434/api/chat"
  api_key: null
  interruptible: true
  audio_io: "sounddevice"
  asr_engine: "tdt"
  wake_word: null
  voice: "glados"
  announcement: "All neural network modules are now loaded. System Operational."
  announcer_model_path: {announcer}
  personality_preprompt:
    - system: "You are GLaDOS."
"""


def _piper_json(sample_rate: int, bos_id: int) -> dict[str, Any]:
    return {
        "audio": {"sample_rate": sample_rate},
        "espeak": {"voice": "en-us"},
        "num_symbols": 256,
        "num_speakers": 1,
        "phoneme_id_map": {"^": [bos_id], "_": [0], "$": [2], "h": [20]},
        "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8},
    }


def _write_piper_pair(directory: Path, sample_rate: int = 22050, bos_id: int = 9) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "announcer.onnx"
    model.write_bytes(b"not-a-real-onnx")
    config = directory / "announcer.onnx.json"
    config.write_text(json.dumps(_piper_json(sample_rate, bos_id)), encoding="utf-8")
    return model


class _FakeSession:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None


class _FakePhonemizer:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        return None


def test_resolve_piper_config_prefers_standard_onnx_json(tmp_path: Path) -> None:
    model = tmp_path / "announcer.onnx"
    standard = tmp_path / "announcer.onnx.json"
    classic = tmp_path / "announcer.json"
    model.write_bytes(b"x")
    standard.write_text("{}", encoding="utf-8")
    classic.write_text("{}", encoding="utf-8")

    assert resolve_piper_config_path(model) == standard
    assert piper_config_candidates(model) == (standard, classic)


def test_resolve_piper_config_falls_back_to_classic_json(tmp_path: Path) -> None:
    model = tmp_path / "glados.onnx"
    classic = tmp_path / "glados.json"
    model.write_bytes(b"x")
    classic.write_text("{}", encoding="utf-8")

    assert resolve_piper_config_path(model) == classic


def test_speech_synthesizer_uses_onnx_json_phoneme_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _write_piper_pair(tmp_path, sample_rate=16000, bos_id=9)
    phonemes = tmp_path / "phoneme_to_id.pkl"
    phonemes.write_bytes(pickle.dumps({"^": [1], "_": [0], "$": [2]}))
    monkeypatch.setattr("glados.TTS.tts_glados.ort.InferenceSession", lambda *_a, **_k: _FakeSession())
    monkeypatch.setattr("glados.TTS.tts_glados.Phonemizer", _FakePhonemizer)

    synthesizer = SpeechSynthesizer(model_path=model, phoneme_path=phonemes, use_config_phoneme_map=True)

    assert synthesizer.sample_rate == 16000
    assert list(synthesizer.id_map["^"]) == [9]


def test_speech_synthesizer_keeps_pickle_map_for_glados(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "glados.onnx"
    model.write_bytes(b"not-a-real-onnx")
    (tmp_path / "glados.json").write_text(json.dumps(_piper_json(22050, 9)), encoding="utf-8")
    phonemes = tmp_path / "phoneme_to_id.pkl"
    phonemes.write_bytes(pickle.dumps({"^": [1], "_": [0], "$": [2], "h": [20]}))
    monkeypatch.setattr("glados.TTS.tts_glados.ort.InferenceSession", lambda *_a, **_k: _FakeSession())
    monkeypatch.setattr("glados.TTS.tts_glados.Phonemizer", _FakePhonemizer)

    synthesizer = SpeechSynthesizer(model_path=model, phoneme_path=phonemes)

    assert synthesizer.sample_rate == 22050
    assert list(synthesizer.id_map["^"]) == [1]


def test_missing_announcer_path_falls_back_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("SpeechSynthesizer should not load when the Announcer path is missing")

    monkeypatch.setattr("glados.TTS.announcer.SpeechSynthesizer", _boom)

    assert try_load_announcer_voice(None) is None
    assert try_load_announcer_voice("") is None
    assert try_load_announcer_voice("   ") is None
    assert try_load_announcer_voice("/no/such/announcer.onnx") is None


def test_missing_announcer_config_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "announcer.onnx"
    model.write_bytes(b"not-a-real-onnx")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("SpeechSynthesizer should not load without a Piper config")

    monkeypatch.setattr("glados.TTS.announcer.SpeechSynthesizer", _boom)

    assert try_load_announcer_voice(str(model)) is None


def test_announcer_load_uses_model_phoneme_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _write_piper_pair(tmp_path)
    captured: dict[str, Any] = {}

    def _capture(
        model_path: Path,
        phoneme_path: Path | None = None,
        speaker_id: int | None = None,
        **kwargs: object,
    ) -> str:
        captured["model_path"] = model_path
        captured["phoneme_path"] = phoneme_path
        captured["kwargs"] = kwargs
        return "loaded"

    monkeypatch.setattr("glados.TTS.announcer.SpeechSynthesizer", _capture)

    assert try_load_announcer_voice(str(model)) == "loaded"
    assert captured["model_path"] == model
    assert captured["kwargs"]["use_config_phoneme_map"] is True


def test_announcer_load_failure_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _write_piper_pair(tmp_path)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("onnx session failed")

    monkeypatch.setattr("glados.TTS.announcer.SpeechSynthesizer", _boom)

    assert try_load_announcer_voice(str(model)) is None


def test_config_reads_announcer_path_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "glados_config.yaml"
    config_file.write_text(_minimal_glados_yaml(DEFAULT_ANNOUNCER_MODEL), encoding="utf-8")
    monkeypatch.delenv("GLADOS_ANNOUNCER_MODEL", raising=False)

    loaded = GladosConfig.from_yaml(config_file)
    assert loaded.voice == "glados"
    assert loaded.announcer_model_path == "models/TTS/announcer.onnx"

    monkeypatch.setenv("GLADOS_ANNOUNCER_MODEL", "models/TTS/custom-announcer.onnx")
    overridden = GladosConfig.from_yaml(config_file)
    assert overridden.announcer_model_path == "models/TTS/custom-announcer.onnx"

    monkeypatch.setenv("GLADOS_ANNOUNCER_MODEL", "  ")
    disabled = GladosConfig.from_yaml(config_file)
    assert disabled.announcer_model_path is None


def test_shipped_config_uses_in_repo_announcer_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GLADOS_ANNOUNCER_MODEL", raising=False)
    loaded = GladosConfig.from_yaml(resource_path("configs/glados_config.yaml"))

    assert loaded.announcer_model_path == DEFAULT_ANNOUNCER_MODEL
    assert loaded.announcer_model_path is not None
    assert not Path(loaded.announcer_model_path).is_absolute()
    assert loaded.notice_chime_on == DEFAULT_NOTICE_CHIME_ON
    assert loaded.notice_chime_off == DEFAULT_NOTICE_CHIME_OFF
    assert loaded.notice_chime_on is not None
    assert not Path(loaded.notice_chime_on).is_absolute()
    assert loaded.notice_chime_off_after_interrupt is True


def test_relative_announcer_path_resolves_from_package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _write_piper_pair(tmp_path / "models" / "TTS")
    monkeypatch.setattr("glados.utils.resources.resource_path", lambda relative: tmp_path / relative)
    captured: dict[str, Path] = {}

    def _capture(model_path: Path, *_args: object, **_kwargs: object) -> str:
        captured["model_path"] = model_path
        return "loaded"

    monkeypatch.setattr("glados.TTS.announcer.SpeechSynthesizer", _capture)

    assert try_load_announcer_voice("models/TTS/announcer.onnx") == "loaded"
    assert captured["model_path"] == model
    assert resolve_announcer_model_path("models/TTS/announcer.onnx") == model


def test_build_tts_models_keeps_conversation_voice_and_loads_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "glados_config.yaml"
    config_file.write_text(_minimal_glados_yaml(DEFAULT_ANNOUNCER_MODEL), encoding="utf-8")
    monkeypatch.delenv("GLADOS_ANNOUNCER_MODEL", raising=False)
    config = GladosConfig.from_yaml(config_file)
    monkeypatch.setattr("glados.core.engine.get_speech_synthesizer", lambda voice: f"conversation:{voice}")
    monkeypatch.setattr("glados.core.engine.try_load_announcer_voice", lambda path: f"notice:{path}")

    conversation, notice = Glados._build_tts_models(config)

    assert conversation == "conversation:glados"
    assert notice == "notice:models/TTS/announcer.onnx"


def test_play_announcement_is_a_notice_line() -> None:
    class _Host:
        def __init__(self) -> None:
            self.announcement = "All neural network modules are now loaded. System Operational."
            self.interruptible = True
            self.processing_active_event = threading.Event()
            self.tts_queue: queue.Queue[str | SpokenLine] = queue.Queue()

    host = _Host()
    Glados.play_announcement(host)  # type: ignore[arg-type]
    item = host.tts_queue.get_nowait()

    assert item == SpokenLine(host.announcement, notice=True)
    assert host.processing_active_event.is_set()


def test_speak_notice_queues_announcer_line_and_ignores_blank() -> None:
    class _Host:
        def __init__(self) -> None:
            self.processing_active_event = threading.Event()
            self.tts_queue: queue.Queue[str | SpokenLine] = queue.Queue()

    host = _Host()
    Glados.speak_notice(host, "   ")  # type: ignore[arg-type]
    assert host.tts_queue.empty()

    Glados.speak_notice(host, " Chamber lockdown. ")  # type: ignore[arg-type]
    assert host.tts_queue.get_nowait() == SpokenLine("Chamber lockdown.", notice=True)


def test_config_defaults_announcer_path_to_repo_drop_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = _minimal_glados_yaml(None).replace("  announcer_model_path: null\n", "")
    config_file = tmp_path / "glados_config.yaml"
    config_file.write_text(text, encoding="utf-8")
    monkeypatch.delenv("GLADOS_ANNOUNCER_MODEL", raising=False)

    loaded = GladosConfig.from_yaml(config_file)

    assert loaded.announcer_model_path == DEFAULT_ANNOUNCER_MODEL


def _collect(audio_queue: queue.Queue[AudioMessage], count: int) -> list[AudioMessage]:
    messages: list[AudioMessage] = []
    deadline = time.time() + 2.0
    while len(messages) < count and time.time() < deadline:
        try:
            messages.append(audio_queue.get(timeout=0.05))
        except queue.Empty:
            continue
    return messages


def test_startup_line_uses_announcer_then_conversation_returns() -> None:
    conversation = _FakeVoice("glados", 22050)
    announcer = _FakeVoice("announcer", 16000)
    incoming: queue.Queue[str | SpokenLine] = queue.Queue()
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    synthesizer = TextToSpeechSynthesizer(
        tts_input_queue=incoming,
        audio_output_queue=outgoing,
        tts_model=conversation,
        stc_instance=_IdentityConverter(),  # type: ignore[arg-type]
        shutdown_event=shutdown,
        pause_time=0.01,
        notice_model=announcer,
    )
    worker = threading.Thread(target=synthesizer.run, daemon=True)
    worker.start()
    incoming.put(SpokenLine("All neural network modules are now loaded.", notice=True))
    incoming.put("The cake is a lie.")

    messages = _collect(outgoing, 2)
    shutdown.set()
    worker.join(timeout=2)

    assert [message.text for message in messages] == [
        "All neural network modules are now loaded.",
        "The cake is a lie.",
    ]
    assert messages[0].sample_rate == 16000
    assert messages[1].sample_rate == 22050
    assert messages[0].speaker == SPEAKER_ANNOUNCER
    assert messages[1].speaker == SPEAKER_GLADOS
    assert messages[0].notice is True
    assert messages[1].notice is False
    assert announcer.calls == ["All neural network modules are now loaded."]
    assert conversation.calls == ["The cake is a lie."]


def test_notice_line_falls_back_to_conversation_voice() -> None:
    conversation = _FakeVoice("glados", 22050)
    incoming: queue.Queue[str | SpokenLine] = queue.Queue()
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    synthesizer = TextToSpeechSynthesizer(
        tts_input_queue=incoming,
        audio_output_queue=outgoing,
        tts_model=conversation,
        stc_instance=_IdentityConverter(),  # type: ignore[arg-type]
        shutdown_event=shutdown,
        pause_time=0.01,
        notice_model=None,
    )
    worker = threading.Thread(target=synthesizer.run, daemon=True)
    worker.start()
    incoming.put(SpokenLine("System Operational.", notice=True))

    messages = _collect(outgoing, 1)
    shutdown.set()
    worker.join(timeout=2)

    assert len(messages) == 1
    assert messages[0].sample_rate == 22050
    assert messages[0].speaker == SPEAKER_GLADOS
    assert messages[0].notice is True
    assert conversation.calls == ["System Operational."]


def test_player_uses_line_sample_rate() -> None:
    audio = _FakeAudio()
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    speaking = threading.Event()
    processing = threading.Event()
    player = SpeechPlayer(
        audio_io=audio,  # type: ignore[arg-type]
        audio_output_queue=outgoing,
        conversation_store=ConversationStore(),
        tts_sample_rate=22050,
        shutdown_event=shutdown,
        currently_speaking_event=speaking,
        processing_active_event=processing,
        pause_time=0.01,
    )
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()
    outgoing.put(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
        )
    )

    deadline = time.time() + 2.0
    while not audio.played and time.time() < deadline:
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=2)

    assert audio.played == [(16000, "")]


def test_play_event_labels_announcer_and_fallback_stays_glados() -> None:
    assert tts_dialog_role({"speaker": SPEAKER_ANNOUNCER}) == "Announcer"
    assert tts_dialog_role({"speaker": SPEAKER_GLADOS}) == "GLaDOS"
    assert tts_dialog_role({}) == "GLaDOS"
    assert tts_dialog_role(None) == "GLaDOS"

    audio = _FakeAudio()
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    bus = ObservabilityBus()
    player = SpeechPlayer(
        audio_io=audio,  # type: ignore[arg-type]
        audio_output_queue=outgoing,
        conversation_store=ConversationStore(),
        tts_sample_rate=22050,
        shutdown_event=shutdown,
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        pause_time=0.01,
        observability_bus=bus,
    )
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()
    outgoing.put(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="All neural network modules are now loaded.",
            speaker=SPEAKER_ANNOUNCER,
        )
    )
    outgoing.put(AudioMessage(audio=np.ones(4, dtype=np.float32), text="The cake is a lie."))

    deadline = time.time() + 2.0
    play_events = []
    while time.time() < deadline:
        play_events = [event for event in bus.snapshot() if event.kind == "play"]
        if len(play_events) >= 2:
            break
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=2)

    assert [tts_dialog_role(event.meta) for event in play_events] == ["Announcer", "GLaDOS"]
    assert [event.message for event in play_events] == [
        "All neural network modules are now loaded.",
        "The cake is a lie.",
    ]


def _chime(length: int, sample_rate: int, source: str = "models/SFX/ding_on.wav") -> NoticeChime:
    return NoticeChime(audio=np.ones(length, dtype=np.float32), sample_rate=sample_rate, source=source)


def _play_one(
    message: AudioMessage,
    *,
    chime_on: NoticeChime | None = None,
    chime_off: NoticeChime | None = None,
    interrupts: list[bool] | None = None,
    extra: list[AudioMessage] | None = None,
    chime_off_after_interrupt: bool = True,
    chime_gap_s: float = 0.0,
    chime_lead_s: float = 0.0,
    chime_tail_s: float = 0.0,
    hold: threading.Event | None = None,
    tts_muted: bool = False,
    percentages: list[int] | None = None,
    audio: _FakeAudio | None = None,
) -> tuple[_FakeAudio, queue.Queue[AudioMessage]]:
    if audio is None:
        audio = _FakeAudio(interrupts, hold, percentages)
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    muted = threading.Event()
    if tts_muted:
        muted.set()
    player = SpeechPlayer(
        audio_io=audio,  # type: ignore[arg-type]
        audio_output_queue=outgoing,
        conversation_store=ConversationStore(),
        tts_sample_rate=22050,
        shutdown_event=shutdown,
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        pause_time=0.01,
        tts_muted_event=muted,
        chime_on=chime_on,
        chime_off=chime_off,
        chime_off_after_interrupt=chime_off_after_interrupt,
        chime_hold_event=hold,
        chime_lead_s=chime_lead_s,
        chime_tail_s=chime_tail_s,
        chime_gap_s=chime_gap_s,
    )
    outgoing.put(message)
    for item in extra or []:
        outgoing.put(item)
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()

    deadline = time.time() + 2.0
    while time.time() < deadline and len(audio.clips) < 1:
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=2)
    return audio, outgoing


def test_notice_line_plays_chime_speech_chime() -> None:
    chime_on = _chime(2, 44100)
    chime_off = _chime(3, 44100)
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        ),
        chime_on=chime_on,
        chime_off=chime_off,
    )

    assert audio.clips == [(44100, 2), (16000, 4), (44100, 3)]
    assert audio.interruptible_flags == [False, True, False]


def test_conversation_line_skips_chimes() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(audio=np.ones(4, dtype=np.float32), text="The cake is a lie.", sample_rate=22050),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100),
    )

    assert audio.clips == [(22050, 4)]
    assert audio.played == [(22050, "")]


def test_missing_chimes_still_speak_the_notice() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        )
    )

    assert audio.clips == [(16000, 4)]


def test_chime_report_of_interrupt_does_not_drop_the_notice() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(8, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        ),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100, "models/SFX/ding_off.wav"),
        interrupts=[True],
    )

    assert audio.clips == [(44100, 2), (16000, 8), (44100, 3)]


def test_interrupt_during_speech_still_plays_ding_off() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(8, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        ),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100),
        interrupts=[False, True],
    )

    assert audio.clips == [(44100, 2), (16000, 8), (44100, 3)]


def test_ding_off_can_be_configured_off_after_interrupt() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(8, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        ),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100, "models/SFX/ding_off.wav"),
        interrupts=[False, True],
        chime_off_after_interrupt=False,
    )

    assert audio.clips == [(44100, 2), (16000, 8)]


def test_load_notice_chime_downmixes_stereo_and_skips_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("glados.utils.resources.resource_path", lambda relative: tmp_path / relative)

    assert load_notice_chime(None) is None
    assert load_notice_chime("  ") is None
    assert load_notice_chime("models/SFX/ding_on.wav") is None

    stereo = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    wav_path = tmp_path / "models" / "SFX" / "ding_on.wav"
    wav_path.parent.mkdir(parents=True)
    sf.write(str(wav_path), stereo, 44100, subtype="FLOAT")

    clip = load_notice_chime("models/SFX/ding_on.wav")

    assert clip is not None
    assert clip.sample_rate == 44100
    assert clip.audio.shape == (3,)
    np.testing.assert_allclose(clip.audio, np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_config_notice_chime_defaults_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = _minimal_glados_yaml(None).replace("  announcer_model_path: null\n", "")
    config_file = tmp_path / "glados_config.yaml"
    config_file.write_text(text, encoding="utf-8")
    monkeypatch.delenv("GLADOS_ANNOUNCER_MODEL", raising=False)
    monkeypatch.delenv("GLADOS_NOTICE_CHIME_ON", raising=False)
    monkeypatch.delenv("GLADOS_NOTICE_CHIME_OFF", raising=False)

    loaded = GladosConfig.from_yaml(config_file)
    assert loaded.notice_chime_on == DEFAULT_NOTICE_CHIME_ON
    assert loaded.notice_chime_off == DEFAULT_NOTICE_CHIME_OFF

    monkeypatch.setenv("GLADOS_NOTICE_CHIME_ON", "  ")
    monkeypatch.setenv("GLADOS_NOTICE_CHIME_OFF", "models/SFX/custom_off.wav")
    overridden = GladosConfig.from_yaml(config_file)
    assert overridden.notice_chime_on is None
    assert overridden.notice_chime_off == "models/SFX/custom_off.wav"
    assert loaded.notice_chime_off_after_interrupt is True


def test_short_clip_is_not_stopped_in_the_buffer_that_contains_it() -> None:
    """The ding must be committed before CallbackStop, matching sounddevice.play."""
    ding = np.linspace(0.1, 0.9, 10, dtype=np.float32)

    first, position, stop, interrupted = fill_output_buffer(
        ding, 0, 32, stop_requested=False, interruptible=False
    )
    assert stop is False
    assert interrupted is False
    assert position == 10
    np.testing.assert_array_equal(first[:10], ding)
    assert np.all(first[10:] == 0)

    second, position, stop, interrupted = fill_output_buffer(
        ding, position, 32, stop_requested=False, interruptible=False
    )
    assert stop is True
    assert interrupted is False
    assert position == 10
    assert np.all(second == 0)

    stopped, _, stop, interrupted = fill_output_buffer(
        ding, 0, 32, stop_requested=True, interruptible=False
    )
    assert stop is False
    assert interrupted is False
    np.testing.assert_array_equal(stopped[:10], ding)

    silenced, _, stop, interrupted = fill_output_buffer(
        ding, 0, 32, stop_requested=True, interruptible=True
    )
    assert stop is True
    assert interrupted is True
    assert np.all(silenced == 0)


def test_chime_edges_keep_the_ding_away_from_the_buffer_ends() -> None:
    clip = NoticeChime(audio=np.ones(10, dtype=np.float32), sample_rate=100, source="ding_on.wav")
    edged = with_chime_edges(clip, lead_s=0.02, tail_s=0.05)

    assert edged.audio.shape == (17,)
    assert np.all(edged.audio[:2] == 0)
    assert np.all(edged.audio[-5:] == 0)
    np.testing.assert_array_equal(edged.audio[2:12], clip.audio)
    assert edged.source == clip.source


def test_notice_chime_logs_and_holds_the_mic(monkeypatch: pytest.MonkeyPatch) -> None:
    messages: list[str] = []
    slept: list[float] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is not None:
            messages.append(str(record["message"]))

    monkeypatch.setattr("glados.core.speech_player.time.sleep", lambda seconds: slept.append(seconds))
    sink_id = logger.add(_sink, level="INFO")
    hold = threading.Event()
    try:
        audio, _outgoing = _play_one(
            AudioMessage(
                audio=np.ones(4, dtype=np.float32),
                text="System Operational.",
                sample_rate=16000,
                notice=True,
            ),
            chime_on=_chime(2, 44100),
            chime_off=_chime(3, 44100, "models/SFX/ding_off.wav"),
            hold=hold,
            chime_gap_s=0.1,
        )
    finally:
        logger.remove(sink_id)

    assert slept == [0.1]
    assert audio.hold_during_start == [True, False, True]
    assert audio.interruptible_flags == [False, True, False]
    started = "PLAYING notice chime ding_on from models/SFX/ding_on.wav: 44100 Hz, 2 samples, 0.00s"
    assert started in messages
    assert "PLAYED notice chime ding_on from models/SFX/ding_on.wav at 100%" in messages
    assert "PLAYED notice chime ding_off from models/SFX/ding_off.wav at 100%" in messages
    assert any(
        "AudioPlayer received:" in text and "notice=True" in text and "ding_on_loaded=True" in text
        for text in messages
    )
    assert any(text.startswith("notice chime start:") and "shape=(2,)" in text for text in messages)
    assert any(text.startswith("notice chime start:") and "shape=(3,)" in text for text in messages)
    assert not hold.is_set()


def test_muted_notice_skips_chimes() -> None:
    messages: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is not None:
            messages.append(str(record["message"]))

    sink_id = logger.add(_sink, level="INFO")
    try:
        audio, _outgoing = _play_one(
            AudioMessage(
                audio=np.ones(4, dtype=np.float32),
                text="System Operational.",
                sample_rate=16000,
                notice=True,
            ),
            chime_on=_chime(2, 44100),
            chime_off=_chime(3, 44100, "models/SFX/ding_off.wav"),
            tts_muted=True,
        )
    finally:
        logger.remove(sink_id)

    assert audio.clips == []
    assert any(text == "Notice chimes skipped: TTS is muted." for text in messages)


def test_chime_hold_and_asr_mute_do_not_cut_playback() -> None:
    class _Io:
        def __init__(self) -> None:
            self.stopped = 0
            self.samples: queue.Queue[tuple[NDArray[np.float32], bool]] = queue.Queue()

        def get_sample_queue(self) -> queue.Queue[tuple[NDArray[np.float32], bool]]:
            return self.samples

        def stop_speaking(self) -> None:
            self.stopped += 1

        def stop_listening(self) -> None:
            return None

    io = _Io()
    hold = threading.Event()
    listener = SpeechListener(
        audio_io=io,  # type: ignore[arg-type]
        llm_queue=queue.Queue(),
        shutdown_event=threading.Event(),
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        asr_model=object(),  # type: ignore[arg-type]
        wake_word=None,
        pause_time=0.01,
        chime_hold_event=hold,
    )
    listener.currently_speaking_event.set()
    sample = np.zeros(8, dtype=np.float32)
    hold.set()
    listener._manage_pre_activation_buffer(sample, True)
    assert io.stopped == 0
    assert listener._recording_started is False

    hold.clear()
    listener._manage_pre_activation_buffer(sample, True)
    assert io.stopped == 1

    muted_io = _Io()
    muted = threading.Event()
    muted.set()
    shutdown = threading.Event()
    muted_listener = SpeechListener(
        audio_io=muted_io,  # type: ignore[arg-type]
        llm_queue=queue.Queue(),
        shutdown_event=shutdown,
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        asr_model=object(),  # type: ignore[arg-type]
        wake_word=None,
        pause_time=0.01,
        asr_muted_event=muted,
    )
    muted_listener.currently_speaking_event.set()
    muted_io.samples.put((sample, True))
    worker = threading.Thread(target=muted_listener.run, daemon=True)
    worker.start()
    deadline = time.time() + 2.0
    while time.time() < deadline and not muted_io.samples.empty():
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=2)

    assert muted_io.stopped == 0


def test_announcer_speaker_chimes_even_when_notice_flag_is_false() -> None:
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            speaker=SPEAKER_ANNOUNCER,
            notice=False,
        ),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100),
    )

    assert audio.clips == [(44100, 2), (16000, 4), (44100, 3)]


def test_zero_percent_chime_is_not_logged_as_played() -> None:
    messages: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is not None:
            messages.append(str(record["message"]))

    sink_id = logger.add(_sink, level="INFO")
    try:
        _play_one(
            AudioMessage(
                audio=np.ones(4, dtype=np.float32),
                text="System Operational.",
                sample_rate=16000,
                notice=True,
            ),
            chime_on=_chime(2, 44100),
            chime_off=_chime(3, 44100, "models/SFX/ding_off.wav"),
            percentages=[0],
        )
    finally:
        logger.remove(sink_id)

    assert any(text.startswith("Notice chime produced no audio (ding_on)") for text in messages)
    assert not any(text.startswith("PLAYED notice chime ding_on") for text in messages)
    assert any(text.startswith("PLAYED notice chime ding_off") for text in messages)


def test_player_uses_backend_chime_playback_when_the_device_provides_it() -> None:
    class _RoutedAudio(_FakeAudio):
        def __init__(self) -> None:
            super().__init__()
            self.routed: list[tuple[int | None, int]] = []

        def play_notice_chime(
            self,
            audio_data: NDArray[np.float32],
            sample_rate: int | None = None,
        ) -> tuple[bool, int]:
            self.routed.append((sample_rate, len(audio_data)))
            return False, 100

    routed = _RoutedAudio()
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
        ),
        chime_on=_chime(2, 44100),
        chime_off=_chime(3, 44100),
        audio=routed,
    )

    assert audio is routed
    assert routed.routed == [(44100, 2), (44100, 3)]
    assert audio.clips == [(16000, 4)]
    assert audio.interruptible_flags == [True]


def test_failed_chime_stream_is_not_reported_as_played() -> None:
    assert finalize_spoken_playback(0, 8000, False, stream_failed=True, chime=True) == (True, 0)
    assert finalize_spoken_playback(0, 8000, False, stream_failed=True, chime=False) == (False, 0)
    assert finalize_spoken_playback(40, 80, False, stream_failed=False, chime=True) == (False, 50)


def test_notice_chime_keeps_the_input_stream_open_and_retries_silence() -> None:
    io = SoundDeviceAudioIO.__new__(SoundDeviceAudioIO)
    mic = object()
    io.input_stream = mic
    io._device_lock = threading.RLock()
    io._pending_audio = None
    io._pending_sample_rate = 44100
    events: list[object] = []

    def start_speaking(
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
        interruptible: bool = True,
    ) -> None:
        events.append(("start", interruptible, tuple(audio_data.shape)))
        io._pending_audio = audio_data
        io._pending_sample_rate = int(sample_rate or 0)

    def measure(total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        events.append(("measure", total_samples, io.input_stream is mic))
        return False, 0

    def blocking(audio_data: NDArray[np.float32], sample_rate: int) -> tuple[bool, int]:
        events.append(("blocking", sample_rate, len(audio_data)))
        return False, 100

    io.start_speaking = start_speaking  # type: ignore[method-assign]
    io.measure_percentage_spoken = measure  # type: ignore[method-assign]
    io._blocking_chime_write = blocking  # type: ignore[method-assign]

    clip = np.ones(8, dtype=np.float32)
    interrupted, percentage = io.play_notice_chime(clip, 44100)

    assert (interrupted, percentage) == (False, 100)
    assert io.input_stream is mic
    assert events == [
        ("start", False, (8,)),
        ("measure", 8, True),
        ("start", False, (8,)),
        ("measure", 8, True),
        ("blocking", 44100, 8),
    ]


def test_notice_chime_callback_success_does_not_touch_the_mic() -> None:
    io = SoundDeviceAudioIO.__new__(SoundDeviceAudioIO)
    mic = object()
    io.input_stream = mic
    io._device_lock = threading.RLock()
    io._pending_audio = None
    io._pending_sample_rate = 44100
    events: list[object] = []

    def start_speaking(
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
        interruptible: bool = True,
    ) -> None:
        events.append("start")
        io._pending_audio = audio_data
        io._pending_sample_rate = int(sample_rate or 0)

    def measure(total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        events.append("measure")
        return False, 100

    io.start_speaking = start_speaking  # type: ignore[method-assign]
    io.measure_percentage_spoken = measure  # type: ignore[method-assign]
    io._blocking_chime_write = lambda *_args: events.append("blocking")  # type: ignore[method-assign]

    interrupted, percentage = io.play_notice_chime(np.ones(4, dtype=np.float32), 44100)

    assert (interrupted, percentage) == (False, 100)
    assert events == ["start", "measure"]
    assert io.input_stream is mic


def test_start_speaking_during_chime_leaves_the_clip_in_place() -> None:
    io = SoundDeviceAudioIO.__new__(SoundDeviceAudioIO)
    pending = np.ones(4, dtype=np.float32)
    io._is_playing = True
    io._playback_interruptible = False
    io._pending_audio = pending
    io._pending_sample_rate = 44100

    io.start_speaking(np.zeros(2, dtype=np.float32), 16000, interruptible=True)

    assert io._is_playing is True
    assert io._playback_interruptible is False
    assert io._pending_audio is pending
    assert io._pending_sample_rate == 44100


def test_chime_playback_suppresses_mic_vad_and_mic_open_waits() -> None:
    assert capture_should_yield_to_chime(True, False) is True
    assert capture_should_yield_to_chime(True, True) is False
    assert capture_should_yield_to_chime(False, False) is False

    io = SoundDeviceAudioIO.__new__(SoundDeviceAudioIO)
    io._is_playing = True
    io._device_lock = threading.RLock()
    opened: list[bool] = []

    def _sleep(_seconds: float) -> None:
        io._is_playing = False

    io._open_input_stream = lambda: opened.append(io._is_playing)  # type: ignore[method-assign]
    original_sleep = sounddevice_io.time.sleep
    sounddevice_io.time.sleep = _sleep
    try:
        io.start_listening()
    finally:
        sounddevice_io.time.sleep = original_sleep

    assert opened == [False]


def test_cli_and_tui_both_pass_loaded_chimes_through_from_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cli_source = (root / "src" / "glados" / "cli.py").read_text(encoding="utf-8")
    tui_source = (root / "src" / "glados" / "tui.py").read_text(encoding="utf-8")
    engine_source = (root / "src" / "glados" / "core" / "engine.py").read_text(encoding="utf-8")

    assert "Glados.from_config" in cli_source
    assert "Glados.from_config" in tui_source
    assert "notice_chime_on=notice_chime_on" in engine_source
    assert "notice_chime_off=notice_chime_off" in engine_source
    assert "chime_on=self._notice_chime_on" in engine_source
    assert "chime_off=self._notice_chime_off" in engine_source
