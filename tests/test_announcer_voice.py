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
from glados.audio_io.sounddevice_io import fill_output_buffer, finalize_spoken_playback
from glados.core.audio_data import SPEAKER_ANNOUNCER, SPEAKER_GLADOS, AudioMessage, tts_dialog_role
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados, GladosConfig, wait_for_startup_notice
from glados.core.speech_player import SpeechPlayer
from glados.core.spoken_line import SpokenLine
from glados.core.tts_synthesizer import TextToSpeechSynthesizer
from glados.observability import ObservabilityBus
from glados.TTS.announcer import (
    DEFAULT_ANNOUNCER_MODEL,
    AnnouncerVoiceUnavailableError,
    require_announcer_voice,
    resolve_announcer_model_path,
    try_load_announcer_voice,
)
from glados.TTS.piper_config import piper_config_candidates, resolve_piper_config_path
from glados.TTS.tts_glados import SpeechSynthesizer
from glados.utils.resources import find_project_root, resolve_repo_path, resource_path


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
        percentages: list[int] | None = None,
    ) -> None:
        self.played: list[tuple[int | None, str]] = []
        self.clips: list[tuple[int | None, int]] = []
        self._interrupts = list(interrupts or [])
        self._percentages = list(percentages or [])

    def start_speaking(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
    ) -> None:
        self.played.append((sample_rate, text))
        self.clips.append((sample_rate, len(audio_data)))

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


def test_require_announcer_voice_refuses_a_missing_model() -> None:
    with pytest.raises(AnnouncerVoiceUnavailableError, match="does not fall back"):
        require_announcer_voice(None)


def test_require_announcer_voice_uses_sidecar_length_scale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = tmp_path / "announcer.onnx"
    model.write_bytes(b"not-a-real-onnx")
    config = _piper_json(16000, 9)
    config["inference"]["length_scale"] = 0.42
    (tmp_path / "announcer.onnx.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr("glados.TTS.tts_glados.ort.InferenceSession", lambda *_a, **_k: _FakeSession())
    monkeypatch.setattr("glados.TTS.tts_glados.Phonemizer", _FakePhonemizer)

    voice = require_announcer_voice(str(model))

    assert voice.config.length_scale == 0.42
    assert voice.sample_rate == 16000


def test_say_announcer_flag_is_plain_speech() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "glados" / "cli.py").read_text(encoding="utf-8")
    say_body = source.split("def say(", 1)[1].split("\ndef ", 1)[0]

    assert "--announcer" in source
    assert "require_announcer_voice" in say_body
    assert "SpeechSynthesizer()" in say_body
    assert "SpokenTextConverter" in say_body
    assert "ding" not in say_body
    assert "play_notice_chime" not in say_body
    assert "return say(args.text, args.config, announcer=args.announcer)" in source


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
    assert loaded.announcement_followup == "Oh. It's you."
    assert loaded.announcement_followup_delay_s == 1.0


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


class _AnnouncementHost:
    def __init__(
        self,
        announcement: str | None,
        followup: str | None,
        delay_s: float = 1.0,
    ) -> None:
        self.announcement = announcement
        self.announcement_followup = followup
        self.announcement_followup_delay_s = delay_s
        self.interruptible = True
        self.processing_active_event = threading.Event()
        self.startup_notice_done = threading.Event()
        self.startup_notice_done.set()
        self.tts_queue: queue.Queue[str | SpokenLine] = queue.Queue()


def _announcement_host(
    announcement: str | None,
    followup: str | None,
    delay_s: float = 1.0,
) -> _AnnouncementHost:
    return _AnnouncementHost(announcement, followup, delay_s)


def test_play_announcement_is_a_notice_line() -> None:
    host = _announcement_host(
        "All neural network modules are now loaded. System Operational.",
        None,
    )
    Glados.play_announcement(host)  # type: ignore[arg-type]
    item = host.tts_queue.get_nowait()

    assert item == SpokenLine(host.announcement, notice=True, ends_startup=True)
    assert host.tts_queue.empty()
    assert host.processing_active_event.is_set()
    assert not host.startup_notice_done.is_set()


def test_play_announcement_queues_glados_followup_after_the_notice() -> None:
    host = _announcement_host(
        "All neural network modules are now loaded. System Operational.",
        "  Oh. It's you.  ",
    )
    Glados.play_announcement(host)  # type: ignore[arg-type]

    assert host.tts_queue.get_nowait() == SpokenLine(
        "All neural network modules are now loaded. System Operational.",
        notice=True,
        ends_startup=False,
    )
    assert host.tts_queue.get_nowait() == SpokenLine(
        "Oh. It's you.",
        notice=False,
        ends_startup=True,
        playback_delay_s=1.0,
    )
    assert host.tts_queue.empty()
    assert not host.startup_notice_done.is_set()


def test_followup_delay_is_skipped_without_a_notice_or_when_set_to_zero() -> None:
    no_notice = _announcement_host(None, "Oh. It's you.", delay_s=1.0)
    Glados.play_announcement(no_notice)  # type: ignore[arg-type]
    assert no_notice.tts_queue.get_nowait() == SpokenLine(
        "Oh. It's you.",
        notice=False,
        ends_startup=True,
        playback_delay_s=0.0,
    )

    zero_delay = _announcement_host("System Operational.", "Oh. It's you.", delay_s=0.0)
    Glados.play_announcement(zero_delay)  # type: ignore[arg-type]
    assert zero_delay.tts_queue.get_nowait().playback_delay_s == 0.0
    followup = zero_delay.tts_queue.get_nowait()
    assert followup == SpokenLine("Oh. It's you.", notice=False, ends_startup=True, playback_delay_s=0.0)


def test_empty_announcement_followup_is_skipped() -> None:
    host = _announcement_host("System Operational.", "   ")
    Glados.play_announcement(host)  # type: ignore[arg-type]

    assert host.tts_queue.get_nowait() == SpokenLine("System Operational.", notice=True, ends_startup=True)
    assert host.tts_queue.empty()


def test_wait_for_startup_notice_returns_when_the_announcement_already_finished() -> None:
    done = threading.Event()
    done.set()

    assert wait_for_startup_notice(done, 0.01) is True


def test_wait_for_startup_notice_waits_until_the_announcement_finishes() -> None:
    done = threading.Event()
    messages: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is not None:
            messages.append(str(record["message"]))

    def _release() -> None:
        time.sleep(0.05)
        done.set()

    sink_id = logger.add(_sink, level="SUCCESS")
    try:
        threading.Thread(target=_release, daemon=True).start()
        assert wait_for_startup_notice(done, 2.0) is True
    finally:
        logger.remove(sink_id)

    assert any("Waiting for the startup announcement before opening the microphone." == text for text in messages)
    assert any("Startup announcement finished. Opening the microphone." == text for text in messages)


def test_wait_for_startup_notice_opens_the_microphone_after_timeout() -> None:
    done = threading.Event()
    messages: list[str] = []

    def _sink(message: object) -> None:
        record = getattr(message, "record", None)
        if record is not None:
            messages.append(str(record["message"]))

    sink_id = logger.add(_sink, level="ERROR")
    try:
        assert wait_for_startup_notice(done, 0.05) is False
    finally:
        logger.remove(sink_id)

    assert any("Startup announcement did not finish. Opening the microphone anyway." == text for text in messages)


def test_speak_notice_queues_announcer_line_and_ignores_blank() -> None:
    class _Host:
        def __init__(self) -> None:
            self.processing_active_event = threading.Event()
            self.tts_queue: queue.Queue[str | SpokenLine] = queue.Queue()

    host = _Host()
    Glados.speak_notice(host, "   ")  # type: ignore[arg-type]
    assert host.tts_queue.empty()

    host.announcement_followup = "Oh. It's you."
    Glados.speak_notice(host, " Chamber lockdown. ")  # type: ignore[arg-type]
    assert host.tts_queue.get_nowait() == SpokenLine("Chamber lockdown.", notice=True, ends_startup=False)
    assert host.tts_queue.empty()


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
    incoming.put(SpokenLine("All neural network modules are now loaded.", notice=True, ends_startup=False))
    incoming.put(SpokenLine("Oh. It's you.", notice=False, ends_startup=True, playback_delay_s=1.0))
    incoming.put("The cake is a lie.")

    messages = _collect(outgoing, 3)
    shutdown.set()
    worker.join(timeout=2)

    assert [message.text for message in messages] == [
        "All neural network modules are now loaded.",
        "Oh. It's you.",
        "The cake is a lie.",
    ]
    assert messages[0].sample_rate == 16000
    assert messages[1].sample_rate == 22050
    assert messages[2].sample_rate == 22050
    assert messages[0].speaker == SPEAKER_ANNOUNCER
    assert messages[1].speaker == SPEAKER_GLADOS
    assert messages[2].speaker == SPEAKER_GLADOS
    assert tts_dialog_role({"speaker": messages[1].speaker}) == "GLaDOS"
    assert messages[0].notice is True
    assert messages[1].notice is False
    assert messages[2].notice is False
    assert messages[0].ends_startup is False
    assert messages[1].ends_startup is True
    assert messages[2].ends_startup is False
    assert messages[0].playback_delay_s == 0.0
    assert messages[1].playback_delay_s == 1.0
    assert messages[2].playback_delay_s == 0.0
    assert announcer.calls == ["All neural network modules are now loaded."]
    assert conversation.calls == ["Oh. It's you.", "The cake is a lie."]


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



def _play_one(
    message: AudioMessage,
    *,
    interrupts: list[bool] | None = None,
    extra: list[AudioMessage] | None = None,
    startup_notice_done: threading.Event | None = None,
    tts_muted: bool = False,
    percentages: list[int] | None = None,
    audio: _FakeAudio | None = None,
) -> tuple[_FakeAudio, queue.Queue[AudioMessage]]:
    if audio is None:
        audio = _FakeAudio(interrupts, percentages)
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
        startup_notice_done=startup_notice_done,
    )
    outgoing.put(message)
    for item in extra or []:
        outgoing.put(item)
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()

    deadline = time.time() + 2.0
    while time.time() < deadline and len(audio.clips) < 1 and not (tts_muted and startup_notice_done and startup_notice_done.is_set()):
        time.sleep(0.01)
    shutdown.set()
    worker.join(timeout=2)
    return audio, outgoing


def test_startup_notice_done_is_set_after_the_notice() -> None:
    done = threading.Event()
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
            ends_startup=True,
        ),
        startup_notice_done=done,
    )

    assert done.is_set()
    assert audio.clips == [(16000, 4)]


def test_microphone_waits_until_the_glados_followup_finishes() -> None:
    audio = _FakeAudio()
    done = threading.Event()
    flags_at_set: list[int] = []
    original_set = done.set

    def _set() -> None:
        flags_at_set.append(len(audio.clips))
        original_set()

    done.set = _set  # type: ignore[method-assign]
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
        startup_notice_done=done,
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            speaker=SPEAKER_ANNOUNCER,
            notice=True,
            ends_startup=False,
        )
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(5, dtype=np.float32),
            text="Oh. It's you.",
            sample_rate=22050,
            speaker=SPEAKER_GLADOS,
            notice=False,
            ends_startup=True,
        )
    )
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()
    assert done.wait(2.0)
    shutdown.set()
    worker.join(timeout=2)

    assert flags_at_set == [2]
    assert audio.clips == [(16000, 4), (22050, 5)]
    play_events = [event for event in bus.snapshot() if event.kind == "play"]
    assert [tts_dialog_role(event.meta) for event in play_events] == ["Announcer", "GLaDOS"]


def test_followup_waits_one_second_after_the_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    audio = _FakeAudio()
    done = threading.Event()
    slept: list[float] = []
    clips_at_delay: list[list[tuple[int | None, int]]] = []
    mic_open_during_delay: list[bool] = []

    def _sleep(seconds: float) -> None:
        slept.append(seconds)
        if seconds == 1.0:
            clips_at_delay.append(list(audio.clips))
            mic_open_during_delay.append(done.is_set())

    monkeypatch.setattr("glados.core.speech_player.time.sleep", _sleep)
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    player = SpeechPlayer(
        audio_io=audio,  # type: ignore[arg-type]
        audio_output_queue=outgoing,
        conversation_store=ConversationStore(),
        tts_sample_rate=22050,
        shutdown_event=shutdown,
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        pause_time=0.01,
        startup_notice_done=done,
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            speaker=SPEAKER_ANNOUNCER,
            notice=True,
        )
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(5, dtype=np.float32),
            text="Oh. It's you.",
            sample_rate=22050,
            speaker=SPEAKER_GLADOS,
            ends_startup=True,
            playback_delay_s=1.0,
        )
    )
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()
    assert done.wait(2.0)
    shutdown.set()
    worker.join(timeout=2)

    assert slept.count(1.0) == 1
    assert clips_at_delay == [[(16000, 4)]]
    assert mic_open_during_delay == [False]
    assert audio.clips == [(16000, 4), (22050, 5)]
    assert done.is_set()


def test_interrupted_notice_still_plays_the_startup_followup() -> None:
    audio = _FakeAudio(interrupts=[True])
    done = threading.Event()
    outgoing: queue.Queue[AudioMessage] = queue.Queue()
    shutdown = threading.Event()
    player = SpeechPlayer(
        audio_io=audio,  # type: ignore[arg-type]
        audio_output_queue=outgoing,
        conversation_store=ConversationStore(),
        tts_sample_rate=22050,
        shutdown_event=shutdown,
        currently_speaking_event=threading.Event(),
        processing_active_event=threading.Event(),
        pause_time=0.01,
        startup_notice_done=done,
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            speaker=SPEAKER_ANNOUNCER,
            notice=True,
        )
    )
    outgoing.put(
        AudioMessage(
            audio=np.ones(5, dtype=np.float32),
            text="Oh. It's you.",
            sample_rate=22050,
            speaker=SPEAKER_GLADOS,
            ends_startup=True,
        )
    )
    worker = threading.Thread(target=player.run, daemon=True)
    worker.start()
    assert done.wait(2.0)
    shutdown.set()
    worker.join(timeout=2)

    assert audio.clips == [(16000, 4), (22050, 5)]


def test_conversation_line_does_not_finish_the_startup_notice() -> None:
    done = threading.Event()
    audio, _outgoing = _play_one(
        AudioMessage(audio=np.ones(4, dtype=np.float32), text="The cake is a lie.", sample_rate=22050),
        startup_notice_done=done,
    )

    assert audio.clips == [(22050, 4)]
    assert not done.is_set()


def test_muted_notice_still_finishes_startup() -> None:
    done = threading.Event()
    audio, _outgoing = _play_one(
        AudioMessage(
            audio=np.ones(4, dtype=np.float32),
            text="System Operational.",
            sample_rate=16000,
            notice=True,
            ends_startup=True,
        ),
        startup_notice_done=done,
        tts_muted=True,
    )

    assert done.is_set()
    assert audio.clips == []


def test_short_clip_is_not_stopped_in_the_buffer_that_contains_it() -> None:
    clip = np.linspace(0.1, 0.9, 10, dtype=np.float32)

    first, position, stop, interrupted = fill_output_buffer(clip, 0, 32, stop_requested=False)
    assert stop is False
    assert interrupted is False
    assert position == 10
    np.testing.assert_array_equal(first[:10], clip)
    assert np.all(first[10:] == 0)

    second, position, stop, interrupted = fill_output_buffer(clip, position, 32, stop_requested=False)
    assert stop is True
    assert interrupted is False
    assert position == 10
    assert np.all(second == 0)

    silenced, _, stop, interrupted = fill_output_buffer(clip, 0, 32, stop_requested=True)
    assert stop is True
    assert interrupted is True
    assert np.all(silenced == 0)


def test_failed_stream_with_no_frames_is_not_reported_as_played() -> None:
    assert finalize_spoken_playback(0, 8000, False, stream_failed=True) == (True, 0)
    assert finalize_spoken_playback(40, 80, False, stream_failed=False) == (False, 50)


def test_installed_layout_resolves_to_the_checkout() -> None:
    repo = Path(__file__).resolve().parents[1]
    installed = repo / ".venv" / "Lib" / "site-packages" / "glados" / "utils" / "resources.py"
    assert find_project_root(installed) == repo
    assert find_project_root(repo / "src" / "glados" / "utils" / "resources.py") == repo


def test_repo_path_falls_back_to_cwd_when_package_root_misses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "models" / "TTS" / "announcer.onnx"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"not-a-real-model")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "glados.utils.resources.resource_path",
        lambda relative: tmp_path / "missing-root" / relative,
    )

    assert resolve_repo_path("models/TTS/announcer.onnx") == wav


def test_cli_start_log_reports_entry(tmp_path: Path) -> None:
    cli_source = (Path(__file__).resolve().parents[1] / "src" / "glados" / "cli.py").read_text(encoding="utf-8")

    assert "tui=False" in cli_source
    assert "log_cli_start(" in cli_source
    assert "glados start entry:" in cli_source
    assert "notice_chime" not in cli_source


def test_cli_and_tui_open_the_microphone_after_startup() -> None:
    root = Path(__file__).resolve().parents[1]
    cli_source = (root / "src" / "glados" / "cli.py").read_text(encoding="utf-8")
    tui_source = (root / "src" / "glados" / "tui.py").read_text(encoding="utf-8")
    engine_source = (root / "src" / "glados" / "core" / "engine.py").read_text(encoding="utf-8")

    assert "Glados.from_config" in cli_source
    assert "log_cli_start(" in cli_source
    assert 'command == "tui"' in cli_source
    assert "Glados.from_config" in tui_source
    assert "notice_chime" not in engine_source
    wait_at = engine_source.index("wait_for_startup_notice(self.startup_notice_done")
    listen_at = engine_source.index("self.audio_io.start_listening()")
    assert wait_at < listen_at
