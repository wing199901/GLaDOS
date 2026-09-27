"""Startup announcement uses a local Announcer Piper voice when configured."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import queue
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

from glados.core.audio_data import SPEAKER_ANNOUNCER, SPEAKER_GLADOS, AudioMessage, tts_dialog_role
from glados.core.conversation_store import ConversationStore
from glados.core.engine import Glados, GladosConfig
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
    def __init__(self) -> None:
        self.played: list[tuple[int | None, str]] = []

    def start_speaking(
        self,
        audio_data: NDArray[np.float32],
        sample_rate: int | None = None,
        text: str = "",
    ) -> None:
        self.played.append((sample_rate, text))

    def measure_percentage_spoken(self, total_samples: int, sample_rate: int | None = None) -> tuple[bool, int]:
        return False, 100


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


def test_relative_announcer_path_resolves_from_package_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _write_piper_pair(tmp_path / "models" / "TTS")
    monkeypatch.setattr("glados.TTS.announcer.resource_path", lambda relative: tmp_path / relative)
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
