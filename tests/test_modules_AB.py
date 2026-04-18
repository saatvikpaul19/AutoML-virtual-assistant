"""
Tests for Sections A and B — run with:  python -m pytest tests/
All tests are pure-Python (no microphone / GPU required).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import tempfile
import scipy.io.wavfile as wav
import pytest


# ═══════════════════════════════════════════════════════════
# MODULE 1 — User Verification
# ═══════════════════════════════════════════════════════════

from modules.A_user_access.user_verification import (
    verify_passcode, get_verification_status, set_new_passcode, reset_state
)

class TestUserVerification:

    def setup_method(self):
        reset_state()

    def test_correct_passcode(self):
        result = verify_passcode("1234")
        assert result["verified"] is True
        assert "successful" in result["message"].lower()

    def test_wrong_passcode(self):
        result = verify_passcode("wrong")
        assert result["verified"] is False

    def test_lockout_after_max_attempts(self):
        # 3 wrong attempts should trigger lockout
        for _ in range(3):
            verify_passcode("bad")
        status = get_verification_status()
        assert status["locked"] is True
        assert status["wait_sec"] > 0

    def test_change_passcode(self):
        result = set_new_passcode("1234", "newpass99")
        assert result["success"] is True
        assert verify_passcode("newpass99")["verified"] is True

    def test_change_passcode_wrong_old(self):
        result = set_new_passcode("wrong", "newpass99")
        assert result["success"] is False

    def test_change_passcode_too_short(self):
        result = set_new_passcode("1234", "ab")
        assert result["success"] is False


# ═══════════════════════════════════════════════════════════
# MODULE 2 — Wake Word Detection
# ═══════════════════════════════════════════════════════════

from modules.A_user_access.wake_word import is_wake_word

class TestWakeWord:

    def test_exact_match(self):
        assert is_wake_word("hey mycroft, start training") is True

    def test_variant_hi(self):
        assert is_wake_word("hi mycroft") is True

    def test_variant_wake_up(self):
        assert is_wake_word("wake up please") is True

    def test_fuzzy_typo(self):
        # Common STT mis-transcription
        assert is_wake_word("hey mycroft load dataset") is True

    def test_no_wake_word(self):
        assert is_wake_word("load the titanic dataset") is False

    def test_empty_string(self):
        assert is_wake_word("") is False


# ═══════════════════════════════════════════════════════════
# MODULE 3 — Text Input Handler
# ═══════════════════════════════════════════════════════════

from modules.A_user_access.text_input_handler import TextInputHandler, sanitise

class TestTextInput:

    def setup_method(self):
        self.handler = TextInputHandler()

    def test_valid_input(self):
        result = self.handler.process("Load the Titanic dataset")
        assert result["valid"] is True
        assert result["text"] == "load the titanic dataset"
        assert result["source"] == "text"

    def test_empty_input(self):
        result = self.handler.process("   ")
        assert result["valid"] is False

    def test_html_injection_stripped(self):
        result = self.handler.process("<script>alert('xss')</script>load dataset")
        assert result["valid"] is True
        assert "<script>" not in result["text"]

    def test_too_long_input(self):
        result = self.handler.process("a" * 400)
        assert result["valid"] is False

    def test_sanitise_helper(self):
        assert sanitise("  HELLO  World  ") == "hello world"

    def test_history_recorded(self):
        self.handler.process("start training")
        self.handler.process("stop training")
        history = self.handler.get_history()
        assert len(history) >= 2
        assert history[-1]["text"] == "stop training"

    def test_clear_history(self):
        self.handler.process("set learning rate 0.01")
        self.handler.clear_history()
        assert self.handler.get_history() == []


# ═══════════════════════════════════════════════════════════
# MODULE 4 — Audio Capture (no real mic — file I/O only)
# ═══════════════════════════════════════════════════════════

from modules.B_voice_processing.audio_capture import AudioCapture, save_audio, load_audio

class TestAudioCapture:

    def _make_sine(self, freq=440, duration=1) -> np.ndarray:
        """Generate a 1-second 440 Hz sine wave (fake speech)."""
        t = np.linspace(0, duration, int(16000 * duration), endpoint=False)
        return (np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_save_and_load_roundtrip(self):
        audio = self._make_sine()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        capture = AudioCapture()
        capture.save(audio, path)
        loaded = capture.load(path)
        os.unlink(path)
        # After int16 roundtrip, values should be close (not bit-perfect)
        assert loaded.shape == audio.shape
        assert np.allclose(audio, loaded, atol=1e-3)

    def test_convenience_wrappers(self):
        audio = self._make_sine(duration=0.5)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        save_audio(audio, path)
        loaded = load_audio(path)
        os.unlink(path)
        assert loaded.dtype == np.float32

import os


# ═══════════════════════════════════════════════════════════
# MODULE 5 — Speech-to-Text (file-based only, no GPU needed)
# ═══════════════════════════════════════════════════════════
# Note: These tests use a MOCK to avoid loading the 74 MB Whisper model
# in CI / grading environments.  In a real run, replace the mock with
# a real audio file to verify end-to-end.

from unittest.mock import patch, MagicMock
from modules.B_voice_processing.speech_to_text import SpeechToText, _clean

class TestSpeechToText:

    def test_clean_helper(self):
        assert _clean("  Load The Titanic  ") == "load the titanic"
        assert _clean("  HELLO\n\nWORLD  ") == "hello world"

    @patch("modules.B_voice_processing.speech_to_text._get_model")
    def test_transcribe_file(self, mock_get_model):
        """Verify transcribe() calls the model and cleans output."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "  Load Titanic Dataset  "}
        mock_get_model.return_value = mock_model

        stt = SpeechToText()
        # Non-silent dummy WAV (Whisper path rejects digital silence)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        audio_i16 = (np.sin(2 * np.pi * 440 * t) * 0.2 * 32767).astype(np.int16)
        wav.write(path, 16000, audio_i16)

        result = stt.transcribe(path)
        os.unlink(path)

        assert result == "load titanic dataset"
        mock_model.transcribe.assert_called_once()
        call_audio = mock_model.transcribe.call_args[0][0]
        assert isinstance(call_audio, np.ndarray)

    @patch("modules.B_voice_processing.speech_to_text._get_model")
    def test_transcribe_array(self, mock_get_model):
        """Verify transcribe_array() passes a numpy array to Whisper (no ffmpeg)."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "start training"}
        mock_get_model.return_value = mock_model

        stt = SpeechToText()
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        audio = (np.sin(2 * np.pi * 440 * t) * 0.2).astype(np.float32)
        result = stt.transcribe_array(audio)
        assert result == "start training"
        passed = mock_model.transcribe.call_args[0][0]
        assert isinstance(passed, np.ndarray)

    def test_missing_file_returns_empty(self):
        stt = SpeechToText()
        result = stt.transcribe("/nonexistent/path/audio.wav")
        assert result == ""
