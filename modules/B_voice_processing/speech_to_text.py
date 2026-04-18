"""
Module 5 — Speech-to-Text (Whisper)
=====================================
Converts captured audio into text that can be passed to the
Intent Detection module.

Whisper (openai-whisper) is used because:
  • It is fully open-source and runs locally — no API key needed.
  • It handles accented English well (important for a diverse class).
  • The "base" model achieves good accuracy with ~1 s latency on CPU.
  • It accepts a WAV/MP3 file path OR a raw numpy float32 array.

Public API
----------
SpeechToText   (class)
    .transcribe(audio_path: str) -> str
        Transcribe a WAV/MP3 file; return cleaned text.

    .transcribe_array(audio: np.ndarray) -> str
        Transcribe a numpy float32 array directly (no temp file needed).

    .transcribe_bytes(audio_bytes: bytes) -> str
        Transcribe raw audio bytes (e.g., from Streamlit audio widget).

transcribe_file(path)  -> str   — module-level convenience wrapper
transcribe_array(arr)  -> str   — module-level convenience wrapper
get_last_transcribe_error()     — last failure reason when result is ""

Design notes
------------
- Audio is loaded with scipy and passed to Whisper as float32 numpy. This avoids
  relying on ffmpeg (Whisper's default path loader shells out to ffmpeg).
- The Whisper model is loaded once (lazy, on first call) and cached.
- fp16=False is forced on CPU to avoid a PyTorch warning.
- If Whisper fails, SpeechRecognition (Google Web API) is tried when a file path exists.
"""

from __future__ import annotations

import os
import re
import tempfile
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

from config import (
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    WHISPER_DEVICE,
    SAMPLE_RATE,
)


# ── Model singleton ──────────────────────────────────────────────────────
_whisper_model = None   # loaded lazily on first transcription call

# Set when transcribe/transcribe_array returns "" so UIs can explain why
_last_transcribe_error: str | None = None


def get_last_transcribe_error() -> str | None:
    """Human-readable reason the last call returned no text, or None."""
    return _last_transcribe_error


def probe_whisper_model() -> tuple[bool, str]:
    """
    Try to load the Whisper model (uses the module singleton cache).
    Returns (success, message) for diagnostics / health checks.
    """
    try:
        _get_model()
        return (
            True,
            f"Whisper model loaded OK (size={WHISPER_MODEL_SIZE!r}, device={WHISPER_DEVICE!r}).",
        )
    except Exception as e:
        return False, repr(e)


def _set_last_transcribe_error(msg: str | None) -> None:
    global _last_transcribe_error
    _last_transcribe_error = msg


def _configure_tls_trust_bundle() -> None:
    """
    Point OpenSSL/urllib at certifi's CA bundle so Whisper model downloads
    and SpeechRecognition's HTTPS calls work when the system store is wrong
    (common on some Python.org macOS installs or corporate proxies).
    """
    try:
        import certifi

        ca = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", ca)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", ca)
    except ImportError:
        pass


_configure_tls_trust_bundle()

def _get_model():
    """Load Whisper model once and cache it."""
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(
            WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
        )
    return _whisper_model


def _clean(text: str) -> str:
    """Strip artefacts, collapse whitespace, lowercase."""
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _load_wav_mono_float32_16k(path: str) -> np.ndarray:
    """Load WAV as mono float32 in [-1, 1] at SAMPLE_RATE (resample if needed)."""
    rate, data = wav.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:
        audio = data.astype(np.float32)
        m = float(np.max(np.abs(audio))) or 1.0
        if m > 1.5:
            audio /= m

    if rate != SAMPLE_RATE:
        n_new = max(1, int(len(audio) * SAMPLE_RATE / rate))
        audio = signal.resample(audio, n_new).astype(np.float32)

    return audio


def _is_effectively_silent(audio: np.ndarray) -> bool:
    return float(np.max(np.abs(audio))) < 1e-6


# ── Main class ───────────────────────────────────────────────────────────

class SpeechToText:
    """
    Whisper-based speech-to-text transcription.
    """

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a WAV/MP3 audio file.

        Returns cleaned text, or "" on failure (see get_last_transcribe_error()).
        """
        _set_last_transcribe_error(None)

        if not os.path.exists(audio_path):
            _set_last_transcribe_error(f"Audio file not found: {audio_path}")
            return ""

        try:
            audio = _load_wav_mono_float32_16k(audio_path)
        except Exception as e:
            fb = self._fallback_transcribe(audio_path, chained=f"Could not read WAV: {e!r}")
            return fb

        return self._transcribe_numpy(audio, fallback_wav_path=audio_path)

    def transcribe_array(self, audio: np.ndarray) -> str:
        """Transcribe a numpy float32 array (mono, ~16 kHz as produced by AudioCapture)."""
        _set_last_transcribe_error(None)
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if len(audio) == 0:
            _set_last_transcribe_error("No audio samples were captured.")
            return ""

        if _is_effectively_silent(audio):
            _set_last_transcribe_error(
                "Captured audio is silent. Check microphone permissions, input device, and volume."
            )
            return ""

        return self._transcribe_numpy(audio, fallback_wav_path=None)

    def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".wav") -> str:
        """Transcribe raw audio bytes (e.g., from Streamlit)."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _transcribe_numpy(
        self,
        audio: np.ndarray,
        fallback_wav_path: str | None,
    ) -> str:
        if _is_effectively_silent(audio):
            _set_last_transcribe_error(
                "Audio is effectively silent after loading. Check recording levels."
            )
            return ""

        whisper_exc: Exception | None = None
        try:
            model = _get_model()
            result = model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                fp16=(WHISPER_DEVICE != "cpu"),
            )
            text = _clean(result.get("text") or "")
            if text:
                return text
            _set_last_transcribe_error(
                "Whisper produced no text. Try speaking louder, closer to the mic, or a longer clip."
            )
        except Exception as e:
            whisper_exc = e

        if fallback_wav_path and os.path.exists(fallback_wav_path):
            chain = (
                str(whisper_exc)
                if whisper_exc
                else "Whisper returned empty transcript"
            )
            fb = self._fallback_transcribe(fallback_wav_path, chained=chain)
            if fb:
                _set_last_transcribe_error(None)
                return fb
        elif whisper_exc:
            _set_last_transcribe_error(
                f"Whisper failed ({whisper_exc!r}). No WAV file available for Google fallback."
            )
        return ""

    def _fallback_transcribe(self, audio_path: str, chained: str = "") -> str:
        """Google Web Speech API via SpeechRecognition (requires network). """
        try:
            import speech_recognition as sr
            recogniser = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                buf = recogniser.record(source)
            text = recogniser.recognize_google(buf)
            _set_last_transcribe_error(None)
            return _clean(text)
        except Exception as e:
            detail = repr(e)
            if "CERTIFICATE_VERIFY_FAILED" in detail or "SSLCertVerificationError" in detail:
                detail += (
                    " — Try: pip install -U certifi; on python.org macOS builds run "
                    "'Install Certificates.command'; corporate networks may need the proxy root CA in certifi or the system keychain."
                )
            if chained:
                _set_last_transcribe_error(f"{chained}; Google fallback failed: {detail}")
            else:
                _set_last_transcribe_error(f"Google Speech fallback failed: {detail}")
            return ""


_stt = SpeechToText()   # singleton


def transcribe_file(path: str) -> str:
    """Transcribe a WAV/MP3 file (module-level convenience)."""
    return _stt.transcribe(path)


def transcribe_array(audio: np.ndarray) -> str:
    """Transcribe a numpy array (module-level convenience)."""
    return _stt.transcribe_array(audio)
