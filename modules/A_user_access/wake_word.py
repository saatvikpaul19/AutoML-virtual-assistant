"""
Module 2 — Wake Word Detection
================================
Activates the assistant only when the user says (or types) the
configured trigger phrase, preventing accidental command execution.

Two detection strategies are provided:

Strategy A — Transcript-based (default, zero extra dependencies)
    The STT module transcribes a short audio clip; this module
    checks whether the transcript contains the wake word.
    Suitable for demo / course-project use.

Strategy B — Continuous background listener (optional upgrade)
    Runs a background thread that permanently listens for the wake
    word and sets a threading.Event when detected.  Requires
    sounddevice + Whisper already installed (same deps as modules 4/5).

Public API
----------
is_wake_word(transcript: str) -> bool
    Check a pre-transcribed string for the wake phrase.

detect_from_audio(audio_path: str) -> bool
    Transcribe an audio file and check for the wake phrase.

WakeWordListener   (class)
    Background-thread continuous listener.
    .start()  — begin listening
    .stop()   — stop background thread
    .detected — threading.Event; call .wait() or .is_set()

Design notes
------------
- Fuzzy matching (difflib) tolerates minor mis-transcriptions like
  "hey assistant" → "hey assistants" or "hey a system".
- The wake word check happens AFTER the user speaks a short clip,
  not in real-time; this is intentional for a course-project scope.
"""

import re
import threading
import difflib
import tempfile
import os
import time

from config import WAKE_WORD, WAKE_WORD_VARIANTS, SAMPLE_RATE, RECORD_DURATION, AUDIO_TMP_PATH


# ── Fuzzy match threshold ────────────────────────────────────────────────
_SIMILARITY_THRESHOLD = 0.72   # 0..1; lower = more permissive


# ── Helpers ──────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_contains(haystack: str, needle: str) -> bool:
    """
    Return True if `needle` appears in `haystack` with at least
    _SIMILARITY_THRESHOLD sequence-match ratio, accounting for
    speech-recognition transcription errors.
    """
    hay = _normalise(haystack)
    ned = _normalise(needle)

    # Exact substring check first (fast path)
    if ned in hay:
        return True

    # Sliding-window fuzzy match
    words_hay = hay.split()
    words_ned = ned.split()
    n = len(words_ned)

    for i in range(len(words_hay) - n + 1):
        window = " ".join(words_hay[i : i + n])
        ratio = difflib.SequenceMatcher(None, window, ned).ratio()
        if ratio >= _SIMILARITY_THRESHOLD:
            return True
    return False


# ── Public API — Strategy A (transcript-based) ────────────────────────────

def is_wake_word(transcript: str) -> bool:
    """
    Check whether a pre-transcribed string contains the wake word
    or any accepted variant.

    Parameters
    ----------
    transcript : str   — Raw text from the STT module

    Returns
    -------
    bool — True if wake word detected
    """
    for variant in WAKE_WORD_VARIANTS:
        if _fuzzy_contains(transcript, variant):
            return True
    return False


def detect_from_audio(audio_path: str) -> bool:
    """
    Transcribe `audio_path` via Whisper and check for the wake word.
    Returns True if the wake word is detected.

    Lazy-imports the STT module to avoid circular dependencies.
    """
    from modules.B_voice_processing.speech_to_text import SpeechToText
    stt = SpeechToText()
    transcript = stt.transcribe(audio_path)
    detected = is_wake_word(transcript)
    return detected


# ── Public API — Strategy B (background thread listener) ─────────────────

class WakeWordListener:
    """
    Continuously listens in a background thread and sets self.detected
    (a threading.Event) the moment the wake word is heard.

    Usage
    -----
    listener = WakeWordListener()
    listener.start()

    # In Streamlit: poll listener.detected.is_set() periodically
    if listener.detected.is_set():
        listener.detected.clear()   # reset for next time
        # → proceed to capture full command

    listener.stop()   # when shutting down
    """

    def __init__(self):
        self.detected = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background listening thread."""
        if self._thread and self._thread.is_alive():
            return   # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background listening thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _listen_loop(self):
        """
        Internal: continuously record SHORT clips, transcribe them,
        check for wake word.  Runs in background thread.
        """
        try:
            import sounddevice as sd
            import numpy as np
            import scipy.io.wavfile as wav
            from modules.B_voice_processing.speech_to_text import SpeechToText

            stt = SpeechToText()
            clip_duration = 2   # seconds — short for responsiveness

            while not self._stop_event.is_set():
                # Record a short clip
                # print("[DEBUG wake_word] Listening...")
                audio = sd.rec(

                    int(clip_duration * SAMPLE_RATE),
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                wav.write(tmp_path, SAMPLE_RATE, audio)

                # Transcribe and check
                transcript = stt.transcribe(tmp_path)
                os.unlink(tmp_path)

                if is_wake_word(transcript):
                    print(f"[DEBUG wake_word] DETECTED: '{transcript}'")
                    self.detected.set()
                else:
                    # print(f"[DEBUG wake_word] No wake word in: '{transcript}'")
                    pass


        except Exception as e:
            print(f"[DEBUG wake_word] _listen_loop ERROR: {e!r}")
            # If mic/library unavailable, silently exit thread
            pass

