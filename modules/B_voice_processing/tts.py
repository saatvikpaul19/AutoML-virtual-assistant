"""
Module 7 — Text-to-Speech (TTS)
=================================
Converts assistant response text into audible speech using Google
Text-to-Speech (gTTS).

gTTS is chosen because:
  • Free — no API key required.
  • Good quality English synthesis.
  • Returns MP3 bytes directly — easy to play in Streamlit.
  • Lightweight dependency.

Public API
----------
speak(text: str) -> str | None
    Generate speech audio for the given text.
    Returns the absolute path to a temporary .mp3 file, or None on failure.

is_tts_available() -> bool
    Returns True if gTTS is installed and importable.

Design notes
------------
- Audio files are cached in a temp directory keyed by a hash of the text,
  so repeated identical responses don't re-generate.
- Errors are caught silently — TTS is a non-critical enhancement.
  The assistant always returns text even if TTS fails.
- The caller (app.py) renders the audio with st.audio(path, autoplay=True).
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


# ── Cache directory ──────────────────────────────────────────────────────
_TTS_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), "automl_tts_cache"
)
os.makedirs(_TTS_CACHE_DIR, exist_ok=True)


def is_tts_available() -> bool:
    """Return True if gTTS can be imported."""
    try:
        from gtts import gTTS  # noqa: F401
        return True
    except ImportError:
        return False


def _text_hash(text: str) -> str:
    """Short SHA-256 hex of the text for cache key."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def speak(text: str) -> str | None:
    """
    Generate speech audio for the given text.

    Parameters
    ----------
    text : str — The text to synthesise into speech.

    Returns
    -------
    str | None — Absolute path to the generated .mp3 file,
                 or None if TTS is unavailable or fails.
    """
    if not text or not text.strip():
        return None

    # Check cache first
    cache_key = _text_hash(text)
    cached_path = os.path.join(_TTS_CACHE_DIR, f"{cache_key}.mp3")
    if os.path.isfile(cached_path):
        return cached_path

    try:
        from gtts import gTTS

        tts = gTTS(text=text.strip(), lang="en", slow=False)
        tts.save(cached_path)
        return cached_path

    except ImportError:
        print("[TTS] gTTS is not installed. Run: pip install gTTS")
        return None
    except Exception as e:
        print(f"[TTS] Speech generation failed: {e!r}")
        # Clean up partial file
        if os.path.isfile(cached_path):
            try:
                os.unlink(cached_path)
            except OSError:
                pass
        return None


def clear_cache() -> int:
    """
    Remove all cached TTS audio files.
    Returns the number of files deleted.
    """
    count = 0
    for f in Path(_TTS_CACHE_DIR).glob("*.mp3"):
        try:
            f.unlink()
            count += 1
        except OSError:
            pass
    return count
