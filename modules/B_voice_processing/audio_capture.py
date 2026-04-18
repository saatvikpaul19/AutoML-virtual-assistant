"""
Module 4 — Audio Capture
=========================
Records the user's voice from the system microphone and returns the
audio in a format that Module 5 (Speech-to-Text / Whisper) can consume.

Two recording modes are provided:

  record_fixed(duration)         — record for exactly N seconds
  record_until_silence(max_sec)  — stop automatically when the user
                                   stops speaking (VAD via RMS energy)

Public API
----------
AudioCapture   (class)
    .record_fixed(duration: int = RECORD_DURATION) -> np.ndarray
    .record_until_silence(max_sec: int = 10) -> np.ndarray
    .save(audio: np.ndarray, path: str) -> None
    .load(path: str) -> np.ndarray

save_audio(audio, path)     — module-level convenience wrapper
load_audio(path) -> array   — module-level convenience wrapper

Design notes
------------
- sounddevice is used instead of pyaudio because it:
    • has no compile-time dependency on portaudio headers on Linux
    • returns numpy arrays natively (no manual byte unpacking)
    • is easier to install in virtual environments
- Audio is always captured as float32 at 16 kHz mono — exactly what
  Whisper expects, so no resampling step is needed.
- The silence-detection VAD works on 50 ms chunks.  RMS below
  SILENCE_THRESHOLD for SILENCE_DURATION seconds → stop recording.
- scipy.io.wavfile is used for saving because it does not require
  installing ffmpeg and handles 16-bit PCM WAV correctly.
"""

import time
import numpy as np
import scipy.io.wavfile as wav

from config import (
    SAMPLE_RATE,
    CHANNELS,
    RECORD_DURATION,
    AUDIO_DTYPE,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    AUDIO_TMP_PATH,
)


def _rms(chunk: np.ndarray) -> float:
    """Root-mean-square energy of an audio chunk (amplitude proxy)."""
    return float(np.sqrt(np.mean(chunk ** 2)))


class AudioCapture:
    """
    Wraps sounddevice to provide clean recording methods for the
    voice-controlled assistant.

    Typical usage
    -------------
    capture = AudioCapture()

    # Simple: record for 5 seconds
    audio = capture.record_fixed(duration=5)
    capture.save(audio, "/tmp/clip.wav")

    # Smart: stop when user stops speaking
    audio = capture.record_until_silence(max_sec=10)
    capture.save(audio, "/tmp/clip.wav")
    """

    def __init__(self):
        # Lazy import so the class can be imported even when sounddevice
        # is not installed (e.g., in CI environments)
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            self._sd = None

    def _check_sd(self):
        if self._sd is None:
            raise ImportError(
                "sounddevice is not installed. "
                "Run: pip install sounddevice"
            )

    # ── Fixed-duration recording ─────────────────────────────────────

    def record_fixed(self, duration: int = RECORD_DURATION) -> np.ndarray:
        """
        Record microphone audio for exactly `duration` seconds.

        Parameters
        ----------
        duration : int   — Recording length in seconds

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
            Single-channel audio normalised to –1.0 … +1.0.
        """
        self._check_sd()
        frames = int(duration * SAMPLE_RATE)
        audio = self._sd.rec(
            frames,
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
        )
        self._sd.wait()                    # blocks until recording finishes
        return audio.flatten()             # (N, 1) → (N,)

    # ── Voice-activity-detection recording ──────────────────────────

    def record_until_silence(self, max_sec: int = 10) -> np.ndarray:
        """
        Record until the user stops speaking (or max_sec is reached).

        The algorithm:
        1. Record in 50 ms chunks.
        2. Compute RMS energy of each chunk.
        3. Once speaking has started (RMS > threshold), monitor for
           a consecutive silence window of SILENCE_DURATION seconds.
        4. Stop and return the collected audio.

        Parameters
        ----------
        max_sec : int   — Hard upper limit on recording length

        Returns
        -------
        np.ndarray, shape (N,), dtype float32
        """
        self._check_sd()

        chunk_duration = 0.05          # 50 ms per chunk
        chunk_frames = int(chunk_duration * SAMPLE_RATE)
        silence_chunks_needed = int(SILENCE_DURATION / chunk_duration)

        collected: list[np.ndarray] = []
        silence_streak = 0
        speaking_started = False
        total_chunks = int(max_sec / chunk_duration)

        with self._sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
        ) as stream:
            for _ in range(total_chunks):
                chunk, _ = stream.read(chunk_frames)
                chunk = chunk.flatten()
                collected.append(chunk)

                energy = _rms(chunk)

                if energy > SILENCE_THRESHOLD:
                    speaking_started = True
                    silence_streak = 0
                elif speaking_started:
                    silence_streak += 1
                    if silence_streak >= silence_chunks_needed:
                        break   # natural end of speech

        return np.concatenate(collected)

    # ── Save / Load ──────────────────────────────────────────────────

    def save(self, audio: np.ndarray, path: str = AUDIO_TMP_PATH) -> None:
        """
        Save a float32 numpy audio array as a 16-bit PCM WAV file.

        Parameters
        ----------
        audio : np.ndarray   — float32, range –1.0 … +1.0
        path  : str          — destination file path (.wav)
        """
        # Convert float32 (–1..+1) to int16 (–32768..+32767)
        audio_int16 = (audio * 32767).astype(np.int16)
        wav.write(path, SAMPLE_RATE, audio_int16)

    def load(self, path: str) -> np.ndarray:
        """
        Load a WAV file and return as float32 numpy array.

        Parameters
        ----------
        path : str   — path to .wav file

        Returns
        -------
        np.ndarray, dtype float32, range –1.0 … +1.0
        """
        rate, data = wav.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if data.ndim > 1:
            data = data[:, 0]   # keep only first channel
        return data


# ── Module-level convenience wrappers ────────────────────────────────────

_capture = AudioCapture()   # singleton for module-level API

def save_audio(audio: np.ndarray, path: str = AUDIO_TMP_PATH) -> None:
    """Save audio array to WAV file (module-level convenience)."""
    _capture.save(audio, path)

def load_audio(path: str) -> np.ndarray:
    """Load a WAV file as float32 array (module-level convenience)."""
    return _capture.load(path)
