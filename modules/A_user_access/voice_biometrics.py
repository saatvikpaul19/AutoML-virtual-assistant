"""
Voice template / verification (Resemblyzer speaker embeddings).
==============================================================
Enrolls a fixed-length utterance embedding; verification uses cosine similarity.

IMPORTANT — what this does and does NOT do:
  ✅  Identifies WHO is speaking by their voice characteristics (pitch, timbre,
      resonance, speaking rhythm).
  ❌  Does NOT compare the words or sentences spoken.  You could say completely
      different sentences at enrollment vs verification and it will still work.
  ❌  Not perfect — very similar-sounding speakers (e.g. siblings) may have
      overlapping similarity scores.

Threshold guidance (VOICE_COSINE_THRESHOLD in config.py):
  • Same person, good mic, quiet room → typically 0.80–0.95
  • Same person, noisy room / different mic → 0.70–0.85
  • Different people → typically 0.40–0.75 (varies a LOT by speaker pair)
  • Setting threshold too high (≥0.90) causes false rejections for the enrolled user.
  • Setting it too low (≤0.70) risks accepting impostors.
  Recommended starting point: 0.80.  Adjust based on your test results.
"""

from __future__ import annotations

import io
import os
import tempfile

import numpy as np

from config import SAMPLE_RATE, VOICE_COSINE_THRESHOLD

_voice_encoder = None

# Path where the last recorded clip is saved for UI playback
LAST_RECORDING_PATH = os.path.join(tempfile.gettempdir(), "automl_last_voice.wav")


def _get_voice_encoder():
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder
        _voice_encoder = VoiceEncoder(device="cpu")
    return _voice_encoder


def save_wav(audio: np.ndarray, path: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Save a float32 mono numpy array to a WAV file for playback."""
    import wave, struct
    wav_arr = np.clip(audio, -1.0, 1.0)
    pcm = (wav_arr * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    print(f"[DEBUG voice] save_wav: saved {len(pcm)} samples to {path}")


def embedding_from_float_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    audio: mono float32 numpy, any length; resemblyzer expects ~16 kHz.

    NOTE: This compares voice IDENTITY — speaker characteristics — not the
    words spoken.  The same person saying different sentences will produce
    similar embeddings.
    """
    wav = np.asarray(audio, dtype=np.float32).flatten()
    print(f"[DEBUG voice] embedding_from_float_audio: {len(wav)} samples "
          f"({len(wav)/sample_rate:.2f}s), sr={sample_rate}")
    if len(wav) < sample_rate * 0.5:
        raise ValueError("Audio too short for voice embedding (need at least ~0.5s).")

    # Save for playback
    try:
        save_wav(wav, LAST_RECORDING_PATH, sample_rate)
    except Exception as e:
        print(f"[DEBUG voice] save_wav failed (non-fatal): {e}")

    enc = _get_voice_encoder()
    emb = enc.embed_utterance(wav)
    result = np.asarray(emb, dtype=np.float32)
    print(f"[DEBUG voice] embedding_from_float_audio: embedding shape={result.shape}, "
          f"norm={np.linalg.norm(result):.4f}")
    return result


def compare_voices(template: np.ndarray, probe: np.ndarray) -> tuple[float, bool]:
    """
    Cosine similarity between L2-normalized Resemblyzer embeddings.

    Score interpretation:
      ≥ threshold  → same speaker (MATCH)
      < threshold  → different speaker (NO MATCH)
    Typical same-speaker score: 0.80–0.95
    Typical different-speaker score: 0.40–0.75
    """
    print(f"[DEBUG voice] compare_voices: template shape={template.shape}, "
          f"probe shape={probe.shape}")
    a = template.astype(np.float64).ravel()
    b = probe.astype(np.float64).ravel()
    a /= np.linalg.norm(a) + 1e-9
    b /= np.linalg.norm(b) + 1e-9
    score = float(np.dot(a, b))
    is_match = score >= VOICE_COSINE_THRESHOLD
    print(f"[DEBUG voice] compare_voices: score={score:.6f} vs "
          f"threshold={VOICE_COSINE_THRESHOLD} → MATCH={is_match}")
    return score, is_match
