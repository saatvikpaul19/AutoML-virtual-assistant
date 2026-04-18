"""
continuous_asr.py — Module B: Background Wake Word + Command Recording
========================================================================
Implements a 3-state ASR lifecycle:

    SLEEP ──(wake word detected)──► ACTIVE ──(recording done)──► LISTENING
      ▲                                                              │
      └──────────────────────────(transcription posted)─────────────┘

State transitions logged to stdout:
    [ASR-STATE] [HH:MM:SS]  SLEEP → ACTIVE (wake word prob=0.86)
    [ASR-STATE] [HH:MM:SS]  ACTIVE → LISTENING (recording captured, 4.2s)
    [ASR-STATE] [HH:MM:SS]  LISTENING → SLEEP (pending_audio posted)

The Streamlit main thread picks up state changes by polling StateManager
and processes the audio via Whisper when pending_audio_path is set.
"""
from __future__ import annotations

import threading
import time
import os
import queue
from datetime import datetime

import numpy as np
import sounddevice as sd

from config import (
    OWW_MODEL_NAME,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_SR,
    AUDIO_TMP_PATH,
)
from modules.D_control.state_manager import StateManager
from modules.B_voice_processing.audio_capture import AudioCapture


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class ContinuousASR:
    """Singleton background thread that manages wake word detection and command recording."""

    _instance = None

    def __new__(cls, sm: StateManager):
        if cls._instance is None:
            cls._instance = super(ContinuousASR, cls).__new__(cls)
            cls._instance._init(sm)
        return cls._instance

    def _init(self, sm: StateManager) -> None:
        self.sm          = sm
        self.is_running  = False
        self._thread     = None
        self._model      = None
        self._q: queue.Queue = queue.Queue()

        # OpenWakeWord consumes 80ms chunks at 16 kHz
        self.samples_per_chunk = int(WAKE_WORD_SR * 0.08)

        print(f"[ContinuousASR] [{_ts()}] Instance created. Chunk size: {self.samples_per_chunk} samples")

    # ── Lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if self.is_running or self._thread is not None:
            print(f"[ContinuousASR] [{_ts()}] Already running — ignored start()")
            return
        print(f"[ContinuousASR] [{_ts()}] Starting background thread …")
        self.is_running  = True
        self.sm.set_asr_state("sleep")
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        print(f"[ContinuousASR] [{_ts()}] Stopping …")
        self.is_running = False
        self.sm.set_asr_state("sleep")
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        print(f"[ContinuousASR] [{_ts()}] Stopped.")

    # ── Audio stream callback (runs in sounddevice thread) ────

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[ContinuousASR] Audio stream status: {status}")
        if self.is_running:
            self._q.put(indata.copy())

    # ── Main worker loop ──────────────────────────────────────

    def _worker_loop(self) -> None:
        # ── Load OpenWakeWord model ───────────────────────────
        try:
            from openwakeword.model import Model as OWWModel
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            os.environ["OMP_NUM_THREADS"]       = "1"
            self._model = OWWModel(
                wakeword_models=[OWW_MODEL_NAME],
                inference_framework="onnx",
            )
            print(f"[ContinuousASR] [{_ts()}] OpenWakeWord model loaded: {OWW_MODEL_NAME}")
        except Exception as e:
            print(f"[ContinuousASR] [{_ts()}] ✗ Failed to load OWW model: {e}")
            self.is_running = False
            return

        print(f"[ContinuousASR] [{_ts()}] Worker loop started. asr_state=SLEEP")

        while self.is_running:
            state = self.sm.get_state()

            # ── Respect mic_active flag ───────────────────────
            if not state.get("mic_active", False):
                time.sleep(0.5)
                continue

            # ── Pause during TTS playback ─────────────────────
            if time.time() < state.get("tts_expected_end_time", 0):
                print(f"[ContinuousASR] [{_ts()}] TTS active — mic suspended")
                time.sleep(0.5)
                continue

            # ── Clear stale queue data ────────────────────────
            flushed = 0
            while not self._q.empty():
                self._q.get_nowait()
                flushed += 1
            if flushed:
                print(f"[ContinuousASR] [{_ts()}] Flushed {flushed} stale audio chunks")

            # ── Open audio stream ─────────────────────────────
            try:
                stream = sd.InputStream(
                    samplerate = WAKE_WORD_SR,
                    channels   = 1,
                    dtype      = "int16",
                    blocksize  = self.samples_per_chunk,
                    callback   = self._audio_callback,
                )
                stream.start()
                print(f"[ContinuousASR] [{_ts()}] Audio stream opened — listening for wake word …")
            except Exception as e:
                print(f"[ContinuousASR] [{_ts()}] ✗ Audio stream failed: {e} — retrying in 2s")
                time.sleep(2)
                continue

            # ── Inner inference loop ──────────────────────────
            try:
                self.sm.set_asr_state("sleep")

                while self.is_running and self.sm.get_state().get("mic_active", False):

                    # Suspend during TTS
                    if time.time() < self.sm.get_state().get("tts_expected_end_time", 0):
                        print(f"[ContinuousASR] [{_ts()}] TTS started mid-stream — breaking inner loop")
                        break

                    try:
                        data = self._q.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    # OWW expects flat (N,) int16 array
                    audio_frame = data.flatten()
                    prediction  = self._model.predict(audio_frame)
                    prob        = prediction.get(OWW_MODEL_NAME, 0.0)

                    # Log any non-trivial probability
                    if prob > 0.05:
                        print(f"[ContinuousASR] [{_ts()}] Wake word probe: {OWW_MODEL_NAME} = {prob:.4f} (threshold={WAKE_WORD_THRESHOLD})")

                    force_active = self.sm.get_state().get("force_active", False)
                    if prob > WAKE_WORD_THRESHOLD or force_active:
                        # ── WAKE WORD DETECTED ────────────────
                        if force_active:
                            print(f"\n[ContinuousASR] [{_ts()}] ★ MIC FORCED ACTIVE (Confirmation loop)")
                            self.sm.set_force_active(False)
                        else:
                            print(f"\n[ContinuousASR] [{_ts()}] ★ WAKE WORD DETECTED  prob={prob:.4f}")
                            
                        self.sm.set_asr_state("active")
                        self.sm.set_wake_detected(True)

                        # Close stream so audio device is free for recording
                        stream.stop()
                        stream.close()
                        self._model.reset()

                        # ── Record command audio ──────────────
                        print(f"[ContinuousASR] [{_ts()}] Recording command audio (max 10s, silence-gated) …")
                        t_rec_start = time.time()
                        try:
                            ac            = AudioCapture()
                            command_audio = ac.record_until_silence(max_sec=10)
                            rec_duration  = round(time.time() - t_rec_start, 2)
                            print(f"[ContinuousASR] [{_ts()}] Recording done: {rec_duration}s captured")
                        except Exception as exc:
                            print(f"[ContinuousASR] [{_ts()}] ✗ Recording failed: {exc}")
                            self.sm.set_asr_state("sleep")
                            self.sm.set_wake_detected(False)
                            break

                        # Save to tmp
                        self.sm.set_asr_state("listening")
                        t_save_start = time.time()
                        ac.save(command_audio, AUDIO_TMP_PATH)
                        print(f"[ContinuousASR] [{_ts()}] Audio saved to {AUDIO_TMP_PATH}")

                        # Hand off to Streamlit
                        self.sm.set_pending_audio_path(AUDIO_TMP_PATH)
                        self.sm.set_listening(False)
                        self.sm.set_wake_detected(False)
                        print(f"[ContinuousASR] [{_ts()}] pending_audio_path set — waiting for Streamlit to consume …")

                        # Wait for Streamlit to consume path
                        wait_start = time.time()
                        while (
                            self.sm.get_state().get("pending_audio_path")
                            and self.is_running
                        ):
                            time.sleep(0.4)
                            if time.time() - wait_start > 30:
                                print(f"[ContinuousASR] [{_ts()}] ⚠ Timeout waiting for Streamlit — clearing path")
                                self.sm.set_pending_audio_path(None)
                                break

                        self.sm.set_asr_state("sleep")
                        print(f"[ContinuousASR] [{_ts()}] Cycle complete — back to SLEEP\n")
                        break   # restart outer loop → fresh stream

            finally:
                if stream.active:
                    stream.stop()
                stream.close()
                print(f"[ContinuousASR] [{_ts()}] Audio stream closed")
