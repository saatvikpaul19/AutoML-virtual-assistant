"""
state_manager.py -- Shared In-Memory State
==========================================
Thread-safe singleton that holds all runtime state for the assistant.
All read/write operations use a threading.Lock -- safe to call from
background threads (ASR, training loop) and the Streamlit main thread.

ASR State Machine values (asr_state):
    "sleep"     -- listening for wake word only
    "active"    -- wake word heard, recording command audio
    "listening" -- audio captured, Whisper transcription in progress
"""
from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime

try:
    from config import DEFAULT_LEARNING_RATE, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
except Exception:
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_BATCH_SIZE    = 32
    DEFAULT_EPOCHS        = 10


class StateManager:
    def __init__(self):
        self._lock  = threading.Lock()
        self._state = {}
        self.reset_all()

    def reset_all(self) -> None:
        with self._lock:
            self._state = {
                # -- Auth / session -------------------------------------
                "verified":            False,

                # -- ASR state machine ----------------------------------
                # Values: "sleep" | "active" | "listening"
                "asr_state":           "sleep",
                "wake_detected":       False,
                "listening":           False,
                "mic_active":          False,
                "pending_audio_path":  None,
                "tts_expected_end_time": 0.0,
                "force_active":        False,

                # -- Conversation ---------------------------------------
                "transcript":          "",
                "assistant_response":  "",

                # -- Dataset -------------------------------------------
                "dataset":             None,
                "dataset_info":        {},
                "dataset_preview":     [],
                "dataset_files":       [],
                "dataset_profile":     {},

                # -- Experiment config ---------------------------------
                "model":               None,
                "learning_rate":       DEFAULT_LEARNING_RATE,
                "batch_size":          DEFAULT_BATCH_SIZE,
                "epochs_total":        DEFAULT_EPOCHS,
                "epoch_current":       0,
                "layers":              3,
                "activation":          "relu",

                # -- Training progress ---------------------------------
                "training_status":     "idle",
                "results_requested":   False,
                "loss_history":        [],
                "accuracy_history":    [],

                # -- Code & outputs ------------------------------------
                "generated_code_py":       "",
                "generated_code_ipynb":    "",
                "reference_code":          "",
                "reference_code_format":   "",
                "code_title":              "",
                "code_output_text":        "",
                "outputs":                 [],

                # -- Stateless results ---------------------------------
                "stateless_results":   [],
                "weather_result":      {},

                # -- Timer ---------------------------------------------
                "timer": {
                    "exists":                   False,
                    "label":                    "timer",
                    "original_duration_seconds": 0,
                    "remaining_seconds":         0,
                    "status":                   "idle",
                    "last_started_at":           None,
                    "completion_announced":      False,
                },

                # -- Event log -----------------------------------------
                "event_log": [],
            }

    def reset_experiment(self) -> None:
        with self._lock:
            self._state["dataset"]             = None
            self._state["dataset_info"]        = {}
            self._state["dataset_preview"]     = []
            self._state["dataset_files"]       = []
            self._state["dataset_profile"]     = {}
            self._state["model"]               = None
            self._state["learning_rate"]       = DEFAULT_LEARNING_RATE
            self._state["batch_size"]          = DEFAULT_BATCH_SIZE
            self._state["epochs_total"]        = DEFAULT_EPOCHS
            self._state["epoch_current"]       = 0
            self._state["layers"]              = 3
            self._state["activation"]          = "relu"
            self._state["training_status"]     = "idle"
            self._state["results_requested"]   = False
            self._state["loss_history"]        = []
            self._state["accuracy_history"]    = []
            self._state["generated_code_py"]   = ""
            self._state["generated_code_ipynb"] = ""
            self._state["reference_code"]      = ""
            self._state["reference_code_format"] = ""
            self._state["code_title"]          = ""
            self._state["code_output_text"]    = ""
            self._state["outputs"]             = []
            self._state["stateless_results"]   = []
            self._state["weather_result"]      = {}

    # -- Read --------------------------------------------------

    def get_state(self) -> dict:
        with self._lock:
            return deepcopy(self._state)

    def get(self, key: str, default=None):
        with self._lock:
            return deepcopy(self._state.get(key, default))

    def get_ui_state(self) -> dict:
        with self._lock:
            loss = list(self._state["loss_history"])
            acc  = list(self._state["accuracy_history"])
            return {
                "dataset":           self._state["dataset"],
                "model":             self._state["model"],
                "learning_rate":     self._state["learning_rate"],
                "batch_size":        self._state["batch_size"],
                "epochs_total":      self._state["epochs_total"],
                "epoch_current":     self._state["epoch_current"],
                "layers":            self._state["layers"],
                "activation":        self._state["activation"],
                "training_status":   self._state["training_status"],
                "results_requested": self._state.get("results_requested", False),
                "loss_latest":       loss[-1] if loss else None,
                "accuracy_latest":   acc[-1]  if acc  else None,
                "n_events":          len(self._state["event_log"]),
            }

    # -- ASR State Machine -------------------------------------

    def set_asr_state(self, state: str) -> None:
        """
        Update the ASR state machine.
        Valid values: "sleep" | "active" | "listening"
        """
        valid = {"sleep", "active", "listening"}
        if state not in valid:
            raise ValueError(f"Invalid asr_state: {state!r}. Must be one of {valid}")
        with self._lock:
            prev = self._state["asr_state"]
            self._state["asr_state"] = state
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[ASR-STATE] [{ts}]  {prev.upper()} -> {state.upper()}")

    def get_asr_state(self) -> str:
        with self._lock:
            return self._state["asr_state"]

    # -- Auth --------------------------------------------------

    def set_verified(self, value: bool) -> None:
        with self._lock:
            self._state["verified"] = value

    # -- Legacy ASR flags (kept for UI compatibility) ----------

    def set_wake_detected(self, value: bool) -> None:
        with self._lock:
            self._state["wake_detected"] = value

    def set_listening(self, value: bool) -> None:
        with self._lock:
            self._state["listening"] = value

    def set_mic_active(self, val: bool) -> None:
        with self._lock:
            self._state["mic_active"] = val
            # Sync asr_state with mic_active toggle
            if not val:
                self._state["asr_state"] = "sleep"

    def set_pending_audio_path(self, val: str | None) -> None:
        with self._lock:
            self._state["pending_audio_path"] = val

    def set_tts_expected_end_time(self, timestamp: float) -> None:
        with self._lock:
            self._state["tts_expected_end_time"] = timestamp

    def set_force_active(self, val: bool) -> None:
        with self._lock:
            self._state["force_active"] = val

    # -- Conversation ------------------------------------------

    def set_transcript(self, text: str) -> None:
        with self._lock:
            self._state["transcript"] = text

    def set_assistant_response(self, text: str) -> None:
        with self._lock:
            self._state["assistant_response"] = text

    # -- Dataset -----------------------------------------------

    def set_dataset(self, name: str) -> None:
        with self._lock:
            self._state["dataset"] = name

    def set_dataset_info(self, info: dict) -> None:
        with self._lock:
            self._state["dataset_info"] = info

    def set_dataset_preview(self, preview) -> None:
        with self._lock:
            self._state["dataset_preview"] = preview

    def set_dataset_files(self, files) -> None:
        with self._lock:
            self._state["dataset_files"] = files

    def set_dataset_profile(self, profile: dict) -> None:
        with self._lock:
            self._state["dataset_profile"] = profile

    # -- Experiment config -------------------------------------

    def set_model(self, name: str) -> None:
        with self._lock:
            self._state["model"] = name

    def set_learning_rate(self, lr: float) -> None:
        with self._lock:
            self._state["learning_rate"] = lr

    def set_batch_size(self, bs: int) -> None:
        with self._lock:
            self._state["batch_size"] = bs

    def set_epochs(self, n: int) -> None:
        with self._lock:
            self._state["epochs_total"] = n

    def set_layers(self, n: int) -> None:
        with self._lock:
            self._state["layers"] = n

    def set_activation(self, act: str) -> None:
        with self._lock:
            self._state["activation"] = act

    # -- Training ----------------------------------------------

    def set_results_requested(self, requested: bool) -> None:
        with self._lock:
            self._state["results_requested"] = requested

    def set_training_status(self, status: str) -> None:
        with self._lock:
            self._state["training_status"] = status

    def set_epoch_current(self, n: int) -> None:
        with self._lock:
            self._state["epoch_current"] = n

    def append_loss(self, value: float) -> None:
        with self._lock:
            self._state["loss_history"].append(value)

    def append_accuracy(self, value: float) -> None:
        with self._lock:
            self._state["accuracy_history"].append(value)

    def reset_metrics(self) -> None:
        with self._lock:
            self._state["loss_history"]     = []
            self._state["accuracy_history"] = []
            self._state["epoch_current"]    = 0

    # -- Code / Outputs ----------------------------------------

    def set_generated_code_py(self, code: str) -> None:
        with self._lock:
            self._state["generated_code_py"] = code

    def set_generated_code_ipynb(self, code: str) -> None:
        with self._lock:
            self._state["generated_code_ipynb"] = code

    def set_reference_code(self, code: str, fmt: str = "", title: str = "") -> None:
        with self._lock:
            self._state["reference_code"]        = code
            self._state["reference_code_format"] = fmt
            self._state["code_title"]            = title

    def set_code_output_text(self, text: str) -> None:
        with self._lock:
            self._state["code_output_text"] = text

    def set_outputs(self, outputs) -> None:
        with self._lock:
            self._state["outputs"] = outputs

    # -- Stateless results -------------------------------------

    def set_stateless_results(self, results) -> None:
        with self._lock:
            self._state["stateless_results"] = results

    def set_weather_result(self, result: dict) -> None:
        with self._lock:
            self._state["weather_result"] = result or {}

    # -- Timer -------------------------------------------------

    def start_timer(self, duration_seconds: int, label: str = "timer") -> None:
        with self._lock:
            self._state["timer"] = {
                "exists":                    True,
                "label":                     label or "timer",
                "original_duration_seconds": int(duration_seconds),
                "remaining_seconds":         int(duration_seconds),
                "status":                    "running",
                "last_started_at":           time.time(),
                "completion_announced":      False,
            }

    def pause_timer(self) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists") or timer.get("status") != "running":
                return False
            elapsed = max(0, int(time.time() - timer["last_started_at"]))
            timer["remaining_seconds"] = max(0, timer["remaining_seconds"] - elapsed)
            timer["status"]            = "paused"
            timer["last_started_at"]   = None
            return True

    def resume_timer(self) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists") or timer.get("status") != "paused":
                return False
            timer["status"]          = "running"
            timer["last_started_at"] = time.time()
            return True

    def stop_timer(self) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists"):
                return False
            timer["status"]            = "stopped"
            timer["remaining_seconds"] = 0
            timer["last_started_at"]   = None
            return True

    def cancel_timer(self) -> None:
        with self._lock:
            self._state["timer"] = {
                "exists":                    False,
                "label":                     "timer",
                "original_duration_seconds": 0,
                "remaining_seconds":         0,
                "status":                    "idle",
                "last_started_at":           None,
                "completion_announced":      False,
            }

    def restart_timer(self) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists"):
                return False
            timer["remaining_seconds"]      = int(timer["original_duration_seconds"])
            timer["status"]                 = "running"
            timer["last_started_at"]        = time.time()
            timer["completion_announced"]   = False
            return True

    def reset_timer(self) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists"):
                return False
            timer["remaining_seconds"]    = int(timer["original_duration_seconds"])
            timer["status"]               = "paused"
            timer["last_started_at"]      = None
            timer["completion_announced"] = False
            return True

    def add_time_to_timer(self, extra_seconds: int) -> bool:
        with self._lock:
            timer = self._state["timer"]
            if not timer.get("exists"):
                return False
            timer["remaining_seconds"]         = int(timer["remaining_seconds"]) + int(extra_seconds)
            timer["original_duration_seconds"] = int(timer["original_duration_seconds"]) + int(extra_seconds)
            if timer["status"] == "completed":
                timer["status"] = "paused"
            timer["completion_announced"] = False
            return True

    def get_timer_info(self) -> dict:
        with self._lock:
            timer = deepcopy(self._state["timer"])

        if not timer.get("exists"):
            return timer

        if timer["status"] == "running" and timer["last_started_at"] is not None:
            elapsed   = max(0, int(time.time() - timer["last_started_at"]))
            remaining = max(0, int(timer["remaining_seconds"] - elapsed))
            timer["remaining_seconds"] = remaining
            if remaining == 0:
                timer["status"]          = "completed"
                timer["last_started_at"] = None
                with self._lock:
                    self._state["timer"]["remaining_seconds"] = 0
                    self._state["timer"]["status"]            = "completed"
                    self._state["timer"]["last_started_at"]   = None

        return timer

    def mark_timer_completion_announced(self) -> None:
        with self._lock:
            self._state["timer"]["completion_announced"] = True

    def timer_completion_announced(self) -> bool:
        with self._lock:
            return bool(self._state["timer"].get("completion_announced", False))

    # -- Event log ---------------------------------------------

    def append_log(self, message: str) -> None:
        with self._lock:
            ts = datetime.now().strftime("%H:%M:%S")
            self._state["event_log"].append(f"[{ts}] {message}")

    def get_event_log(self, last_n: int | None = None):
        with self._lock:
            logs = list(self._state["event_log"])
        return logs[-last_n:] if last_n else logs

    def clear_event_log(self) -> None:
        with self._lock:
            self._state["event_log"] = []


# -- Singleton -------------------------------------------------

_instance      = None
_instance_lock = threading.Lock()


def get_state_manager() -> StateManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = StateManager()
    return _instance