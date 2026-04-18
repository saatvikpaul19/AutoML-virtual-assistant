"""
config.py — Module 21: Config / Constants
==========================================
Single source of truth for all fixed values used across the system.
Import this everywhere instead of hardcoding values.
"""

import hashlib
import os
from dotenv import load_dotenv

# Load .env file (picks up KAGGLE_USERNAME, KAGGLE_KEY, etc.)
load_dotenv()

# ─────────────────────────────────────────────────────────
# WAKE WORD
# ─────────────────────────────────────────────────────────
WAKE_WORD = "hey mycroft"

# The assistant will also accept these natural variants
WAKE_WORD_VARIANTS = [
    "hey mycroft",
    "hi mycroft",
    "okay mycroft",
    "wake up",
    "hello mycroft",
]

# Continuous Streaming OpenWakeWord Model Configuration
OWW_MODEL_NAME = "hey_mycroft_v0.1"
WAKE_WORD_THRESHOLD = 0.5
WAKE_WORD_SR = 16000

# ─────────────────────────────────────────────────────────
# USER VERIFICATION
# ─────────────────────────────────────────────────────────
_DEFAULT_PASSCODE = "1234"
PASSCODE_HASH = hashlib.sha256(_DEFAULT_PASSCODE.encode()).hexdigest()

MAX_FAILED_ATTEMPTS = 3
LOCKOUT_SECONDS = 30

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENROLLMENT_ROOT = os.path.join(_BASE_DIR, "data", "enrollment")

FACE_CROP_SIZE = 160
FACE_COSINE_THRESHOLD = 0.8
VOICE_COSINE_THRESHOLD = 0.9

DEFAULT_USER_ID = "default"

# ─────────────────────────────────────────────────────────
# AUDIO CAPTURE
# ─────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHANNELS = 1
RECORD_DURATION = 5
AUDIO_DTYPE = "float32"
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 1.5
import tempfile as _tempfile
AUDIO_TMP_PATH = os.path.join(_tempfile.gettempdir(), "automl_captured_audio.wav")

# ─────────────────────────────────────────────────────────
# WHISPER / SPEECH-TO-TEXT
# ─────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = "en"
WHISPER_DEVICE = "cpu"

# ─────────────────────────────────────────────────────────
# ML EXPERIMENT DEFAULTS
# ─────────────────────────────────────────────────────────
SUPPORTED_MODELS = ["xgboost", "random_forest", "logistic_regression", "cnn", "mlp", "resnet"]
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20

# ─────────────────────────────────────────────────────────
# SUPPORTED INTENTS
# ─────────────────────────────────────────────────────────
SUPPORTED_INTENTS = [
    # Stateful control intents
    "load_dataset",
    "select_model",
    "set_learning_rate",
    "set_batch_size",
    "set_epochs",
    "set_layers",
    "load_code",
    "run_code",
    "show_output",
    "start_training",
    "pause_training",
    "stop_training",
    "resume_training",
    "show_status",
    "show_accuracy",
    "show_loss_curve",
    "tell_results",
    "set_activation",
    "clean_dataset",
    "split_dataset",
    "download_weights",
    "set_timer",
    "check_timer",
    "cancel_timer",

    # Stateless info intents
    "search_dataset",
    "get_dataset_info",
    "show_competition",
    "show_leaderboard",
    "search_code",
    "suggest_model",
    "suggest_hyperparameters",
    "get_weather",

    # Utility intents
    "help",
    "repeat",
    "greetings",
    "farewell",
    "out_of_scope",
]

# ─────────────────────────────────────────────────────────
# SUPPORTED DATASETS
# ─────────────────────────────────────────────────────────
SUPPORTED_DATASETS = [
    "titanic", "iris", "mnist", "cifar10",
    "boston", "wine", "diabetes", "breast_cancer",
]

# ─────────────────────────────────────────────────────────
# TRAINING SIMULATION
# ─────────────────────────────────────────────────────────
TRAINING_SIMULATION_INTERVAL = 0.5

# ─────────────────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────────────────
WEATHER_TIMEOUT_SECONDS = 15
WEATHER_FORECAST_DAYS = 7