from __future__ import annotations

import inspect
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import DEFAULT_USER_ID, WAKE_WORD, OWW_MODEL_NAME
from modules.A_user_access.user_verification import (
    enrollment_status,
    enroll_face,
    verify_face,
    enroll_voice,
    verify_voice,
    enroll_password,
    verify_profile_password,
)
from modules.A_user_access.wake_word import is_wake_word
from modules.A_user_access.text_input_handler import TextInputHandler
from modules.B_voice_processing.audio_capture import AudioCapture
from modules.B_voice_processing.speech_to_text import (
    SpeechToText,
    get_last_transcribe_error,
)
from modules.C_nlu.nlu_pipeline import understand
from modules.D_control.command_router import route_command
from modules.D_control.state_manager import get_state_manager

# TTS import (graceful — works even if gTTS is not installed)
try:
    from modules.B_voice_processing.tts import speak as tts_speak, is_tts_available
except ImportError:
    tts_speak = lambda text: None  # noqa: E731
    is_tts_available = lambda: False  # noqa: E731

st.set_page_config(page_title="AutoML Workspace Assistant", page_icon="🤖", layout="wide")

_CONTAINER_KW = {"border": True} if "border" in inspect.signature(st.container).parameters else {}

# ── Premium CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global font ─────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Animated background ─────────────────────────────── */
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp {
    background: linear-gradient(135deg, #0a0e17 0%, #111827 25%, #0d1321 50%, #151c2e 75%, #0a0e17 100%);
    background-size: 400% 400%;
    animation: gradient-shift 25s ease infinite;
}

/* ── Header ──────────────────────────────────────────── */
header[data-testid="stHeader"] {
    background: linear-gradient(135deg, rgba(10, 14, 23, 0.95) 0%, rgba(21, 28, 46, 0.95) 100%) !important;
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(88, 166, 255, 0.08);
}

/* ── Panel fade-in animation ─────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Animated shimmer for titles ─────────────────────── */
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}

h1 {
    background: linear-gradient(120deg, #58a6ff, #a78bfa, #f472b6, #58a6ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    font-weight: 800 !important;
    letter-spacing: -0.8px;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

h4 {
    color: #c9d1d9 !important;
    font-weight: 600 !important;
    letter-spacing: -0.3px;
}

/* ── Container glassmorphism ─────────────────────────── */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15, 20, 30, 0.75) !important;
    border: 1px solid rgba(56, 97, 150, 0.2) !important;
    border-radius: 18px !important;
    backdrop-filter: blur(16px) saturate(1.4);
    box-shadow:
        0 4px 30px rgba(0, 0, 0, 0.25),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeSlideIn 0.5s ease-out;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(88, 166, 255, 0.25) !important;
    box-shadow:
        0 8px 40px rgba(0, 0, 0, 0.3),
        0 0 30px rgba(88, 166, 255, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.05);
    transform: translateY(-1px);
}

/* ── Primary buttons ─────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1d6ce0 0%, #3b82f6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.2px;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 2px 10px rgba(59, 130, 246, 0.25);
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 6px 25px rgba(59, 130, 246, 0.4) !important;
    filter: brightness(1.1);
}

.stButton > button:active {
    transform: translateY(0) scale(0.99);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #4f8afa) !important;
    box-shadow: 0 2px 12px rgba(37, 99, 235, 0.35);
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(37, 99, 235, 0.5) !important;
}

/* ── Tab styling ─────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    background: rgba(10, 14, 23, 0.5);
    border-radius: 14px;
    padding: 4px 5px;
    border: 1px solid rgba(48, 54, 61, 0.3);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #7c8ca0 !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #c9d1d9 !important;
    background: rgba(88, 166, 255, 0.06) !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.2), rgba(88, 166, 255, 0.15)) !important;
    color: #79b8ff !important;
    border: 1px solid rgba(88, 166, 255, 0.2) !important;
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.1);
}

/* ── Chat messages ───────────────────────────────────── */
.stChatMessage {
    border-radius: 14px !important;
    margin-bottom: 6px !important;
    border: 1px solid rgba(48, 54, 61, 0.35) !important;
    backdrop-filter: blur(8px);
}

/* User messages */
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: linear-gradient(135deg, rgba(37, 99, 235, 0.08), rgba(59, 130, 246, 0.04)) !important;
    border-color: rgba(59, 130, 246, 0.15) !important;
}

/* Assistant messages */
.stChatMessage[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: rgba(15, 20, 30, 0.5) !important;
    border-color: rgba(48, 54, 61, 0.4) !important;
}

/* ── Input fields ────────────────────────────────────── */
.stTextInput > div > div > input {
    background: rgba(10, 14, 23, 0.75) !important;
    border: 1px solid rgba(56, 97, 150, 0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12), 0 0 20px rgba(59, 130, 246, 0.08) !important;
}

/* ── Chat input ──────────────────────────────────────── */
.stChatInput {
    border-radius: 14px !important;
}

.stChatInput > div {
    border-radius: 14px !important;
    border-color: rgba(56, 97, 150, 0.25) !important;
    background: rgba(10, 14, 23, 0.8) !important;
    transition: all 0.25s ease;
}

.stChatInput > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.08) !important;
}

/* ── Dataframe styling ───────────────────────────────── */
.stDataFrame {
    border-radius: 14px !important;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

/* ── Download buttons ────────────────────────────────── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    box-shadow: 0 2px 10px rgba(124, 58, 237, 0.25);
}

.stDownloadButton > button:hover {
    box-shadow: 0 6px 25px rgba(124, 58, 237, 0.4) !important;
    transform: translateY(-2px);
}

/* ── Expander ────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(15, 20, 30, 0.5) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    transition: all 0.2s ease;
}

.streamlit-expanderHeader:hover {
    color: #c9d1d9 !important;
    background: rgba(15, 20, 30, 0.7) !important;
}

/* ── Code blocks ─────────────────────────────────────── */
.stCodeBlock {
    border-radius: 14px !important;
}

/* ── Metric / caption ────────────────────────────────── */
.stCaption, caption {
    color: #5b6578 !important;
}

/* ── JSON viewer ─────────────────────────────────────── */
.stJson {
    border-radius: 14px !important;
}

/* ── Slider ──────────────────────────────────────────── */
.stSlider > div > div > div {
    background: rgba(59, 130, 246, 0.3) !important;
}

/* ── Toggle ──────────────────────────────────────────── */
.stToggle label {
    color: #94a3b8 !important;
}

/* ── Status badges ───────────────────────────────────── */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}
.status-idle      { background: rgba(100, 116, 139, 0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.2); }
.status-training  { background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); animation: pulse-glow 2s ease-in-out infinite; }
.status-completed { background: rgba(34, 197, 94, 0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.status-paused    { background: rgba(234, 179, 8, 0.12); color: #fbbf24; border: 1px solid rgba(234,179,8,0.25); }
.status-stopped   { background: rgba(239, 68, 68, 0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.2); }
    50%      { box-shadow: 0 0 12px 2px rgba(59, 130, 246, 0.15); }
}

/* ── Verified badge ──────────────────────────────────── */
.verified-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 14px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.badge-yes { background: rgba(34, 197, 94, 0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.badge-no  { background: rgba(239, 68, 68, 0.1); color: #f87171; border: 1px solid rgba(239,68,68,0.15); }

/* ── Recording pulse animation ───────────────────────── */
@keyframes pulse-red {
    0%   { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
    70%  { box-shadow: 0 0 0 14px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.recording-indicator {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08), rgba(239, 68, 68, 0.04));
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 12px;
    color: #f87171;
    font-weight: 600;
    font-size: 0.88rem;
}

.recording-dot {
    width: 11px;
    height: 11px;
    background: #ef4444;
    border-radius: 50%;
    animation: pulse-red 1.5s infinite;
}

/* ── Typing indicator ────────────────────────────────── */
@keyframes typing-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
    40% { transform: translateY(-7px); opacity: 1; }
}

.typing-indicator {
    display: inline-flex;
    gap: 5px;
    padding: 10px 14px;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: #3b82f6;
    border-radius: 50%;
    animation: typing-bounce 1.2s ease-in-out infinite;
}

.typing-indicator span:nth-child(2) { animation-delay: 0.15s; }
.typing-indicator span:nth-child(3) { animation-delay: 0.3s; }

/* ── Pipeline stage badges ───────────────────────────── */
.pipeline-stage {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 10px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 4px;
}
.stage-active  { background: rgba(59, 130, 246, 0.1); color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
.stage-done    { background: rgba(34, 197, 94, 0.1); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.stage-pending { background: rgba(100, 116, 139, 0.08); color: #64748b; border: 1px solid rgba(100,116,139,0.15); }

/* ── Quick command cards ─────────────────────────────── */
.cmd-card {
    display: inline-block;
    padding: 6px 14px;
    margin: 3px 4px;
    background: rgba(15, 20, 30, 0.6);
    border: 1px solid rgba(56, 97, 150, 0.2);
    border-radius: 10px;
    color: #94a3b8;
    font-size: 0.78rem;
    font-family: 'Inter', monospace;
    cursor: default;
    transition: all 0.2s ease;
}

.cmd-card:hover {
    border-color: rgba(88, 166, 255, 0.3);
    color: #c9d1d9;
    background: rgba(15, 20, 30, 0.8);
    transform: translateY(-1px);
}

/* ── Divider styling ─────────────────────────────────── */
hr {
    border-color: rgba(56, 97, 150, 0.15) !important;
    margin: 12px 0 !important;
}

/* ── Scrollbar styling ───────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(10, 14, 23, 0.4);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: rgba(88, 166, 255, 0.15);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(88, 166, 255, 0.3);
}

/* ── Info/Warning/Success boxes ───────────────────────── */
.stAlert {
    border-radius: 12px !important;
}

</style>
""", unsafe_allow_html=True)

sm = get_state_manager()
handler = TextInputHandler()

if "profile_id" not in st.session_state:
    st.session_state.profile_id = ""
if "verified" not in st.session_state:
    st.session_state.verified = False
if "chat_open" not in st.session_state:
    st.session_state.chat_open = True
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "👋 Welcome to the AutoML Assistant! Sign in or create an account to get started."}
    ]
if "timer_completion_audio_path" not in st.session_state:
    st.session_state.timer_completion_audio_path = None
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False
if "last_tts_path" not in st.session_state:
    st.session_state.last_tts_path = None
# Prevents duplicate button render while _handle_voice_command is running
if "recording_active" not in st.session_state:
    st.session_state.recording_active = False

# Streamlit hack for live training metrics
if sm.get("training_status") == "training":
    time.sleep(0.5)
    st.rerun()

def log(msg: str):
    sm.append_log(msg)


def strip_wake_or_bypass(text: str) -> tuple[bool, str]:
    """Check for wake word and strip it. Returns (wake_detected, cleaned_text)."""
    cleaned = text.strip()
    if is_wake_word(cleaned):
        stripped = re.sub(
            r"^(hey|hi|okay|wake up|hello)\s+mycroft[\s,:\-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        sm.set_wake_detected(True)
        return True, stripped
    sm.set_wake_detected(False)
    return False, cleaned


def _strip_markdown_for_tts(text: str) -> str:
    """Remove markdown / emoji for cleaner TTS audio."""
    clean = re.sub(r"[*_`#]", "", text)
    clean = re.sub(r"\[.*?\]\(.*?\)", "", clean)           # links
    clean = re.sub(r":[a-z_]+:", "", clean)                 # :emoji_codes:
    # Remove common emoji ranges
    clean = re.sub(
        r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF\u2700-\u27BF\uFE00-\uFE0F\u200D]",
        "", clean,
    )
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def _format_timer_display(seconds: int) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def maybe_announce_timer_completion():
    timer = sm.get_timer_info()
    if not timer.get("exists"):
        return

    if timer.get("status") == "completed" and not sm.timer_completion_announced():
        msg = f"{timer.get('label', 'timer').title()} completed."
        sm.set_assistant_response(msg)
        sm.append_log(f"🔔 Timer completed: {timer.get('label', 'timer')}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"⏰ {msg}"})

        if st.session_state.tts_enabled:
            path = tts_speak(msg)
            if path:
                st.session_state.timer_completion_audio_path = path

        sm.mark_timer_completion_announced()

def process_command(user_text: str, bypass_wake_word: bool = False):
    """
    Core command processing pipeline.
    Enforces wake word — rejects commands without it (unless bypassed).
    """
    st.session_state.chat_history.append({"role": "user", "content": user_text})

    # ── Sleep command gate (before any NLU) ──────────────────
    # If the user types a sleep/stop command, silence the VA immediately
    # regardless of mic state — don't even bother routing through NLU.
    _SLEEP_PHRASES = {
        "sleep", "go to sleep", "stop listening", "be quiet", "quiet",
        "mute", "standby", "go standby", "rest", "pause listening",
        "stop the microphone", "turn off mic", "back to sleep", "sleep mode",
        "hey mycroft sleep", "hey mycroft stop listening", "hey mycroft be quiet",
    }
    if user_text.strip().lower() in _SLEEP_PHRASES:
        sm.set_asr_state("sleep")
        sm.set_wake_detected(False)
        sm.append_log("💤 VA put to sleep by chat command")
        sleep_msg = "Going to sleep. 💤 Say *'hey mycroft'* to wake me up again."
        st.session_state.chat_history.append({"role": "assistant", "content": sleep_msg})
        return

    # ── Wake word gate ───────────────────────────────────────
    ok, cleaned = strip_wake_or_bypass(user_text)
    
    if bypass_wake_word and not ok:
        cleaned = user_text.strip() # keep user text since no text-based wake word was found, just audio
        ok = True
    
    # Allow pure greetings/farewells through without a wake word (quality of life for UI chat)
    ui_bypasses = {
        "hello", "hi", "hey", "good morning", "good evening", "good afternoon", "greetings", "morning", "sup",
        "bye", "goodbye", "good bye", "see ya", "see you later", "good night", "farewell"
    }
    if not ok and user_text.strip().lower() in ui_bypasses:
        ok = True
        cleaned = user_text.strip().lower()

    if not ok:
        reject_msg = (
            "🔇 I only respond to commands that start with the wake word **'hey mycroft'**.\n\n"
            "Try saying: *'hey mycroft load iris dataset'*"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": reject_msg})
        log(f"⚠️ Wake word missing — rejected: '{user_text[:40]}'")
        return

    # If the user only said the wake word ("Hey mycroft"), convert it into a greeting
    if not cleaned:
        cleaned = "hello"

    # ── NLU + Routing ────────────────────────────────────────
    nlu = understand(cleaned)
    result = route_command(nlu)

    sm.set_transcript(user_text)
    sm.set_assistant_response(result["message"])
    log(f"Chat command → {nlu['intent']}")

    # Build rich response showing what was understood
    intent_badge = nlu["intent"].replace("_", " ").title()
    assistant_text = (
        f"🧠 **Intent:** `{intent_badge}`\n\n"
        f"{result['message']}"
    )

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})

    # ── TTS ──────────────────────────────────────────────────
    if st.session_state.tts_enabled:
        tts_clean = _strip_markdown_for_tts(result["message"])
        if tts_clean:
            path = tts_speak(tts_clean)
            if path:
                st.session_state.last_tts_path = path


def _status_class(status: str) -> str:
    return {
        "idle": "status-idle",
        "training": "status-training",
        "completed": "status-completed",
        "paused": "status-paused",
        "stopped": "status-stopped",
    }.get(status, "status-idle")

def render_timer_panel():
    st.markdown("#### ⏲️ Timer")

    timer_info = sm.get_timer_info()

    if not timer_info:
        st.caption("No active timer.")
        return

    remaining = int(timer_info.get("remaining_seconds", 0))
    mins, secs = divmod(remaining, 60)
    hours, mins = divmod(mins, 60)

    label = timer_info.get("label", "timer").title()
    status = timer_info.get("status", "running")

    if hours > 0:
        time_text = f"{hours:02d}:{mins:02d}:{secs:02d}"
    else:
        time_text = f"{mins:02d}:{secs:02d}"

    status_color = "#3fb950" if status == "running" else "#f85149"

    st.markdown(
        f"""
        <div style="
            background: rgba(22, 27, 34, 0.88);
            border: 1px solid rgba(48, 54, 61, 0.75);
            border-radius: 16px;
            padding: 18px 16px;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        ">
            <div style="
                font-size: 0.85rem;
                color: #8b949e;
                margin-bottom: 8px;
                font-weight: 600;
                letter-spacing: 0.4px;
            ">
                ACTIVE TIMER
            </div>
            <div style="
                font-size: 1rem;
                color: #c9d1d9;
                margin-bottom: 10px;
                font-weight: 600;
            ">
                {label}
            </div>
            <div style="
                font-size: 2.2rem;
                font-weight: 700;
                color: #58a6ff;
                line-height: 1.1;
                margin-bottom: 10px;
                font-variant-numeric: tabular-nums;
            ">
                {time_text}
            </div>
            <div style="
                font-size: 0.85rem;
                font-weight: 700;
                color: {status_color};
                text-transform: uppercase;
                letter-spacing: 0.6px;
            ">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_dataset_panel(state: dict):
    st.markdown("#### 📚 Dataset")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        ds_name = state.get("dataset") or "—"
        st.markdown(f"**Current dataset:** `{ds_name}`")
        info = state.get("dataset_info", {})
        if info:
            cols_display = {
                "Rows": info.get("rows", "—"),
                "Columns": info.get("columns", "—"),
                "Target": info.get("target_name") or "—",
            }
            st.markdown(" · ".join(f"**{k}:** {v}" for k, v in cols_display.items()))
            profile = info.get("profile", {})
            if profile:
                st.caption(f"Task: {profile.get('task_family', '—')} · Model: {profile.get('suggested_model', '—')}")
        else:
            st.markdown(
                '<div style="padding:12px 0;color:#475569;font-size:0.88rem;">'
                '📭 No dataset loaded yet<br>'
                '<span style="font-size:0.8rem;color:#64748b;">Try: <code>hey mycroft load iris dataset</code></span>'
                '</div>',
                unsafe_allow_html=True,
            )
    with col_b:
        preview = state.get("dataset_preview", [])
        if preview:
            st.dataframe(pd.DataFrame(preview), use_container_width=True, height=220)
        else:
            st.markdown(
                '<div style="padding:30px 0;text-align:center;color:#475569;font-size:0.85rem;">'
                '📋 Preview will appear here once a dataset is loaded'
                '</div>',
                unsafe_allow_html=True,
            )


def render_code_panel(state: dict):
    st.markdown("#### 💻 Code")
    tab_py, tab_ipynb, tab_ref = st.tabs(["🐍 Runnable .py", "📓 Notebook (JSON)", "📘 Kaggle Reference"])

    with tab_py:
        code = state.get("generated_code_py", "")
        if code:
            st.code(code, language="python")
            st.download_button(
                "⬇️ Download .py",
                data=code.encode("utf-8"),
                file_name=f"{state.get('dataset') or 'experiment'}.py",
                mime="text/x-python",
                key="dl_py_code",
            )
        else:
            st.markdown(
                '<div style="padding:20px 0;text-align:center;color:#475569;font-size:0.85rem;">'
                '🐍 No Python code generated yet<br>'
                '<span style="font-size:0.8rem;color:#64748b;">Try: <code>hey mycroft load corresponding code</code></span>'
                '</div>',
                unsafe_allow_html=True,
            )

    with tab_ipynb:
        notebook = state.get("generated_code_ipynb", "")
        if notebook:
            st.code(notebook, language="json")
            st.download_button(
                "⬇️ Download .ipynb",
                data=notebook.encode("utf-8"),
                file_name=f"{state.get('dataset') or 'experiment'}.ipynb",
                mime="application/x-ipynb+json",
                key="dl_ipynb_code",
            )
        else:
            st.markdown(
                '<div style="padding:16px 0;text-align:center;color:#475569;font-size:0.85rem;">'
                '📓 No notebook generated yet'
                '</div>',
                unsafe_allow_html=True,
            )

    with tab_ref:
        ref_code = state.get("reference_code", "")
        ref_fmt = state.get("reference_code_format", "text")
        title = state.get("code_title", "")
        if title:
            st.markdown(f"**Source:** {title}")
        if ref_code:
            st.code(ref_code[:30000], language="python" if ref_fmt == "py" else "json")
        else:
            st.markdown(
                '<div style="padding:16px 0;text-align:center;color:#475569;font-size:0.85rem;">'
                '📘 No Kaggle reference code loaded'
                '</div>',
                unsafe_allow_html=True,
            )


def render_outputs_panel(state: dict):
    st.markdown("#### 🖨️ Output")
    outputs = state.get("outputs", [])

    status = state.get("training_status", "idle")
    model_name = state.get("model") or "—"
    epoch_cur = state.get("epoch_current", 0)
    epoch_tot = state.get("epochs_total", 0)

    status_cls = _status_class(status)
    st.markdown(
        f'<span class="status-badge {status_cls}">{status.upper()}</span> '
        f'&nbsp; Epoch **{epoch_cur}** / **{epoch_tot}** &nbsp;·&nbsp; Model: **{model_name}**',
        unsafe_allow_html=True
    )

    # Output details are hidden until requested by user (or code manually ran)
    if not state.get("results_requested", False) and state.get("training_status") != "idle":
        st.info("Metrics are being tracked. Say 'hey mycroft show results' or 'tell the results' to reveal them when ready.")
        
        # Real time chart sneak peek
        if state.get("loss_history") or state.get("accuracy_history"):
            chart_df = pd.DataFrame({
                "Loss": state.get("loss_history", []),
                "Accuracy": state.get("accuracy_history", []),
            })
            chart_df.index = range(1, len(chart_df) + 1)
            st.line_chart(chart_df, use_container_width=True)
            
        return

    if state.get("loss_history") or state.get("accuracy_history"):
        chart_df = pd.DataFrame(
            {
                "Loss": state.get("loss_history", []),
                "Accuracy": state.get("accuracy_history", []),
            }
        )
        chart_df.index = range(1, len(chart_df) + 1)
        st.line_chart(chart_df, use_container_width=True)

    if not outputs:
        code_output = state.get("code_output_text", "")
        if code_output:
            st.text(code_output)
        else:
            if state.get("training_status") == "idle":
                st.markdown(
                    '<div style="padding:16px 0;text-align:center;color:#475569;font-size:0.85rem;">'
                    '📤 No output yet<br>'
                    '<span style="font-size:0.8rem;color:#64748b;">Ask me to run code or start training</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            
        # Download weights — only show after training has completed
        if state.get("training_status") in ("completed",) and os.path.exists("artifacts/model_weights.pkl"):
            st.markdown("---")
            with open("artifacts/model_weights.pkl", "rb") as f:
                st.download_button(
                    "⬇️ Download Model Weights (.pkl)",
                    data=f.read(),
                    file_name=f"{state.get('dataset', 'model')}_weights.pkl",
                    mime="application/octet-stream",
                    key="dl_weights",
                )
        return

    for idx, item in enumerate(outputs):
        kind = item.get("type")
        title = item.get("title", f"Output {idx + 1}")
        st.markdown(f"**{title}**")

        if kind == "text":
            st.text(item.get("content", ""))
        elif kind == "table":
            st.dataframe(pd.DataFrame(item.get("records", [])), use_container_width=True)
        elif kind == "image":
            path = item.get("path")
            if path and Path(path).exists():
                st.image(path, use_container_width=True)
        elif kind == "file":
            path = item.get("path")
            if path and Path(path).exists():
                with open(path, "rb") as f:
                    st.download_button(
                        f"⬇️ Download {Path(path).name}",
                        data=f.read(),
                        file_name=Path(path).name,
                        mime=item.get("mime", "application/octet-stream"),
                        key=f"download_{idx}_{Path(path).name}",
                    )


# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION PANEL — Camera/Voice gated behind explicit action
# ═══════════════════════════════════════════════════════════════

def render_auth_panel():
    st.markdown(
        '<div style="text-align:center;padding:8px 0 4px;">'
        '<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;">🔐 Welcome to Mycroft</div>'
        '<div style="font-size:0.8rem;color:#64748b;margin-top:4px;">Sign in or create an account to start</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    auth_tab_signin, auth_tab_signup, auth_tab_bio = st.tabs(["🔑 Sign In", "📝 Sign Up", "🧬 Biometrics"])

    # ── Sign Up ──────────────────────────────────────────────
    with auth_tab_signup:
        new_user = st.text_input("Choose a username", key="chat_signup_user", placeholder="e.g. john_doe")
        new_pw = st.text_input("New password", type="password", key="chat_signup_password")
        conf_pw = st.text_input("Confirm password", type="password", key="chat_signup_confirm")
        if st.button("Create Account", key="chat_signup_btn", type="primary", use_container_width=True):
            uid = new_user.strip() if new_user.strip() else DEFAULT_USER_ID
            if new_pw != conf_pw:
                st.error("Passwords do not match.")
            elif len(new_pw) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                result = enroll_password(uid, new_pw)
                if result["success"]:
                    st.session_state.profile_id = uid
                    st.success(f"✅ Account created for **{uid}**. You can sign in now.")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": f"Account created for **{uid}**. Sign in to start using the assistant."}
                    )
                else:
                    st.error(result["message"])

    # ── Sign In ──────────────────────────────────────────────
    with auth_tab_signin:
        login_user = st.text_input("Username", key="chat_signin_user", placeholder="e.g. john_doe")
        password = st.text_input("Password", type="password", key="chat_signin_password")
        if st.button("Sign In", key="chat_signin_btn", type="primary", use_container_width=True):
            uid = login_user.strip() if login_user.strip() else DEFAULT_USER_ID
            result = verify_profile_password(uid, password)
            if result["verified"]:
                st.session_state.profile_id = uid
                st.session_state.verified = True
                sm.set_verified(True)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"✅ Welcome back, **{uid}**! You can now use voice or text commands. Try: *'hey mycroft load iris dataset'*"}
                )
                st.rerun()
            else:
                st.error(result["message"])

    # ── Biometrics — gated behind explicit user action ───────
    with auth_tab_bio:
        bio_user = st.text_input(
            "Username for biometrics", key="chat_bio_user",
            value=st.session_state.profile_id or DEFAULT_USER_ID,
        )
        uid = bio_user.strip() if bio_user.strip() else DEFAULT_USER_ID
        est = enrollment_status(uid)
        st.caption(
            f"Face enrolled: {'✅' if est['face_enrolled'] else '❌'} · "
            f"Voice enrolled: {'✅' if est['voice_enrolled'] else '❌'}"
        )

        bio_mode = st.radio(
            "Choose biometric method:",
            ["📸 Face", "🎤 Voice"],
            key="bio_mode_radio",
            horizontal=True,
        )

        # ── FACE BIOMETRICS ──────────────────────────────────
        if "Face" in bio_mode:
            bio_action = st.radio(
                "Action:", ["Enroll Face", "Verify & Sign In"],
                key="face_action_radio", horizontal=True,
            )

            # Camera only opens when the user clicks the button
            if st.button("📷 Open Camera", key="btn_open_camera", type="primary"):
                st.session_state.show_camera = True

            if st.session_state.show_camera:
                face_img = st.camera_input("Capture your face", key="chat_face_camera")
                if face_img is not None:
                    if "Enroll" in bio_action:
                        if st.button("✅ Enroll My Face", key="btn_do_enroll_face", type="primary"):
                            with st.spinner("🔍 Detecting face and creating template..."):
                                arr = np.array(Image.open(face_img).convert("RGB"))
                                result = enroll_face(uid, arr)
                            if result["success"]:
                                st.success(result["message"])
                                st.session_state.show_camera = False
                                st.balloons()
                            else:
                                st.error(result["message"])
                    else:  # Verify
                        if st.button("🔓 Verify Face & Sign In", key="btn_do_verify_face", type="primary"):
                            with st.spinner("🔍 Comparing against stored template..."):
                                arr = np.array(Image.open(face_img).convert("RGB"))
                                result = verify_face(uid, arr)
                            if result["verified"]:
                                st.session_state.profile_id = uid
                                st.session_state.verified = True
                                sm.set_verified(True)
                                st.session_state.show_camera = False
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": f"✅ Face verified. Welcome, **{uid}**!"}
                                )
                                st.rerun()
                            else:
                                st.error(result["message"])
                else:
                    st.caption("👆 Take a photo above to proceed.")

        # ── VOICE BIOMETRICS ─────────────────────────────────
        else:
            bio_action = st.radio(
                "Action:", ["Enroll Voice", "Verify & Sign In"],
                key="voice_action_radio", horizontal=True,
            )
            duration = st.slider(
                "Recording duration (seconds)", 3, 12, 7,
                key="chat_voice_seconds",
            )
            st.info(
                "🎙️ Speak naturally — say anything. Your voice *characteristics* "
                "(pitch, timbre) are compared, not the words."
            )

            btn_label = "🎤 Record & Enroll Voice" if "Enroll" in bio_action else "🎤 Record & Verify Voice"
            if st.button(btn_label, key="btn_bio_voice_action", type="primary"):
                # Show recording indicator
                st.markdown(
                    f'<div class="recording-indicator">'
                    f'<div class="recording-dot"></div>'
                    f'🔴 Recording for {duration}s — please speak now…'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                try:
                    with st.spinner(f"🔴 Recording for {duration}s..."):
                        cap = AudioCapture()
                        audio = cap.record_fixed(duration=duration)

                    peak = float(np.max(np.abs(audio)))
                    if peak < 0.01:
                        st.warning("⚠️ Very quiet recording. Check mic and speak louder.")

                    # Save and play back the recording
                    from modules.A_user_access.voice_biometrics import LAST_RECORDING_PATH, save_wav
                    save_wav(np.asarray(audio, dtype=np.float32).flatten(), LAST_RECORDING_PATH)
                    if os.path.isfile(LAST_RECORDING_PATH):
                        st.markdown("🔊 **Listen to your recording:**")
                        st.audio(LAST_RECORDING_PATH, format="audio/wav")

                    if "Enroll" in bio_action:
                        with st.spinner("Creating speaker embedding..."):
                            result = enroll_voice(uid, audio)
                        if result["success"]:
                            st.success(result["message"])
                            st.balloons()
                        else:
                            st.error(result["message"])
                    else:
                        with st.spinner("Comparing against stored voiceprint..."):
                            result = verify_voice(uid, audio)
                        if result["verified"]:
                            st.session_state.profile_id = uid
                            st.session_state.verified = True
                            sm.set_verified(True)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": f"✅ Voice verified. Welcome, **{uid}**!"}
                            )
                            st.rerun()
                        else:
                            st.error(result["message"])

                except Exception as e:
                    st.error(f"Microphone error: {e}")


# ═══════════════════════════════════════════════════════════════
# CHAT PANEL — Scrollable, with voice recording UX and TTS
# ═══════════════════════════════════════════════════════════════

def render_timer_panel_compact():
    timer = sm.get_timer_info()

    if not timer.get("exists"):
        st.markdown(
            """
            <div style="
                background: rgba(12, 16, 25, 0.75);
                border: 1px solid rgba(56, 97, 150, 0.15);
                border-radius: 14px;
                padding: 10px 14px;
                min-height: 60px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-bottom: 10px;
                backdrop-filter: blur(8px);
            ">
                <div style="font-size:0.74rem;color:#64748b;font-weight:600;letter-spacing:0.4px;text-transform:uppercase;">
                    ⏲️ Timer
                </div>
                <div style="font-size:0.85rem;color:#475569;margin-top:4px;">
                    No active timer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    label = str(timer.get("label", "timer")).title()
    status = str(timer.get("status", "idle")).upper()

    remaining_seconds = int(timer.get("remaining_seconds", 0))
    hours, rem = divmod(remaining_seconds, 3600)
    mins, secs = divmod(rem, 60)

    if hours > 0:
        remaining = f"{hours:02d}:{mins:02d}:{secs:02d}"
    else:
        remaining = f"{mins:02d}:{secs:02d}"

    status_color = {
        "RUNNING": "#4ade80",
        "PAUSED": "#fbbf24",
        "COMPLETED": "#f87171",
        "STOPPED": "#94a3b8",
    }.get(status, "#94a3b8")

    timer_border = "rgba(59, 130, 246, 0.2)" if status == "RUNNING" else "rgba(56, 97, 150, 0.15)"
    timer_glow = "0 0 20px rgba(59, 130, 246, 0.06)" if status == "RUNNING" else "none"

    st.markdown(
        f"""
        <div style="
            background: rgba(12, 16, 25, 0.8);
            border: 1px solid {timer_border};
            border-radius: 14px;
            padding: 10px 14px;
            min-height: 64px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-bottom: 10px;
            box-shadow: {timer_glow};
            backdrop-filter: blur(8px);
            transition: all 0.3s ease;
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                gap:8px;
            ">
                <div style="font-size:0.74rem;color:#64748b;font-weight:600;letter-spacing:0.4px;text-transform:uppercase;">
                    ⏲️ Timer · {label}
                </div>
                <div style="font-size:0.68rem;font-weight:700;color:{status_color};letter-spacing:0.5px;text-transform:uppercase;">
                    {status}
                </div>
            </div>
            <div style="
                font-size:1.5rem;
                font-weight:800;
                color:#60a5fa;
                margin-top:5px;
                line-height:1.1;
                font-variant-numeric: tabular-nums;
                letter-spacing: 1px;
            ">
                {remaining}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_chat_panel():
    with st.container(**_CONTAINER_KW):
        # ── Header ───────────────────────────────────────────
        st.markdown("### 💬 Assistant")
        uid = st.session_state.profile_id or "not signed in"
        verified = st.session_state.verified
        badge_cls = "badge-yes" if verified else "badge-no"
        badge_txt = "Verified ✓" if verified else "Not signed in"
        st.markdown(
            f'👤 **{uid}** &nbsp; <span class="verified-badge {badge_cls}">{badge_txt}</span>',
            unsafe_allow_html=True
        )

        if not st.session_state.verified:
            render_auth_panel()
            return

        # ── Example commands ─────────────────────────────────
        with st.expander("💡 Quick commands", expanded=False):
            examples = [
                ("📂", "load iris dataset"),
                ("💻", "load corresponding code"),
                ("⚙️", "set learning rate to 0.01"),
                ("🏗️", "set layers to 4"),
                ("🚀", "start training"),
                ("📊", "show results"),
                ("⏲️", "set a timer for 2 min"),
                ("🌦️", "weather in Ottawa"),
                ("🔍", "search dataset fraud"),
                ("❓", "help"),
            ]
            cards_html = ''.join(
                f'<span class="cmd-card">{icon} hey mycroft {cmd}</span>'
                for icon, cmd in examples
            )
            st.markdown(
                f'<div style="line-height:2.2;">{cards_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Chat history in a scrollable container ───────────
        chat_container = st.container(height=420)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # ── TTS playback (autoplay last response) ────────────
        if st.session_state.timer_completion_audio_path and os.path.isfile(st.session_state.timer_completion_audio_path):
            st.audio(st.session_state.timer_completion_audio_path, format="audio/mp3", autoplay=True)
            st.session_state.timer_completion_audio_path = None
        
        if st.session_state.get("last_tts_path") and os.path.isfile(st.session_state.last_tts_path):
            import time
            current = time.time()
            
            # New audio triggered? Compute the math once
            if st.session_state.get("playing_tts_path") != st.session_state.last_tts_path:
                try:
                    size = os.path.getsize(st.session_state.last_tts_path)
                    # Using ~3000 bytes/sec to guarantee Streamlit overestimates duration
                    duration_sec = (size / 3000.0) + 1.0
                    sm.set_tts_expected_end_time(current + duration_sec)
                    st.session_state.playing_tts_path = st.session_state.last_tts_path
                except Exception:
                    pass
            
            # Keep audio element mounted while playing to avoid Streamlit rerun wipeouts
            if current < sm.get_state().get("tts_expected_end_time", 0):
                st.audio(st.session_state.last_tts_path, format="audio/mp3", autoplay=True)
            else:
                # Playback has run its course, clean up variables explicitly
                st.session_state.last_tts_path = None
                st.session_state.playing_tts_path = None
        
        # ── Voice recording & Audio controls ───────────────────
        st.markdown("---")

        action_col, setting_col = st.columns([1.6, 1.4], gap="medium")
        
        with setting_col:
            voice_duration = st.slider(
                "⏱️ Manual Record Length", 2, 10, 5,
                key="voice_cmd_duration",
                help="How long to record when you manually press the mic button",
            )
            st.session_state.tts_enabled = st.toggle(
                "🔊 Speak Responses",
                value=st.session_state.tts_enabled,
                key="tts_toggle",
                help="Assistant will reply using Text-to-Speech",
            )

        with action_col:
            st.markdown("<div style='margin-top: 6px;'></div>", unsafe_allow_html=True)
            
            # The toggle automatically triggers a script rerun in Streamlit when changed
            mic_active = st.toggle(
                "🎙️ Hands-Free Mode (Wake Word)",
                value=sm.get_state().get("mic_active", False),
                key="mic_toggle",
                help="Enable continuous background mic to listen for the wake word",
            )
            
            # Detect changes and spin up/down the background thread
            if mic_active != sm.get_state().get("mic_active", False):
                sm.set_mic_active(mic_active)
                from modules.B_voice_processing.continuous_asr import ContinuousASR
                casr = ContinuousASR(sm)
                if mic_active:
                    casr.start()
                else:
                    casr.stop()
                # Rely on natural rerun

            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)

            if sm.get_state().get("mic_active"):
                asr_state = sm.get_asr_state()
                if asr_state == "active":
                    st.info("🟢 **Active** — Recording command…")
                elif asr_state == "listening":
                    st.warning("⏳ **Listening** — Transcribing…")
                else:
                    st.success(f"💤 **Sleep** — Waiting for *'{WAKE_WORD.title()}'*")
            
            # Show manual record button regardless of hands-free state
            if not st.session_state.get("recording_active"):
                if st.button(
                    f"🎤 Record Command ({voice_duration}s)",
                    key="chat_record_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.recording_active = True
                    st.rerun()
            else:
                # If recording_active is True, handle voice command cleanly
                st.markdown(
                    '<div class="recording-indicator" style="margin-bottom:8px;"><div class="recording-dot"></div>Recording… Speak now</div>',
                    unsafe_allow_html=True,
                )
                _handle_voice_command(voice_duration)



        # ── Action buttons row ───────────────────────────────
        clear_col, log_col = st.columns(2)
        if clear_col.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Chat cleared. Ready for the next command."}
            ]
            st.rerun()
        if log_col.button("🔒 Sign Out", key="signout_btn", use_container_width=True):
            st.session_state.verified = False
            st.session_state.profile_id = ""
            sm.set_verified(False)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": "🔒 Signed out. Please sign in again to continue."}
            )
            st.rerun()

        # ── Text input ───────────────────────────────────────
        if st.session_state.get("pending_transcript_approval") and not st.session_state.get("confirming_intent"):
            with st.form("edit_transcript_form", clear_on_submit=True):
                edited_prompt = st.text_input(
                    "🎙️ Let's fix that. Edit your command below:", 
                    value=st.session_state.pending_transcript_approval
                )
                submitted = st.form_submit_button("➡️ Send Command")
                
                if submitted:
                    st.session_state.pending_transcript_approval = None
                    st.session_state.confirming_intent = False
                    if edited_prompt.strip():
                        process_command(edited_prompt.strip(), bypass_wake_word=True)
                    st.rerun()
        else:
            prompt = st.chat_input("Type a command (e.g. 'hey mycroft load iris dataset')")
            if prompt:
                process_command(prompt)
                st.rerun()

        # Check if the Continuous thread dropped off a pending action
        if sm.get_state().get("pending_audio_path"):
            audio_path = sm.get_state().get("pending_audio_path")
            
            from modules.B_voice_processing.speech_to_text import SpeechToText
            st.session_state.chat_history.append(
                {"role": "user", "content": "🎙️ [Command received from voice]"}
            )
            st.markdown(
                '<span class="pipeline-stage stage-active">⏳ Transcribing with Whisper...</span>',
                unsafe_allow_html=True,
            )
            
            try:
                stt = SpeechToText()
                transcript = stt.transcribe(audio_path)
                
                # Strip implicit wake words if perfectly captured
                import re
                clean = re.sub(r'^(hey|hi|okay|wake up|hello)\s+mycroft[\s,:\-]*', '', transcript, flags=re.IGNORECASE).strip()
                
                if clean:
                    if st.session_state.get("confirming_intent"):
                        # THIS IS A VOICE REPLY TO A CONFIRMATION PROMPT
                        ans = clean.lower()
                        # Very simple intent check for yes/no
                        if any(w in ans for w in ["yes", "yeah", "yep", "sure", "correct", "do it", "ok", "okay"]):
                            st.session_state.chat_history.pop()  # Remove "Command received"
                            st.session_state.chat_history.append({"role": "user", "content": f"🎙️ {clean}"})
                            
                            cmd = st.session_state.get("pending_transcript_approval", "")
                            st.session_state.confirming_intent = False
                            st.session_state.pending_transcript_approval = None
                            
                            process_command(cmd, bypass_wake_word=True)
                        elif any(w in ans for w in ["no", "nah", "nope", "incorrect", "cancel", "stop", "don't"]):
                            st.session_state.chat_history.pop()
                            st.session_state.chat_history.append({"role": "user", "content": f"🎙️ {clean}"})
                            
                            # Announce cancellation, but leave pending_transcript_approval active so they can edit it manually
                            from modules.B_voice_processing.tts import speak
                            tts_path = speak("Okay, I've cancelled that. You can edit it manually.")
                            if tts_path:
                                st.session_state.last_tts_path = tts_path
                                
                            st.session_state.confirming_intent = False
                            st.session_state.chat_history.append({"role": "assistant", "content": "Voice command cancelled. You can edit the text below."})
                        else:
                            st.session_state.chat_history.pop()
                            st.session_state.chat_history.append({"role": "assistant", "content": f"I didn't catch a clear yes or no (heard '{clean}'). Try using the execute or cancel buttons below."})
                            st.session_state.confirming_intent = False
                    
                    else:
                        # THIS IS A BRAND NEW COMMAND
                        st.session_state.chat_history[-1]["content"] = f"🎙️ {clean}"
                        st.session_state.chat_history.pop() 
                        
                        # INSTEAD OF EXECUTING, PROMPT USER FOR CONFIRMATION
                        st.session_state.pending_transcript_approval = clean
                        st.session_state.confirming_intent = True
                        
                        # Trigger TTS confirmation
                        from modules.B_voice_processing.tts import speak
                        tts_path = speak(f"I heard you say: {clean}... Is that correct?")
                        if tts_path:
                            st.session_state.last_tts_path = tts_path
                        
                        # Tell ASR to turn on IMMEDIATELY after TTS finishes (skip Wake Word)
                        sm.set_force_active(True)
                else:
                    st.session_state.chat_history[-1]["content"] = "🔇 Ignored (Only heard wake word)."
                    
            except Exception as e:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Voice command failed: {e}"})
                
            # Clear flag to free stream loop
            sm.set_pending_audio_path(None)
            st.rerun()

        # ── Transcript Confirmation Block ────────────────────
        if st.session_state.get("pending_transcript_approval"):
            if st.session_state.get("confirming_intent"):
                st.info("🎙️ I heard you say: **" + st.session_state.pending_transcript_approval + "**")
                st.markdown(
                    '<span class="pipeline-stage stage-active">Listening for your answer (Yes/No)...</span>',
                    unsafe_allow_html=True,
                )


def _handle_voice_command(duration: int):
    """
    Full voice command pipeline with staged feedback:
    1. Recording → indicator + spinner
    2. Playback → st.audio
    3. Transcription → staged progress
    4. Wake word check + command processing
    """
    # ── Stage 1: Recording ───────────────────────────────
    st.markdown(
        f'<div class="recording-indicator">'
        f'<div class="recording-dot"></div>'
        f'Recording for {duration}s — speak now…'
        f'</div>',
        unsafe_allow_html=True,
    )

    audio = None
    try:
        with st.spinner(f"🔴 Recording for {duration}s..."):
            cap = AudioCapture()
            audio = cap.record_fixed(duration=duration)
    except Exception as e:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"⚠️ Microphone error: {e}"}
        )
        return

    if audio is None or len(audio) == 0:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "⚠️ No audio was captured. Check your microphone."}
        )
        return

    peak = float(np.max(np.abs(audio)))
    if peak < 0.005:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "🔇 Recording is silent. Check that your microphone is enabled and not muted."}
        )
        return

    # ── Stage 2: Save & Playback ─────────────────────────
    tmp_path = Path("artifacts") / "last_command.wav"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    cap.save(audio, str(tmp_path))
    st.markdown("🔊 **Your recording:**")
    st.audio(str(tmp_path), format="audio/wav")

    # ── Stage 3: Transcription ───────────────────────────
    st.markdown(
        '<span class="pipeline-stage stage-active">⏳ Transcribing with Whisper...</span>',
        unsafe_allow_html=True,
    )

    transcript = ""
    with st.spinner("⏳ Transcribing with Whisper..."):
        try:
            stt = SpeechToText()
            transcript = stt.transcribe(str(tmp_path))
        except Exception as e:
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ Transcription error: {e}"}
            )
            return

    if not transcript.strip():
        hint = get_last_transcribe_error() or "No speech detected."
        st.session_state.chat_history.append(
            {"role": "assistant", "content": f"🔇 Could not transcribe the recording.\n\n*Reason: {hint}*"}
        )
        return

    # Show transcription result
    st.markdown(
        f'<span class="pipeline-stage stage-done">✅ Transcribed</span>',
        unsafe_allow_html=True,
    )
    st.success(f"🗣️ **Heard:** *\"{transcript}\"*")
    log(f"🗣️ Voice transcribed: '{transcript}'")

    # ── Stage 4: Process command ─────────────────────────
    st.markdown(
        '<span class="pipeline-stage stage-active">🧠 Processing command...</span>',
        unsafe_allow_html=True,
    )

    with st.spinner("🧠 Processing command..."):
        process_command(transcript)

    st.session_state.recording_active = False
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<h1 style="margin-bottom:0">🤖 AutoML Workspace Assistant</h1>'
    '<p style="color:#64748b;margin-top:4px;font-size:0.88rem;font-weight:400;">'
    'Workspace on the left &nbsp;·&nbsp; Assistant on the right'
    '&nbsp;&nbsp;<span style="display:inline-block;padding:2px 8px;border-radius:8px;'
    'font-size:0.68rem;font-weight:700;background:rgba(59,130,246,0.1);'
    'color:#60a5fa;border:1px solid rgba(59,130,246,0.2);letter-spacing:0.3px;'
    'vertical-align:middle;">v2.0</span></p>',
    unsafe_allow_html=True,
)

left, right = st.columns([2.25, 1], gap="large")

state = sm.get_state()

with left:
    with st.container(**_CONTAINER_KW):
        render_dataset_panel(state)

    with st.container(**_CONTAINER_KW):
        render_code_panel(state)

    with st.container(**_CONTAINER_KW):
        render_outputs_panel(sm.get_state())

with right:
    render_timer_panel_compact()
    render_chat_panel()

# Auto-refresh UI during active training simulator
maybe_announce_timer_completion()

state_now = sm.get_state()
timer_now = sm.get_timer_info()

if state_now.get("training_status") == "training" or timer_now.get("status") == "running" or state_now.get("mic_active"):
    import time
    time.sleep(1.0)
    st.rerun()