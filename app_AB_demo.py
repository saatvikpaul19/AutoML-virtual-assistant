"""
app_AB_demo.py — Streamlit Demo for Sections A + B
====================================================
Demonstrates Modules 1-5 working end-to-end inside a Streamlit UI.
This is NOT the full app — it is the integration testbed for the
voice pipeline before the backend (Section D) is connected.

Run with:  streamlit run app_AB_demo.py

Sections shown
--------------
  ① Verification Gate        — password, face, or voice enrollment + sign-in (Module 1)
  ② Voice or Text Command    — record audio (Module 4)
                               or type a command (Module 3)
  ③ Wake Word Check          — Module 2
  ④ Transcription            — Module 5
  ⑤ Raw Command Output       — ready for Module 6 (Intent Detection)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import inspect
import shutil
import urllib.request

import streamlit as st
import numpy as np
from PIL import Image

from config import DEFAULT_USER_ID

# Bordered containers need Streamlit ≥1.29; omit kwarg on older versions.
_CONTAINER_KW = (
    {"border": True}
    if "border" in inspect.signature(st.container).parameters
    else {}
)

# ── Streamlit page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Voice Assistant — A+B Demo",
    page_icon="🎙️",
    layout="wide",
)

# ── Module imports ──────────────────────────────────────────────────────
from modules.A_user_access.user_verification import (
    verify_passcode,
    reset_state as reset_verification,
    enrollment_status,
    enroll_face,
    verify_face,
    enroll_voice,
    verify_voice,
    enroll_password,
    verify_profile_password,
    clear_biometric_enrollment,
)
from modules.A_user_access.wake_word import is_wake_word
from modules.A_user_access.text_input_handler import TextInputHandler
from modules.B_voice_processing.audio_capture import AudioCapture
from modules.B_voice_processing.speech_to_text import (
    SpeechToText,
    get_last_transcribe_error,
    probe_whisper_model,
)

# ── Session state initialisation ────────────────────────────────────────
if "verified" not in st.session_state:
    st.session_state.verified = False
if "awake" not in st.session_state:
    st.session_state.awake = False
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "raw_command" not in st.session_state:
    st.session_state.raw_command = ""
if "event_log" not in st.session_state:
    st.session_state.event_log = []
if "profile_id" not in st.session_state:
    st.session_state.profile_id = DEFAULT_USER_ID

handler = TextInputHandler()


def log(msg: str):
    """Append timestamped message to event log."""
    from datetime import datetime
    st.session_state.event_log.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )


def run_stt_health_checks() -> dict:
    """Mic level, TLS to Google, Whisper load, SpeechRecognition import, ffmpeg on PATH."""
    out: dict = {}

    try:
        capture = AudioCapture()
        audio = np.asarray(capture.record_fixed(duration=1), dtype=np.float32).flatten()
        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        if peak < 1e-5:
            out["microphone_1s_capture"] = {
                "ok": False,
                "detail": f"Capture looks silent (peak≈{peak:.1e}). Check mic device, volume, and macOS permissions.",
            }
        else:
            out["microphone_1s_capture"] = {
                "ok": True,
                "detail": f"Captured 1s — peak={peak:.4f}, rms={rms:.4f}.",
            }
    except Exception as e:
        out["microphone_1s_capture"] = {"ok": False, "detail": repr(e)}

    try:
        with urllib.request.urlopen("https://www.google.com", timeout=8) as resp:
            code = getattr(resp, "status", None) or resp.getcode()
        out["https_google_tls"] = {
            "ok": True,
            "detail": f"TLS request to www.google.com OK (HTTP {code}). Google STT fallback needs working HTTPS.",
        }
    except Exception as e:
        out["https_google_tls"] = {"ok": False, "detail": repr(e)}

    whisper_ok, whisper_msg = probe_whisper_model()
    out["whisper_model_load"] = {"ok": whisper_ok, "detail": whisper_msg}

    try:
        import speech_recognition as sr  # noqa: F401

        out["speech_recognition_import"] = {"ok": True, "detail": "Package imports OK."}
    except Exception as e:
        out["speech_recognition_import"] = {"ok": False, "detail": repr(e)}

    ffmpeg = shutil.which("ffmpeg")
    out["ffmpeg_on_path"] = {
        "ok": bool(ffmpeg),
        "detail": ffmpeg or "Not found (optional when passing numpy/WAV to Whisper; some installs still expect ffmpeg).",
    }

    can_whisper = bool(out["whisper_model_load"]["ok"])
    can_fallback = (
        bool(out["microphone_1s_capture"]["ok"])
        and bool(out["https_google_tls"]["ok"])
        and bool(out["speech_recognition_import"]["ok"])
    )
    passed = sum(1 for k, v in out.items() if k != "_summary" and v.get("ok"))
    total = len([k for k in out if k != "_summary"])
    out["_summary"] = {
        "ok": can_whisper or can_fallback,
        "detail": (
            f"{passed}/{total} checks green. "
            f"Local Whisper: {'OK' if can_whisper else 'not usable'}. "
            f"Google fallback path (mic + HTTPS + SpeechRecognition): {'OK' if can_fallback else 'not usable'}."
        ),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════════════

st.title("🎙️ AutoML Voice Assistant — Pipeline Demo (Modules 1–5)")
st.caption("CSI5180 Project · Sections A + B · Voice Pipeline End-to-End")

col_left, col_right = st.columns([1.6, 1], gap="large")

# ── LEFT COLUMN: Pipeline Stages ────────────────────────────────────────
with col_left:

    # ══════════════════════════════════════════════════════════════════
    # STAGE 1 — User Verification  (Module 1)
    # ══════════════════════════════════════════════════════════════════
    with st.container(**_CONTAINER_KW):
        st.subheader("① User Verification  (Module 1)")

        # ── Current status indicator ──
        status_icon = "🟢 Verified" if st.session_state.verified else "🔴 Not Verified"
        st.markdown(f"**Status:** {status_icon}")

        # ── Profile ID ──
        pid_in = st.text_input(
            "Profile ID (each profile has its own enrollment data)",
            value=st.session_state.profile_id,
            key="profile_id_input",
            help="Enrollment data is stored under data/enrollment/<profile_id>/",
        )
        st.session_state.profile_id = (pid_in.strip() or DEFAULT_USER_ID)

        # ── Enrollment status badges ──
        est = enrollment_status(st.session_state.profile_id)
        print(f"[DEBUG app] Enrollment status for '{st.session_state.profile_id}': {est}")

        badge_f, badge_v, badge_p = st.columns(3)
        with badge_f:
            if est["face_enrolled"]:
                st.markdown("📸 **Face:** :green[Enrolled ✅]")
            else:
                st.markdown("📸 **Face:** :orange[Not Enrolled]")
        with badge_v:
            if est["voice_enrolled"]:
                st.markdown("🎤 **Voice:** :green[Enrolled ✅]")
            else:
                st.markdown("🎤 **Voice:** :orange[Not Enrolled]")
        with badge_p:
            if est["password_enrolled"]:
                st.markdown("🔑 **Password:** :green[Set ✅]")
            else:
                st.markdown("🔑 **Password:** :orange[Not Set (default 1234)]")

        # ── Main authentication flow ──
        if not st.session_state.verified:
            st.divider()

            # ════════════════════════════════════════════════════════
            # EXPLICIT CHOICE: Sign Up or Sign In
            # ════════════════════════════════════════════════════════
            action = st.radio(
                "What would you like to do?",
                ["🆕 Sign Up (First-Time Enrollment)", "🔓 Sign In (Verify & Login)"],
                key="action_radio",
                horizontal=True,
            )
            print(f"[DEBUG app] Action selected: {action}")

            # ────────────────────────────────────────────────────────
            # SIGN UP — First-Time Enrollment
            # ────────────────────────────────────────────────────────
            if "Sign Up" in action:
                st.markdown("---")
                st.markdown("### 🆕 First-Time Enrollment")
                st.caption(
                    "Enroll your password, face, or voice. "
                    "All data is stored locally on this machine only."
                )

                enroll_what = st.radio(
                    "What to enroll:",
                    ["🔑 Password", "📸 Face", "🎤 Voice"],
                    key="enroll_what_radio",
                )

                # ── PASSWORD ENROLLMENT ──
                if "Password" in enroll_what:
                    st.markdown("#### 🔑 Set Your Profile Password")
                    if est["password_enrolled"]:
                        st.warning("⚠️ A password is already set for this profile. Setting a new one will overwrite it.")
                    else:
                        st.info("🆕 No password set yet — create one below.")
                    st.caption("Your password is hashed with SHA-256 and stored in `meta.json`. Never stored in plaintext.")
                    new_pw  = st.text_input("New password (min 4 characters)", type="password", key="new_pw")
                    conf_pw = st.text_input("Confirm password", type="password", key="conf_pw")
                    if st.button("✅ Save Password", key="btn_enroll_pw", type="primary"):
                        print("[DEBUG app] === PASSWORD ENROLL START ===")
                        if new_pw != conf_pw:
                            st.error("❌ Passwords do not match.")
                        else:
                            er = enroll_password(st.session_state.profile_id, new_pw)
                            print(f"[DEBUG app] enroll_password result: {er}")
                            st.info(f"📊 **Debug result:** `{er}`")
                            if er["success"]:
                                log(f"🔑 Password set — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.rerun()
                            else:
                                st.error(er["message"])

                # ── FACE ENROLLMENT ──
                elif "Face" in enroll_what:
                    st.markdown("#### 📸 Face Enrollment")
                    if est["face_enrolled"]:
                        st.warning(
                            "⚠️ Face is **already enrolled** for this profile. "
                            "Re-enrolling will **overwrite** your existing template."
                        )

                    with st.expander("ℹ️ How face enrollment works"):
                        st.markdown(
                            "1. **Capture** — Webcam takes a photo.  \n"
                            "2. **Detect & Align** — MTCNN neural network finds and aligns the face.  \n"
                            "3. **Encode** — FaceNet (InceptionResnetV1) creates a 512-dim identity embedding.  \n"
                            "4. **Store** — Embedding saved as `face_encoding.npy`.  \n"
                            "5. **Future logins** — New photo → new embedding → cosine similarity ≥ threshold.  \n"
                            "   Same person typically scores 0.70–0.95; different people score 0.0–0.50."
                        )

                    face_img = st.camera_input(
                        "📷 Capture your face (clear, frontal, good lighting)",
                        key="face_cam_enroll",
                    )

                    if face_img is not None:
                        if st.button("✅ Enroll My Face", key="btn_do_enroll_face", type="primary"):
                            print(f"[DEBUG app] === FACE ENROLLMENT START ===")
                            print(f"[DEBUG app] Profile: {st.session_state.profile_id}")
                            with st.spinner("Detecting face and saving template…"):
                                arr = np.array(Image.open(face_img).convert("RGB"))
                                print(f"[DEBUG app] Image array shape: {arr.shape}")
                                er = enroll_face(st.session_state.profile_id, arr)
                            print(f"[DEBUG app] enroll_face result: {er}")
                            print(f"[DEBUG app] === FACE ENROLLMENT END ===")
                            st.info(f"📊 **Debug result:** `{er}`")
                            if er["success"]:
                                log(f"📸 Face enrolled — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(er["message"])
                    else:
                        st.caption("👆 Take a photo above to begin enrollment.")

                # ── VOICE ENROLLMENT ──
                else:
                    st.markdown("#### 🎤 Voice Enrollment")
                    if est["voice_enrolled"]:
                        st.warning(
                            "⚠️ Voice is **already enrolled** for this profile. "
                            "Re-enrolling will **overwrite** your existing voiceprint."
                        )

                    with st.expander("ℹ️ How voice enrollment works — READ THIS"):
                        st.markdown(
                            "**What gets compared:** Speaker *identity* (pitch, timbre, resonance) — "
                            "**NOT** the words you say. You can speak any sentence.  \n\n"
                            "1. **Record** — Microphone captures your voice at 16 kHz.  \n"
                            "2. **Embed** — Resemblyzer creates a 256-dim speaker embedding.  \n"
                            "3. **Store** — Saved as `voice_embedding.npy`.  \n\n"
                            "**Typical scores:**  \n"
                            "- Same person → 0.75–0.95  \n"
                            "- Different people → 0.40–0.75  \n"
                            "If many different voices pass, raise the threshold in `config.py`."
                        )

                    from config import VOICE_COSINE_THRESHOLD
                    st.caption(f"Current voice threshold: **{VOICE_COSINE_THRESHOLD}** (set in `config.py`)")
                    vdur = st.slider("Recording duration (seconds)", 3, 12, 7, key="voice_dur_enroll")
                    st.info('🎙️ Speak naturally — say anything. Longer clips give better results.')

                    if st.button("🎤 Record & Enroll My Voice", key="btn_do_enroll_voice", type="primary"):
                        print(f"[DEBUG app] === VOICE ENROLLMENT START ===")
                        print(f"[DEBUG app] Profile: {st.session_state.profile_id}")
                        try:
                            with st.spinner(f"🔴 Recording for {vdur}s — please speak now…"):
                                cap = AudioCapture()
                                audio = cap.record_fixed(duration=vdur)
                            peak = float(np.max(np.abs(audio)))
                            print(f"[DEBUG app] Audio captured: {len(audio)} samples, peak={peak:.4f}")
                            if peak < 0.01:
                                st.warning("⚠️ Recording is very quiet. Check your microphone.")
                            # Show playback widget
                            from modules.A_user_access.voice_biometrics import LAST_RECORDING_PATH, save_wav
                            save_wav(np.asarray(audio, dtype=np.float32).flatten(), LAST_RECORDING_PATH)
                            if os.path.isfile(LAST_RECORDING_PATH):
                                st.markdown("🔊 **Listen to your recording:**")
                                st.audio(LAST_RECORDING_PATH, format="audio/wav")
                            with st.spinner("Creating speaker embedding…"):
                                er = enroll_voice(st.session_state.profile_id, audio)
                            print(f"[DEBUG app] enroll_voice result: {er}")
                            print(f"[DEBUG app] === VOICE ENROLLMENT END ===")
                            st.info(f"📊 **Debug result:** `{er}`")
                            if er["success"]:
                                log(f"🎙️ Voice enrolled — profile={st.session_state.profile_id}")
                                st.success(er["message"])
                                st.balloons()
                                st.rerun()
                            else:
                                st.error(er["message"])
                        except Exception as e:
                            print(f"[DEBUG app] Voice enrollment EXCEPTION: {e}")
                            st.error(f"Microphone error: {e}")

            # ────────────────────────────────────────────────────────
            # SIGN IN — Verify & Login
            # ────────────────────────────────────────────────────────
            else:
                st.markdown("---")
                st.markdown("### 🔓 Sign In — Verify Your Identity")

                signin_method = st.radio(
                    "Verification method:",
                    ["🔑 Password", "📸 Face Recognition", "🎤 Voice Recognition"],
                    key="signin_method_radio",
                )
                print(f"[DEBUG app] Sign-in method: {signin_method}")

                # ── PASSWORD SIGN IN ──
                if "Password" in signin_method:
                    st.markdown("#### 🔑 Password Sign In")
                    if est["password_enrolled"]:
                        st.success("✅ Profile password set. Enter it below.")
                    else:
                        st.warning(
                            "⚠️ No profile password set yet — using **default passcode (1234)**. "
                            "Go to Sign Up → Password to set your own."
                        )
                    passcode = st.text_input("Password", type="password", key="passcode_input")
                    if st.button("🔓 Verify Password & Sign In", key="btn_verify_pw", type="primary"):
                        print(f"[DEBUG app] === PASSWORD VERIFY START ===")
                        result = verify_profile_password(st.session_state.profile_id, passcode)
                        print(f"[DEBUG app] verify_profile_password result: {result}")
                        print(f"[DEBUG app] === PASSWORD VERIFY END ===")
                        st.info(f"📊 **Debug result:** `{result}`")
                        if result["verified"]:
                            st.session_state.verified = True
                            log(f"✅ Verified (password) profile={st.session_state.profile_id}")
                            st.success(result["message"])
                            st.rerun()
                        else:
                            st.error(result["message"])

                # ── FACE SIGN IN ──
                elif "Face" in signin_method:
                    st.markdown("#### 📸 Face Recognition Sign In")
                    if not est["face_enrolled"]:
                        st.error(
                            "❌ **Face is not enrolled** for this profile!  \n"
                            "Please go to **Sign Up** first to enroll your face."
                        )
                    else:
                        st.success("✅ Face template found. Capture a new photo to verify.")
                        face_img = st.camera_input("📷 Verify your face", key="face_cam_verify")
                        if face_img is not None:
                            if st.button("🔓 Verify Face & Sign In", key="btn_do_verify_face", type="primary"):
                                print(f"[DEBUG app] === FACE VERIFY START ===")
                                print(f"[DEBUG app] Profile: {st.session_state.profile_id}")
                                with st.spinner("Comparing against stored template…"):
                                    arr = np.array(Image.open(face_img).convert("RGB"))
                                    print(f"[DEBUG app] Image array shape: {arr.shape}")
                                    result = verify_face(st.session_state.profile_id, arr)
                                print(f"[DEBUG app] verify_face result: {result}")
                                print(f"[DEBUG app] === FACE VERIFY END ===")
                                st.info(f"📊 **Debug result:** `{result}`")
                                if result["verified"]:
                                    st.session_state.verified = True
                                    log(f"✅ Verified (face) profile={st.session_state.profile_id}")
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                        else:
                            st.caption("👆 Take a photo above to verify your identity.")

                # ── VOICE SIGN IN ──
                elif "Voice" in signin_method:
                    st.markdown("#### 🎤 Voice Recognition Sign In")
                    if not est["voice_enrolled"]:
                        st.error(
                            "❌ **Voice is not enrolled** for this profile!  \n"
                            "Please go to **Sign Up** first to enroll your voice."
                        )
                    else:
                        from config import VOICE_COSINE_THRESHOLD
                        st.success("✅ Voiceprint found. Record any short phrase to verify.")
                        st.info(
                            "💡 **Note:** This matches your voice *characteristics* (who you are), "
                            "not the specific words you say. Say anything in your normal voice."
                        )
                        st.caption(
                            f"Threshold: **{VOICE_COSINE_THRESHOLD}** — "
                            f"same speaker typically 0.75–0.95, different speakers 0.40–0.75."
                        )
                        vdur = st.slider("Recording duration (seconds)", 3, 12, 7, key="voice_dur_verify")

                        if st.button("🎤 Record & Verify Voice", key="btn_do_verify_voice", type="primary"):
                            print(f"[DEBUG app] === VOICE VERIFY START ===")
                            print(f"[DEBUG app] Profile: {st.session_state.profile_id}")
                            try:
                                with st.spinner(f"🔴 Recording for {vdur}s — please speak now…"):
                                    cap = AudioCapture()
                                    audio = cap.record_fixed(duration=vdur)
                                peak = float(np.max(np.abs(audio)))
                                print(f"[DEBUG app] Audio captured: {len(audio)} samples, peak={peak:.4f}")
                                if peak < 0.01:
                                    st.warning("⚠️ Very quiet recording. Check mic and speak louder.")

                                # Save + show playback before comparison
                                from modules.A_user_access.voice_biometrics import LAST_RECORDING_PATH, save_wav
                                save_wav(np.asarray(audio, dtype=np.float32).flatten(), LAST_RECORDING_PATH)
                                if os.path.isfile(LAST_RECORDING_PATH):
                                    st.markdown("🔊 **Listen to your recording:**")
                                    st.audio(LAST_RECORDING_PATH, format="audio/wav")

                                with st.spinner("Comparing against stored voiceprint…"):
                                    result = verify_voice(st.session_state.profile_id, audio)
                                print(f"[DEBUG app] verify_voice result: {result}")
                                print(f"[DEBUG app] === VOICE VERIFY END ===")

                                # Extract and display score as metric
                                import re
                                m = re.search(r"similarity=([\d.]+)", result.get("message", ""))
                                if m:
                                    score = float(m.group(1))
                                    delta = score - VOICE_COSINE_THRESHOLD
                                    c1, c2 = st.columns(2)
                                    c1.metric("🎤 Similarity Score", f"{score:.3f}",
                                              delta=f"{delta:+.3f} vs threshold",
                                              delta_color="normal")
                                    c2.metric("🎯 Threshold", f"{VOICE_COSINE_THRESHOLD}")

                                st.info(f"📊 **Debug result:** `{result}`")
                                if result["verified"]:
                                    st.session_state.verified = True
                                    log(f"✅ Verified (voice) profile={st.session_state.profile_id}")
                                    st.success(result["message"])
                                    st.rerun()
                                else:
                                    st.error(result["message"])
                            except Exception as e:
                                print(f"[DEBUG app] Voice verify EXCEPTION: {e}")
                                st.error(f"Microphone error: {e}")


            # ── Enrollment data management ────────────────────────
            with st.expander("🔧 Enrollment Data & Privacy"):
                st.markdown(
                    "Biometrics are stored **only on this machine** under "
                    "`data/enrollment/<profile_id>/`:\n\n"
                    "| File | Contents |\n"
                    "|------|----------|\n"
                    "| `face_encoding.npy` | 512-dim FaceNet identity embedding |\n"
                    "| `voice_embedding.npy` | 256-dim Resemblyzer speaker embedding |\n"
                    "| `meta.json` | Enrollment flags and metadata |\n\n"
                    "Passwords are compared to a **SHA-256 hash** defined in "
                    "`config.py` — never stored in plaintext."
                )
                c1, c2 = st.columns(2)
                if c1.button("🗑️ Clear Face Data", key="clr_face"):
                    print(f"[DEBUG app] Clearing face enrollment for '{st.session_state.profile_id}'")
                    clear_biometric_enrollment(st.session_state.profile_id, "face")
                    log("🗑️ Face enrollment cleared")
                    st.rerun()
                if c2.button("🗑️ Clear Voice Data", key="clr_voice"):
                    print(f"[DEBUG app] Clearing voice enrollment for '{st.session_state.profile_id}'")
                    clear_biometric_enrollment(st.session_state.profile_id, "voice")
                    log("🗑️ Voice enrollment cleared")
                    st.rerun()
                if st.button("🗑️ Clear ALL Biometric Data for This Profile", key="clr_bio_all"):
                    print(f"[DEBUG app] Clearing ALL enrollment for '{st.session_state.profile_id}'")
                    clear_biometric_enrollment(st.session_state.profile_id, "all")
                    log("🗑️ All biometric data cleared")
                    st.rerun()

        else:
            # ── Already verified — show log-out button ──
            if st.button("🔒 Lock / Log Out", key="btn_logout"):
                st.session_state.verified = False
                st.session_state.awake = False
                reset_verification()
                log("🔒 Session locked.")
                st.rerun()

    st.divider()

    # ── Stage 2: Input (Voice or Text) ───────────────────────────────
    with st.container(**_CONTAINER_KW):
        st.subheader("② Input — Voice or Text  (Modules 3 & 4)")

        if not st.session_state.verified:
            st.warning("⚠️ Verify your identity first.")
        else:
            input_mode = st.radio(
                "Input mode",
                ["📝 Type a command", "🎤 Record voice"],
                horizontal=True,
                key="input_mode",
            )

            raw_text = ""

            if input_mode == "📝 Type a command":
                typed = st.text_input(
                    "Type your command:",
                    placeholder='e.g.  "hey mycroft set learning rate to 0.01"',
                    key="typed_cmd",
                )
                if st.button("Submit Text Command", key="btn_text"):
                    processed = handler.process(typed)
                    if processed["valid"]:
                        raw_text = processed["text"]
                        log(f"📝 Text input received: '{raw_text}'")
                    else:
                        st.warning(processed["message"])

            else:  # Voice recording
                duration = st.slider("Recording duration (seconds)", 2, 10, 5)
                if st.button("🎤 Start Recording", key="btn_record"):
                    audio = None
                    with st.spinner(f"Recording for {duration}s…"):
                        try:
                            capture = AudioCapture()
                            audio = capture.record_fixed(duration=duration)
                            capture.save(audio, "/tmp/demo_clip.wav")
                            log(f"🎤 Audio captured ({duration}s).")
                        except Exception as e:
                            st.error(f"Microphone error: {e}")
                            audio = None

                    if audio is not None:
                        with st.spinner("Transcribing with Whisper…"):
                            try:
                                stt = SpeechToText()
                                # Use file path so Google fallback is available if Whisper init fails.
                                transcript = stt.transcribe("/tmp/demo_clip.wav")
                                st.session_state.transcript = transcript
                                raw_text = transcript
                                log(f"🗣️ Transcribed: '{transcript}'")
                                if not transcript.strip():
                                    hint = get_last_transcribe_error() or "No transcript returned."
                                    st.warning(hint)
                                    log(f"⚠️ Transcription empty: {hint}")
                                else:
                                    st.success(f"**Heard:** {transcript}")
                            except Exception as e:
                                st.error(f"Transcription error: {e}")

            # ── Push raw_text into session if we got something ──────
            if raw_text:
                st.session_state.raw_command = raw_text

    st.divider()

    # ── Stage 3: Wake Word Check ──────────────────────────────────────
    with st.container(**_CONTAINER_KW):
        st.subheader("③ Wake Word Detection  (Module 2)")

        cmd = st.session_state.raw_command
        if cmd:
            woke = is_wake_word(cmd)
            st.session_state.awake = woke
            if woke:
                st.success(f"✅ Wake word detected in: *\"{cmd}\"*")
                log("🔔 Wake word detected.")
            else:
                st.warning(
                    f"⚠️ No wake word found in: *\"{cmd}\"*  "
                    "— Try starting with **'hey mycroft'**"
                )
        else:
            st.info("Waiting for input…")

    st.divider()

    # ── Stage 4: Output (ready for Intent Detection) ──────────────────
    with st.container(**_CONTAINER_KW):
        st.subheader("④ Pipeline Output — Ready for Module 6")

        if st.session_state.awake and st.session_state.raw_command:
            st.markdown("**Raw command text (passed to Intent Detection):**")
            st.code(st.session_state.raw_command, language="text")

            # Strip the wake word from the command before passing downstream
            import re
            clean_cmd = re.sub(
                r"(hey|hi|okay|wake up|hello)\s+mycroft\s*",
                "",
                st.session_state.raw_command,
            ).strip()
            st.markdown("**Command (wake word stripped):**")
            st.code(clean_cmd, language="text")
            st.success("✅ Command ready for Intent Detection (Module 6)")
        else:
            st.info("Awaiting a verified wake-word command…")


# ── RIGHT COLUMN: System State + Event Log ──────────────────────────────
with col_right:

    st.subheader("🖥️ System State")
    state_data = {
        "verified": st.session_state.verified,
        "awake": st.session_state.awake,
        "transcript": st.session_state.transcript or "—",
        "raw_command": st.session_state.raw_command or "—",
    }
    for k, v in state_data.items():
        icon = "✅" if v is True else ("❌" if v is False else "📄")
        val_str = str(v)
        st.markdown(f"**`{k}`** {icon}  \n`{val_str}`")

    st.divider()
    st.subheader("📋 Event Log")
    if st.session_state.event_log:
        for entry in reversed(st.session_state.event_log[-15:]):
            st.text(entry)
    else:
        st.caption("No events yet.")

    if st.button("Clear Log", key="btn_clear_log"):
        st.session_state.event_log = []
        st.rerun()

    st.divider()
    st.subheader("🩺 STT health check")
    st.caption("Records ~1s from the mic, verifies HTTPS to Google, tries to load Whisper, and checks optional ffmpeg.")
    if st.button("Run STT health checks", key="btn_stt_health"):
        with st.spinner("Running checks (Whisper load may take a while the first time)…"):
            report = run_stt_health_checks()
        st.json(report)
        summ = report.get("_summary", {})
        if summ.get("ok"):
            st.success(summ.get("detail", "Checks complete."))
        else:
            st.warning(summ.get("detail", "Some checks failed — see JSON for details."))
        log(f"🩺 STT health: {summ.get('detail', report)}")

    st.divider()
    st.subheader("🧪 Quick Test (no mic)")
    st.caption("Paste any text to test the pipeline without a microphone.")
    test_text = st.text_area(
        "Test text",
        value='hey mycroft set learning rate to 0.01',
        height=70,
    )
    if st.button("Run Test", key="btn_test"):
        p = handler.process(test_text)
        if p["valid"]:
            woke = is_wake_word(p["text"])
            st.write(f"Wake word detected: **{woke}**")
            import re
            clean = re.sub(r"(hey|hi|okay|wake up|hello)\s+mycroft\s*", "", p["text"]).strip()
            st.write(f"Command: `{clean}`")
            st.json({"valid": p["valid"], "wake_word": woke, "command": clean})
        else:
            st.warning(p["message"])
