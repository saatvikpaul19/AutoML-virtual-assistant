"""
Module 1 — User Identification / Verification
==============================================
Supports:
  • Password (SHA-256 hash, lockout on failures)
  • Face (OpenCV Haar + enrolled grayscale template, cosine match)
  • Voice (Resemblyzer embedding + cosine match)

First-time users enroll face and/or voice; data is stored on disk under
`data/enrollment/<user_id>/`. Password uses config hash only (no enrollment file).
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np

from config import (
    PASSCODE_HASH,
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_SECONDS,
    SAMPLE_RATE,
)

from modules.A_user_access import enrollment_store
from modules.A_user_access.face_biometrics import compare_faces, extract_face_encoding
from modules.A_user_access.voice_biometrics import compare_voices, embedding_from_float_audio


# ── Internal state (module-level, persists within one Python session) ──
_failed_attempts: int = 0
_lockout_until: float = 0.0
_current_hash: str = PASSCODE_HASH


def _hash(passcode: str) -> str:
    return hashlib.sha256(passcode.strip().encode()).hexdigest()


def _is_locked_out() -> bool:
    return time.time() < _lockout_until


def _seconds_remaining() -> int:
    remaining = _lockout_until - time.time()
    return max(0, int(remaining))


def _lockout_response() -> dict[str, Any] | None:
    if _is_locked_out():
        wait = _seconds_remaining()
        return {
            "verified": False,
            "message": f"Too many failed attempts. Please wait {wait}s.",
            "locked": True,
            "wait_sec": wait,
        }
    return None


def _success_verification(message: str) -> dict[str, Any]:
    global _failed_attempts
    _failed_attempts = 0
    return {
        "verified": True,
        "message": message,
        "locked": False,
        "wait_sec": 0,
    }


def _failed_biometric(message: str) -> dict[str, Any]:
    """Increment failures; lock out after MAX_FAILED_ATTEMPTS wrong biometrics."""
    global _failed_attempts, _lockout_until
    _failed_attempts += 1
    if _failed_attempts >= MAX_FAILED_ATTEMPTS:
        _lockout_until = time.time() + LOCKOUT_SECONDS
        _failed_attempts = 0
        return {
            "verified": False,
            "message": f"{message} Too many failures — locked for {LOCKOUT_SECONDS}s.",
            "locked": True,
            "wait_sec": LOCKOUT_SECONDS,
        }
    remaining = MAX_FAILED_ATTEMPTS - _failed_attempts
    return {
        "verified": False,
        "message": f"{message} {remaining} attempt(s) left.",
        "locked": False,
        "wait_sec": 0,
    }


# ── Password ────────────────────────────────────────────────────────────

def verify_passcode(passcode: str) -> dict[str, Any]:
    global _failed_attempts, _lockout_until

    lo = _lockout_response()
    if lo:
        return lo

    if _hash(passcode) == _current_hash:
        _failed_attempts = 0
        return _success_verification("Verification successful. Welcome!")

    _failed_attempts += 1
    if _failed_attempts >= MAX_FAILED_ATTEMPTS:
        _lockout_until = time.time() + LOCKOUT_SECONDS
        _failed_attempts = 0
        return {
            "verified": False,
            "message": (
                f"Incorrect passcode. Too many failures — locked for {LOCKOUT_SECONDS}s."
            ),
            "locked": True,
            "wait_sec": LOCKOUT_SECONDS,
        }
    remaining = MAX_FAILED_ATTEMPTS - _failed_attempts
    return {
        "verified": False,
        "message": f"Incorrect passcode. {remaining} attempt(s) left.",
        "locked": False,
        "wait_sec": 0,
    }


def get_verification_status() -> dict[str, Any]:
    return {
        "locked": _is_locked_out(),
        "wait_sec": _seconds_remaining(),
        "attempts": _failed_attempts,
    }


def set_new_passcode(old_passcode: str, new_passcode: str) -> dict[str, Any]:
    global _current_hash

    if len(new_passcode.strip()) < 4:
        return {"success": False, "message": "New passcode must be at least 4 characters."}

    result = verify_passcode(old_passcode)
    if not result["verified"]:
        return {"success": False, "message": "Old passcode is incorrect."}

    _current_hash = _hash(new_passcode)
    return {"success": True, "message": "Passcode updated successfully."}


def reset_state() -> None:
    global _failed_attempts, _lockout_until
    _failed_attempts = 0
    _lockout_until = 0.0


# ── Enrollment status ─────────────────────────────────────────────────────

def enrollment_status(user_id: str | None = None) -> dict[str, Any]:
    uid = enrollment_store.safe_user_id(user_id)
    m = enrollment_store.load_meta(uid)
    result = {
        "user_id": uid,
        "face_enrolled": bool(m.get("face_enrolled")),
        "voice_enrolled": bool(m.get("voice_enrolled")),
        "password_enrolled": bool(m.get("password_enrolled")),
    }
    print(f"[DEBUG verify] enrollment_status('{uid}'): "
          f"face={result['face_enrolled']}, voice={result['voice_enrolled']}, "
          f"password={result['password_enrolled']}")
    return result


# ── Per-profile Password ──────────────────────────────────────────────────

def enroll_password(user_id: str | None, new_password: str) -> dict[str, Any]:
    """
    Save a SHA-256 hash of new_password to this profile's meta.json.
    Called during Sign Up so each profile has its own password.
    """
    uid = enrollment_store.safe_user_id(user_id)
    pw = new_password.strip()
    if len(pw) < 4:
        return {"success": False, "message": "Password must be at least 4 characters."}
    hashed = _hash(pw)
    enrollment_store.save_meta(uid, {"password_enrolled": True, "password_hash": hashed})
    print(f"[DEBUG verify] enroll_password: password set for '{uid}'")
    return {"success": True, "message": "Password saved. Use it to Sign In next time."}


def verify_profile_password(user_id: str | None, passcode: str) -> dict[str, Any]:
    """
    Verify passcode against the per-profile hash stored in meta.json.
    Falls back to the global config hash if the profile has no password set.
    """
    uid = enrollment_store.safe_user_id(user_id)
    print(f"[DEBUG verify] verify_profile_password: uid='{uid}'")

    lo = _lockout_response()
    if lo:
        return lo

    global _failed_attempts, _lockout_until
    meta = enrollment_store.load_meta(uid)
    stored_hash = meta.get("password_hash")
    password_enrolled = bool(meta.get("password_enrolled"))

    if not password_enrolled or not stored_hash:
        print(f"[DEBUG verify] verify_profile_password: no profile password for '{uid}', "
              f"falling back to global hash")
        result = verify_passcode(passcode)
        if not result["verified"]:
            result["message"] = f"{result['message']} (using default passcode — set a profile password via Sign Up first)"
        return result

    entered_hash = _hash(passcode)
    print(f"[DEBUG verify] verify_profile_password: comparing hashes")
    if entered_hash == stored_hash:
        _failed_attempts = 0
        return _success_verification("Password verified. Welcome!")

    _failed_attempts += 1
    if _failed_attempts >= MAX_FAILED_ATTEMPTS:
        _lockout_until = time.time() + LOCKOUT_SECONDS
        _failed_attempts = 0
        return {
            "verified": False,
            "message": f"Wrong password. Too many failures — locked for {LOCKOUT_SECONDS}s.",
            "locked": True,
            "wait_sec": LOCKOUT_SECONDS,
        }
    remaining = MAX_FAILED_ATTEMPTS - _failed_attempts
    return {
        "verified": False,
        "message": f"Wrong password. {remaining} attempt(s) left.",
        "locked": False,
        "wait_sec": 0,
    }


# ── Face ─────────────────────────────────────────────────────────────────

def enroll_face(user_id: str | None, image_rgb: np.ndarray) -> dict[str, Any]:
    uid = enrollment_store.safe_user_id(user_id)
    print(f"[DEBUG verify] enroll_face: uid='{uid}', image shape={image_rgb.shape}")
    try:
        encoding = extract_face_encoding(image_rgb)
    except Exception as e:
        print(f"[DEBUG verify] enroll_face: FAILED — {e}")
        return {"success": False, "message": f"Image error: {e}"}
    if encoding is None:
        print("[DEBUG verify] enroll_face: FAILED — no face detected")
        return {
            "success": False,
            "message": "No face detected. Face the camera with good lighting.",
        }
    path = enrollment_store.face_array_path(uid)
    np.save(path, encoding)
    enrollment_store.save_meta(uid, {"face_enrolled": True})
    print(f"[DEBUG verify] enroll_face: SUCCESS — template saved to {path}")
    return {
        "success": True,
        "message": "Face enrolled. Next time, choose Face verification to sign in.",
    }


def verify_face(user_id: str | None, image_rgb: np.ndarray) -> dict[str, Any]:
    uid = enrollment_store.safe_user_id(user_id)
    print(f"[DEBUG verify] verify_face: uid='{uid}', image shape={image_rgb.shape}")

    lo = _lockout_response()
    if lo:
        print("[DEBUG verify] verify_face: LOCKED OUT")
        return lo

    meta = enrollment_store.load_meta(uid)
    if not meta.get("face_enrolled"):
        print(f"[DEBUG verify] verify_face: face NOT enrolled for '{uid}'")
        return {
            "verified": False,
            "message": "No face enrolled. Enroll your face first (first-time setup).",
            "locked": False,
            "wait_sec": 0,
        }

    path = enrollment_store.face_array_path(uid)
    if not path.is_file():
        print(f"[DEBUG verify] verify_face: face_gray.npy MISSING at {path}")
        return {
            "verified": False,
            "message": "Face enrollment data missing. Please enroll again.",
            "locked": False,
            "wait_sec": 0,
        }

    template = np.load(path)
    print(f"[DEBUG verify] verify_face: loaded template shape={template.shape}, "
          f"mean={template.mean():.2f}, std={template.std():.2f}")
    try:
        probe = extract_face_encoding(image_rgb)
    except Exception as e:
        print(f"[DEBUG verify] verify_face: probe extraction FAILED — {e}")
        return _failed_biometric(f"Image error: {e}")

    if probe is None:
        print("[DEBUG verify] verify_face: no face in probe image")
        return _failed_biometric("No face detected in frame.")

    print(f"[DEBUG verify] verify_face: probe shape={probe.shape}, "
          f"mean={probe.mean():.2f}, std={probe.std():.2f}")
    score, is_match = compare_faces(template, probe)
    print(f"[DEBUG verify] verify_face: FINAL score={score:.6f}, match={is_match}")
    if is_match:
        return _success_verification(f"Face verified (similarity={score:.3f}). Welcome!")
    return _failed_biometric(f"Face mismatch (similarity={score:.3f}).")


# ── Voice ────────────────────────────────────────────────────────────────

def enroll_voice(
    user_id: str | None,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    uid = enrollment_store.safe_user_id(user_id)
    print(f"[DEBUG verify] enroll_voice: uid='{uid}', audio length={len(audio)} samples")
    try:
        emb = embedding_from_float_audio(audio, sample_rate)
    except ValueError as e:
        print(f"[DEBUG verify] enroll_voice: FAILED — {e}")
        return {"success": False, "message": str(e)}
    except Exception as e:
        print(f"[DEBUG verify] enroll_voice: FAILED — {e!r}")
        return {"success": False, "message": f"Voice embedding failed: {e!r}"}

    path = enrollment_store.voice_embedding_path(uid)
    np.save(path, emb)
    enrollment_store.save_meta(uid, {"voice_enrolled": True})
    print(f"[DEBUG verify] enroll_voice: SUCCESS — embedding saved to {path}")
    return {
        "success": True,
        "message": "Voice enrolled. Next time, choose Voice verification to sign in.",
    }


def verify_voice(
    user_id: str | None,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, Any]:
    uid = enrollment_store.safe_user_id(user_id)
    print(f"[DEBUG verify] verify_voice: uid='{uid}', audio length={len(audio)} samples")

    lo = _lockout_response()
    if lo:
        print("[DEBUG verify] verify_voice: LOCKED OUT")
        return lo

    meta = enrollment_store.load_meta(uid)
    if not meta.get("voice_enrolled"):
        print(f"[DEBUG verify] verify_voice: voice NOT enrolled for '{uid}'")
        return {
            "verified": False,
            "message": "No voice enrolled. Enroll your voice first (first-time setup).",
            "locked": False,
            "wait_sec": 0,
        }

    path = enrollment_store.voice_embedding_path(uid)
    if not path.is_file():
        print(f"[DEBUG verify] verify_voice: voice_embedding.npy MISSING at {path}")
        return {
            "verified": False,
            "message": "Voice enrollment data missing. Please enroll again.",
            "locked": False,
            "wait_sec": 0,
        }

    template = np.load(path)
    print(f"[DEBUG verify] verify_voice: loaded template shape={template.shape}")
    try:
        emb = embedding_from_float_audio(audio, sample_rate)
    except ValueError as e:
        print(f"[DEBUG verify] verify_voice: embedding FAILED — {e}")
        return _failed_biometric(str(e))
    except Exception as e:
        print(f"[DEBUG verify] verify_voice: embedding FAILED — {e!r}")
        return _failed_biometric(f"Voice embedding failed: {e!r}")

    score, is_match = compare_voices(template, emb)
    print(f"[DEBUG verify] verify_voice: FINAL score={score:.6f}, match={is_match}")
    if is_match:
        return _success_verification(f"Voice verified (similarity={score:.3f}). Welcome!")
    return _failed_biometric(f"Voice mismatch (similarity={score:.3f}).")



def clear_biometric_enrollment(user_id: str | None = None, which: str = "all") -> None:
    """Delete stored face/voice templates (password unchanged)."""
    uid = enrollment_store.safe_user_id(user_id)
    enrollment_store.clear_enrollment(uid, which)
