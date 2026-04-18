"""
Enrollment storage — on-disk datasets for face and voice verification.
======================================================================
Each user_id gets a folder under config.ENROLLMENT_ROOT with:
  - meta.json          — flags and timestamps
  - face_gray.npy      — enrolled face (fixed-size grayscale)
  - voice_embedding.npy — resemblyzer speaker embedding vector
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from config import ENROLLMENT_ROOT


def _ensure_root() -> Path:
    root = Path(ENROLLMENT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    return root


def profile_dir(user_id: str) -> Path:
    """Directory for one user's enrollment files."""
    p = _ensure_root() / _safe_id(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_id(user_id: str) -> str:
    s = (user_id or "default").strip()
    if not s or ".." in s or "/" in s or "\\" in s:
        return "default"
    return s


def safe_user_id(user_id: str | None) -> str:
    """Public helper for canonical profile folder name."""
    return _safe_id(user_id or "default")


def meta_path(user_id: str) -> Path:
    return profile_dir(user_id) / "meta.json"


def load_meta(user_id: str) -> dict[str, Any]:
    path = meta_path(user_id)
    if not path.is_file():
        print(f"[DEBUG enrollment_store.load_meta] No meta.json for '{user_id}' at {path}")
        return {
            "face_enrolled": False,
            "voice_enrolled": False,
            "password_enrolled": False,
            "password_hash": None,
            "user_id": user_id,
        }
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("face_enrolled", False)
    data.setdefault("voice_enrolled", False)
    data.setdefault("password_enrolled", False)
    data.setdefault("password_hash", None)
    print(f"[DEBUG enrollment_store.load_meta] '{user_id}' -> "
          f"face={data['face_enrolled']}, voice={data['voice_enrolled']}, "
          f"password={data['password_enrolled']}")
    return data


def save_meta(user_id: str, updates: dict[str, Any]) -> None:
    print(f"[DEBUG enrollment_store.save_meta] '{user_id}' updates={updates}")
    path = meta_path(user_id)
    current = load_meta(user_id)
    current.update(updates)
    current["user_id"] = user_id
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)
    os.replace(tmp, path)


def face_array_path(user_id: str) -> Path:
    return profile_dir(user_id) / "face_encoding.npy"


def voice_embedding_path(user_id: str) -> Path:
    return profile_dir(user_id) / "voice_embedding.npy"


def clear_enrollment(user_id: str, which: str = "all") -> None:
    """Remove face and/or voice enrollment files for a user."""
    uid = _safe_id(user_id)
    pd = profile_dir(uid)
    if which == "all":
        for name in ("face_encoding.npy", "face_gray.npy", "voice_embedding.npy", "meta.json"):
            fp = pd / name
            if fp.is_file():
                fp.unlink()
        try:
            pd.rmdir()
        except OSError:
            pass
        return

    if which == "password":
        save_meta(uid, {"password_enrolled": False, "password_hash": None})
        return

    updates: dict[str, Any] = {}
    if which == "face":
        fp = face_array_path(uid)
        if fp.is_file():
            fp.unlink()
        updates["face_enrolled"] = False
    elif which == "voice":
        vp = voice_embedding_path(uid)
        if vp.is_file():
            vp.unlink()
        updates["voice_enrolled"] = False
    if updates:
        save_meta(uid, updates)
