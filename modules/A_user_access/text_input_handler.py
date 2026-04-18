"""
Module 3 — Manual Text Input Fallback
======================================
Provides a clean Python interface for the Streamlit text-input widget,
so the typed-command path and the voice path share the same downstream
processing logic (Intent Detection → Slot Filling → Command Router).

Public API
----------
TextInputHandler   (class)
    .process(raw_text: str) -> dict
        Validates and normalises the typed command.
        Returns a dict ready for the Intent Detection module.

sanitise(text: str) -> str
    Stand-alone helper: clean raw text input.

Design notes
------------
- Normalisation (lowercase, strip, collapse spaces) ensures the same
  intent-detection logic handles both transcribed speech and typed text.
- Basic injection / injection-guard strips HTML/script tags so the
  text is safe to echo back in the Streamlit UI.
- A command history list is maintained (in-memory) to let the UI show
  recent typed commands without needing a database.
"""

import re
from datetime import datetime

# ── Internal command history (module-level list, in-memory) ──────────────
_history: list[dict] = []   # [{"text": ..., "timestamp": ...}, ...]
HISTORY_LIMIT = 50           # keep only the last N entries


# ── Helpers ──────────────────────────────────────────────────────────────

def sanitise(text: str) -> str:
    """
    Clean raw text:
    1. Strip leading/trailing whitespace.
    2. Collapse internal whitespace to single spaces.
    3. Remove HTML / script tags (prevents XSS echo in Streamlit).
    4. Lowercase for consistent downstream processing.
    """
    text = text.strip()
    text = re.sub(r"<[^>]+>", "", text)          # strip HTML tags
    text = re.sub(r"\s+", " ", text)              # collapse whitespace
    text = text.lower()
    return text


def _is_empty(text: str) -> bool:
    return len(text.strip()) == 0


def _is_too_long(text: str, max_chars: int = 300) -> bool:
    return len(text) > max_chars


# ── Public API ──────────────────────────────────────────────────────────

class TextInputHandler:
    """
    Wraps the Streamlit text-input field with validation and
    normalisation so the voice path and text path produce identical
    output for downstream modules.

    Usage (inside Streamlit app)
    ----------------------------
    handler = TextInputHandler()
    raw = st.text_input("Type a command", key="cmd_input")
    if raw:
        result = handler.process(raw)
        if result["valid"]:
            # Pass result["text"] to Intent Detection module
            intent_module.detect(result["text"])
        else:
            st.warning(result["message"])
    """

    def process(self, raw_text: str) -> dict:
        """
        Validate and normalise a raw typed command.

        Parameters
        ----------
        raw_text : str   — Directly from st.text_input()

        Returns
        -------
        dict with keys:
            valid    (bool)   — False if input should be rejected
            text     (str)    — Cleaned, normalised text (empty string if invalid)
            message  (str)    — Status / error description for the UI
            source   (str)    — Always "text" (vs "voice" from the STT path)
        """
        if _is_empty(raw_text):
            return {"valid": False, "text": "", "message": "Empty input.", "source": "text"}

        if _is_too_long(raw_text):
            return {
                "valid": False,
                "text": "",
                "message": f"Input too long ({len(raw_text)} chars). Max 300.",
                "source": "text",
            }

        cleaned = sanitise(raw_text)

        # Log to history
        _history.append({"text": cleaned, "timestamp": datetime.now().isoformat()})
        if len(_history) > HISTORY_LIMIT:
            _history.pop(0)

        return {
            "valid": True,
            "text": cleaned,
            "message": "OK",
            "source": "text",
        }

    def get_history(self) -> list[dict]:
        """Return the most-recent typed commands (newest last)."""
        return list(_history)

    def clear_history(self):
        """Clear the command history."""
        _history.clear()
