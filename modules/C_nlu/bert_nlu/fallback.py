"""
fallback.py -- BERT Confidence Gate
=====================================
Determines whether the BERT NLU pipeline result is reliable enough to
use directly, or whether we should fall back to the regex-based system.

Logs the exact reason for every fallback decision so you can diagnose
why a particular utterance triggered regex instead of BERT.
"""
from __future__ import annotations

from .config import CONFIDENCE_THRESHOLD

# -- Intents that MUST have specific slots to be trusted -------
_CRITICAL_SLOTS: dict[str, list[str]] = {
    "load_dataset":      ["dataset"],
    "select_model":      ["model"],
    "set_learning_rate": ["learning_rate"],
    "set_batch_size":    ["batch_size"],
    "set_epochs":        ["epochs"],
    "set_layers":        ["layers"],
    "get_weather":       ["city"],       # regex has the best city extractor
    "set_timer":         ["duration_seconds"],  # regex handles "5 mins" -> seconds
    "add_time_to_timer": ["duration_seconds"],
}


def should_fallback(bert_result: dict, text: str = "") -> tuple[bool, str]:
    """
    Evaluate whether the BERT result should be trusted or rejected.

    Args:
        bert_result: Adapted output from adapt_nlu_output()
                     Keys: intent, slots, intent_confidence, fallback_used
        text:        Original user text (for logging context only)

    Returns:
        (should_fallback: bool, reason: str)
        reason explains why fallback was triggered (empty string if not).

    Examples logged:
        [NLU-FALLBACK] PASS  -- intent=start_training conf=0.92 >= 0.70, no critical slots required
        [NLU-FALLBACK] FAIL  -- intent=get_weather conf=0.45 < threshold=0.70  -> FALLBACK
        [NLU-FALLBACK] FAIL  -- intent=set_timer conf=0.84 but missing slot 'duration_seconds' -> FALLBACK
    """
    intent = bert_result.get("intent", "out_of_scope")
    slots  = bert_result.get("slots", {})
    conf   = bert_result.get("intent_confidence", 0.0)

    print(f"[NLU-FALLBACK] Evaluating: intent={intent!r} | conf={conf:.4f} | threshold={CONFIDENCE_THRESHOLD}")

    # -- Rule 1: Low intent confidence ------------------------
    if conf < CONFIDENCE_THRESHOLD:
        reason = (
            f"BERT confidence {conf:.4f} < threshold {CONFIDENCE_THRESHOLD}  "
            f"(intent={intent!r})"
        )
        print(f"[NLU-FALLBACK] [ERR] FALLBACK -> {reason}")
        return True, reason

    # -- Rule 2: Critical slots missing -----------------------
    required_slots = _CRITICAL_SLOTS.get(intent, [])
    missing = [s for s in required_slots if s not in slots]
    if missing:
        reason = (
            f"Intent {intent!r} (conf={conf:.4f}) is missing critical slot(s): "
            f"{missing}  -- falling back for better slot extraction"
        )
        print(f"[NLU-FALLBACK] [ERR] FALLBACK -> {reason}")
        return True, reason

    # -- Rule 3: Numeric consistency check (Decimal Safety Valve) ----
    # If BERT extracted '1.0' or '1' but the raw text contains '0.01', '0.001', etc.
    # BERT often misses the leading '0.' and we want to catch that.
    if intent == "set_learning_rate" and "learning_rate" in slots:
        import re
        bert_val_str = str(slots["learning_rate"])
        # If BERT got 1.0, 1, 5.0, 5 but the text has something like 0.01 or 0.05
        if bert_val_str in ["1.0", "1", "5.0", "5", "0.1", "0.5"]:
            # Look for suspicious decimals in the text that BERT might have missed
            suspicious = re.findall(r"0\.\d+", text)
            if suspicious and bert_val_str.endswith(suspicious[0].split('.')[-1]):
                reason = f"BERT numeric extraction suspicious: got {bert_val_str} but text contains {suspicious[0]}"
                print(f"[NLU-FALLBACK] [ERR] FALLBACK -> {reason}")
                return True, reason

    # -- All checks passed -------------------------------------
    print(
        f"[NLU-FALLBACK] [OK] PASS    -- intent={intent!r} conf={conf:.4f} >= {CONFIDENCE_THRESHOLD}"
        + (f", slots present: {list(slots.keys())}" if slots else ", no slots required")
    )
    return False, ""
