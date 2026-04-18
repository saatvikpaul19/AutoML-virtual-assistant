"""
nlu_pipeline.py -- Main NLU Entry Point
========================================
Orchestrates the full intent + slot extraction pipeline:

  1. BERT NLU (primary)  -- DistilBERT joint model
  2. Regex fallback       -- if BERT confidence < threshold or critical slots missing

Every step is logged so you can trace exactly what happened:

  [NLU] ==============================================
  [NLU] Input: "set learning rate to 0.001"
  [NLU] -- BERT pass ------------------------------
  [NLU] BERT raw -> intent=set_learning_rate (conf=0.9321)
  [NLU] BERT raw -> slots={LEARNING_RATE: '0.001'}
  ...adapter logs...
  [NLU] -- Fallback gate --------------------------
  ...fallback logs...
  [NLU] [OK] Using BERT result
  [NLU] Final -> intent=set_learning_rate | slots={'learning_rate': 0.001}
  [NLU] ==============================================

  OR (when fallback triggers):

  [NLU] -- Regex fallback -------------------------
  [NLU] Regex intent  -> set_learning_rate
  [NLU] Regex slots   -> {'learning_rate': 0.001}
  [NLU] [OK] Using regex result (FALLBACK)
  [NLU] Final (FALLBACK) -> intent=set_learning_rate | slots={'learning_rate': 0.001}
"""
from __future__ import annotations

from modules.C_nlu.intent_detection import detect_intent, get_intent_category, normalize_text
from modules.C_nlu.slot_filling import fill_slots

# -- BERT NLU -- loaded once at import time --------------------
_BERT_NLU    = None
_BERT_READY  = False
_BERT_ERROR  = ""

try:
    from modules.C_nlu.bert_nlu.inference import BERTNLUInference
    from modules.C_nlu.bert_nlu.integration_adapter import adapt_nlu_output
    from modules.C_nlu.bert_nlu.fallback import should_fallback
    _BERT_NLU   = BERTNLUInference()
    _BERT_READY = True
    print("[NLU] [OK] BERT NLU engine loaded successfully")
except Exception as e:
    _BERT_ERROR = str(e)
    print(f"[NLU] [ERR] BERT NLU initialization failed: {e}")
    print("[NLU]   -> All requests will use regex fallback")

# -- Required slots per intent (for validation only, NOT extraction) --
_REQUIRED: dict[str, list[str]] = {
    "load_dataset":      ["dataset"],
    "search_dataset":    ["query"],
    "get_dataset_info":  [],
    "load_code":         [],
    "search_code":       ["query"],
    "select_model":      ["model"],
    "set_learning_rate": ["learning_rate"],
    "set_batch_size":    ["batch_size"],
    "set_epochs":        ["epochs"],
    "set_layers":        ["layers"],
    "get_weather":       ["city"],
    "set_timer":         ["duration_seconds"],
    "add_time_to_timer": ["duration_seconds"],
    "check_timer":       [],
    "pause_timer":       [],
    "resume_timer":      [],
    "stop_timer":        [],
    "restart_timer":     [],
    "reset_timer":       [],
    "cancel_timer":      [],
    "help":              [],
    "repeat":            [],
    "greetings":         [],
    "farewell":          [],
    "thanks":            [],
    "sleep_va":          [],
    "out_of_scope":      [],
}


def understand(text: str) -> dict:
    """
    Parse user text into a structured NLU result.

    Returns:
        {
            "intent":            str,
            "intent_category":   str,   # stateful | stateless | utility
            "intent_confidence": float,
            "slots":             dict,
            "missing_slots":     list[str],
            "invalid_slots":     dict,
            "raw_text":          str,
            "normalised_text":   str,
            "fallback_used":     bool,
            "fallback_reason":   str,   # why fallback was used (empty = BERT used)
        }
    """
    normalised    = normalize_text(text)
    intent        = "out_of_scope"
    slots: dict   = {}
    fallback_used = False
    fallback_reason = ""
    intent_confidence = 0.0

    print(f"\n[NLU] {'='*56}")
    print(f"[NLU] Input       : {text!r}")
    print(f"[NLU] Normalised  : {normalised!r}")
    print(f"[NLU] BERT ready  : {_BERT_READY}" + (f" (error: {_BERT_ERROR})" if not _BERT_READY else ""))

    # -- 1. Primary: BERT NLU ---------------------------------
    if _BERT_NLU and _BERT_READY:
        print(f"[NLU] -- BERT pass --------------------------------------")
        try:
            bert_raw = _BERT_NLU.parse(text)

            print(f"[NLU] BERT raw -> intent={bert_raw['intent']!r} (conf={bert_raw['intent_confidence']:.4f})")
            print(f"[NLU] BERT raw -> slots={bert_raw['slots']}")
            print(f"[NLU] BERT raw -> token_slots (sample, first 10):")
            for entry in bert_raw.get("token_slots", [])[:10]:
                if entry["label"] != "O":
                    print(f"[NLU]              token={entry['token']!r:12s} -> label={entry['label']}")

            adapted = adapt_nlu_output(bert_raw)

            print(f"[NLU] -- Fallback gate ----------------------------------")
            use_fallback, reason = should_fallback(adapted, text)

            if not use_fallback:
                intent            = adapted["intent"]
                slots             = adapted["slots"]
                intent_confidence = adapted["intent_confidence"]
                fallback_used     = False
                fallback_reason   = ""
                print(f"[NLU] [OK] Using BERT result")
            else:
                fallback_used   = True
                fallback_reason = reason
                print(f"[NLU] [SKIP] BERT rejected -- triggering regex fallback")

        except Exception as e:
            fallback_used   = True
            fallback_reason = f"BERT inference exception: {e}"
            print(f"[NLU] [ERR] BERT inference error: {e}  -> triggering regex fallback")
    else:
        fallback_used   = True
        fallback_reason = f"BERT engine not loaded: {_BERT_ERROR}"

    # -- 2. Fallback: Regex -----------------------------------
    if fallback_used:
        print(f"[NLU] -- Regex fallback ---------------------------------")
        intent            = detect_intent(text)
        slots             = fill_slots(text, intent)
        intent_confidence = 1.0   # regex is "deterministic" -- no probability
        print(f"[NLU] Regex intent  -> {intent!r}")
        print(f"[NLU] Regex slots   -> {slots}")
        print(f"[NLU] [OK] Using regex result (FALLBACK)")
        print(f"[NLU] Fallback reason: {fallback_reason}")

    # -- 3. Slot validation -----------------------------------
    missing_slots: list[str] = []
    invalid_slots: dict      = {}

    for slot_name in _REQUIRED.get(intent, []):
        if slot_name not in slots:
            missing_slots.append(slot_name)

    if "learning_rate" in slots:
        lr = slots["learning_rate"]
        if isinstance(lr, (int, float)) and lr <= 0:
            invalid_slots["learning_rate"] = "must be > 0"
    if "batch_size" in slots:
        bs = slots["batch_size"]
        if isinstance(bs, int) and bs <= 0:
            invalid_slots["batch_size"] = "must be > 0"
    if "epochs" in slots:
        ep = slots["epochs"]
        if isinstance(ep, int) and ep <= 0:
            invalid_slots["epochs"] = "must be > 0"
    if "layers" in slots:
        ly = slots["layers"]
        if isinstance(ly, int) and ly <= 0:
            invalid_slots["layers"] = "must be > 0"
    if "duration_seconds" in slots:
        ds = slots["duration_seconds"]
        if isinstance(ds, int) and ds <= 0:
            invalid_slots["duration_seconds"] = "must be > 0"

    # -- 4. Final log -----------------------------------------
    fb_tag = " [FALLBACK]" if fallback_used else ""
    print(f"[NLU] Final{fb_tag} -> intent={intent!r} | conf={intent_confidence:.4f}")
    print(f"[NLU] Final{fb_tag} -> slots={slots}")
    if missing_slots:
        print(f"[NLU] Missing slots : {missing_slots}")
    if invalid_slots:
        print(f"[NLU] Invalid slots : {invalid_slots}")
    print(f"[NLU] {'='*56}\n")

    return {
        "intent":            intent,
        "intent_category":   get_intent_category(intent),
        "intent_confidence": intent_confidence,
        "slots":             slots,
        "missing_slots":     missing_slots,
        "invalid_slots":     invalid_slots,
        "raw_text":          text,
        "normalised_text":   normalised,
        "fallback_used":     fallback_used,
        "fallback_reason":   fallback_reason,
    }