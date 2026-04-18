"""
integration_adapter.py -- BERT -> App Schema Translator
=======================================================
Translates BERT NLU output (UPPER_CASE slot keys) to the format
expected by CommandRouter and ExperimentController (lowercase keys).

Also applies type conversions (e.g. "0.01" -> 0.01 for learning_rate)
and delegates timer duration conversion to the regex slot_filling helper.

Logging: every adaptation step is printed so you can follow
exactly how BERT's raw output becomes the app's NLU payload.
"""
from __future__ import annotations

from .labels import INTENTS
from ..slot_filling import _extract_timer_duration_seconds

# -- Slot key mapping: BERT UPPER_CASE -> app lowercase ---------
_SLOT_MAPPING: dict[str, str] = {
    "DATASET_NAME":      "dataset",
    "DATASET_PATH":      "dataset",       # alias
    "MODEL_NAME":        "model",
    "LEARNING_RATE":     "learning_rate",
    "BATCH_SIZE":        "batch_size",
    "EPOCHS":            "epochs",
    "LAYERS":            "layers",
    "ACTIVATION":        "activation",
    "WEATHER_LOCATION":  "city",
    "WEATHER_CONDITION": "weather_condition",  # NEW
    "DATE":              "day",                # NEW
    "QUERY":             "query",
    "COMPETITION_NAME":  "query",             # alias for show_leaderboard
    "SPLIT_RATIO":       "ratio",
    "ORIGINAL_REQUEST":  "original_request",  # NEW -- for out_of_scope
    "RESULT_TYPE":       "result_type",
    "TIMER_NAME":        "label",
}

# -- Numeric type coercions -------------------------------------
_FLOAT_SLOTS = {"learning_rate", "ratio"}
_INT_SLOTS   = {"batch_size", "epochs", "layers"}


def adapt_nlu_output(bert_output: dict) -> dict:
    """
    Convert raw BERT inference output to the app's NLU schema.

    Input (from BERTNLUInference.parse):
        {
            "intent":            str,
            "intent_confidence": float,
            "slots":             {"UPPER_CASE_KEY": "value", ...},
            "token_slots":       [...],
            "fallback_used":     bool,
        }

    Output (consumed by nlu_pipeline.understand -> CommandRouter):
        {
            "intent":            str,
            "slots":             {"lowercase_key": value, ...},
            "intent_confidence": float,
            "fallback_used":     bool,
        }
    """
    raw_intent      = bert_output.get("intent", "out_of_scope")
    intent_conf     = bert_output.get("intent_confidence", 0.0)
    bert_slots      = bert_output.get("slots", {})
    fallback_used   = bert_output.get("fallback_used", False)

    print(f"\n[NLU-ADAPTER] -- Adapting BERT output ----------------------")
    print(f"[NLU-ADAPTER]  Raw intent   : {raw_intent} (conf={intent_conf:.4f})")
    print(f"[NLU-ADAPTER]  Raw slots    : {bert_slots}")

    adapted_slots: dict = {}

    for bert_key, app_key in _SLOT_MAPPING.items():
        if bert_key not in bert_slots:
            continue

        raw_val = bert_slots[bert_key]
        value   = raw_val

        # Type coercion
        if app_key in _FLOAT_SLOTS:
            try:
                value = float(raw_val)
                print(f"[NLU-ADAPTER]    {bert_key} -> {app_key} = {value!r}  (float coercion)")
            except (ValueError, TypeError):
                print(f"[NLU-ADAPTER]    {bert_key} -> {app_key}: float coercion FAILED for {raw_val!r}, skipped")
                continue
        elif app_key in _INT_SLOTS:
            try:
                value = int(float(raw_val))
                print(f"[NLU-ADAPTER]    {bert_key} -> {app_key} = {value!r}  (int coercion)")
            except (ValueError, TypeError):
                print(f"[NLU-ADAPTER]    {bert_key} -> {app_key}: int coercion FAILED for {raw_val!r}, skipped")
                continue
        else:
            print(f"[NLU-ADAPTER]    {bert_key} -> {app_key} = {value!r}")

        adapted_slots[app_key] = value

    # Special: TIMER_DURATION needs seconds conversion (not just a raw string passthrough)
    if "TIMER_DURATION" in bert_slots:
        raw_dur = bert_slots["TIMER_DURATION"]
        seconds = _extract_timer_duration_seconds(raw_dur)
        if seconds:
            adapted_slots["duration_seconds"] = seconds
            print(f"[NLU-ADAPTER]    TIMER_DURATION -> duration_seconds = {seconds}s  (converted from '{raw_dur}')")
        else:
            print(f"[NLU-ADAPTER]    TIMER_DURATION: could not convert '{raw_dur}' to seconds, skipped")

    print(f"[NLU-ADAPTER]  Adapted slots: {adapted_slots}")
    print(f"[NLU-ADAPTER] -------------------------------------------------\n")

    return {
        "intent":            raw_intent,
        "slots":             adapted_slots,
        "intent_confidence": intent_conf,
        "fallback_used":     fallback_used,
    }
