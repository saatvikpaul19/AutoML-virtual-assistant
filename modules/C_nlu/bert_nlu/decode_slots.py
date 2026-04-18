"""
decode_slots.py — BIO Slot Decoder with WordPiece Subword Awareness
====================================================================
Converts the per-token BIO predictions from BERT back into clean
slot key → value mappings.

Handles:
- Standard word tokens:     "iris" → "iris"
- WordPiece continuations:  "##ris" → joined without space (e.g. "iris")
- Numbers split by `.`:     "0", ".", "001" → "0.001"
- Multi-token phrases:      "red", "wine" → "red wine"
"""
from __future__ import annotations

from .labels import id2slot


def _is_subword(token: str) -> bool:
    """DistilBERT/BERT subword tokens start with ##."""
    return token.startswith("##")


def _join_tokens(tokens: list[str]) -> str:
    """
    Join a list of tokens into a clean string, handling:
    - ## subword continuation (no space before)
    - Punctuation (no space before . , : ; etc.)
    """
    result = ""
    for i, tok in enumerate(tokens):
        tok_clean = tok.replace("##", "")
        if i == 0:
            result = tok_clean
        elif _is_subword(tok) or tok_clean in {".", ",", ":", ";", "-", "_", "/"}:
            # Attach without space
            result += tok_clean
        elif result and result[-1] in {"/", "-", "_"}:
            result += tok_clean
        else:
            result += " " + tok_clean
    return result.strip()


def decode_slots(tokens: list[str], slot_ids) -> dict[str, str]:
    """
    Decode per-token BIO slot predictions into a dictionary.

    Args:
        tokens:   Token list from tokenizer (includes [CLS], [SEP], [PAD])
        slot_ids: Numpy array of slot class IDs, same length as tokens

    Returns:
        {"SLOT_TYPE": "slot value", ...}

    Example:
        tokens   = ["[CLS]", "load", "iris", "dataset", "[SEP]", ...]
        slot_ids = [O,        O,      B-4,    O,          O      ...]
        → {"DATASET_NAME": "iris"}
    """
    _SKIP_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}

    slots: dict[str, str]   = {}
    current_slot: str | None = None
    current_tokens: list[str] = []

    for token, slot_id in zip(tokens, slot_ids):
        if token in _SKIP_TOKENS:
            # Flush any open slot
            if current_slot and current_tokens:
                slots[current_slot] = _join_tokens(current_tokens)
                current_slot  = None
                current_tokens = []
            continue

        label = id2slot[int(slot_id)]

        if label.startswith("B-"):
            # Flush previous slot
            if current_slot and current_tokens:
                slots[current_slot] = _join_tokens(current_tokens)
            # Start new slot
            current_slot   = label[2:]
            current_tokens = [token]

        elif label.startswith("I-") and current_slot == label[2:]:
            # Continuation — append
            current_tokens.append(token)

        elif current_slot and _is_subword(token):
            # Subword continuation even if labeled O (tokenisation artefact)
            # This helps keep "0.001" → "0", ".", "##001" together if the
            # "." token gets labelled O accidentally.
            current_tokens.append(token)

        else:
            # O or mismatched I → flush
            if current_slot and current_tokens:
                slots[current_slot] = _join_tokens(current_tokens)
                current_slot   = None
                current_tokens = []

    # Final flush
    if current_slot and current_tokens:
        slots[current_slot] = _join_tokens(current_tokens)

    return slots
