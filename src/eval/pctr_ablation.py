"""
Experiment 7.5: pCTR Model Ablation — User Intent Evolution.

Trains multiple pCTR variants with different input configurations:
  - full: complete dialogue history + turn embedding (numeric feature)
  - no_history: only current-turn user query (truncate history)
  - no_turn_emb: full history but remove turn-number numeric features
  - static: only ad features (no dialogue input at all)

Then evaluates each variant's:
  - AUC on click prediction
  - Δ pCTR = pCTR(turn=last) - pCTR(turn=1) per session (intent evolution signal)
  - Downstream mechanism revenue when plugged into the full pipeline
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.data.data_utils import (
    build_dialogue_map,
    history_until_turn,
    candidate_ads_text,
    LabeledAdSample,
)


# ---------------------------------------------------------------------------
# Ablated input-text constructors
# ---------------------------------------------------------------------------

def pctr_input_text_full(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> str:
    """Full model: complete history + all features (same as original)."""
    return (
        f"Original query: {dialogue.get('query', '')}\n"
        f"Dialogue history before current assistant response:\n{history_until_turn(dialogue, turn_number)}\n\n"
        f"Candidate ads:\n{candidate_ads_text(dialogue)}\n\n"
        f"Scored ad title: {ad.get('title', '')}\n"
        f"Scored ad id: {ad.get('ad_id', '')}\n"
        # Mechanism fix: q_i(h_t) must be bid-clean.
        # f"Scored ad bid: {float(ad.get('bid', 0.0)):.2f}\n"
        f"Current turn: {turn_number}\n"
    )


def pctr_input_text_no_history(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> str:
    """Ablation: only current-turn user query, no prior dialogue history."""
    current_user = ""
    for t in dialogue.get("turns", []):
        if int(t.get("turn", 0)) == int(turn_number):
            current_user = t.get("user", "")
            break
    if not current_user:
        current_user = dialogue.get("query", "")
    return (
        f"User query: {current_user}\n\n"
        f"Scored ad title: {ad.get('title', '')}\n"
        f"Scored ad id: {ad.get('ad_id', '')}\n"
        # Mechanism fix: q_i(h_t) must be bid-clean.
        # f"Scored ad bid: {float(ad.get('bid', 0.0)):.2f}\n"
        f"Current turn: {turn_number}\n"
    )


def pctr_input_text_static(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> str:
    """Ablation: only ad features, no dialogue context at all."""
    return (
        f"Scored ad title: {ad.get('title', '')}\n"
        f"Scored ad id: {ad.get('ad_id', '')}\n"
        # Mechanism fix: q_i(h_t) must be bid-clean.
        # f"Scored ad bid: {float(ad.get('bid', 0.0)):.2f}\n"
    )


def numeric_features_full(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[float]:
    """Full numeric features including turn position."""
    n_turns = max(1, int(dialogue.get("n_turns", len(dialogue.get("turns", [])) or 1)))
    # Mechanism fix: pCTR numeric features must not include bid.
    # bid = float(ad.get("bid", 0.0))
    # max_bid = max([float(a.get("bid", 0.0)) for a in dialogue.get("candidate_ads", [])] or [1.0])
    return [
        float(turn_number) / 5.0,
        float(turn_number) / float(max(1, n_turns)),
        # bid,
        # bid / max(1e-6, max_bid),
        0.0,
        0.0,
        float(n_turns) / 5.0,
    ]


def numeric_features_no_turn(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[float]:
    """Ablation: remove turn-number features (set to 0)."""
    # Mechanism fix: remove bid from quality features.
    # bid = float(ad.get("bid", 0.0))
    # max_bid = max([float(a.get("bid", 0.0)) for a in dialogue.get("candidate_ads", [])] or [1.0])
    return [
        0.0,
        0.0,
        # bid,
        # bid / max(1e-6, max_bid),
        0.0,
        0.0,
        0.0,
    ]


def numeric_features_static(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[float]:
    # Old description:
    # """Ablation: only bid-related features."""
    """Ablation: ad-only text, no bid/context numeric signal."""
    # Mechanism fix: static pCTR may use ad text/id, but not bid.
    # bid = float(ad.get("bid", 0.0))
    # max_bid = max([float(a.get("bid", 0.0)) for a in dialogue.get("candidate_ads", [])] or [1.0])
    return [
        0.0,
        0.0,
        # bid,
        # bid / max(1e-6, max_bid),
        0.0,
        0.0,
        0.0,
    ]


# ---------------------------------------------------------------------------
# Ablation variant registry
# ---------------------------------------------------------------------------

ABLATION_VARIANTS = {
    "full": {
        "text_fn": pctr_input_text_full,
        "num_fn": numeric_features_full,
        "description": "Full history + turn embedding",
    },
    "no_history": {
        "text_fn": pctr_input_text_no_history,
        "num_fn": numeric_features_full,
        "description": "Current-turn query only (no prior dialogue)",
    },
    "no_turn_emb": {
        "text_fn": pctr_input_text_full,
        "num_fn": numeric_features_no_turn,
        "description": "Full history but no turn-number features",
    },
    "static": {
        "text_fn": pctr_input_text_static,
        "num_fn": numeric_features_static,
        "description": "Ad features only (no dialogue context)",
    },
}


def build_pctr_ablation_data(
    dialogues: List[Dict[str, Any]],
    samples: List[LabeledAdSample],
    session_filter: set,
    variant: str = "full",
) -> Tuple[List[str], List[List[float]], List[float]]:
    """Build pCTR training data using a specific ablation variant's text/num functions."""
    cfg = ABLATION_VARIANTS[variant]
    text_fn = cfg["text_fn"]
    num_fn = cfg["num_fn"]
    dialogue_map = build_dialogue_map(dialogues)

    texts, nums, ys = [], [], []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        texts.append(text_fn(d, s.turn, ad))
        nums.append(num_fn(d, s.turn, ad))
        ys.append(float(s.clicked))
    return texts, nums, ys
