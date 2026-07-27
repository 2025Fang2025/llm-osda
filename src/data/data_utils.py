from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


@dataclass
class LabeledAdSample:
    session_id: str
    turn: int
    ad_id: str
    title: str
    bid: float
    response: str
    clicked: float
    gamma_score: float
    sample_type: str
    source: str


def build_dialogue_map(dialogues: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {d["session_id"]: d for d in dialogues}


def get_turn(dialogue: Dict[str, Any], turn_number: int) -> Optional[Dict[str, Any]]:
    for t in dialogue.get("turns", []):
        if int(t.get("turn", -1)) == int(turn_number):
            return t
    return None


def get_ad_from_dialogue(dialogue: Dict[str, Any], ad_id: str, fallback_title: str = "", fallback_bid: float = 0.0) -> Dict[str, Any]:
    for ad in dialogue.get("candidate_ads", []):
        if ad.get("ad_id") == ad_id:
            return ad
    return {"ad_id": ad_id, "title": fallback_title, "bid": float(fallback_bid)}


def history_until_turn(dialogue: Dict[str, Any], turn_number: int) -> str:
    """History visible to the platform immediately after user gives query at turn_number.

    It includes all previous user/assistant messages and the current user query,
    but not the current assistant response.
    """
    parts: List[str] = []
    for t in dialogue.get("turns", []):
        tn = int(t.get("turn", 0))
        if tn < turn_number:
            parts.append(f"User: {t.get('user', '')}")
            parts.append(f"Assistant: {t.get('assistant', '')}")
        elif tn == turn_number:
            parts.append(f"User: {t.get('user', '')}")
            break
    return "\n".join(parts)


def candidate_ads_text(dialogue: Dict[str, Any]) -> str:
    lines = []
    for idx, ad in enumerate(dialogue.get("candidate_ads", []), start=1):
        lines.append(f"{idx}. {ad.get('title', '')}")
    return "\n".join(lines)


def pctr_input_text(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> str:
    return (
        f"Dialogue history:\n{history_until_turn(dialogue, turn_number)}\n\n"
        f"Ad: {ad.get('title', '')}\n"
    )


def judge_input_text(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> str:
    parts = [f"Dialogue history:\n{history_until_turn(dialogue, turn_number)}"]
    profile = dialogue.get("user_profile", "")
    if profile:
        parts.append(f"User: {profile}")
    parts.append(f"Ad: {ad.get('title', '')}")
    parts.append(f"Assistant response:\n{response}")
    return "\n\n".join(parts) + "\n"


def numeric_features(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[float]:
    return []


def collect_labeled_samples(dialogues: List[Dict[str, Any]], counterfactuals: List[Dict[str, Any]]) -> List[LabeledAdSample]:
    dialogue_map = build_dialogue_map(dialogues)
    samples: Dict[Tuple[str, int, str, str], LabeledAdSample] = {}

    # Counterfactual/user-LLM labels.
    for cf in counterfactuals:
        # Skip target_early: 100% negative due to timing, not relevance.
        if cf.get("sample_type") == "target_early":
            continue
        sid = cf["session_id"]
        if sid not in dialogue_map:
            continue
        key = (sid, int(cf["turn"]), cf["ad_id"], "cf")
        samples[key] = LabeledAdSample(
            session_id=sid,
            turn=int(cf["turn"]),
            ad_id=cf["ad_id"],
            title=cf.get("title", ""),
            bid=float(cf.get("bid", 0.0)),
            response=cf.get("response", ""),
            clicked=float(bool(cf.get("clicked", False))),
            gamma_score=float(cf.get("gamma_score", 0.5)),
            sample_type=cf.get("sample_type", "counterfactual"),
            source="counterfactuals",
        )

    # Observed SOFT turns from dialogues. This improves label coverage.
    for d in dialogues:
        sid = d["session_id"]
        for t in d.get("turns", []):
            if t.get("action") != "SOFT":
                continue
            judge = t.get("judge") or {}
            gamma = float(judge.get("gamma_score", 0.5))
            for ad in t.get("ads_inserted", []):
                clicked = 0.0
                for item in judge.get("ads", []) or []:
                    if item.get("ad_id") == ad.get("ad_id"):
                        clicked = float(bool(item.get("clicked", False)))
                        break
                key = (sid, int(t.get("turn", 0)), ad.get("ad_id", ""), "obs")
                samples[key] = LabeledAdSample(
                    session_id=sid,
                    turn=int(t.get("turn", 0)),
                    ad_id=ad.get("ad_id", ""),
                    title=ad.get("title", ""),
                    bid=float(ad.get("bid", 0.0)),
                    response=t.get("assistant", ""),
                    clicked=clicked,
                    gamma_score=gamma,
                    sample_type=t.get("sample_type", "observed_soft"),
                    source="dialogues_observed_soft",
                )

    return list(samples.values())


def build_label_map(samples: List[LabeledAdSample]) -> Dict[Tuple[str, int, str], LabeledAdSample]:
    # Prefer actual observed soft if duplicate exists, otherwise counterfactual.
    priority = {"dialogues_observed_soft": 2, "counterfactuals": 1}
    out: Dict[Tuple[str, int, str], LabeledAdSample] = {}
    for s in samples:
        key = (s.session_id, int(s.turn), s.ad_id)
        if key not in out or priority.get(s.source, 0) >= priority.get(out[key].source, 0):
            out[key] = s
    return out


def split_sessions(dialogues: List[Dict[str, Any]], test_ratio: float = 0.2, seed: int = 42) -> Tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    sessions = np.array([d["session_id"] for d in dialogues])
    rng.shuffle(sessions)
    n_test = max(1, int(len(sessions) * test_ratio))
    test = set(sessions[:n_test].tolist())
    train = set(sessions[n_test:].tolist())
    return train, test


def extract_ad_id_from_response(response: str) -> Optional[str]:
    m = re.search(r"@[^@]+@\[([^\]]+)\]", response or "")
    if m:
        return m.group(1)
    return None
