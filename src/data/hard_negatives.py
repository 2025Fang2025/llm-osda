"""Build hard negative samples for pCTR training.

For each (session, turn), adds random ads from the global pool as negative samples.
This makes the matching task harder and prevents AUC inflation.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List


def build_hard_negatives(
    counterfactuals: List[Dict[str, Any]],
    n_extra_negatives: int = 10,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Add hard negative samples to counterfactual data.

    Returns augmented list (original + hard negatives).
    """
    rng = random.Random(seed)

    # Filter out target_early
    rows = [r for r in counterfactuals if r.get("sample_type") != "target_early"]

    # Build ad pool
    ad_pool = {}
    for r in rows:
        aid = r.get("ad_id", "")
        if aid and aid not in ad_pool:
            ad_pool[aid] = {"ad_id": aid, "title": r.get("title", ""), "bid": r.get("bid", 1.0)}
    all_ad_ids = list(ad_pool.keys())

    # Group by session+turn
    session_turn = defaultdict(list)
    for r in rows:
        key = (r["session_id"], r["turn"])
        session_turn[key].append(r)

    # Build augmented data
    augmented = list(rows)  # keep originals

    for (sid, turn), samples in session_turn.items():
        existing_ads = set(r["ad_id"] for r in samples)
        available = [aid for aid in all_ad_ids if aid not in existing_ads]
        n_sample = min(n_extra_negatives, len(available))
        extra_ads = rng.sample(available, n_sample)

        base_gamma = min(r.get("gamma_score", 0.5) for r in samples)

        for aid in extra_ads:
            ad = ad_pool[aid]
            augmented.append({
                "session_id": sid,
                "turn": turn,
                "sample_type": "hard_negative",
                "ad_id": aid,
                "title": ad["title"],
                "bid": ad["bid"],
                "response": "",
                "clicked": False,
                "gamma_score": max(0.3, base_gamma - 0.1),
                "judge": {},
                "trajectory": [],
            })

    return augmented
