"""Iterative Mechanism-Aware Refinement.

Round 2+: After initial training (Round 1), close the feedback loop:
  Step 4: Trained LLM regenerates responses for training data
  Step 5: UnifiedUserModel re-scores new responses
  Step 6: Retrain UnifiedUserModel + Stop Policy on updated data
  Step 7: Best-of-N: generate N responses, pick highest msm reward, retrain LLM
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def regenerate_responses(
    renderer,
    dialogues: List[Dict],
    samples,
    session_filter: set,
    dialogue_map: Dict,
    batch_size: int = 1,
    max_samples: int = 0,
    seed: int = 42,
) -> Dict[Tuple[str, int, str], str]:
    """Step 4: Use trained LLM to regenerate responses for training samples."""
    import random

    results = {}
    items = []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        if not s.response:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        items.append((s, d, ad))

    # Subsample to keep runtime reasonable
    if max_samples > 0 and len(items) > max_samples:
        rng = random.Random(seed)
        items = rng.sample(items, max_samples)

    log.info(f"  Regenerating {len(items)} responses with trained LLM...")

    # Try batch generation with render_batch if available
    if hasattr(renderer, 'render_batch') and len(items) > 10:
        render_items = [(d, s.turn, ad) for s, d, ad in items]
        log.info(f"  Using batch rendering ({len(render_items)} items)...")
        rendered = renderer.render_batch(render_items, batch_size=16)
        for (s, d, ad), (text, tag) in zip(items, rendered):
            key = (s.session_id, s.turn, s.ad_id)
            results[key] = text
    else:
        for s, d, ad in tqdm(items, desc="regenerate responses"):
            try:
                response, _ = renderer.render(d, s.turn, ad)
            except Exception:
                response = s.response
            key = (s.session_id, s.turn, s.ad_id)
            results[key] = response

    log.info(f"  Regenerated {len(results)} responses")
    return results


def rescore_samples(
    user_model,
    samples,
    new_responses: Dict,
    dialogue_map: Dict,
    session_filter: set,
    user_id_map: Dict = None,
) -> List:
    """Step 5: Re-score samples with new responses using UnifiedUserModel Judge mode."""
    from src.data.data_utils import judge_input_text, numeric_features, LabeledAdSample

    updated = []
    for s in samples:
        if s.session_id not in session_filter:
            updated.append(s)
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            updated.append(s)
            continue

        key = (s.session_id, s.turn, s.ad_id)
        new_resp = new_responses.get(key)

        if new_resp:
            ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
            text = judge_input_text(d, s.turn, ad, new_resp)
            nums = [numeric_features(d, s.turn, ad)]
            uid = (user_id_map or {}).get(d.get("profile_id", ""), 0)
            click_prob, gamma_pred = user_model.predict([text], nums, user_ids=[uid])

            updated.append(LabeledAdSample(
                session_id=s.session_id,
                turn=s.turn,
                ad_id=s.ad_id,
                title=s.title,
                bid=s.bid,
                response=new_resp,
                clicked=float(click_prob[0]),
                gamma_score=float(gamma_pred[0]),
                sample_type=s.sample_type + "_rescored",
                source="iterative_refinement",
            ))
        else:
            updated.append(s)

    n_rescored = sum(1 for s in updated if s.source == "iterative_refinement")
    log.info(f"  Re-scored {n_rescored}/{len(updated)} samples")
    return updated


def best_of_n_selection(
    renderer,
    user_model,
    samples,
    dialogue_map: Dict,
    session_filter: set,
    n_candidates: int = 4,
    mechanism_auction_fn=None,
    max_samples: int = 0,
    seed: int = 42,
    user_id_map: Dict = None,
) -> List[Dict]:
    """Step 7: Generate N responses per sample, pick highest mechanism reward.

    Returns cleaned SFT training data with best responses.
    """
    from src.data.data_utils import judge_input_text, numeric_features
    from src.data.sft_cleaner import make_system_prompt_with_ad
    import re
    import random

    best_sft_data = []

    items_with_response = [
        s for s in samples
        if s.session_id in session_filter and s.response
        and dialogue_map.get(s.session_id)
    ]

    # Subsample
    if max_samples > 0 and len(items_with_response) > max_samples:
        rng = random.Random(seed)
        items_with_response = rng.sample(items_with_response, max_samples)

    log.info(f"  Best-of-{n_candidates} selection on {len(items_with_response)} samples...")

    # Batch generate all candidates at once
    all_render_items = []
    item_indices = []  # maps render result back to sample
    for idx, s in enumerate(items_with_response):
        d = dialogue_map[s.session_id]
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        for _ in range(n_candidates):
            all_render_items.append((d, s.turn, ad))
            item_indices.append(idx)

    log.info(f"  Generating {len(all_render_items)} candidate responses in batch...")
    if hasattr(renderer, 'render_batch') and len(all_render_items) > 10:
        all_responses = renderer.render_batch(all_render_items, batch_size=0)
        all_resp_texts = [r[0] for r in all_responses]
    else:
        all_resp_texts = []
        for d, tn, ad in tqdm(all_render_items, desc="generating candidates"):
            try:
                resp, _ = renderer.render(d, tn, ad)
            except Exception:
                resp = ""
            all_resp_texts.append(resp)

    # Group by sample and score
    from collections import defaultdict
    candidates_by_idx = defaultdict(list)
    for i, resp in enumerate(all_resp_texts):
        candidates_by_idx[item_indices[i]].append(resp)

    for idx, s in enumerate(tqdm(items_with_response, desc="scoring candidates")):
        d = dialogue_map[s.session_id]
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}

        candidates = candidates_by_idx[idx]
        candidates.append(s.response)  # include original

        best_resp = s.response
        best_score = -float("inf")
        uid = (user_id_map or {}).get(d.get("profile_id", ""), 0)

        for resp in candidates:
            if not resp:
                continue
            text = judge_input_text(d, s.turn, ad, resp)
            nums = [numeric_features(d, s.turn, ad)]
            click_prob, gamma_pred = user_model.predict([text], nums, user_ids=[uid])
            click_val = float(click_prob[0])

            # Mechanism reward: bid * click (proxy for P * q)
            score = float(ad["bid"]) * click_val
            if score > best_score:
                best_score = score
                best_resp = resp

        # Build SFT sample with best response
        # Fix ad format if needed (untrained LLM may output wrong format)
        from src.eval.closed_loop_eval import _fix_ad_format
        best_resp = _fix_ad_format(best_resp, ad)
        has_ad = bool(re.search(r"@[^@]+@\[[^\]]+\]", best_resp))
        if has_ad:
            # Build conversation history
            history = []
            for t in d.get("turns", []):
                tn = int(t.get("turn", 0))
                if tn < s.turn:
                    history.append(("user", t.get("user", "")))
                    history.append(("assistant", t.get("assistant", "")))
                elif tn == s.turn:
                    history.append(("user", t.get("user", "")))
                    break

            system = make_system_prompt_with_ad(s.title, s.ad_id)
            messages = [{"role": "system", "content": system}]
            for role, content in history:
                messages.append({"role": role, "content": content})
            messages.append({"role": "assistant", "content": best_resp})

            best_sft_data.append({
                "session_id": s.session_id,
                "profile_id": "",
                "turn_index": s.turn,
                "has_ad": True,
                "messages": messages,
                "metadata": {"best_of_n_score": best_score},
            })

    log.info(f"  Selected {len(best_sft_data)} best-of-N samples for SFT")
    return best_sft_data
