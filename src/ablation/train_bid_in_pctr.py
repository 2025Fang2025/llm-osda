"""Ablation-1: Train a pCTR variant where bid enters the model input.

This violates assumption (A2) of the DSIC layering: q_i(h_t) should be
bid-blind so that advertisers cannot manipulate quality scores via bidding.

The only difference from the baseline training is that pctr_input_text and
numeric_features here INCLUDE bid information. Everything else (model arch,
training procedure, data splits) stays identical for a controlled comparison.

Run:
  python -m src.ablation.train_bid_in_pctr \
    --dialogues data/dialogues_all.jsonl \
    --counterfactuals data/counterfactuals_with_judge.jsonl \
    --output_dir outputs/ablation_bid_in_pctr \
    --base_model_dir outputs/final_qwen3emb/round_2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from src.config import ExperimentConfig
from src.data.data_utils import (
    build_dialogue_map,
    collect_labeled_samples,
    history_until_turn,
    candidate_ads_text,
    LabeledAdSample,
    split_sessions,
)
from src.models.text_mlp import train_unified_user_model, UnifiedUserModel
from src.utils.io import ensure_dir, load_jsonl, save_json, resolve_device, set_seed


# ---------------------------------------------------------------------------
# Bid-aware input constructors (the ONLY difference from baseline)
# ---------------------------------------------------------------------------

def pctr_input_text_with_bid(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> str:
    """pCTR input that INCLUDES bid — violates bid-blind assumption (A2)."""
    return (
        f"Dialogue history:\n{history_until_turn(dialogue, turn_number)}\n\n"
        f"Ad: {ad.get('title', '')}\n"
        f"Ad bid: {float(ad.get('bid', 0.0)):.2f}\n"
    )


def numeric_features_with_bid(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> List[float]:
    """Numeric features that INCLUDE bid — violates bid-blind assumption (A2)."""
    bid = float(ad.get("bid", 0.0))
    all_bids = [float(a.get("bid", 0.0)) for a in dialogue.get("candidate_ads", [])]
    max_bid = max(all_bids) if all_bids else 1.0
    return [
        bid,
        bid / max(1e-6, max_bid),
    ]


def judge_input_text_with_bid(dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> str:
    """Judge input that includes bid (for unified model training consistency)."""
    parts = [f"Dialogue history:\n{history_until_turn(dialogue, turn_number)}"]
    profile = dialogue.get("user_profile", "")
    if profile:
        parts.append(f"User: {profile}")
    parts.append(f"Ad: {ad.get('title', '')}")
    parts.append(f"Ad bid: {float(ad.get('bid', 0.0)):.2f}")
    parts.append(f"Assistant response:\n{response}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Data builders using bid-aware features
# ---------------------------------------------------------------------------

def build_pctr_data_with_bid(dialogues, samples, session_filter):
    dialogue_map = build_dialogue_map(dialogues)
    texts, nums, ys = [], [], []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        texts.append(pctr_input_text_with_bid(d, s.turn, ad))
        nums.append(numeric_features_with_bid(d, s.turn, ad))
        ys.append(float(s.clicked))
    return texts, nums, ys


def build_judge_data_with_bid(dialogues, samples, session_filter, user_id_map=None):
    dialogue_map = build_dialogue_map(dialogues)
    texts, nums, y_click, y_gamma, user_ids = [], [], [], [], []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        if not s.response:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        texts.append(judge_input_text_with_bid(d, s.turn, ad, s.response))
        nums.append(numeric_features_with_bid(d, s.turn, ad))
        y_click.append(float(s.clicked))
        y_gamma.append(float(s.gamma_score))
        if user_id_map is not None:
            pid = d.get("profile_id", "")
            user_ids.append(user_id_map.get(pid, 0))
        else:
            user_ids.append(0)
    return texts, nums, y_click, y_gamma, user_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Ablation-1: Train pCTR with bid in input")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--output_dir", default="outputs/ablation_bid_in_pctr")
    parser.add_argument("--base_model_dir", default="", help="Base model dir to reuse data_splits.json from")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_tfidf_features", type=int, default=4096)
    parser.add_argument("--text_encoder", choices=["tfidf", "qwen3_embedding"], default="tfidf")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    t_start = time.time()

    log.info(f"Loading data ...")
    dialogues = load_jsonl(args.dialogues)
    counterfactuals_raw = load_jsonl(args.counterfactuals)

    # Hard negatives
    hard_neg_path = Path(args.counterfactuals).parent / (Path(args.counterfactuals).stem + "_hard_neg.jsonl")
    if hard_neg_path.exists():
        counterfactuals = load_jsonl(str(hard_neg_path))
        log.info(f"  Reusing hard negatives: {hard_neg_path} ({len(counterfactuals)} samples)")
    else:
        counterfactuals = counterfactuals_raw

    # User profiles
    profiles_path = Path(args.dialogues).parent / "profiles_.jsonl"
    user_id_map = {}
    if profiles_path.exists():
        profiles_raw = {p["profile_id"]: p for p in load_jsonl(str(profiles_path))}
        for idx, pid in enumerate(sorted(profiles_raw.keys()), start=1):
            user_id_map[pid] = idx
        for d in dialogues:
            pid = d.get("profile_id", "")
            if pid in profiles_raw:
                tags = profiles_raw[pid].get("tags", {})
                d["user_profile"] = f"age: {tags.get('age', '')}, budget: {tags.get('budget', '')}"
        log.info(f"  Loaded {len(profiles_raw)} user profiles")

    samples = collect_labeled_samples(dialogues, counterfactuals)

    # Data splits: reuse from base model if available
    splits_path = None
    if args.base_model_dir:
        candidate = Path(args.base_model_dir) / "data_splits.json"
        if candidate.exists():
            splits_path = candidate
        else:
            candidate = Path(args.base_model_dir).parent / "data_splits.json"
            if candidate.exists():
                splits_path = candidate

    if splits_path and splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        train_sessions = set(sp["train_sessions"])
        val_sessions = set(sp["val_sessions"])
        test_sessions = set(sp["test_sessions"])
        log.info(f"  Reusing splits from {splits_path}: train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)}")
    else:
        train_val_sessions, test_sessions = split_sessions(dialogues, test_ratio=args.test_ratio, seed=args.seed)
        train_val_dialogues = [d for d in dialogues if d["session_id"] in train_val_sessions]
        train_sessions, val_sessions = split_sessions(train_val_dialogues, test_ratio=0.125, seed=args.seed + 17)
        log.info(f"  Created splits: train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)}")

    save_json({
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "test_sessions": sorted(test_sessions),
    }, output_dir / "data_splits.json")

    # Build training data with bid-aware features
    log.info("Building bid-aware pCTR training data ...")
    tr_pctr_t, tr_pctr_n, tr_pctr_y = build_pctr_data_with_bid(dialogues, samples, train_sessions)
    va_pctr_t, va_pctr_n, va_pctr_y = build_pctr_data_with_bid(dialogues, samples, val_sessions)
    log.info(f"  pCTR: train={len(tr_pctr_t)}, val={len(va_pctr_t)}")

    log.info("Building bid-aware Judge training data ...")
    tr_j_t, tr_j_n, tr_j_c, tr_j_g, tr_j_uid = build_judge_data_with_bid(dialogues, samples, train_sessions, user_id_map)
    va_j_t, va_j_n, va_j_c, va_j_g, va_j_uid = build_judge_data_with_bid(dialogues, samples, val_sessions, user_id_map)
    log.info(f"  Judge: train={len(tr_j_t)}, val={len(va_j_t)}")

    # Train unified model (same architecture, just different input features)
    log.info("Training UnifiedUserModel with bid in input ...")
    user_model, metrics = train_unified_user_model(
        tr_pctr_t, tr_pctr_n, tr_pctr_y,
        tr_j_t, tr_j_n, tr_j_c, tr_j_g,
        va_pctr_t, va_pctr_n, va_pctr_y,
        va_j_t, va_j_n, va_j_c, va_j_g,
        device=device,
        max_features=args.max_tfidf_features,
        hidden_dim=256,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        text_encoder_type=args.text_encoder,
        embedding_model=args.embedding_model,
        patience=args.patience,
        judge_user_ids=tr_j_uid,
        val_judge_user_ids=va_j_uid,
        n_users=len(user_id_map) + 1,
    )
    user_model.save(output_dir)
    save_json(metrics, output_dir / "user_model_metrics.json")

    # Also train stop policy using the bid-aware pCTR (for complete mechanism evaluation)
    log.info("Training Stop Policy (using bid-aware pCTR) ...")
    from src.policy.stop_policy import StopState, stop_features_from_auction, train_stop_policy
    from src.auction.auction import run_quality_weighted_allocation
    from collections import defaultdict

    train_dialogues = [d for d in dialogues if d["session_id"] in train_sessions]
    states_by_session = defaultdict(list)

    for d in train_dialogues:
        n_turns = max(1, int(d.get("n_turns", len(d.get("turns", [])) or 1)))
        for idx, turn in enumerate(d.get("turns", [])):
            tn = int(turn.get("turn", idx + 1))
            ads = d.get("candidate_ads", [])
            # Compute pCTR using bid-aware model
            texts = [pctr_input_text_with_bid(d, tn, ad) for ad in ads]
            nums = [numeric_features_with_bid(d, tn, ad) for ad in ads]
            pctrs = user_model.predict_proba(texts, nums)
            pctrs = np.clip(pctrs, 1e-4, 1.0).tolist()

            mech_result = run_quality_weighted_allocation(ads, pctrs, psi_mode="identity")
            if mech_result is None:
                continue
            bids = [float(a.get("bid", 0.0)) for a in ads]
            auction_meta = {
                "pctrs": mech_result.all_pctrs,
                "scores": mech_result.all_scores,
                "bids": bids,
                "winner_index": mech_result.winner_index,
                "winner_ad_id": mech_result.winner_ad.get("ad_id", ""),
                "winner_title": mech_result.winner_ad.get("title", ""),
                "winner_bid": float(mech_result.winner_ad.get("bid", 0.0)),
                "winner_pctr": mech_result.pctr,
                "winner_score": mech_result.score,
                "second_score": mech_result.second_score,
                "payment": 0.0,
            }
            features = stop_features_from_auction(tn, n_turns, auction_meta)
            states_by_session[d["session_id"]].append(StopState(
                session_id=d["session_id"],
                turn=tn,
                n_turns=n_turns,
                features=features,
                immediate_reward=float(mech_result.score),
                auction_meta=auction_meta,
            ))

    stop_policy, stop_metrics = train_stop_policy(
        states_by_session,
        device=device,
        hidden_dim=128,
        epochs=40,
        batch_size=args.batch_size,
        lr=args.lr,
        discount=0.95,
        patience=10,
    )
    stop_policy.save(output_dir)
    save_json(stop_metrics, output_dir / "stop_policy_metrics.json")

    elapsed = time.time() - t_start
    log.info(f"Done in {elapsed:.0f}s. Models saved to: {output_dir}")
    log.info(f"  pCTR val AUC: {metrics.get('pctr_val_auc', 'N/A')}")
    log.info(f"  Stop policy action_match_acc: {stop_metrics.get('train_action_match_acc', 'N/A')}")

    save_json({
        "variant": "bid_in_pctr",
        "description": "Ablation-1: bid enters pCTR input, violates assumption (A2)",
        "user_model_metrics": metrics,
        "stop_policy_metrics": stop_metrics,
        "elapsed_seconds": elapsed,
    }, output_dir / "ablation_meta.json")


if __name__ == "__main__":
    main()
