from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.auction.auction import run_quality_adjusted_second_price, run_virtual_value_auction, run_quality_weighted_allocation
from src.config import ExperimentConfig
from src.data.data_utils import (
    build_dialogue_map,
    collect_labeled_samples,
    build_label_map,
    split_sessions,
    pctr_input_text,
    judge_input_text,
    numeric_features,
)
from src.eval.closed_loop_eval import (
    simulate_first_turn_auction,
    simulate_random_turn_auction,
    simulate_last_turn_auction,
    simulate_learned_stop_policy,
    aggregate_results,
    batch_evaluate,
)
from src.generation.sft_proxy import SoftInsertionRenderer
from src.generation.base_llm_renderer import BaseLLMRenderer, BaseTemplateRenderer
from src.models.text_mlp import train_unified_user_model, UnifiedUserModel
from src.policy.stop_policy import StopState, stop_features_from_auction, train_stop_policy, StopPolicy
from src.user_simulator.judge_proxy import UserJudgeProxy
from src.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl, resolve_device, set_seed
from src.utils.plotting import plot_all_training_curves, plot_unified_user_model_curves, plot_stop_policy_curves


def build_pctr_training_data(dialogues, samples, session_filter):
    dialogue_map = build_dialogue_map(dialogues)
    texts, nums, ys = [], [], []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        texts.append(pctr_input_text(d, s.turn, ad))
        nums.append(numeric_features(d, s.turn, ad))
        ys.append(float(s.clicked))
    return texts, nums, ys


def build_judge_training_data(dialogues, samples, session_filter, user_id_map=None):
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
        texts.append(judge_input_text(d, s.turn, ad, s.response))
        nums.append(numeric_features(d, s.turn, ad))
        y_click.append(float(s.clicked))
        y_gamma.append(float(s.gamma_score))
        if user_id_map is not None:
            pid = d.get("profile_id", "")
            user_ids.append(user_id_map.get(pid, 0))
        else:
            user_ids.append(0)
    return texts, nums, y_click, y_gamma, user_ids


def predict_pctrs_for_dialogue_turn(pctr_model, dialogue, turn_number):
    ads = dialogue.get("candidate_ads", [])
    texts = [pctr_input_text(dialogue, turn_number, ad) for ad in ads]
    nums = [numeric_features(dialogue, turn_number, ad) for ad in ads]
    pctrs = pctr_model.predict_proba(texts, nums)
    return np.clip(pctrs, 1e-4, 1.0).astype(float).tolist()


def immediate_reward_for_train_state(dialogue, turn_number, auction_result, auction_meta, renderer, judge_proxy, lambda_gamma):
    ad = auction_result.winner_ad
    response, response_source = renderer.render(dialogue, turn_number, ad)
    judge_out = judge_proxy.judge(dialogue, turn_number, ad, response)
    click_value = float(judge_out["click_value"])
    gamma = float(judge_out["gamma_score"])
    revenue = float(auction_meta["payment"]) * click_value
    reward = revenue - float(lambda_gamma) * (1.0 - gamma)
    return reward, {
        "response_source": response_source, "judge_source": judge_out["source"],
        "click_value": click_value, "gamma_score": gamma, "revenue": revenue,
    }


def build_stop_training_states(train_dialogues, pctr_model, renderer, judge_proxy, lambda_gamma, gamma=0.95, theta_min=0.1, theta_max=4.1, randomize_bids=True, bid_noise_ratio=0.5, seed=42):
    import random as _rng
    _rng.seed(seed)
    states_by_session = defaultdict(list)
    for d in tqdm(train_dialogues, desc="build Bellman states"):
        n_turns = max(1, int(d.get("n_turns", len(d.get("turns", [])) or 1)))
        for idx, turn in enumerate(d.get("turns", [])):
            tn = int(turn.get("turn", idx + 1))
            ads = d.get("candidate_ads", [])

            # Randomize bids to prevent Stop Policy from memorizing ad-specific bid values
            if randomize_bids:
                ads = [dict(a) for a in ads]
                for a in ads:
                    orig = float(a.get("bid", 1.0))
                    lo = max(0.01, orig * (1.0 - bid_noise_ratio))
                    hi = orig * (1.0 + bid_noise_ratio)
                    a["bid"] = _rng.uniform(lo, hi)

            pctrs = predict_pctrs_for_dialogue_turn(pctr_model, d, tn)
            # Mechanism fix: stop-policy Bellman target should use
            # W_t=max_i psi(b_i)q_i(h_t), not local Myerson payment/revenue.
            # vv_result = run_virtual_value_auction(ads, pctrs, theta_min=theta_min, theta_max=theta_max, gamma=gamma, cv_minus_winner=0.0)
            mech_result = run_quality_weighted_allocation(ads, pctrs, psi_mode="identity")
            if mech_result is None:
                continue
            bids = [float(a.get("bid", 0.0)) for a in ads]
            auction_meta = {
                # Old virtual-value metadata:
                # "pctrs": vv_result.all_pctrs, "scores": vv_result.all_virtual_scores, "bids": bids,
                # "winner_index": vv_result.winner_index, "winner_ad_id": vv_result.winner_ad.get("ad_id", ""),
                # "winner_title": vv_result.winner_ad.get("title", ""), "winner_bid": float(vv_result.winner_ad.get("bid", 0.0)),
                # "winner_pctr": vv_result.pctr, "winner_score": vv_result.virtual_score,
                # "second_score": vv_result.theta_threshold, "payment": vv_result.payment_per_click,
                "pctrs": mech_result.all_pctrs, "scores": mech_result.all_scores, "bids": bids,
                "winner_index": mech_result.winner_index, "winner_ad_id": mech_result.winner_ad.get("ad_id", ""),
                "winner_title": mech_result.winner_ad.get("title", ""), "winner_bid": float(mech_result.winner_ad.get("bid", 0.0)),
                "winner_pctr": mech_result.pctr, "winner_score": mech_result.score,
                "second_score": mech_result.second_score, "payment": 0.0,
            }
            # Old reward-trained stopping objective:
            # imm_reward, feedback = immediate_reward_for_train_state(d, tn, vv_result, auction_meta, renderer, judge_proxy, lambda_gamma=lambda_gamma)
            imm_reward = float(mech_result.score)
            feedback = {
                "response_source": "mechanism_no_generation",
                "judge_source": "mechanism_no_judge",
                "click_value": float(mech_result.pctr),
                "gamma_score": 1.0,
                "revenue": 0.0,
            }
            features = stop_features_from_auction(tn, n_turns, auction_meta)
            auction_meta.update(feedback)
            states_by_session[d["session_id"]].append(StopState(
                session_id=d["session_id"], turn=tn, n_turns=n_turns,
                features=features, immediate_reward=float(imm_reward), auction_meta=auction_meta,
            ))
    return states_by_session


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger(__name__)

    def log_gpu():
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            log.info(f"GPU memory: allocated={alloc:.1f}GB, reserved={reserved:.1f}GB")

    def log_step(msg):
        log.info(msg)
        log_gpu()

    t_start = time.time()

    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="LLM-OSDA final closed-loop training/evaluation")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--output_dir", default="outputs/final_closed_loop")
    parser.add_argument("--model_dir", default="",
                        help="Eval-mode only: load models from here instead of output_dir "
                             "(e.g. a round_N subdir). Splits/results still use output_dir.")
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--pctr_epochs", type=int, default=10)
    parser.add_argument("--judge_epochs", type=int, default=10)
    parser.add_argument("--stop_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--lambda_gamma", type=float, default=0.25)
    parser.add_argument("--reserve_price", type=float, default=0.0)
    parser.add_argument("--max_tfidf_features", type=int, default=4096)
    # Old default was qwen3_embedding, which depends on a machine-specific local model path.
    # parser.add_argument("--text_encoder", choices=["qwen3_embedding", "tfidf"], default="qwen3_embedding")
    parser.add_argument("--text_encoder", choices=["qwen3_embedding", "tfidf"], default="tfidf")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_force_auction_last_turn", action="store_true")

    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--eval_with_proxy_renderer", action="store_true")
    parser.add_argument("--skip_llm_sft", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster LLM inference during eval.")
    parser.add_argument("--llm_adapter_dir", default="")
    parser.add_argument("--llm_max_samples", type=int, default=0)
    parser.add_argument("--llm_epochs", type=float, default=3.0)
    parser.add_argument("--llm_max_length", type=int, default=8192)
    parser.add_argument("--llm_batch_size", type=int, default=4)
    parser.add_argument("--llm_grad_accum", type=int, default=32)
    parser.add_argument("--llm_lr", type=float, default=2e-5)
    parser.add_argument("--llm_lora_r", type=int, default=8)
    parser.add_argument("--llm_lora_alpha", type=int, default=16)
    parser.add_argument("--llm_load_in_4bit", action="store_true")
    parser.add_argument("--llm_max_new_tokens", type=int, default=512)
    parser.add_argument("--max_eval_sessions", type=int, default=0)

    parser.add_argument("--judge_mode", choices=["proxy", "local_llm"], default="proxy")
    parser.add_argument("--use_exact_judge_labels", action="store_true")
    parser.add_argument("--include_ablation", action="store_true")
    parser.add_argument("--train_rounds", type=int, default=1)
    parser.add_argument("--eval_rounds", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=8)

    args = parser.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed, test_ratio=args.test_ratio, max_tfidf_features=args.max_tfidf_features,
        text_encoder=args.text_encoder, embedding_model=args.embedding_model,
        batch_size=args.batch_size, lr=args.lr, discount=args.discount,
        lambda_gamma=args.lambda_gamma, reserve_price=args.reserve_price,
        force_auction_last_turn=not args.no_force_auction_last_turn, device=args.device,
    )
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = ensure_dir(args.output_dir)

    log_step(f"Loading data on device={device} ...")
    dialogues = load_jsonl(args.dialogues)
    counterfactuals_raw = load_jsonl(args.counterfactuals)
    sft_rows = load_jsonl(args.sft)

    # Load user profiles — short text for Judge, build user_id_map for learnable embedding
    profiles_path = Path(args.dialogues).parent / "profiles_.jsonl"
    user_id_map = {}
    if profiles_path.exists():
        profiles_raw = {p["profile_id"]: p for p in load_jsonl(str(profiles_path))}
        # Build user_id_map: profile_id → integer index (0 = padding/unknown)
        for idx, pid in enumerate(sorted(profiles_raw.keys()), start=1):
            user_id_map[pid] = idx
        for d in dialogues:
            pid = d.get("profile_id", "")
            if pid in profiles_raw:
                tags = profiles_raw[pid].get("tags", {})
                d["user_profile"] = f"age: {tags.get('age', '')}, budget: {tags.get('budget', '')}"
        log.info(f"  Loaded {len(profiles_raw)} user profiles, user_id_map size={len(user_id_map)}")
        save_json(user_id_map, output_dir / "user_id_map.json")
    else:
        log.info(f"  No profiles file found at {profiles_path}, skipping")

    # Auto-generate hard negatives if not already done
    hard_neg_path = Path(args.counterfactuals).parent / (Path(args.counterfactuals).stem + "_hard_neg.jsonl")
    if not hard_neg_path.exists():
        log.info(f"  Building hard negatives → {hard_neg_path}")
        from src.data.hard_negatives import build_hard_negatives
        counterfactuals = build_hard_negatives(counterfactuals_raw, n_extra_negatives=10, seed=cfg.seed)
        save_jsonl(counterfactuals, hard_neg_path)
        orig = sum(1 for r in counterfactuals if r.get("sample_type") != "hard_negative")
        hard = sum(1 for r in counterfactuals if r.get("sample_type") == "hard_negative")
        log.info(f"  Hard negatives: {orig} original + {hard} added = {len(counterfactuals)} total")
    else:
        counterfactuals = load_jsonl(str(hard_neg_path))
        log.info(f"  Reusing hard negatives: {hard_neg_path} ({len(counterfactuals)} samples)")

    # Auto-clean SFT data if not already cleaned
    sft_clean_path = Path(args.sft).parent / (Path(args.sft).stem + "_clean.jsonl")
    if not sft_clean_path.exists():
        log.info(f"  Cleaning SFT data → {sft_clean_path}")
        from src.data.sft_cleaner import clean_sft_row
        cleaned_sft = []
        for r in sft_rows:
            cleaned_sft.extend(clean_sft_row(r))
        save_jsonl(cleaned_sft, sft_clean_path)
        sft_rows = cleaned_sft
        log.info(f"  Cleaned: {len(sft_rows)} training samples ({sum(1 for s in sft_rows if s.get('has_ad'))} with ad, {sum(1 for s in sft_rows if not s.get('has_ad'))} without)")
    else:
        sft_rows = load_jsonl(str(sft_clean_path))
        log.info(f"  Reusing cleaned SFT: {sft_clean_path} ({len(sft_rows)} samples)")

    samples = collect_labeled_samples(dialogues, counterfactuals)
    label_map = build_label_map(samples)

    # --- Data split: reuse if exists, create otherwise ---
    splits_path = output_dir / "data_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            sp = json.load(f)
        train_sessions = set(sp["train_sessions"])
        val_sessions = set(sp["val_sessions"])
        test_sessions = set(sp["test_sessions"])
        log.info(f"  Reusing splits: train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)}")
    else:
        if args.mode == "eval":
            raise FileNotFoundError(f"data_splits.json not found in {output_dir}. Run --mode train first.")
        train_val_sessions, test_sessions = split_sessions(dialogues, test_ratio=cfg.test_ratio, seed=cfg.seed)
        train_val_dialogues = [d for d in dialogues if d["session_id"] in train_val_sessions]
        train_sessions, val_sessions = split_sessions(train_val_dialogues, test_ratio=0.125, seed=cfg.seed + 17)
        save_json({"train_sessions": sorted(train_sessions), "val_sessions": sorted(val_sessions), "test_sessions": sorted(test_sessions)}, splits_path)
        log.info(f"  Created splits: train={len(train_sessions)}, val={len(val_sessions)}, test={len(test_sessions)}")

        # Save split data files (model-input format for inspection)
        data_dir = Path(args.dialogues).parent
        train_val_set = train_sessions | val_sessions

        dialogue_map = build_dialogue_map(dialogues)
        for split_name, sess_set in [("train", train_val_set), ("test", test_sessions)]:
            user_model_records = []
            for s in samples:
                if s.session_id not in sess_set:
                    continue
                d = dialogue_map.get(s.session_id)
                if not d:
                    continue
                ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
                has_response = bool(s.response)
                record = {
                    "session_id": s.session_id, "turn": s.turn,
                    "ad_id": s.ad_id, "ad_title": s.title,
                    "sample_type": s.sample_type,
                    "has_response": has_response,
                    "label_clicked": s.clicked,
                    "label_gamma": s.gamma_score if has_response else None,
                    "pctr_input_text": pctr_input_text(d, s.turn, ad),
                    "judge_input_text": judge_input_text(d, s.turn, ad, s.response) if has_response else None,
                    "model_input_numeric": numeric_features(d, s.turn, ad),
                    "response": s.response if has_response else None,
                }
                user_model_records.append(record)
            save_jsonl(user_model_records, data_dir / f"user_model_input_{split_name}.jsonl")
            log.info(f"    {split_name}: {len(user_model_records)} samples ({sum(1 for r in user_model_records if r['has_response'])} with response)")

        def _build_sft_io_record(s):
            messages = s.get("messages", [])
            # Input = all messages except last assistant
            # Output = last assistant message
            input_msgs = []
            output_text = ""
            for i, m in enumerate(messages):
                if m["role"] == "assistant" and i == len(messages) - 1:
                    output_text = m["content"]
                else:
                    input_msgs.append(m)

            input_text = "\n".join(
                f"[{m['role']}]: {m['content']}" for m in input_msgs
            )
            return {
                "session_id": s.get("session_id", ""),
                "has_ad": s.get("has_ad", False),
                "input": input_text,
                "output": output_text,
                "metadata": s.get("metadata", {}),
            }

        sft_train = [_build_sft_io_record(s) for s in sft_rows if s.get("session_id") in train_val_set]
        sft_test = [_build_sft_io_record(s) for s in sft_rows if s.get("session_id") in test_sessions]
        save_jsonl(sft_train, data_dir / "sft_model_input_train.jsonl")
        save_jsonl(sft_test, data_dir / "sft_model_input_test.jsonl")
        log.info(f"  Saved model input files to {data_dir}/")

    train_dialogues = [d for d in dialogues if d["session_id"] in train_sessions]
    test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]
    train_val_sessions_all = train_sessions | val_sessions
    train_samples = [s for s in samples if s.session_id in train_sessions]
    non_test_samples = [s for s in samples if s.session_id in train_val_sessions_all]
    train_sft_rows = [s for s in sft_rows if s.get("session_id") in train_sessions]
    non_test_sft_rows = [s for s in sft_rows if s.get("session_id") in train_val_sessions_all]
    if args.max_eval_sessions and args.max_eval_sessions > 0 and len(test_dialogues) > args.max_eval_sessions:
        test_dialogues = test_dialogues[:args.max_eval_sessions]

    save_json({
        "n_dialogues": len(dialogues), "n_counterfactuals": len(counterfactuals), "n_sft": len(sft_rows),
        "n_labeled_samples": len(samples), "n_train": len(train_sessions), "n_val": len(val_sessions),
        "n_test_used": len(test_dialogues), "config": cfg.__dict__, "llm": vars(args),
    }, output_dir / "data_summary.json")

    # --- Rounds ---
    n_train = args.train_rounds if args.mode in ("train", "all") else 0
    n_eval = args.eval_rounds if args.mode in ("eval", "all") else 0
    total_rounds = max(n_train, n_eval)

    for ri in range(1, total_rounds + 1):
        do_train = ri <= n_train
        do_eval = ri <= n_eval
        use_rounds = total_rounds > 1
        round_dir = output_dir / f"round_{ri}" if use_rounds else output_dir
        if use_rounds:
            round_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"\n{'='*60}\n  Round {ri}/{total_rounds}  train={'yes' if do_train else 'skip'} eval={'yes' if do_eval else 'skip'}\n{'='*60}")

        # =================== TRAIN ===================
        if do_train:
            set_seed(cfg.seed + ri - 1)

            # For Round 2+: copy Round 1 models as starting point for iterative refinement
            if ri > 1 and use_rounds:
                prev_dir = output_dir / f"round_{ri - 1}" if ri > 2 else output_dir / "round_1"
                if not prev_dir.exists():
                    prev_dir = output_dir  # fallback to base output_dir
                # Copy models from previous round if not already in round_dir
                import shutil
                for fname in ["user_model.pt", "text_encoder_meta.json", "text_encoder_tfidf.joblib",
                              "stop_policy.pt", "stop_scaler.joblib"]:
                    src = prev_dir / fname
                    dst = round_dir / fname
                    if src.exists() and not dst.exists():
                        shutil.copy2(str(src), str(dst))
                # Copy SFT model directory
                for sft_name in ["llm_full_sft", "llm_lora_adapter"]:
                    src = prev_dir / sft_name
                    dst = round_dir / sft_name
                    if src.exists() and not dst.exists():
                        shutil.copytree(str(src), str(dst))
                if (prev_dir / "llm_sft_meta.json").exists() and not (round_dir / "llm_sft_meta.json").exists():
                    shutil.copy2(str(prev_dir / "llm_sft_meta.json"), str(round_dir / "llm_sft_meta.json"))
                log.info(f"  Copied Round {ri-1} models to {round_dir} as starting point for iterative refinement")

            # --- Unified User Model (pCTR + Judge): skip if already trained ---
            if (round_dir / "user_model.pt").exists():
                log.info("[train 1/5] Unified user model already exists, loading.")
                user_model = UnifiedUserModel.load(round_dir, device, hidden_dim=cfg.pctr_hidden_dim)
                user_m = None
            else:
                log_step("[train 1/5] Training unified user model (pCTR + Judge) ...")
                tr_pctr_t, tr_pctr_n, tr_pctr_y = build_pctr_training_data(dialogues, samples, train_sessions)
                va_pctr_t, va_pctr_n, va_pctr_y = build_pctr_training_data(dialogues, samples, val_sessions)
                tr_judge_t, tr_judge_n, tr_judge_c, tr_judge_g, tr_judge_uids = build_judge_training_data(dialogues, samples, train_sessions, user_id_map)
                va_judge_t, va_judge_n, va_judge_c, va_judge_g, va_judge_uids = build_judge_training_data(dialogues, samples, val_sessions, user_id_map)
                user_model, user_m = train_unified_user_model(
                    tr_pctr_t, tr_pctr_n, tr_pctr_y,
                    tr_judge_t, tr_judge_n, tr_judge_c, tr_judge_g,
                    va_pctr_t, va_pctr_n, va_pctr_y,
                    va_judge_t, va_judge_n, va_judge_c, va_judge_g,
                    device=device,
                    max_features=cfg.max_tfidf_features, hidden_dim=cfg.pctr_hidden_dim,
                    epochs=args.pctr_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                    text_encoder_type=cfg.text_encoder, embedding_model=cfg.embedding_model,
                    patience=args.patience,
                    judge_user_ids=tr_judge_uids, val_judge_user_ids=va_judge_uids,
                    n_users=len(user_id_map) + 1,
                )
                user_model.save(round_dir)
                save_json(user_m, round_dir / "user_model_metrics.json")

            # Aliases for downstream code
            pctr_model = user_model
            judge_model = user_model

            # --- Stop Policy: skip if already trained ---
            if (round_dir / "stop_policy.pt").exists():
                log.info("[train 3/5] Stop Policy already exists, loading.")
                stop_policy = StopPolicy.load(round_dir, device)
                stop_m = None
            else:
                log_step("[train 2/5] Building Bellman states (W_t = max score) ...")
                log_step("[train 3/5] Training Stop Policy ...")
                train_states = build_stop_training_states(
                    train_dialogues, pctr_model=pctr_model, renderer=None,
                    judge_proxy=None, lambda_gamma=cfg.lambda_gamma, gamma=cfg.discount,
                )
                stop_policy, stop_m = train_stop_policy(
                    train_states, device=device, hidden_dim=cfg.stop_hidden_dim,
                    epochs=args.stop_epochs, batch_size=cfg.batch_size, lr=cfg.lr, discount=cfg.discount,
                    patience=args.patience * 2,
                )
                stop_policy.save(round_dir)
                save_json(stop_m, round_dir / "stop_policy_metrics.json")

            # Plot training curves
            if user_m:
                plot_unified_user_model_curves(user_m, round_dir)
            if stop_m:
                plot_stop_policy_curves(stop_m, round_dir)
            if user_m and stop_m:
                save_json({"user_model": user_m, "stop_policy": stop_m}, round_dir / "training_metrics.json")

            # --- SFT: skip if already trained ---
            has_sft = (round_dir / "llm_full_sft").exists() or (round_dir / "llm_lora_adapter").exists()
            if has_sft:
                log.info("[train 5/6] SFT model already exists, skipping.")
            elif args.eval_with_proxy_renderer:
                log_step("[train 5/6] Proxy mode: skipping LLM SFT.")
            elif not args.skip_llm_sft:
                log_step("[train 5/6] Training LLM renderer ...")
                try:
                    from src.llm_sft.train_lora_sft import train_sft
                    train_sft(
                        sft_path=args.sft, output_dir=round_dir, base_model=args.base_model,
                        train_session_ids=set(train_sessions), max_samples=args.llm_max_samples,
                        max_length=args.llm_max_length, epochs=args.llm_epochs,
                        batch_size=args.llm_batch_size, grad_accum=args.llm_grad_accum,
                        lr=args.llm_lr, use_lora=args.use_lora, lora_r=args.llm_lora_r,
                        lora_alpha=args.llm_lora_alpha, seed=args.seed, fp16=True, gradient_checkpointing=True,
                        load_in_4bit=args.llm_load_in_4bit,
                    )
                except Exception:
                    log.exception("[train 5/6] LLM SFT FAILED")
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log_step("[train 5/6] LLM SFT complete.")

            # --- Iterative Mechanism-Aware Refinement (Round 2+) ---
            if ri > 1 and not args.eval_with_proxy_renderer and not args.skip_llm_sft:
                log_step(f"[train 5.5/6] Iterative Mechanism-Aware Refinement (Round {ri}) ...")
                try:
                    from src.training.iterative_refinement import (
                        regenerate_responses, rescore_samples, best_of_n_selection,
                    )

                    dialogue_map = build_dialogue_map(dialogues)

                    # Load trained renderer for regeneration
                    sft_meta_path = round_dir / "llm_sft_meta.json"
                    sft_dir = ""
                    if sft_meta_path.exists():
                        with open(sft_meta_path) as f:
                            sft_dir = json.load(f).get("save_dir", "")
                    if not sft_dir:
                        prev_round = output_dir / f"round_{ri-1}" if total_rounds > 1 else output_dir
                        for candidate in [prev_round / "llm_full_sft", prev_round / "llm_lora_adapter"]:
                            if candidate.exists():
                                sft_dir = str(candidate)
                                break

                    if sft_dir:
                        if args.use_vllm:
                            from src.generation.base_llm_renderer import VLLMRenderer
                            renderer = VLLMRenderer(
                                base_model=sft_dir,
                                max_new_tokens=args.llm_max_new_tokens,
                        temperature=0.3, top_p=0.9,
                                gpu_memory_utilization=0.45,
                            )
                        else:
                            from src.generation.base_llm_renderer import BaseLLMRenderer
                            renderer = BaseLLMRenderer(
                                base_model=sft_dir, device=device,
                                max_new_tokens=args.llm_max_new_tokens,
                                temperature=0.3, top_p=0.9,
                            )

                        # Step 4: Regenerate responses
                        new_responses = regenerate_responses(
                            renderer, dialogues, samples, train_sessions,
                            dialogue_map,
                        )

                        # Step 5: Re-score with UnifiedUserModel
                        updated_samples = rescore_samples(
                            user_model, samples, new_responses,
                            dialogue_map, train_sessions,
                            user_id_map=user_id_map,
                        )

                        # Step 6: Retrain UnifiedUserModel on updated data
                        log_step("[train 5.5a] Retraining UnifiedUserModel on re-scored data ...")
                        tr_pctr_t, tr_pctr_n, tr_pctr_y = build_pctr_training_data(dialogues, updated_samples, train_sessions)
                        va_pctr_t, va_pctr_n, va_pctr_y = build_pctr_training_data(dialogues, updated_samples, val_sessions)
                        tr_judge_t, tr_judge_n, tr_judge_c, tr_judge_g, tr_judge_uids = build_judge_training_data(dialogues, updated_samples, train_sessions, user_id_map)
                        va_judge_t, va_judge_n, va_judge_c, va_judge_g, va_judge_uids = build_judge_training_data(dialogues, updated_samples, val_sessions, user_id_map)
                        user_model, user_m = train_unified_user_model(
                            tr_pctr_t, tr_pctr_n, tr_pctr_y,
                            tr_judge_t, tr_judge_n, tr_judge_c, tr_judge_g,
                            va_pctr_t, va_pctr_n, va_pctr_y,
                            va_judge_t, va_judge_n, va_judge_c, va_judge_g,
                            device=device,
                            max_features=cfg.max_tfidf_features, hidden_dim=cfg.pctr_hidden_dim,
                            epochs=args.pctr_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                            text_encoder_type=cfg.text_encoder, embedding_model=cfg.embedding_model,
                            patience=args.patience,
                            judge_user_ids=tr_judge_uids, val_judge_user_ids=va_judge_uids,
                            n_users=len(user_id_map) + 1,
                        )
                        user_model.save(round_dir)
                        save_json(user_m, round_dir / "user_model_metrics.json")
                        pctr_model = user_model
                        judge_model = user_model

                        # Retrain Stop Policy
                        log_step("[train 5.5b] Retraining Stop Q-Network ...")
                        train_states = build_stop_training_states(
                            train_dialogues, pctr_model=pctr_model, renderer=None,
                            judge_proxy=None, lambda_gamma=cfg.lambda_gamma, gamma=cfg.discount,
                        )
                        stop_policy, stop_m = train_stop_policy(
                            train_states, device=device, hidden_dim=cfg.stop_hidden_dim,
                            epochs=args.stop_epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                            discount=cfg.discount, patience=args.patience * 2,
                        )
                        stop_policy.save(round_dir)

                        # Step 7: Best-of-N selection → retrain LLM
                        log_step("[train 5.5c] Best-of-N selection ...")
                        best_sft = best_of_n_selection(
                            renderer, user_model, updated_samples,
                            dialogue_map, train_sessions, n_candidates=4,
                            user_id_map=user_id_map,
                        )
                        if best_sft:
                            best_sft_path = round_dir / "sft_best_of_n.jsonl"
                            save_jsonl(best_sft, best_sft_path)
                            log_step(f"[train 5.5d] Retraining LLM on {len(best_sft)} best-of-N samples ...")
                            from src.llm_sft.train_lora_sft import train_sft
                            train_sft(
                                sft_path=str(best_sft_path), output_dir=round_dir,
                                base_model=args.base_model,
                                train_session_ids=None, max_samples=args.llm_max_samples,
                                max_length=args.llm_max_length, epochs=args.llm_epochs,
                                batch_size=args.llm_batch_size, grad_accum=args.llm_grad_accum,
                                lr=args.llm_lr, use_lora=args.use_lora,
                                lora_r=args.llm_lora_r, lora_alpha=args.llm_lora_alpha,
                                seed=args.seed, fp16=True, gradient_checkpointing=True,
                                load_in_4bit=args.llm_load_in_4bit,
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        del renderer
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        log_step(f"[train 5.5] Iterative refinement complete.")
                    else:
                        log.warning("  No SFT model found for iterative refinement, skipping.")
                except Exception:
                    log.exception("[train 5.5] Iterative refinement FAILED")

            elapsed = time.time() - t_start
            log_step(f"[train 6/6] Training complete ({elapsed:.0f}s). Models saved to: {round_dir}")
            if args.mode == "train":
                continue

        # =================== EVAL ===================
        if do_eval:
            model_dir = round_dir if do_train else output_dir
            # Eval-only override: models may live in a round_N subdir while
            # data_splits.json / results stay in output_dir.
            if not do_train and args.model_dir:
                model_dir = Path(args.model_dir)
            if not do_train:
                log_step(f"[eval 1/4] Loading pre-trained models from {model_dir} ...")
                user_model = UnifiedUserModel.load(model_dir, device, hidden_dim=cfg.pctr_hidden_dim)
                pctr_model = user_model
                judge_model = user_model
                stop_policy = StopPolicy.load(model_dir, device)

            # Mechanism fix: proxy renderer must not retrieve exact test responses.
            # proxy_renderer = SoftInsertionRenderer(samples, sft_rows)
            proxy_renderer = SoftInsertionRenderer(non_test_samples, non_test_sft_rows)

            sft_meta_path = model_dir / "llm_sft_meta.json"
            adapter_dir = args.llm_adapter_dir.strip()
            if not adapter_dir and sft_meta_path.exists():
                with open(sft_meta_path) as f:
                    adapter_dir = json.load(f).get("save_dir", "")

            log_step("[eval 2/4] Loading renderers ...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.eval_with_proxy_renderer:
                untrained_renderer = BaseTemplateRenderer()
                trained_renderer = proxy_renderer
                renderer_setup = "proxy"
            elif args.use_vllm:
                log.info("  vLLM mode: will load models sequentially to save GPU memory")
                trained_renderer = "vllm_deferred_trained"
                untrained_renderer = "vllm_deferred_untrained"
                renderer_setup = "vllm"
            else:
                if args.use_lora:
                    from src.generation.llm_renderer import TrainedLLMRenderer
                    trained_renderer = TrainedLLMRenderer(
                        base_model=args.base_model, adapter_dir=adapter_dir, device=device,
                        max_new_tokens=args.llm_max_new_tokens, temperature=0.2, top_p=0.9,
                        load_in_4bit=args.llm_load_in_4bit,
                    )
                else:
                    from src.generation.base_llm_renderer import BaseLLMRenderer as _FullSFT
                    trained_renderer = _FullSFT(
                        base_model=adapter_dir, device=device,
                        max_new_tokens=args.llm_max_new_tokens, temperature=0.2, top_p=0.9,
                    )
                untrained_renderer = BaseLLMRenderer(
                    base_model=args.base_model, device=device,
                    max_new_tokens=args.llm_max_new_tokens, temperature=0.2, top_p=0.9,
                )
                renderer_setup = "trained_lora" if args.use_lora else "trained_full_sft"

            if args.judge_mode == "local_llm":
                from src.user_simulator.local_llm_judge import LocalUserLLMJudge
                user_judge = LocalUserLLMJudge(
                    base_model=args.base_model, label_map=label_map,
                    fallback_judge_model=judge_model, device=device,
                )
                judge_setup = "local_llm"
            else:
                user_judge = UserJudgeProxy(label_map, judge_model, use_exact_labels=args.use_exact_judge_labels, user_id_map=user_id_map)
                judge_setup = "proxy" + ("_exact" if args.use_exact_judge_labels else "")

            log_step("[eval 3/4] Closed-loop evaluation ...")
            log.info(f"  Eval batch_size={args.eval_batch_size}, sessions={len(test_dialogues)}")
            # Frozen embedding => same ad text re-encoded many times across
            # policies / bid bisection / cv computation. Cache it so Phase1
            # doesn't re-run the 0.6B forward (tfidf is cheap; embedding is not).
            enc = getattr(pctr_model, "encoder", None)
            if enc is not None and hasattr(enc, "enable_cache"):
                enc.enable_cache(True)
            rows = batch_evaluate(
                test_dialogues, pctr_model, stop_policy,
                trained_renderer, untrained_renderer, user_judge,
                lambda_gamma=cfg.lambda_gamma, gamma=cfg.discount,
                seed=cfg.seed + ri,
                force_auction_last_turn=cfg.force_auction_last_turn,
                include_ablation=args.include_ablation,
                eval_batch_size=args.eval_batch_size,
                vllm_trained_model=adapter_dir if (args.use_vllm and adapter_dir) else None,
                vllm_untrained_model=args.base_model if args.use_vllm else None,
                vllm_max_new_tokens=args.llm_max_new_tokens,
            )

            session_df = pd.DataFrame(rows)
            summary_df = aggregate_results(rows)
            session_df["round"] = ri
            session_df.to_csv(round_dir / "final_evaluation_sessions.csv", index=False)
            summary_df.to_csv(round_dir / "final_evaluation_summary.csv", index=False)
            save_json({"renderer": renderer_setup, "judge": judge_setup, "round": ri}, round_dir / "final_eval_readme.json")

            elapsed = time.time() - t_start
            log.info(f"[eval 4/4] Round {ri} done ({elapsed:.0f}s):")
            log.info("\n" + summary_df.to_string(index=False))
            log.info(f"  Outputs: {round_dir}")

    total_time = time.time() - t_start
    log.info(f"All done in {total_time:.0f}s ({total_time/60:.1f}min). Results in: {output_dir}")


if __name__ == "__main__":
    main()
