from __future__ import annotations

import argparse
import gc
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.auction.auction import run_quality_adjusted_second_price
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
    simulate_learned_stop_policy,
    aggregate_results,
)
from src.generation.sft_proxy import SoftInsertionRenderer
from src.models.text_mlp import train_pctr_text_model, train_judge_text_model
from src.policy.stop_policy import StopState, stop_features_from_auction, train_stop_policy
from src.user_simulator.judge_proxy import UserJudgeProxy
from src.utils.io import ensure_dir, load_jsonl, save_json, resolve_device, set_seed


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


def build_judge_training_data(dialogues, samples, session_filter):
    dialogue_map = build_dialogue_map(dialogues)
    texts, nums, y_click, y_gamma = [], [], [], []
    for s in samples:
        if s.session_id not in session_filter:
            continue
        d = dialogue_map.get(s.session_id)
        if not d:
            continue
        ad = {"ad_id": s.ad_id, "title": s.title, "bid": s.bid}
        texts.append(judge_input_text(d, s.turn, ad, s.response))
        nums.append(numeric_features(d, s.turn, ad))
        y_click.append(float(s.clicked))
        y_gamma.append(float(s.gamma_score))
    return texts, nums, y_click, y_gamma


def predict_pctrs_for_dialogue_turn(pctr_model, dialogue, turn_number):
    ads = dialogue.get("candidate_ads", [])
    texts = [pctr_input_text(dialogue, turn_number, ad) for ad in ads]
    nums = [numeric_features(dialogue, turn_number, ad) for ad in ads]
    pctrs = pctr_model.predict_proba(texts, nums)
    return np.clip(pctrs, 1e-4, 1.0).astype(float).tolist()


def immediate_reward_for_train_state(
    dialogue: Dict[str, Any],
    turn_number: int,
    auction_result,
    auction_meta: Dict[str, Any],
    renderer: SoftInsertionRenderer,
    judge_proxy: UserJudgeProxy,
    lambda_gamma: float,
) -> Tuple[float, Dict[str, Any]]:
    # Training the stopping policy with LLM generations for every possible state is slow.
    # We use the exact/proxy SFT renderer for Bellman-target construction, and reserve
    # the trained LLM renderer for the final held-out closed-loop evaluation.
    ad = auction_result.winner_ad
    response, response_source = renderer.render(dialogue, turn_number, ad)
    judge_out = judge_proxy.judge(dialogue, turn_number, ad, response)
    click_value = float(judge_out["click_value"])
    gamma = float(judge_out["gamma_score"])
    revenue = float(auction_meta["payment"]) * click_value
    reward = revenue - float(lambda_gamma) * (1.0 - gamma)
    return reward, {
        "response_source": response_source,
        "judge_source": judge_out["source"],
        "judge_exact": bool(judge_out["exact"]),
        "click_value": click_value,
        "gamma_score": gamma,
        "revenue": revenue,
    }


def build_stop_training_states(
    train_dialogues: List[Dict[str, Any]],
    pctr_model,
    renderer: SoftInsertionRenderer,
    judge_proxy: UserJudgeProxy,
    reserve_price: float,
    lambda_gamma: float,
) -> Dict[str, List[StopState]]:
    states_by_session: Dict[str, List[StopState]] = defaultdict(list)
    for d in tqdm(train_dialogues, desc="build stop-policy Bellman states"):
        n_turns = max(1, int(d.get("n_turns", len(d.get("turns", [])) or 1)))
        for idx, turn in enumerate(d.get("turns", [])):
            tn = int(turn.get("turn", idx + 1))
            pctrs = predict_pctrs_for_dialogue_turn(pctr_model, d, tn)
            result = run_quality_adjusted_second_price(d.get("candidate_ads", []), pctrs, reserve_price=reserve_price)
            auction_meta = {
                "pctrs": result.all_pctrs,
                "scores": result.all_scores,
                "bids": [float(a.get("bid", 0.0)) for a in d.get("candidate_ads", [])],
                "winner_index": result.winner_index,
                "winner_ad_id": result.winner_ad.get("ad_id", ""),
                "winner_title": result.winner_ad.get("title", ""),
                "winner_bid": float(result.winner_ad.get("bid", 0.0)),
                "winner_pctr": result.pctr,
                "winner_score": result.score,
                "second_score": result.second_score,
                "payment": result.payment_per_click,
            }
            imm_reward, feedback = immediate_reward_for_train_state(
                d, tn, result, auction_meta, renderer, judge_proxy, lambda_gamma=lambda_gamma
            )
            features = stop_features_from_auction(tn, n_turns, auction_meta)
            auction_meta.update(feedback)
            states_by_session[d["session_id"]].append(StopState(
                session_id=d["session_id"],
                turn=tn,
                n_turns=n_turns,
                features=features,
                immediate_reward=float(imm_reward),
                auction_meta=auction_meta,
            ))
    return states_by_session


def main():
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--sft", required=True)
    parser.add_argument("--output_dir", default="outputs/closed_loop_with_sft")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--pctr_epochs", type=int, default=20)
    parser.add_argument("--judge_epochs", type=int, default=20)
    parser.add_argument("--stop_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--lambda_gamma", type=float, default=0.25)
    parser.add_argument("--reserve_price", type=float, default=0.0)
    parser.add_argument("--max_tfidf_features", type=int, default=4096)
    parser.add_argument("--text_encoder", choices=["qwen3_embedding", "tfidf"], default="qwen3_embedding")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no_force_auction_last_turn", action="store_true")

    # LLM SFT arguments. Defaults are chosen for RTX 4060 8GB.
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B")
    parser.add_argument("--skip_llm_sft", action="store_true")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA instead of full SFT for the LLM renderer.")
    parser.add_argument("--llm_adapter_dir", default="")
    parser.add_argument("--eval_with_proxy_renderer", action="store_true")
    parser.add_argument("--llm_max_samples", type=int, default=600)
    parser.add_argument("--llm_epochs", type=float, default=1.0)
    parser.add_argument("--llm_max_length", type=int, default=1024)
    parser.add_argument("--llm_batch_size", type=int, default=1)
    parser.add_argument("--llm_grad_accum", type=int, default=8)
    parser.add_argument("--llm_lr", type=float, default=2e-4)
    parser.add_argument("--llm_lora_r", type=int, default=8)
    parser.add_argument("--llm_lora_alpha", type=int, default=16)
    parser.add_argument("--llm_max_new_tokens", type=int, default=140)
    parser.add_argument("--max_eval_sessions", type=int, default=80)

    args = parser.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        test_ratio=args.test_ratio,
        max_tfidf_features=args.max_tfidf_features,
        text_encoder=args.text_encoder,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        lr=args.lr,
        discount=args.discount,
        lambda_gamma=args.lambda_gamma,
        reserve_price=args.reserve_price,
        force_auction_last_turn=not args.no_force_auction_last_turn,
        device=args.device,
    )
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = ensure_dir(args.output_dir)

    print(f"[1/9] Loading data on device={device} ...")
    dialogues = load_jsonl(args.dialogues)
    counterfactuals = load_jsonl(args.counterfactuals)
    sft_rows = load_jsonl(args.sft)
    samples = collect_labeled_samples(dialogues, counterfactuals)
    label_map = build_label_map(samples)
    train_val_sessions, test_sessions = split_sessions(dialogues, test_ratio=cfg.test_ratio, seed=cfg.seed)
    train_val_dialogues = [d for d in dialogues if d["session_id"] in train_val_sessions]
    train_sessions, val_sessions = split_sessions(train_val_dialogues, test_ratio=0.125, seed=cfg.seed + 17)
    train_dialogues = [d for d in dialogues if d["session_id"] in train_sessions]
    val_dialogues = [d for d in dialogues if d["session_id"] in val_sessions]
    test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]
    if args.max_eval_sessions and len(test_dialogues) > args.max_eval_sessions:
        test_dialogues = test_dialogues[:args.max_eval_sessions]

    save_json({
        "n_dialogues": len(dialogues),
        "n_counterfactuals": len(counterfactuals),
        "n_sft": len(sft_rows),
        "n_labeled_samples": len(samples),
        "n_label_map": len(label_map),
        "n_train_sessions": len(train_sessions),
        "n_val_sessions": len(val_sessions),
        "n_test_sessions_used": len(test_dialogues),
        "config": cfg.__dict__,
        "llm": {
            "base_model": args.base_model,
            "skip_llm_sft": args.skip_llm_sft,
            "llm_max_samples": args.llm_max_samples,
            "llm_epochs": args.llm_epochs,
            "llm_max_length": args.llm_max_length,
            "llm_batch_size": args.llm_batch_size,
            "llm_grad_accum": args.llm_grad_accum,
        }
    }, output_dir / "data_summary.json")

    print("[2/9] Training pCTR model from pre-auction dialogue/ad features ...")
    tr_text, tr_num, tr_y = build_pctr_training_data(dialogues, samples, train_sessions)
    va_text, va_num, va_y = build_pctr_training_data(dialogues, samples, val_sessions)
    pctr_model, pctr_metrics = train_pctr_text_model(
        tr_text, tr_num, tr_y,
        va_text, va_num, va_y,
        device=device,
        max_features=cfg.max_tfidf_features,
        hidden_dim=cfg.pctr_hidden_dim,
        epochs=args.pctr_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        text_encoder_type=cfg.text_encoder,
        embedding_model=cfg.embedding_model,
    )
    pctr_model.save(output_dir)
    save_json(pctr_metrics, output_dir / "pctr_metrics.json")
    print("pCTR metrics:", pctr_metrics)

    print("[3/9] Training user-LLM judge proxy for click/gamma after generated response ...")
    jt, jn, jc, jg = build_judge_training_data(dialogues, samples, train_sessions)
    vt, vn, vc, vg = build_judge_training_data(dialogues, samples, val_sessions)
    judge_model, judge_metrics = train_judge_text_model(
        jt, jn, jc, jg,
        vt, vn, vc, vg,
        device=device,
        max_features=cfg.max_tfidf_features,
        hidden_dim=cfg.judge_hidden_dim,
        epochs=args.judge_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        text_encoder_type=cfg.text_encoder,
        embedding_model=cfg.embedding_model,
    )
    judge_model.save(output_dir)
    save_json(judge_metrics, output_dir / "judge_metrics.json")
    print("Judge metrics:", judge_metrics)

    print("[4/9] Building proxy renderer for stop-policy reward construction ...")
    proxy_renderer = SoftInsertionRenderer(samples, sft_rows)
    judge_proxy = UserJudgeProxy(label_map, judge_model)

    print("[5/9] Training optimal-stopping network WAIT/AUCTION with Bellman targets ...")
    train_states = build_stop_training_states(
        train_dialogues,
        pctr_model=pctr_model,
        renderer=proxy_renderer,
        judge_proxy=judge_proxy,
        reserve_price=cfg.reserve_price,
        lambda_gamma=cfg.lambda_gamma,
    )
    stop_policy, stop_metrics = train_stop_policy(
        train_states,
        device=device,
        hidden_dim=cfg.stop_hidden_dim,
        epochs=args.stop_epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        discount=cfg.discount,
    )
    stop_policy.save(output_dir)
    save_json(stop_metrics, output_dir / "stop_policy_metrics.json")
    print("Stop policy metrics:", stop_metrics)

    adapter_dir = args.llm_adapter_dir.strip()
    if not args.eval_with_proxy_renderer:
        print("[6/9] Training LLM renderer on sft_conversations ...")
        if not args.skip_llm_sft:
            from src.llm_sft.train_lora_sft import train_sft
            meta = train_sft(
                sft_path=args.sft,
                output_dir=output_dir,
                base_model=args.base_model,
                train_session_ids=set(train_sessions),
                max_samples=args.llm_max_samples,
                max_length=args.llm_max_length,
                epochs=args.llm_epochs,
                batch_size=args.llm_batch_size,
                grad_accum=args.llm_grad_accum,
                lr=args.llm_lr,
                use_lora=args.use_lora,
                lora_r=args.llm_lora_r,
                lora_alpha=args.llm_lora_alpha,
                seed=args.seed,
                fp16=True,
                gradient_checkpointing=True,
            )
            adapter_dir = meta["save_dir"]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif not adapter_dir:
            raise ValueError("--skip_llm_sft requires --llm_adapter_dir, or use --eval_with_proxy_renderer.")
    else:
        print("[6/9] Skipping LLM renderer evaluation; using proxy renderer by request.")

    print("[7/9] Loading evaluation renderer ...")
    if args.eval_with_proxy_renderer:
        eval_renderer = proxy_renderer
        renderer_type = "proxy_renderer"
    else:
        if args.use_lora:
            from src.generation.llm_renderer import TrainedLLMRenderer
            eval_renderer = TrainedLLMRenderer(
                base_model=args.base_model,
                adapter_dir=adapter_dir,
                device=device,
                max_new_tokens=args.llm_max_new_tokens,
                temperature=0.2,
                top_p=0.9,
            )
            renderer_type = "trained_lora_llm"
        else:
            from src.generation.base_llm_renderer import BaseLLMRenderer as _FullSFTRenderer
            eval_renderer = _FullSFTRenderer(
                base_model=adapter_dir,
                device=device,
                max_new_tokens=args.llm_max_new_tokens,
                temperature=0.2,
                top_p=0.9,
            )
            renderer_type = "trained_full_sft_llm"

    print("[8/9] Closed-loop evaluation on held-out sessions: first-turn vs learned stopping ...")
    rows = []
    for d in tqdm(test_dialogues, desc="evaluate"):
        rows.append(simulate_first_turn_auction(
            d, pctr_model, eval_renderer, judge_proxy,
            reserve_price=cfg.reserve_price,
            lambda_gamma=cfg.lambda_gamma,
        ))
        rows.append(simulate_learned_stop_policy(
            d, pctr_model, stop_policy, eval_renderer, judge_proxy,
            reserve_price=cfg.reserve_price,
            lambda_gamma=cfg.lambda_gamma,
            force_auction_last_turn=cfg.force_auction_last_turn,
        ))

    session_df = pd.DataFrame(rows)
    summary_df = aggregate_results(rows)
    session_df["renderer_type"] = renderer_type
    session_df.to_csv(output_dir / "evaluation_sessions.csv", index=False)
    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)

    print("[9/9] Done. Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nRenderer used in evaluation: {renderer_type}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
