"""
run_experiments_p1p2.py
Run P1 (virtual-value mechanism) and P2 (IC verification + pCTR ablation) experiments.

Usage:
  python -m src.run_experiments_p1p2 \
    --dialogues data/dialogues_shard0000.jsonl \
    --counterfactuals data/counterfactuals_shard0000.jsonl \
    --model_dir outputs/with_sft_main \
    --output_dir outputs/experiments_p1p2
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.auction.auction import run_virtual_value_auction
from src.data.data_utils import (
    build_dialogue_map,
    collect_labeled_samples,
    build_label_map,
    split_sessions,
)
from src.eval.closed_loop_eval import compute_pctrs_for_state
from src.eval.ic_verification import run_ic_verification, summarize_ic_results
from src.eval.pctr_ablation import ABLATION_VARIANTS, build_pctr_ablation_data
from src.models.text_mlp import PCTRModel, train_pctr_text_model
from src.policy.stop_policy import StopPolicy, stop_features_from_auction
from src.utils.io import ensure_dir, load_jsonl, save_json, resolve_device, set_seed


def run_pctr_ablation_experiment(
    dialogues, samples, train_sessions, val_sessions, test_sessions, device, output_dir,
    max_tfidf_features=4096, hidden_dim=256, epochs=10, batch_size=64, lr=1e-3,
    text_encoder_type="qwen3_embedding", embedding_model="Qwen/Qwen3-Embedding-0.6B",
):
    """Experiment 7.5: Train all pCTR ablation variants and compare."""
    results = {}
    for variant_name in ABLATION_VARIANTS:
        print(f"  Training pCTR variant: {variant_name} ...")
        tr_text, tr_num, tr_y = build_pctr_ablation_data(dialogues, samples, train_sessions, variant_name)
        va_text, va_num, va_y = build_pctr_ablation_data(dialogues, samples, val_sessions, variant_name)

        model, metrics = train_pctr_text_model(
            tr_text, tr_num, tr_y,
            va_text, va_num, va_y,
            device=device,
            max_features=max_tfidf_features,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            text_encoder_type=text_encoder_type,
            embedding_model=embedding_model,
        )
        metrics["variant"] = variant_name
        metrics["description"] = ABLATION_VARIANTS[variant_name]["description"]
        results[variant_name] = metrics
        print(f"    {variant_name}: AUC={metrics.get('val_auc', 'N/A'):.4f}, log_loss={metrics['val_log_loss']:.4f}")

    # Compute Δ pCTR (turn=last minus turn=1) for full and no_history variants
    print("  Computing Δ pCTR across turns ...")
    dialogue_map = build_dialogue_map(dialogues)
    for variant_name in ["full", "no_history", "static"]:
        cfg = ABLATION_VARIANTS[variant_name]
        text_fn = cfg["text_fn"]
        num_fn = cfg["num_fn"]

        delta_pctrs = []
        test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]
        for d in test_dialogues[:200]:
            turns = d.get("turns", [])
            if len(turns) < 2:
                continue
            ads = d.get("candidate_ads", [])
            if not ads:
                continue
            ad = ads[0]
            first_turn = int(turns[0].get("turn", 1))
            last_turn = int(turns[-1].get("turn", len(turns)))
            text_first = text_fn(d, first_turn, ad)
            text_last = text_fn(d, last_turn, ad)
            delta_pctrs.append({
                "session_id": d["session_id"],
                "text_first_len": len(text_first),
                "text_last_len": len(text_last),
            })
        results[variant_name]["n_delta_sessions"] = len(delta_pctrs)

    return results


def main():
    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "4")))
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="P1+P2 experiments: virtual-value + IC + ablation")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--counterfactuals", required=True)
    parser.add_argument("--model_dir", default="outputs/final_qwen3emb/round_2",
                        help="Directory with pre-trained pctr_model and stop_policy")
    parser.add_argument("--output_dir", default="outputs/experiments_p1p2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--theta_min", type=float, default=0.1)
    parser.add_argument("--theta_max", type=float, default=2.18)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max_ic_sessions", type=int, default=3000)
    parser.add_argument("--pctr_epochs", type=int, default=5)
    parser.add_argument("--text_encoder", choices=["qwen3_embedding", "tfidf"], default="tfidf")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print("[1/5] Loading data and pre-trained models ...")
    dialogues = load_jsonl(args.dialogues)
    counterfactuals = load_jsonl(args.counterfactuals)
    samples = collect_labeled_samples(dialogues, counterfactuals)

    train_val_sessions, test_sessions = split_sessions(dialogues, test_ratio=args.test_ratio, seed=args.seed)
    train_val_dialogues = [d for d in dialogues if d["session_id"] in train_val_sessions]
    train_sessions, val_sessions = split_sessions(train_val_dialogues, test_ratio=0.125, seed=args.seed + 17)
    test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]

    pctr_model = PCTRModel.load(args.model_dir, device)
    stop_policy = StopPolicy.load(args.model_dir, device)
    print(f"  Loaded pCTR and StopPolicy from {args.model_dir}")
    print(f"  Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)} sessions")

    # -------------------------------------------------------------------
    print("[2/5] P1: Virtual-value mechanism demo on test sessions ...")
    vv_rows = []
    for d in tqdm(test_dialogues[:args.max_ic_sessions], desc="virtual-value auction"):
        ads = d.get("candidate_ads", [])
        turns = d.get("turns", [])
        if not turns or not ads:
            continue
        last_turn = int(turns[-1].get("turn", len(turns)))
        n_turns = max(1, int(d.get("n_turns", len(turns))))
        pctrs = compute_pctrs_for_state(pctr_model, d, last_turn)

        # Run GSP baseline
        from src.auction.auction import run_quality_adjusted_second_price
        gsp_result = run_quality_adjusted_second_price(ads, pctrs)

        # Compute CV^{-i*} for winner
        winner_idx = gsp_result.winner_index
        from src.eval.ic_verification import compute_cv_minus_winner
        cv_minus = compute_cv_minus_winner(
            d, last_turn, winner_idx, pctr_model, stop_policy, n_turns,
            args.theta_min, args.theta_max,
        )

        # Run virtual-value auction
        vv_result = run_virtual_value_auction(
            ads, pctrs,
            theta_min=args.theta_min, theta_max=args.theta_max,
            gamma=args.gamma, cv_minus_winner=cv_minus,
        )

        vv_rows.append({
            "session_id": d["session_id"],
            "gsp_winner": gsp_result.winner_ad.get("ad_id", ""),
            "gsp_payment": gsp_result.payment_per_click,
            "vv_winner": vv_result.winner_ad.get("ad_id", "") if vv_result else "",
            "vv_payment": vv_result.payment_per_click if vv_result else 0.0,
            "vv_theta_threshold": vv_result.theta_threshold if vv_result else 0.0,
            "cv_minus_winner": cv_minus,
            "vv_filtered": vv_result is None,
        })

    vv_df = pd.DataFrame(vv_rows)
    vv_df.to_csv(output_dir / "p1_virtual_value_comparison.csv", index=False)
    print(f"  GSP avg payment: {vv_df['gsp_payment'].mean():.4f}")
    print(f"  VV  avg payment: {vv_df['vv_payment'].mean():.4f}")
    print(f"  VV  filtered (no eligible): {vv_df['vv_filtered'].sum()}/{len(vv_df)}")
    print(f"  Avg CV^{{-i*}}: {vv_df['cv_minus_winner'].mean():.4f}")

    # -------------------------------------------------------------------
    print("[3/5] P2-a: IC verification (bid perturbation) ...")
    ic_df = run_ic_verification(
        test_dialogues,
        pctr_model=pctr_model,
        stop_policy=stop_policy,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        gamma=args.gamma,
        max_sessions=args.max_ic_sessions,
        seed=args.seed,
    )
    ic_df.to_csv(output_dir / "p2_ic_verification_raw.csv", index=False)
    ic_summary = summarize_ic_results(ic_df)
    ic_summary.to_csv(output_dir / "p2_ic_verification_summary.csv", index=False)
    print("  IC Summary (utility by bid multiplier):")
    print(ic_summary.to_string(index=False))

    # -------------------------------------------------------------------
    print("[4/5] P2-b: pCTR ablation experiment ...")
    ablation_results = run_pctr_ablation_experiment(
        dialogues, samples, train_sessions, val_sessions, test_sessions,
        device=device, output_dir=output_dir,
        epochs=args.pctr_epochs,
        text_encoder_type=args.text_encoder,
        embedding_model=args.embedding_model,
    )
    save_json(ablation_results, output_dir / "p2_pctr_ablation_metrics.json")
    print("  Ablation results:")
    for v, m in ablation_results.items():
        auc = m.get("val_auc", "N/A")
        auc_str = f"{auc:.4f}" if isinstance(auc, float) else str(auc)
        print(f"    {v}: AUC={auc_str}")

    # -------------------------------------------------------------------
    print("[5/5] Done. All outputs saved to:", output_dir)
    save_json({
        "theta_min": args.theta_min,
        "theta_max": args.theta_max,
        "gamma": args.gamma,
        "n_test_sessions": len(test_sessions),
        "experiments": ["p1_virtual_value", "p2_ic_verification", "p2_pctr_ablation"],
    }, output_dir / "experiment_config.json")


if __name__ == "__main__":
    main()
