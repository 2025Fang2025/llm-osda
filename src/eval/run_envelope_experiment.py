"""
Route B+ Envelope Payment Experiment.

Loads frozen models from a completed training run (Round 2) and recomputes
payment for every triggered test session using the Myerson envelope formula.

Usage:
    cd /path/to/llm_osda
    python -m src.eval.run_envelope_experiment \
        --model_dir outputs/final_qwen3emb/round_2 \
        --splits_dir outputs/final_qwen3emb \
        --dialogues data/dialogues_all.jsonl \
        --output_dir outputs/envelope_revision \
        --n_grid 30 \
        --max_sessions 0
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eval.envelope_payment import (
    EnvelopeResult,
    run_envelope_for_session,
    DEFAULT_THETA_MIN,
    DEFAULT_THETA_MAX,
    DEFAULT_N_GRID,
)
from src.eval.envelope_numerics import build_random_turn_map, random_turn_map_from_outcomes
from src.eval.closed_loop_eval import (
    _run_policy_allocation,
    compute_global_critical_cpc,
    compute_pctrs_for_state,
)
from src.models.text_mlp import UnifiedUserModel
from src.policy.stop_policy import StopPolicy
from src.utils.io import load_jsonl, save_json, ensure_dir, resolve_device, set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def find_winner_for_session(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model,
    stop_policy: StopPolicy,
    gamma: float,
    theta_max: float,
    force_auction_last_turn: bool = True,
    fixed_turn: int | None = None,
) -> Dict[str, Any] | None:
    """Run the mechanism once to identify the winner and GDC payment."""
    chosen_turn, result, meta, q_wait, q_auction = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        fixed_turn=fixed_turn,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity",
        theta_max=theta_max,
        gamma=gamma,
    )
    if chosen_turn is None or result is None:
        return None

    # Compute GDC payment
    gdc_payment = compute_global_critical_cpc(
        policy_name, dialogue, pctr_model, stop_policy,
        winner_index=result.winner_index,
        winner_bid=float(result.winner_ad.get("bid", 0.0)),
        fixed_turn=None if policy_name == "learned_stop" else chosen_turn,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity",
        theta_max=theta_max,
        gamma=gamma,
    )

    return {
        "chosen_turn": chosen_turn,
        "winner_index": result.winner_index,
        "winner_bid": float(result.winner_ad.get("bid", 0.0)),
        "winner_pctr": float(meta.get("winner_pctr", result.pctr)),
        "gdc_payment": gdc_payment,
    }


def run_envelope_experiment(
    test_dialogues: List[Dict[str, Any]],
    pctr_model,
    stop_policy: StopPolicy,
    output_dir: Path,
    policies: List[str] = None,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    n_grid: int = DEFAULT_N_GRID,
    force_auction_last_turn: bool = True,
    max_sessions: int = 0,
    seed: int = 42,
    random_turn_schedule: Dict[str, int] | None = None,
):
    """Main experiment loop."""
    if policies is None:
        policies = ["learned_stop", "first_turn", "last_turn", "random_turn"]

    if max_sessions > 0:
        test_dialogues = test_dialogues[:max_sessions]

    all_results: List[Dict[str, Any]] = []
    all_curves: List[Dict[str, Any]] = []
    # Prefer a schedule supplied from the frozen outcomes so the replay aligns
    # with the evaluated turns; fall back to a seeded schedule otherwise.
    if random_turn_schedule is not None:
        random_turns = random_turn_schedule
        log.info("Random-turn schedule: loaded from frozen outcome sessions")
    else:
        random_turns = build_random_turn_map(test_dialogues, seed)
    if "random_turn" in policies:
        random_turn_counts = Counter(random_turns.values())
        has_multi_turn_session = any(
            len(dialogue.get("turns", [])) > 1 for dialogue in test_dialogues
        )
        if has_multi_turn_session and len(random_turn_counts) <= 1:
            raise RuntimeError(
                f"Degenerate random_turn schedule: {dict(random_turn_counts)}"
            )
        log.info(f"Random-turn schedule: {dict(sorted(random_turn_counts.items()))}")

    for policy_name in policies:
        log.info(f"\n{'='*60}\nPolicy: {policy_name}\n{'='*60}")

        policy_results = []
        n_triggered = 0
        n_monotone = 0

        for dialogue in tqdm(test_dialogues, desc=f"[{policy_name}] envelope"):
            if policy_name == "random_turn":
                fixed_turn = random_turns.get(dialogue["session_id"])
                if fixed_turn is None:
                    # Session not scheduled (e.g., no-trigger in the frozen
                    # outcomes); skip so the replay stays aligned.
                    continue
            else:
                fixed_turn = None
            # Step 1: Identify winner under this policy
            winner_info = find_winner_for_session(
                policy_name, dialogue, pctr_model, stop_policy,
                gamma=gamma, theta_max=theta_max,
                force_auction_last_turn=force_auction_last_turn,
                fixed_turn=fixed_turn,
            )
            if winner_info is None:
                continue
            if policy_name == "random_turn" and int(
                winner_info["chosen_turn"]
            ) != int(fixed_turn):
                raise RuntimeError(
                    "random_turn winner selection ignored its scheduled turn: "
                    f"session={dialogue['session_id']}, scheduled={fixed_turn}, "
                    f"chosen={winner_info['chosen_turn']}"
                )

            n_triggered += 1

            # Step 2: Run envelope payment computation
            env_result = run_envelope_for_session(
                policy_name=policy_name,
                dialogue=dialogue,
                pctr_model=pctr_model,
                stop_policy=stop_policy,
                winner_index=winner_info["winner_index"],
                true_bid=winner_info["winner_bid"],
                gdc_cpc=winner_info["gdc_payment"],
                gamma=gamma,
                theta_min=theta_min,
                theta_max=theta_max,
                n_grid=n_grid,
                force_auction_last_turn=force_auction_last_turn,
                fixed_turn=fixed_turn,
            )

            if env_result.is_monotone:
                n_monotone += 1

            # Collect session-level result
            row = {
                "session_id": env_result.session_id,
                "policy": env_result.policy,
                "winner_index": env_result.winner_index,
                "winner_ad_id": dialogue.get("candidate_ads", [])[env_result.winner_index].get("ad_id", ""),
                "true_bid": env_result.true_bid,
                "winner_pctr": winner_info["winner_pctr"],
                "chosen_turn": winner_info["chosen_turn"],
                "x_hat_at_bid": env_result.x_hat_at_bid,
                "envelope_transfer": env_result.envelope_transfer,
                "envelope_cpc": env_result.envelope_cpc,
                "gdc_cpc": env_result.gdc_cpc,
                "is_monotone": int(env_result.is_monotone),
                "n_violations": env_result.n_violations,
                "max_regret": env_result.max_regret,
                "ir_violation": int(env_result.ir_violation),
                "negative_payment": int(env_result.negative_payment),
                # Recomputed metrics using envelope CPC
                "env_net_revenue": env_result.envelope_cpc * env_result.x_hat_at_bid,
                "gdc_net_revenue": env_result.gdc_cpc * env_result.x_hat_at_bid,
                "env_info_rent": (env_result.true_bid - env_result.envelope_cpc) * env_result.x_hat_at_bid,
                "gdc_info_rent": (env_result.true_bid - env_result.gdc_cpc) * env_result.x_hat_at_bid,
            }
            policy_results.append(row)
            all_results.append(row)

            # Collect allocation curves (for plotting)
            for z, x_hat in zip(env_result.bid_grid, env_result.x_hat_values):
                all_curves.append({
                    "session_id": env_result.session_id,
                    "policy": policy_name,
                    "bid_probe": z,
                    "x_hat": x_hat,
                    "true_bid": env_result.true_bid,
                })

        # Policy-level summary
        if policy_results:
            df_policy = pd.DataFrame(policy_results)
            monotone_rate = n_monotone / max(n_triggered, 1)
            avg_env_cpc = df_policy["envelope_cpc"].mean()
            avg_gdc_cpc = df_policy["gdc_cpc"].mean()
            avg_regret = df_policy["max_regret"].mean()
            median_regret = df_policy["max_regret"].median()
            p95_regret = df_policy["max_regret"].quantile(0.95)

            log.info(f"  Triggered: {n_triggered}/{len(test_dialogues)}")
            log.info(f"  Monotonicity rate: {monotone_rate:.3f} ({n_monotone}/{n_triggered})")
            log.info(f"  Avg Envelope CPC: {avg_env_cpc:.4f}  vs  Avg GDC CPC: {avg_gdc_cpc:.4f}")
            log.info(f"  Regret — mean: {avg_regret:.6f}, median: {median_regret:.6f}, 95th: {p95_regret:.6f}")
            log.info(f"  Avg Env Net Revenue: {df_policy['env_net_revenue'].mean():.4f}")
            log.info(f"  Avg GDC Net Revenue: {df_policy['gdc_net_revenue'].mean():.4f}")

    results_df = pd.DataFrame(all_results)
    if {"first_turn", "random_turn"}.issubset(set(policies)):
        first_turns = results_df.loc[
            results_df["policy"] == "first_turn", ["session_id", "chosen_turn"]
        ].rename(columns={"chosen_turn": "first_turn"})
        random_turn_results = results_df.loc[
            results_df["policy"] == "random_turn", ["session_id", "chosen_turn"]
        ].rename(columns={"chosen_turn": "random_turn"})
        turn_audit = first_turns.merge(random_turn_results, on="session_id")
        if not turn_audit.empty and (
            turn_audit["first_turn"] == turn_audit["random_turn"]
        ).all():
            # Equality is mandatory for T=1 and can occur by chance in a
            # finite multi-turn sample. Validate against the frozen random
            # schedule above, not against another policy's observed choices.
            log.info(
                "random_turn matches first_turn for every matched session; "
                "this is expected for T=1 and is not a replay error"
            )

    # Save results
    results_df.to_csv(output_dir / "envelope_results.csv", index=False)

    curves_df = pd.DataFrame(all_curves)
    curves_df.to_csv(output_dir / "allocation_curves.csv", index=False)

    # Generate summary
    summary = generate_summary(results_df, gamma)
    save_json(summary, output_dir / "summary_metrics.json")

    log.info(f"\nResults saved to: {output_dir}")
    log.info(f"  envelope_results.csv: {len(results_df)} rows")
    log.info(f"  allocation_curves.csv: {len(curves_df)} rows")

    return results_df, curves_df, summary


def generate_summary(results_df: pd.DataFrame, gamma: float) -> Dict[str, Any]:
    """Generate summary metrics mimicking Table 1 format."""
    summary = {"gamma": gamma, "policies": {}}

    for policy, g in results_df.groupby("policy"):
        n = len(g)
        summary["policies"][policy] = {
            "n_sessions": n,
            # Original GDC-based
            "gdc_avg_payment": float(g["gdc_cpc"].mean()),
            "gdc_avg_net_revenue": float(g["gdc_net_revenue"].mean()),
            "gdc_avg_info_rent": float(g["gdc_info_rent"].mean()),
            # New envelope-based
            "env_avg_payment": float(g["envelope_cpc"].mean()),
            "env_avg_net_revenue": float(g["env_net_revenue"].mean()),
            "env_avg_info_rent": float(g["env_info_rent"].mean()),
            # Diagnostics
            "monotonicity_rate": float(g["is_monotone"].mean()),
            "avg_n_violations": float(g["n_violations"].mean()),
            "avg_max_regret": float(g["max_regret"].mean()),
            "median_max_regret": float(g["max_regret"].median()),
            "p95_max_regret": float(g["max_regret"].quantile(0.95)),
            "p99_max_regret": float(g["max_regret"].quantile(0.99)),
            "max_max_regret": float(g["max_regret"].max()),
            "zero_regret_rate": float((g["max_regret"] <= 1e-12).mean()),
            "ir_violation_rate": float(g["ir_violation"].mean()),
            "negative_payment_rate": float(g["negative_payment"].mean()),
            # Allocation info
            "avg_x_hat_at_bid": float(g["x_hat_at_bid"].mean()),
            "avg_winner_bid": float(g["true_bid"].mean()),
            "avg_winner_pctr": float(g["winner_pctr"].mean()),
            "avg_chosen_turn": float(g["chosen_turn"].mean()),
            "chosen_turn_counts": {
                str(int(turn)): int(count)
                for turn, count in g["chosen_turn"].value_counts().sort_index().items()
            },
            # Comparison
            "avg_cpc_gap": float((g["envelope_cpc"] - g["gdc_cpc"]).mean()),
            "avg_abs_cpc_gap": float((g["envelope_cpc"] - g["gdc_cpc"]).abs().mean()),
            "cpc_gap_ratio_of_means_pct": float(
                (g["envelope_cpc"].mean() / g["gdc_cpc"].mean() - 1) * 100
            ),
            "avg_cpc_gap_pct": float(((g["envelope_cpc"] - g["gdc_cpc"]) / g["gdc_cpc"].clip(lower=1e-6)).mean() * 100),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Route B+ Envelope Payment Experiment")
    parser.add_argument("--model_dir", required=True, help="Directory with frozen models (user_model.pt, stop_policy.pt)")
    parser.add_argument("--splits_dir", default="", help="Directory with data_splits.json (defaults to model_dir parent)")
    parser.add_argument("--dialogues", required=True, help="Path to dialogues_all.jsonl")
    parser.add_argument("--output_dir", default="outputs/envelope_revision")
    parser.add_argument("--n_grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--theta_min", type=float, default=DEFAULT_THETA_MIN)
    parser.add_argument("--theta_max", type=float, default=DEFAULT_THETA_MAX)
    parser.add_argument("--max_sessions", type=int, default=0, help="0 = all test sessions")
    parser.add_argument("--policies", nargs="+", default=["learned_stop", "first_turn", "last_turn", "random_turn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outcome_sessions", default="", help="Frozen final_evaluation_sessions.csv; if given, random_turn reuses its scheduled turns")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    model_dir = Path(args.model_dir)

    # Determine splits directory
    splits_dir = Path(args.splits_dir) if args.splits_dir else model_dir.parent
    splits_path = splits_dir / "data_splits.json"
    if not splits_path.exists():
        splits_path = model_dir / "data_splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Cannot find data_splits.json in {splits_dir} or {model_dir}")

    log.info(f"Loading splits from: {splits_path}")
    with open(splits_path) as f:
        sp = json.load(f)
    test_sessions = set(sp["test_sessions"])

    log.info(f"Loading dialogues from: {args.dialogues}")
    dialogues = load_jsonl(args.dialogues)
    test_dialogues = [d for d in dialogues if d["session_id"] in test_sessions]
    log.info(f"  Total dialogues: {len(dialogues)}, test: {len(test_dialogues)}")

    log.info(f"Loading models from: {model_dir}")
    pctr_model = UnifiedUserModel.load(model_dir, device, hidden_dim=256)
    stop_policy = StopPolicy.load(model_dir, device)
    log.info("  Models loaded successfully.")

    t_start = time.time()

    random_turn_schedule = None
    if args.outcome_sessions:
        log.info(f"Loading random-turn schedule from: {args.outcome_sessions}")
        outcome_df = pd.read_csv(args.outcome_sessions)
        random_turn_schedule = random_turn_map_from_outcomes(outcome_df)
        log.info(f"  Loaded {len(random_turn_schedule)} scheduled random_turn sessions")

    results_df, curves_df, summary = run_envelope_experiment(
        test_dialogues=test_dialogues,
        pctr_model=pctr_model,
        stop_policy=stop_policy,
        output_dir=output_dir,
        policies=args.policies,
        gamma=args.gamma,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        n_grid=args.n_grid,
        force_auction_last_turn=True,
        max_sessions=args.max_sessions,
        seed=args.seed,
        random_turn_schedule=random_turn_schedule,
    )

    elapsed = time.time() - t_start
    log.info(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"\n{'='*60}\nSummary:\n{'='*60}")
    log.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
