"""Metric: rule-agreement between llm-stop and the committed Bellman rule.

For each dialogue turn, we compare two decisions:
  - Bellman decision from the baseline stop policy (Q_wait vs Q_auction, i.e.
    the committed W_t vs gamma*CV_t comparison).
  - LLM-stop decision from the SFT'd Qwen (STOP or WAIT).

A3 says the stopping rule is a committed formula from the bid-blind quality
signal. If llm-stop truly implements A3 it must agree with the Bellman rule
on every turn (up to numerical ties). Disagreement quantifies the A3 violation.

Usage:
  python -m src.eval.rule_agreement_test \
    --dialogues data/dialogues_all.jsonl \
    --baseline_dir outputs/final_qwen3emb/round_2 \
    --llm_stop_dir outputs/exp7_6_layering/ablation_llm_stop \
    --output_dir outputs/exp7_6_layering/ablation_llm_stop \
    --n_sessions 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.eval.closed_loop_eval import (
    mechanism_meta_at_turn,
    compute_pctrs_for_state,
)
from src.models.text_mlp import UnifiedUserModel
from src.policy.stop_policy import StopPolicy, stop_features_from_auction
from src.utils.io import ensure_dir, load_jsonl, resolve_device, save_json


def bellman_stop_bit(
    dialogue: Dict[str, Any],
    turn_number: int,
    pctr_model,
    stop_policy: StopPolicy,
    n_turns: int,
) -> str:
    """Return STOP/WAIT from the committed Q-net (Bellman rule)."""
    result, meta = mechanism_meta_at_turn(
        dialogue, turn_number, pctr_model,
        psi_mode="identity",
    )
    if result is None:
        return "WAIT"
    features = stop_features_from_auction(turn_number, n_turns, meta)
    q = stop_policy.q_values(features)
    # q[0] = Q_wait, q[1] = Q_auction. auction dominates -> STOP.
    return "STOP" if float(q[1]) >= float(q[0]) else "WAIT"


def llm_stop_bit(dialogue: Dict[str, Any], turn_number: int, llm_policy) -> str:
    """Return STOP/WAIT from the SFT'd LLM."""
    return llm_policy.decide(dialogue, turn_number)


def test_rule_agreement(
    dialogues: List[Dict[str, Any]],
    pctr_model,
    stop_policy: StopPolicy,
    llm_policy,
    n_sessions: int,
) -> pd.DataFrame:
    sessions = dialogues[:n_sessions]
    rows: List[Dict[str, Any]] = []
    for dialogue in tqdm(sessions, desc="rule-agreement: llm-stop vs Bellman"):
        turns = dialogue.get("turns", [])
        if not turns:
            continue
        n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
        for idx, turn in enumerate(turns):
            tn = int(turn.get("turn", idx + 1))
            is_last = idx == len(turns) - 1
            bellman = bellman_stop_bit(dialogue, tn, pctr_model, stop_policy, n_turns)
            # Bellman rule forces auction at last turn (matches _choose_learned_stop_turn).
            if is_last:
                bellman = "STOP"
            llm = llm_stop_bit(dialogue, tn, llm_policy)
            rows.append({
                "session_id": dialogue["session_id"],
                "turn": tn,
                "is_last": bool(is_last),
                "bellman": bellman,
                "llm_stop": llm,
                "agree": bellman == llm,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"n_turns": 0, "turn_agreement": 0.0, "session_agreement": 0.0}
    turn_agreement = float(df["agree"].mean())
    # Session-level: whole trajectory of decisions must match.
    per_session = df.groupby("session_id")["agree"].all()
    session_agreement = float(per_session.mean())
    return {
        "n_turns": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "turn_agreement": turn_agreement,
        "session_agreement": session_agreement,
        "disagree_stop_from_llm": float(
            ((df["bellman"] == "WAIT") & (df["llm_stop"] == "STOP")).mean()
        ),
        "disagree_wait_from_llm": float(
            ((df["bellman"] == "STOP") & (df["llm_stop"] == "WAIT")).mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--baseline_dir", required=True,
                        help="Baseline model dir with user_model.pt + stop_policy.pt.")
    parser.add_argument("--llm_stop_dir", required=True,
                        help="Ablation model dir with the SFT'd LLM stop policy.")
    parser.add_argument("--output_dir", default="outputs/rule_agreement")
    parser.add_argument("--n_sessions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print(f"[1/3] Loading baseline pCTR + StopPolicy from {args.baseline_dir}")
    pctr_model = UnifiedUserModel.load(args.baseline_dir, device)
    stop_policy = StopPolicy.load(args.baseline_dir, device)

    print(f"[2/3] Loading LLM-stop from {args.llm_stop_dir}")
    from src.ablation.train_llm_stop import LLMStopPolicy
    llm_policy = LLMStopPolicy(args.llm_stop_dir, device=str(device))

    print(f"[3/3] Loading dialogues ...")
    dialogues = load_jsonl(args.dialogues)

    df = test_rule_agreement(dialogues, pctr_model, stop_policy, llm_policy,
                             n_sessions=args.n_sessions)
    df.to_csv(output_dir / "rule_agreement.csv", index=False)

    summary = summarize(df)
    save_json(summary, output_dir / "rule_agreement_summary.json")

    print("\n=== Rule-agreement summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
