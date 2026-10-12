"""Experiment 7.4: Multi-turn qualitative case study visualization.

Picks a handful of representative test sessions where the learned stop policy
chose to wait then exercise, and produces three artifacts per session:

  fig1_decision.png   per-turn W_t / CV_t curves with the chosen stop turn
                      marked.  Shows the Bellman crossing visually.
  fig3_decoupling.png winner-vs-others quality-weighted virtual scores at the
                      stop turn, with the winner's bid / actual payment /
                      information rent annotated.  Mirrors §4.0's b-vs-P
                      decoupling figure on real data.
  meta.json           the underlying numbers (pctrs / scores / bids / payment
                      / threshold breakdown / chosen turn) so the figures can
                      be reproduced or rendered differently later.

Text-comparison (fig 2 in the paper) is intentionally left out here: it needs
the trained renderer LLM and is run separately via the closed-loop eval; once
those responses exist this script just consumes them.

Run:
  python -m src.eval.case_study_visualize \
    --dialogues data/dialogues_all.jsonl \
    --model_dir outputs/final_qwen3emb/round_2 \
    --output_dir outputs/case_studies \
    --n_picks 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.auction.auction import run_virtual_value_auction
from src.eval.closed_loop_eval import (
    compute_cv_minus_winner,
    compute_pctrs_for_state,
    mechanism_meta_at_turn,
)
from src.models.text_mlp import PCTRModel
from src.policy.stop_policy import StopPolicy, stop_features_v2
from src.utils.io import ensure_dir, load_jsonl, resolve_device


def per_turn_decision_trace(
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    theta_max: float,
    gamma: float,
) -> List[Dict[str, Any]]:
    """For each turn return W_t (q_auction) and CV_t (q_wait + payoff at stop).

    We use the StopPolicy's two Q-values as the empirical proxy for W_t and
    CV_t — that's exactly what the trained policy compares to decide.
    """
    turns = dialogue.get("turns", [])
    n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
    trace = []
    for idx, turn in enumerate(turns):
        tn = int(turn.get("turn", idx + 1))
        result, meta = mechanism_meta_at_turn(
            dialogue, tn, pctr_model, psi_mode="identity", theta_max=theta_max,
        )
        if result is None:
            continue
        features = stop_features_v2(tn, meta)
        q = stop_policy.q_values(features)
        q_wait = float(q[0])
        q_auction = float(q[1])
        winner_pctr = float(result.pctr)
        trace.append(
            {
                "turn": tn,
                "q_wait_cv": q_wait,
                "q_auction_w": q_auction,
                "winner_pctr": winner_pctr,
                "winner_score": float(result.score),
                "winner_index": int(result.winner_index),
                "all_scores": list(result.all_scores),
                "all_pctrs": list(result.all_pctrs),
            }
        )
    return trace


def first_stop_turn(trace: List[Dict[str, Any]]) -> int | None:
    """Return the first turn where the policy would choose AUCTION."""
    for entry in trace:
        if entry["q_auction_w"] >= entry["q_wait_cv"]:
            return entry["turn"]
    return None


import re

# Pattern to extract @ad_title@[ad_id] markers from rendered dialogues.
# Data-generation step embeds these in the assistant text whenever an ad
# was placed; we use them to filter for cases where the mechanism's winner
# matches the ad the dialogue actually recommends.
AD_MENTION_RE = re.compile(r"@[^@]+@\[(A_\d+)\]")


def mentioned_ad_ids(dialogue: Dict[str, Any]) -> set:
    """Return the set of A_xxxx ad ids that appear inside @...@[A_id] markers
    in any assistant turn of this dialogue."""
    ids = set()
    for t in dialogue.get("turns", []):
        for ad_id in AD_MENTION_RE.findall(t.get("assistant", "") or ""):
            ids.add(ad_id)
    return ids


def pick_representative_sessions(
    dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    theta_max: float,
    gamma: float,
    n_picks: int,
    min_pctr_lift: float = 0.05,
    require_winner_matches_dialogue: bool = True,
) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]], int]]:
    """Score sessions by (winner pctr at stop) - (winner pctr at turn 1) and
    return the top n_picks where the policy stops at turn 2 or later.

    `require_winner_matches_dialogue=True` (default) additionally filters out
    sessions whose mechanism-rerun winner doesn't match the ad explicitly
    mentioned in the dialogue's assistant text. This is the cleanest way to
    avoid showing cases where the pCTR model misranks a topic-irrelevant ad
    above the topic-relevant one — the dialogue is the ground truth for
    "which ad belongs here".
    """
    scored = []
    skipped_mismatch = 0
    skipped_other = 0
    for d in dialogues:
        turns = d.get("turns", [])
        ads = d.get("candidate_ads", [])
        if len(turns) < 3 or len(ads) < 2:
            skipped_other += 1
            continue
        trace = per_turn_decision_trace(d, pctr_model, stop_policy, theta_max, gamma)
        if not trace or len(trace) < 2:
            skipped_other += 1
            continue
        stop_t = first_stop_turn(trace)
        if stop_t is None or stop_t == trace[0]["turn"]:
            skipped_other += 1
            continue  # we want sessions where the policy actually waits first
        stop_entry = next((e for e in trace if e["turn"] == stop_t), trace[-1])
        first_entry = trace[0]
        # Compare the eventual winner's pctr at stop vs that ad's pctr at turn 1
        winner_idx = stop_entry["winner_index"]
        if winner_idx >= len(first_entry["all_pctrs"]):
            skipped_other += 1
            continue
        lift = stop_entry["all_pctrs"][winner_idx] - first_entry["all_pctrs"][winner_idx]
        if lift < min_pctr_lift:
            skipped_other += 1
            continue
        # NEW: require the mechanism winner to match the ad mentioned in the dialogue.
        if require_winner_matches_dialogue:
            winner_ad_id = ads[winner_idx].get("ad_id", "")
            mentioned = mentioned_ad_ids(d)
            if not mentioned or winner_ad_id not in mentioned:
                skipped_mismatch += 1
                continue
        scored.append((lift, d, trace, stop_t))
    scored.sort(key=lambda x: x[0], reverse=True)
    if require_winner_matches_dialogue:
        print(f"  [filter] skipped {skipped_mismatch} sessions where mechanism winner != "
              f"dialogue-mentioned ad; {skipped_other} sessions failed other criteria")
    return [(d, trace, stop_t) for _, d, trace, stop_t in scored[:n_picks]]


def plot_decision_chain(trace: List[Dict[str, Any]], stop_t: int, save_path: Path) -> None:
    turns = [e["turn"] for e in trace]
    w = [e["q_auction_w"] for e in trace]
    cv = [e["q_wait_cv"] for e in trace]
    pctrs = [e["winner_pctr"] for e in trace]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(turns, w, "o-", color="#d62728", label="$W_t$ immediate (Q_AUCTION)")
    ax1.plot(turns, cv, "s--", color="#1f77b4", label="$CV_t$ waiting (Q_WAIT)")
    ax1.axvline(stop_t, color="black", linestyle=":", alpha=0.7, label=f"stop at $t^*={stop_t}$")
    ax1.set_xlabel("dialogue turn $t$")
    ax1.set_ylabel("Bellman value")
    ax1.set_xticks(turns)
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(turns, pctrs, "^-", color="#2ca02c", alpha=0.7, label="winner $q_{i^*}(h_t)$")
    ax2.set_ylabel("winner pCTR", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_b_p_rent_decoupling(
    dialogue: Dict[str, Any],
    stop_t: int,
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    theta_min: float,
    theta_max: float,
    gamma: float,
    save_path: Path,
) -> Dict[str, Any] | None:
    """Identity-mode allocation (b·q) consistent with run_final_experiment training.

    Selects the winner by argmax (b·q), no reserve filter — matching
    psi_mode="identity" used everywhere in the training pipeline.
    Payment is the standard quality-adjusted second-price formula:
      P = max(0, second_score, gamma·CV^{-i*}) / q_winner, capped at b_winner.
    """
    from src.auction.auction import run_quality_weighted_allocation

    ads = dialogue.get("candidate_ads", [])
    if not ads:
        return None
    pctrs = compute_pctrs_for_state(pctr_model, dialogue, stop_t)
    n_turns = max(1, int(dialogue.get("n_turns", len(dialogue.get("turns", [])) or 1)))

    # First pass: identity-mode auction to find the winner.
    pre = run_quality_weighted_allocation(ads, pctrs, psi_mode="identity")
    if pre is None:
        return None
    winner = int(pre.winner_index)

    # Inter-temporal externality: continuation value with winner removed.
    cv_minus = compute_cv_minus_winner(
        dialogue, stop_t, winner, pctr_model, stop_policy, n_turns,
    )

    # Composite threshold: max(0, second-best b·q, gamma·CV^{-i*}).
    bids = [float(a.get("bid", 0.0)) for a in ads]
    scores = [bids[i] * pctrs[i] for i in range(len(ads))]
    other_scores = [s for i, s in enumerate(scores) if i != winner]
    lateral = max(other_scores) if other_scores else 0.0
    inter = float(gamma) * float(cv_minus)
    theta_threshold = max(0.0, lateral, inter)

    # Critical-value payment: P = threshold / q_winner, capped at bid.
    q_w = float(pctrs[winner])
    raw_p = theta_threshold / max(q_w, 1e-9)
    payment = float(max(theta_min, min(raw_p, bids[winner])))

    rent = (bids[winner] - payment) * q_w
    titles = [str(a.get("title", ""))[:24] or f"ad{i}" for i, a in enumerate(ads)]

    # Figure: bar chart of b·q scores, winner highlighted.
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#d62728" if i == winner else "#aaaaaa" for i in range(len(ads))]
    ax.bar(range(len(ads)), scores, color=colors)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(ads)))
    ax.set_xticklabels(titles, rotation=20, ha="right")
    ax.set_ylabel(r"$b \cdot q$ (quality-weighted score, identity mode)")
    ax.set_title(
        f"winner: {titles[winner]}   bid b={bids[winner]:.3f}   "
        f"payment P={payment:.3f}   rent (b-P)q={rent:.4f}"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "winner_index": int(winner),
        "winner_title": titles[winner],
        "bid": bids[winner],
        "payment": payment,
        "rent": rent,
        "theta_threshold": float(theta_threshold),
        "cv_minus_winner": float(cv_minus),
        "all_scores": scores,
        "all_pctrs": list(pctrs),
        "all_titles": titles,
        "all_bids": bids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 7.4 case-study visualizer")
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--model_dir", required=True, help="Directory with trained pctr_model + stop_policy")
    parser.add_argument("--output_dir", default="outputs/case_studies")
    parser.add_argument("--n_picks", type=int, default=5)
    parser.add_argument("--theta_min", type=float, default=0.1)
    parser.add_argument("--theta_max", type=float, default=2.18)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=2000,
        help="upper bound on sessions scanned for representativeness ranking",
    )
    parser.add_argument(
        "--no_dialogue_match",
        action="store_true",
        help="Disable the 'mechanism winner must match dialogue-mentioned ad' filter. "
             "Default behavior keeps the filter on so demo cases stay self-consistent.",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)

    print(f"[1/3] Loading models from {args.model_dir} ...")
    pctr_model = PCTRModel.load(args.model_dir, device)
    stop_policy = StopPolicy.load(args.model_dir, device)

    print(f"[2/3] Loading dialogues and scanning at most {args.scan_limit} sessions ...")
    dialogues = load_jsonl(args.dialogues)[: args.scan_limit]
    picks = pick_representative_sessions(
        dialogues,
        pctr_model,
        stop_policy,
        theta_max=args.theta_max,
        gamma=args.gamma,
        n_picks=args.n_picks,
        require_winner_matches_dialogue=not args.no_dialogue_match,
    )
    print(f"  picked {len(picks)} representative sessions")

    print(f"[3/3] Rendering figures into {output_dir} ...")
    summary = []
    for d, trace, stop_t in picks:
        sid = d["session_id"]
        sub = ensure_dir(output_dir / sid)
        plot_decision_chain(trace, stop_t, sub / "fig1_decision.png")
        deco = plot_b_p_rent_decoupling(
            d,
            stop_t,
            pctr_model,
            stop_policy,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            gamma=args.gamma,
            save_path=sub / "fig3_decoupling.png",
        )
        meta = {
            "session_id": sid,
            "stop_turn": int(stop_t),
            "n_turns": len(d.get("turns", [])),
            "trace": trace,
            "decoupling": deco,
            "user_query": (d.get("turns") or [{}])[0].get("user", ""),
            "dialogue_turns": [
                {
                    "turn": int(t.get("turn", i + 1)),
                    "user": t.get("user", ""),
                    "assistant": t.get("assistant", ""),
                }
                for i, t in enumerate(d.get("turns", []))
            ],
            "user_profile": d.get("user_profile", ""),
            "candidate_ads": [
                {
                    "ad_id": a.get("ad_id", ""),
                    "title": a.get("title", ""),
                    "bid": float(a.get("bid", 0.0)),
                    "category": a.get("category", ""),
                }
                for a in d.get("candidate_ads", [])
            ],
        }
        with open(sub / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        summary.append({"session_id": sid, "stop_turn": int(stop_t)})
        print(f"  {sid}: stop_turn={stop_t}  ->  {sub}")

    with open(output_dir / "case_study_index.json", "w", encoding="utf-8") as fh:
        json.dump({"theta_max": args.theta_max, "gamma": args.gamma, "picks": summary},
                  fh, ensure_ascii=False, indent=2)

    print(f"\nDone. Summary: {output_dir / 'case_study_index.json'}")


if __name__ == "__main__":
    main()
