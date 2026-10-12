"""
Experiment 7.3: Incentive Compatibility (IC) Verification under Single Bid.

For each test session, fix one advertiser's true valuation θ_i, then sweep
reported bids b_i ∈ {0.5θ, 0.75θ, θ, 1.25θ, 1.5θ, 2.0θ} and measure:
  - Win rate (does misreporting help win more?)
  - Expected utility = (θ_i - P_i) * click_prob  if won, else 0
  - Payment P_i under the virtual-value mechanism

Expected result: truthful bid b_i = θ_i maximizes expected utility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.auction.auction import run_virtual_value_auction
from src.eval.closed_loop_eval import (
    compute_pctrs_for_state,
    _run_policy_allocation,
    compute_global_critical_cpc,
)
from src.models.text_mlp import PCTRModel
from src.policy.stop_policy import StopPolicy, stop_features_v2


BID_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def compute_cv_minus_winner(
    dialogue: Dict[str, Any],
    turn_number: int,
    winner_index: int,
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    n_turns: int,
    theta_min: float,
    theta_max: float,
) -> float:
    """Estimate CV^{-i*}: continuation value after removing winner from candidate set."""
    ads = dialogue.get("candidate_ads", [])
    if len(ads) <= 1:
        return 0.0
    ads_reduced = [a for idx, a in enumerate(ads) if idx != winner_index]
    pctrs_reduced = compute_pctrs_for_state(pctr_model, dialogue, turn_number)
    pctrs_reduced = [p for idx, p in enumerate(pctrs_reduced) if idx != winner_index]
    if not ads_reduced:
        return 0.0
    bids_reduced = [float(a.get("bid", 0.0)) for a in ads_reduced]
    scores_reduced = [b * p for b, p in zip(bids_reduced, pctrs_reduced)]
    meta_reduced = {
        "pctrs": pctrs_reduced,
        "scores": scores_reduced,
        "bids": bids_reduced,
        "winner_index": int(np.argmax(scores_reduced)),
        "winner_pctr": pctrs_reduced[int(np.argmax(scores_reduced))],
        "winner_score": max(scores_reduced),
        "second_score": sorted(scores_reduced, reverse=True)[1] if len(scores_reduced) > 1 else 0.0,
        "payment": 0.0,
    }
    features_reduced = stop_features_v2(turn_number, meta_reduced)
    q_vals = stop_policy.q_values(features_reduced)
    return max(0.0, float(q_vals[0]))


def run_ic_verification(
    test_dialogues: List[Dict[str, Any]],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    theta_min: float = 0.1,
    theta_max: float = 4.1,
    gamma: float = 0.95,
    max_sessions: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep bid multipliers for each advertiser in each session, measure utility."""
    rng = np.random.default_rng(seed)
    results = []

    sessions_to_eval = test_dialogues[:max_sessions]

    for dialogue in tqdm(sessions_to_eval, desc="IC verification"):
        ads = dialogue.get("candidate_ads", [])
        if len(ads) < 2:
            continue
        turns = dialogue.get("turns", [])
        n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
        last_turn = int(turns[-1].get("turn", len(turns))) if turns else 1

        # pCTR is bid-clean: same (turn, ads) -> same pctr regardless of the bid
        # perturbations below. Precompute once per session so the inner
        # allocation/critical-cpc reruns hit the cache instead of re-running the
        # pctr forward dozens of times (the IC bottleneck).
        pctr_cache = {}
        for idx_t, turn in enumerate(turns):
            tn = int(turn.get("turn", idx_t + 1))
            pctr_cache[tn] = compute_pctrs_for_state(pctr_model, dialogue, tn)
        if last_turn in pctr_cache:
            pctrs = pctr_cache[last_turn]
        else:
            pctrs = compute_pctrs_for_state(pctr_model, dialogue, last_turn)

        for target_idx, target_ad in enumerate(ads):
            true_bid = float(target_ad.get("bid", 0.0))
            if true_bid < 1e-6:
                continue

            for mult in BID_MULTIPLIERS:
                perturbed_bid = true_bid * mult
                perturbed_ads = []
                for idx, ad in enumerate(ads):
                    if idx == target_idx:
                        perturbed_ads.append({**ad, "bid": perturbed_bid})
                    else:
                        perturbed_ads.append(ad)

                # Old IC path used local virtual-value pricing:
                # cv_minus = compute_cv_minus_winner(
                #     dialogue, last_turn, target_idx, pctr_model,
                #     stop_policy, n_turns, theta_min, theta_max,
                # )
                # auction_result = run_virtual_value_auction(
                #     perturbed_ads, pctrs,
                #     theta_min=theta_min, theta_max=theta_max,
                #     gamma=gamma, cv_minus_winner=cv_minus,
                # )
                chosen_turn, auction_result, meta, _, _ = _run_policy_allocation(
                    "learned_stop",
                    dialogue,
                    pctr_model,
                    stop_policy,
                    candidate_ads=perturbed_ads,
                    force_auction_last_turn=True,
                    psi_mode="identity",
                    theta_max=theta_max,
                    pctr_cache=pctr_cache,
                )

                if auction_result is None or chosen_turn is None:
                    won = False
                    payment = 0.0
                    click_prob = 0.0
                else:
                    won = (auction_result.winner_index == target_idx)
                    payment = compute_global_critical_cpc(
                        "learned_stop",
                        dialogue,
                        pctr_model,
                        stop_policy,
                        winner_index=target_idx,
                        winner_bid=perturbed_bid,
                        fixed_turn=None,
                        force_auction_last_turn=True,
                        psi_mode="identity",
                        theta_max=theta_max,
                        pctr_cache=pctr_cache,
                    ) if won else 0.0
                    click_prob = float(pctrs[target_idx])

                utility = (true_bid - payment) * click_prob if won else 0.0

                results.append({
                    "session_id": dialogue["session_id"],
                    "ad_id": target_ad.get("ad_id", ""),
                    "true_bid": true_bid,
                    "bid_multiplier": mult,
                    "reported_bid": perturbed_bid,
                    "won": int(won),
                    "payment": payment,
                    "click_prob": click_prob,
                    "utility": utility,
                })

    return pd.DataFrame(results)


def summarize_ic_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IC results by bid_multiplier."""
    summary = df.groupby("bid_multiplier").agg(
        win_rate=("won", "mean"),
        avg_payment=("payment", "mean"),
        avg_utility=("utility", "mean"),
        median_utility=("utility", "median"),
        n=("utility", "count"),
    ).reset_index()
    return summary
