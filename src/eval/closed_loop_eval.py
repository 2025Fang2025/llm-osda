from __future__ import annotations

from typing import Any, Dict, List, Tuple, Protocol

import numpy as np
import pandas as pd

from src.auction.auction import run_virtual_value_auction, VirtualValueAuctionResult, run_quality_weighted_allocation
from src.data.data_utils import pctr_input_text, numeric_features
from src.models.text_mlp import PCTRModel
from src.policy.stop_policy import StopPolicy, stop_features_from_auction


# Default prior bounds (Uniform). Passed through from config / CLI.
DEFAULT_THETA_MIN = 0.1
DEFAULT_THETA_MAX = 4.1


class RendererLike(Protocol):
    def render(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any]) -> Tuple[str, str]: ...


class JudgeLike(Protocol):
    def judge(self, dialogue: Dict[str, Any], turn_number: int, ad: Dict[str, Any], response: str) -> Dict[str, Any]: ...


def compute_pctrs_for_state(pctr_model: PCTRModel, dialogue: Dict[str, Any], turn_number: int) -> List[float]:
    ads = dialogue.get("candidate_ads", [])
    texts = [pctr_input_text(dialogue, turn_number, ad) for ad in ads]
    nums = [numeric_features(dialogue, turn_number, ad) for ad in ads]
    pctrs = pctr_model.predict_proba(texts, nums)
    return np.clip(pctrs, 1e-4, 1.0).astype(float).tolist()


def compute_cv_minus_winner(
    dialogue: Dict[str, Any],
    turn_number: int,
    winner_index: int,
    pctr_model: PCTRModel,
    stop_policy: StopPolicy | None,
    n_turns: int,
) -> float:
    """Estimate CV^{-i*}: continuation value with winner removed from candidate set."""
    if stop_policy is None:
        return 0.0
    ads = dialogue.get("candidate_ads", [])
    if len(ads) <= 1:
        return 0.0
    pctrs_full = compute_pctrs_for_state(pctr_model, dialogue, turn_number)
    ads_reduced = [a for idx, a in enumerate(ads) if idx != winner_index]
    pctrs_reduced = [p for idx, p in enumerate(pctrs_full) if idx != winner_index]
    if not ads_reduced:
        return 0.0
    bids_reduced = [float(a.get("bid", 0.0)) for a in ads_reduced]
    scores_reduced = [b * p for b, p in zip(bids_reduced, pctrs_reduced)]
    best_idx = int(np.argmax(scores_reduced))
    sorted_scores = sorted(scores_reduced, reverse=True)
    meta_reduced = {
        "pctrs": pctrs_reduced,
        "scores": scores_reduced,
        "bids": bids_reduced,
        "winner_index": best_idx,
        "winner_pctr": pctrs_reduced[best_idx],
        "winner_score": sorted_scores[0],
        "second_score": sorted_scores[1] if len(sorted_scores) > 1 else 0.0,
        "payment": 0.0,
    }
    features_reduced = stop_features_from_auction(turn_number, n_turns, meta_reduced)
    q_vals = stop_policy.q_values(features_reduced)
    return max(0.0, float(q_vals[0]))


def vv_auction_meta(
    dialogue: Dict[str, Any],
    turn_number: int,
    pctr_model: PCTRModel,
    stop_policy: StopPolicy | None,
    gamma: float,
    theta_min: float,
    theta_max: float,
) -> Tuple[VirtualValueAuctionResult | None, Dict[str, Any]]:
    """Run virtual-value auction with CV^{-i*} for a given dialogue turn."""
    ads = dialogue.get("candidate_ads", [])
    pctrs = compute_pctrs_for_state(pctr_model, dialogue, turn_number)
    n_turns = max(1, int(dialogue.get("n_turns", len(dialogue.get("turns", [])) or 1)))

    # First pass without CV to find winner index
    preliminary = run_virtual_value_auction(
        ads, pctrs, theta_min=theta_min, theta_max=theta_max, gamma=gamma, cv_minus_winner=0.0,
    )
    if preliminary is None:
        return None, _empty_meta(ads, pctrs)

    # Compute CV^{-i*} using stop policy
    cv_minus = compute_cv_minus_winner(
        dialogue, turn_number, preliminary.winner_index,
        pctr_model, stop_policy, n_turns,
    )

    # Final auction with CV^{-i*}
    result = run_virtual_value_auction(
        ads, pctrs, theta_min=theta_min, theta_max=theta_max, gamma=gamma, cv_minus_winner=cv_minus,
    )
    if result is None:
        return None, _empty_meta(ads, pctrs)

    meta = {
        "pctrs": result.all_pctrs,
        "scores": result.all_virtual_scores,
        "bids": [float(a.get("bid", 0.0)) for a in ads],
        "winner_index": result.winner_index,
        "winner_ad_id": result.winner_ad.get("ad_id", ""),
        "winner_title": result.winner_ad.get("title", ""),
        "winner_bid": float(result.winner_ad.get("bid", 0.0)),
        "winner_pctr": result.pctr,
        "winner_score": result.virtual_score,
        "second_score": result.theta_threshold,
        "payment": result.payment_per_click,
        "cv_minus_winner": result.cv_minus_winner,
        "theta_threshold": result.theta_threshold,
    }
    return result, meta


def _empty_meta(ads, pctrs):
    return {
        "pctrs": pctrs,
        "scores": [0.0] * len(ads),
        "bids": [float(a.get("bid", 0.0)) for a in ads],
        "winner_index": 0,
        "winner_ad_id": "",
        "winner_title": "",
        "winner_bid": 0.0,
        "winner_pctr": 0.0,
        "winner_score": 0.0,
        "second_score": 0.0,
        "payment": 0.0,
        "cv_minus_winner": 0.0,
        "theta_threshold": 0.0,
    }


# Keep GSP version for stop-policy feature extraction (training compatibility)
def auction_meta_from_result_gsp(dialogue, turn_number, pctr_model, reserve_price):
    from src.auction.auction import run_quality_adjusted_second_price
    ads = dialogue.get("candidate_ads", [])
    pctrs = compute_pctrs_for_state(pctr_model, dialogue, turn_number)
    result = run_quality_adjusted_second_price(ads, pctrs, reserve_price=reserve_price)
    meta = {
        "pctrs": result.all_pctrs,
        "scores": result.all_scores,
        "bids": [float(a.get("bid", 0.0)) for a in ads],
        "winner_index": result.winner_index,
        "winner_ad_id": result.winner_ad.get("ad_id", ""),
        "winner_title": result.winner_ad.get("title", ""),
        "winner_bid": float(result.winner_ad.get("bid", 0.0)),
        "winner_pctr": result.pctr,
        "winner_score": result.score,
        "second_score": result.second_score,
        "payment": result.payment_per_click,
    }
    return result, meta


# Backward-compatible alias used by training code
auction_meta_from_result = auction_meta_from_result_gsp


# ---------------------------------------------------------------------------
# Final LLM-OSDA mechanism: bid-aware stopping + allocation + global critical CPC
# ---------------------------------------------------------------------------

def mechanism_meta_at_turn(
    dialogue: Dict[str, Any],
    turn_number: int,
    pctr_model: PCTRModel,
    candidate_ads: List[Dict[str, Any]] | None = None,
    psi_mode: str = "virtual",
    theta_max: float = DEFAULT_THETA_MAX,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> Tuple[Any | None, Dict[str, Any]]:
    """Allocation at a fixed turn using s_i,t = psi(b_i) q_i(h_t)."""
    # Mechanism fix: this replaces the old mixed GSP/VV feature path for final eval.
    ads = candidate_ads if candidate_ads is not None else dialogue.get("candidate_ads", [])
    # q_i(h_t) is bid-clean (pctr_input_text/numeric_features carry no bid), so a
    # session-level cache keyed by turn stays valid across bid perturbations.
    if pctr_cache is not None and turn_number in pctr_cache:
        pctrs = pctr_cache[turn_number]
    elif candidate_ads is None:
        pctrs = compute_pctrs_for_state(pctr_model, dialogue, turn_number)
    else:
        d2 = {**dialogue, "candidate_ads": candidate_ads}
        pctrs = compute_pctrs_for_state(pctr_model, d2, turn_number)
    result = run_quality_weighted_allocation(
        ads, pctrs, psi_mode=psi_mode, theta_max=theta_max,
    )
    if result is None:
        return None, _empty_meta(ads, pctrs)
    meta = {
        "pctrs": result.all_pctrs,
        "scores": result.all_scores,
        "bids": [float(a.get("bid", 0.0)) for a in ads],
        "winner_index": result.winner_index,
        "winner_ad_id": result.winner_ad.get("ad_id", ""),
        "winner_title": result.winner_ad.get("title", ""),
        "winner_bid": float(result.winner_ad.get("bid", 0.0)),
        "winner_pctr": result.pctr,
        "winner_score": result.score,
        "second_score": result.second_score,
        "payment": 0.0,
        "cv_minus_winner": 0.0,
        "theta_threshold": 0.0,
    }
    return result, meta


def _ads_with_replaced_bid(dialogue: Dict[str, Any], target_idx: int, bid: float) -> List[Dict[str, Any]]:
    ads = []
    for idx, ad in enumerate(dialogue.get("candidate_ads", [])):
        if idx == target_idx:
            ads.append({**ad, "bid": float(bid)})
        else:
            ads.append(dict(ad))
    return ads


def _choose_learned_stop_turn(
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    candidate_ads: List[Dict[str, Any]] | None = None,
    force_auction_last_turn: bool = True,
    psi_mode: str = "virtual",
    theta_max: float = DEFAULT_THETA_MAX,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> Tuple[int | None, float, float]:
    turns = dialogue.get("turns", [])
    n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
    for idx, turn in enumerate(turns):
        tn = int(turn.get("turn", idx + 1))
        result, meta = mechanism_meta_at_turn(
            dialogue, tn, pctr_model, candidate_ads=candidate_ads,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        if result is None:
            continue
        features = stop_features_from_auction(tn, n_turns, meta)
        is_last = idx == len(turns) - 1
        if force_auction_last_turn and is_last:
            return tn, np.nan, np.nan
        q = stop_policy.q_values(features)
        if float(q[1]) >= float(q[0]):
            return tn, float(q[0]), float(q[1])
    return None, np.nan, np.nan


def _choose_myopic_threshold_turn(
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    threshold: float,
    candidate_ads: List[Dict[str, Any]] | None = None,
    force_auction_last_turn: bool = True,
    psi_mode: str = "virtual",
    theta_max: float = DEFAULT_THETA_MAX,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> int | None:
    """Myopic adaptive baseline: stop at the first turn whose winner
    expected click value b_i * q_i exceeds a fixed threshold (greedy, no
    Bellman look-ahead). Reuses the same G_eta / allocation as other policies.
    """
    turns = dialogue.get("turns", [])
    for idx, turn in enumerate(turns):
        tn = int(turn.get("turn", idx + 1))
        result, meta = mechanism_meta_at_turn(
            dialogue, tn, pctr_model, candidate_ads=candidate_ads,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        if result is None:
            continue
        is_last = idx == len(turns) - 1
        if force_auction_last_turn and is_last:
            return tn
        score = float(meta["winner_bid"]) * float(meta["winner_pctr"])
        if score >= threshold:
            return tn
    return None


def _run_policy_allocation(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy | None,
    candidate_ads: List[Dict[str, Any]] | None = None,
    fixed_turn: int | None = None,
    force_auction_last_turn: bool = True,
    psi_mode: str = "virtual",
    theta_max: float = DEFAULT_THETA_MAX,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> Tuple[int | None, Any | None, Dict[str, Any], float, float]:
    turns = dialogue.get("turns", [])
    if not turns:
        turns = [{"turn": 1}]

    q_wait, q_auction = np.nan, np.nan
    if policy_name in ("first_turn", "first_turn_trained_llm"):
        chosen_turn = int(turns[0].get("turn", 1))
    elif policy_name == "last_turn":
        chosen_turn = int(turns[-1].get("turn", len(turns)))
    elif policy_name in ("random_turn",):
        if fixed_turn is None:
            raise ValueError("random_turn requires an explicit fixed_turn")
        valid_turns = {
            int(turn.get("turn", index + 1))
            for index, turn in enumerate(turns)
        }
        chosen_turn = int(fixed_turn)
        if chosen_turn not in valid_turns:
            raise ValueError(
                f"random_turn fixed_turn={chosen_turn} is not in {sorted(valid_turns)}"
            )
    elif policy_name in ("learned_stop", "learned_stop_untrained_llm"):
        if stop_policy is None:
            return None, None, _empty_meta(candidate_ads or dialogue.get("candidate_ads", []), []), q_wait, q_auction
        chosen_turn, q_wait, q_auction = _choose_learned_stop_turn(
            dialogue, pctr_model, stop_policy, candidate_ads=candidate_ads,
            force_auction_last_turn=force_auction_last_turn,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        if chosen_turn is None:
            return None, None, _empty_meta(candidate_ads or dialogue.get("candidate_ads", []), []), q_wait, q_auction
    elif policy_name == "myopic_threshold":
        import os
        thr = float(os.environ.get("MYOPIC_THRESHOLD", "0.4"))
        chosen_turn = _choose_myopic_threshold_turn(
            dialogue, pctr_model, threshold=thr, candidate_ads=candidate_ads,
            force_auction_last_turn=force_auction_last_turn,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        if chosen_turn is None:
            return None, None, _empty_meta(candidate_ads or dialogue.get("candidate_ads", []), []), q_wait, q_auction
    else:
        chosen_turn = int(fixed_turn if fixed_turn is not None else turns[-1].get("turn", len(turns)))

    result, meta = mechanism_meta_at_turn(
        dialogue, chosen_turn, pctr_model, candidate_ads=candidate_ads,
        psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
    )
    return chosen_turn, result, meta, q_wait, q_auction


def _wins_under_bid(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy | None,
    target_idx: int,
    bid: float,
    fixed_turn: int | None,
    force_auction_last_turn: bool,
    psi_mode: str,
    theta_max: float,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> bool:
    ads = _ads_with_replaced_bid(dialogue, target_idx, bid)
    _, result, _, _, _ = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        candidate_ads=ads, fixed_turn=fixed_turn,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
    )
    return bool(result is not None and int(result.winner_index) == int(target_idx))


def compute_global_critical_cpc(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy | None,
    winner_index: int,
    winner_bid: float,
    fixed_turn: int | None = None,
    force_auction_last_turn: bool = True,
    psi_mode: str = "virtual",
    theta_max: float = DEFAULT_THETA_MAX,
    n_iter: int = 24,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> float:
    """Global critical CPC kappa_i(b_-i) via complete mechanism reruns.

    This implements kappa_i = inf{z: x_i(z,b_-i)>0} with a deterministic
    grid/binary-search style audit over the full stopping+allocation rule.
    """
    if winner_bid <= 0:
        return 0.0
    if _wins_under_bid(policy_name, dialogue, pctr_model, stop_policy, winner_index, 0.0,
                       fixed_turn, force_auction_last_turn, psi_mode, theta_max, pctr_cache):
        return 0.0
    if not _wins_under_bid(policy_name, dialogue, pctr_model, stop_policy, winner_index, winner_bid,
                           fixed_turn, force_auction_last_turn, psi_mode, theta_max, pctr_cache):
        return float(winner_bid)

    lo, hi = 0.0, float(winner_bid)
    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        if _wins_under_bid(policy_name, dialogue, pctr_model, stop_policy, winner_index, mid,
                           fixed_turn, force_auction_last_turn, psi_mode, theta_max, pctr_cache):
            hi = mid
        else:
            lo = mid
    return float(hi)


def _fix_ad_format(response: str, ad: Dict[str, Any]) -> str:
    """Ensure response contains the correct @title@[ad_id] format."""
    import re
    if re.search(r'@[^@]+@\[[^\]]+\]', response):
        return response
    title = ad.get("title", "")
    ad_id = ad.get("ad_id", "")
    bad_patterns = [
        f"@{ad_id}@", f"@ad_title@{ad_id}", f"@ad_title@",
        f"[{ad_id}]", f"({ad_id})",
    ]
    for pat in bad_patterns:
        if pat in response:
            response = response.replace(pat, f"@{title}@[{ad_id}]", 1)
            return response
    response = response.rstrip(". ") + f" @{title}@[{ad_id}]."
    return response


def settle_episode(
    policy_name: str,
    dialogue: Dict[str, Any],
    turn_number: int,
    auction_result,
    auction_meta: Dict[str, Any],
    renderer: RendererLike,
    judge: JudgeLike,
    lambda_gamma: float,
    gamma: float = 1.0,
    pre_rendered_response: str | None = None,
) -> Dict[str, Any]:
    ad = auction_result.winner_ad
    if pre_rendered_response is not None:
        response = pre_rendered_response
        response_source = getattr(renderer, "renderer_tag", "pre_rendered")
    else:
        response, response_source = renderer.render(dialogue, turn_number, ad)
    response = _fix_ad_format(response, ad)
    judge_out = judge.judge(dialogue, turn_number, ad, response)
    click_value = float(judge_out["click_value"])
    gamma_score = float(judge_out["gamma_score"])
    payment = float(auction_meta["payment"])
    # Mechanism fix: document Sec. 5 defines discounted expected clicks,
    # x_i = gamma^(tau-1) q_i(h_tau). The old evaluation used
    # undiscounted clicks and systematically favored last_turn.
    # revenue = payment * click_value
    time_discount = float(gamma) ** max(0, int(turn_number) - 1)
    expected_click = time_discount * click_value
    revenue = payment * expected_click
    winner_bid = float(ad.get("bid", 0.0))
    # info_rent = (winner_bid - payment) * click_value
    info_rent = (winner_bid - payment) * expected_click
    # reward = revenue - float(lambda_gamma) * (1.0 - gamma)
    reward = revenue - float(lambda_gamma) * (1.0 - gamma_score)
    return {
        "policy": policy_name,
        "session_id": dialogue["session_id"],
        "triggered": 1,
        "trigger_turn": int(turn_number),
        "winner_ad_id": ad.get("ad_id", ""),
        "winner_title": ad.get("title", ""),
        "winner_bid": winner_bid,
        "winner_pctr": float(auction_meta["winner_pctr"]),
        "winner_score": float(auction_meta["winner_score"]),
        "payment": payment,
        "click_value": click_value,
        "time_discount": time_discount,
        "expected_click": expected_click,
        "clicked_binary": int(judge_out["clicked_binary"]),
        "gamma_score": gamma_score,
        "revenue": revenue,
        "info_rent": info_rent,
        # "gross_revenue": winner_bid * click_value,
        "gross_revenue": winner_bid * expected_click,
        "cv_minus_winner": float(auction_meta.get("cv_minus_winner", 0.0)),
        "theta_threshold": float(auction_meta.get("theta_threshold", 0.0)),
        "reward": reward,
        "judge_source": judge_out.get("source", ""),
        "judge_mode": judge_out.get("judge_mode", ""),
        "judge_exact": int(bool(judge_out.get("exact", False))),
        "response_source": response_source,
        "response": response,
        "judge_reason": judge_out.get("reason", ""),
        "raw_user_llm_output": judge_out.get("raw_user_llm_output", ""),
    }


def _no_trigger_row(policy_name, dialogue):
    return {
        "policy": policy_name,
        "session_id": dialogue["session_id"],
        "triggered": 0,
        "trigger_turn": np.nan,
        "winner_ad_id": "", "winner_title": "", "winner_bid": 0.0,
        "winner_pctr": 0.0, "winner_score": 0.0,
        "payment": 0.0, "click_value": 0.0, "clicked_binary": 0,
        "gamma_score": 1.0, "revenue": 0.0, "info_rent": 0.0,
        "gross_revenue": 0.0, "cv_minus_winner": 0.0, "theta_threshold": 0.0,
        "time_discount": 0.0, "expected_click": 0.0,
        "reward": 0.0, "judge_source": "no_ad", "judge_mode": "no_ad",
        "judge_exact": 0, "response_source": "no_ad", "response": "",
        "judge_reason": "", "raw_user_llm_output": "",
        "q_wait": np.nan, "q_auction": np.nan,
    }


# ---------------------------------------------------------------------------
# Evaluation policies 鈥?all use virtual-value pricing
# ---------------------------------------------------------------------------

def simulate_first_turn_auction(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    renderer: RendererLike,
    judge: JudgeLike,
    lambda_gamma: float,
    stop_policy: StopPolicy | None = None,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    **_kwargs,
) -> Dict[str, Any]:
    first_turn = int(dialogue.get("turns", [{}])[0].get("turn", 1))
    # Old path:
    # result, meta = vv_auction_meta(
    #     dialogue, first_turn, pctr_model, stop_policy, gamma, theta_min, theta_max,
    # )
    _, result, meta, q_wait, q_auction = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        fixed_turn=first_turn, force_auction_last_turn=True,
        psi_mode="identity", theta_max=theta_max,
    )
    if result is None:
        return _no_trigger_row(policy_name, dialogue)
    meta["payment"] = compute_global_critical_cpc(
        policy_name, dialogue, pctr_model, stop_policy, result.winner_index,
        float(result.winner_ad.get("bid", 0.0)), fixed_turn=first_turn,
        psi_mode="identity", theta_max=theta_max,
    )
    row = settle_episode(policy_name, dialogue, first_turn, result, meta, renderer, judge, lambda_gamma, gamma=gamma)
    row["q_wait"] = q_wait
    row["q_auction"] = q_auction
    return row


def simulate_random_turn_auction(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    renderer: RendererLike,
    judge: JudgeLike,
    lambda_gamma: float,
    rng: np.random.Generator | None = None,
    stop_policy: StopPolicy | None = None,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    **_kwargs,
) -> Dict[str, Any]:
    turns = dialogue.get("turns", [])
    if not turns:
        turns = [{"turn": 1}]
    if rng is None:
        rng = np.random.default_rng()
    chosen_idx = int(rng.integers(0, len(turns)))
    chosen_turn = int(turns[chosen_idx].get("turn", chosen_idx + 1))
    # Old path:
    # result, meta = vv_auction_meta(
    #     dialogue, chosen_turn, pctr_model, stop_policy, gamma, theta_min, theta_max,
    # )
    _, result, meta, q_wait, q_auction = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        fixed_turn=chosen_turn, force_auction_last_turn=True,
        psi_mode="identity", theta_max=theta_max,
    )
    if result is None:
        return _no_trigger_row(policy_name, dialogue)
    meta["payment"] = compute_global_critical_cpc(
        policy_name, dialogue, pctr_model, stop_policy, result.winner_index,
        float(result.winner_ad.get("bid", 0.0)), fixed_turn=chosen_turn,
        psi_mode="identity", theta_max=theta_max,
    )
    row = settle_episode(policy_name, dialogue, chosen_turn, result, meta, renderer, judge, lambda_gamma, gamma=gamma)
    row["q_wait"] = q_wait
    row["q_auction"] = q_auction
    return row


def simulate_last_turn_auction(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    renderer: RendererLike,
    judge: JudgeLike,
    lambda_gamma: float,
    stop_policy: StopPolicy | None = None,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    **_kwargs,
) -> Dict[str, Any]:
    turns = dialogue.get("turns", [])
    if not turns:
        turns = [{"turn": 1}]
    last_turn = int(turns[-1].get("turn", len(turns)))
    # Old path:
    # result, meta = vv_auction_meta(
    #     dialogue, last_turn, pctr_model, stop_policy, gamma, theta_min, theta_max,
    # )
    _, result, meta, q_wait, q_auction = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        fixed_turn=last_turn, force_auction_last_turn=True,
        psi_mode="identity", theta_max=theta_max,
    )
    if result is None:
        return _no_trigger_row(policy_name, dialogue)
    meta["payment"] = compute_global_critical_cpc(
        policy_name, dialogue, pctr_model, stop_policy, result.winner_index,
        float(result.winner_ad.get("bid", 0.0)), fixed_turn=last_turn,
        psi_mode="identity", theta_max=theta_max,
    )
    row = settle_episode(policy_name, dialogue, last_turn, result, meta, renderer, judge, lambda_gamma, gamma=gamma)
    row["q_wait"] = q_wait
    row["q_auction"] = q_auction
    return row


def simulate_learned_stop_policy(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    renderer: RendererLike,
    judge: JudgeLike,
    lambda_gamma: float,
    force_auction_last_turn: bool = True,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    **_kwargs,
) -> Dict[str, Any]:
    # Old learned-stop implementation mixed GSP decision features and VV pricing.
    # turns = dialogue.get("turns", [])
    # n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
    # for idx, turn in enumerate(turns):
    #     ...
    chosen_turn, result, meta, q_wait, q_auction = _run_policy_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity", theta_max=theta_max,
    )
    if result is None or chosen_turn is None:
        row = _no_trigger_row(policy_name, dialogue)
        row["q_wait"] = q_wait
        row["q_auction"] = q_auction
        return row
    meta["payment"] = compute_global_critical_cpc(
        policy_name, dialogue, pctr_model, stop_policy, result.winner_index,
        float(result.winner_ad.get("bid", 0.0)), fixed_turn=None,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity", theta_max=theta_max,
    )
    row = settle_episode(policy_name, dialogue, chosen_turn, result, meta, renderer, judge, lambda_gamma, gamma=gamma)
    row["q_wait"] = q_wait
    row["q_auction"] = q_auction
    return row


def aggregate_results(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    summaries = []
    for policy, g in df.groupby("policy"):
        triggered = g[g["triggered"] == 1]
        summaries.append({
            "policy": policy,
            "n_sessions": int(len(g)),
            "trigger_rate": float(g["triggered"].mean()),
            "avg_trigger_turn": float(triggered["trigger_turn"].mean()) if len(triggered) else np.nan,
            "avg_revenue": float(g["revenue"].mean()),
            "avg_gross_revenue": float(g["gross_revenue"].mean()),
            "avg_info_rent": float(g["info_rent"].mean()),
            "click_rate": float(triggered["click_value"].mean()) if len(triggered) else 0.0,
            "avg_expected_click": float(triggered["expected_click"].mean()) if len(triggered) else 0.0,
            "binary_click_rate": float(triggered["clicked_binary"].mean()) if len(triggered) else 0.0,
            "avg_gamma": float(triggered["gamma_score"].mean()) if len(triggered) else 1.0,
            "avg_reward": float(g["reward"].mean()),
            "avg_payment": float(triggered["payment"].mean()) if len(triggered) else 0.0,
            "avg_winner_bid": float(triggered["winner_bid"].mean()) if len(triggered) else 0.0,
            "avg_winner_pctr": float(triggered["winner_pctr"].mean()) if len(triggered) else 0.0,
            "avg_cv_minus_winner": float(triggered["cv_minus_winner"].mean()) if len(triggered) else 0.0,
            "exact_judge_coverage": float(triggered["judge_exact"].mean()) if len(triggered) else 0.0,
        })
    return pd.DataFrame(summaries).sort_values("policy")


# ---------------------------------------------------------------------------
# Two-phase batch evaluation
# ---------------------------------------------------------------------------

def _resolve_auction_for_policy(
    policy_name, dialogue, pctr_model, stop_policy, lambda_gamma, gamma,
    random_turn=None, force_auction_last_turn=True, include_ablation=False,
):
    """Phase 1: determine auction decision + winner for each policy. No LLM calls."""
    turns = dialogue.get("turns", [])
    n_turns = max(1, int(dialogue.get("n_turns", len(turns) or 1)))
    results = []

    def _auction_at_turn(pname, turn_number):
        # Old pricing path used local virtual-value pricing:
        # result, meta = vv_auction_meta(
        #     dialogue, turn_number, pctr_model, stop_policy, gamma,
        #     DEFAULT_THETA_MIN, DEFAULT_THETA_MAX,
        # )
        chosen_turn, result, meta, q_wait, q_auction = _run_policy_allocation(
            pname, dialogue, pctr_model, stop_policy,
            fixed_turn=turn_number,
            force_auction_last_turn=force_auction_last_turn,
            psi_mode="identity",
            theta_max=DEFAULT_THETA_MAX,
        )
        if result is None:
            row = _no_trigger_row(pname, dialogue)
            row["q_wait"] = q_wait
            row["q_auction"] = q_auction
            return {"row": row, "render_item": None}
        payment = compute_global_critical_cpc(
            pname, dialogue, pctr_model, stop_policy,
            winner_index=result.winner_index,
            winner_bid=float(result.winner_ad.get("bid", 0.0)),
            fixed_turn=chosen_turn,
            force_auction_last_turn=force_auction_last_turn,
            psi_mode="identity",
            theta_max=DEFAULT_THETA_MAX,
        )
        meta["payment"] = payment
        return {
            "row_partial": {
                "policy": pname, "dialogue": dialogue, "turn": chosen_turn,
                "result": result, "meta": meta, "lambda_gamma": lambda_gamma,
                "q_wait": q_wait, "q_auction": q_auction, "gamma": gamma,
            },
            "render_item": (dialogue, chosen_turn, result.winner_ad),
        }

    # first_turn
    first_turn = int(turns[0].get("turn", 1)) if turns else 1
    results.append(("first_turn", _auction_at_turn("first_turn", first_turn), "untrained"))

    # random_turn
    if random_turn is None:
        raise ValueError("random_turn schedule must be provided for reproducibility")
    valid_turns = {int(t.get("turn", idx + 1)) for idx, t in enumerate(turns)} or {1}
    chosen_turn = int(random_turn)
    if chosen_turn not in valid_turns:
        raise ValueError(
            f"random_turn={chosen_turn} not in session turns {sorted(valid_turns)}"
        )
    results.append(("random_turn", _auction_at_turn("random_turn", chosen_turn), "untrained"))

    # last_turn
    last_turn = int(turns[-1].get("turn", len(turns))) if turns else 1
    results.append(("last_turn", _auction_at_turn("last_turn", last_turn), "untrained"))

    # myopic_threshold (adaptive greedy baseline: stop when b_i*q_i >= threshold)
    myo_tn, _, _, _, _ = _run_policy_allocation(
        "myopic_threshold", dialogue, pctr_model, stop_policy,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity",
        theta_max=DEFAULT_THETA_MAX,
    )
    if myo_tn is not None:
        results.append(("myopic_threshold", _auction_at_turn("myopic_threshold", myo_tn), "untrained"))
    else:
        results.append(("myopic_threshold", {"row": _no_trigger_row("myopic_threshold", dialogue), "render_item": None}, "untrained"))

    # learned_stop
    # Old learned-stop path mixed GSP features with virtual-value pricing:
    # chosen_tn = None
    # for idx, turn in enumerate(turns):
    #     tn = int(turn.get("turn", idx + 1))
    #     gsp_result, gsp_meta = auction_meta_from_result_gsp(dialogue, tn, pctr_model, 0.0)
    #     features = stop_features_from_auction(tn, n_turns, gsp_meta)
    #     ...
    chosen_tn, _, _, _, _ = _run_policy_allocation(
        "learned_stop", dialogue, pctr_model, stop_policy,
        force_auction_last_turn=force_auction_last_turn,
        psi_mode="identity",
        theta_max=DEFAULT_THETA_MAX,
    )
    if chosen_tn is not None:
        results.append(("learned_stop", _auction_at_turn("learned_stop", chosen_tn), "trained"))
    else:
        results.append(("learned_stop", {"row": _no_trigger_row("learned_stop", dialogue), "render_item": None}, "trained"))

    if include_ablation:
        results.append(("first_turn_trained_llm", _auction_at_turn("first_turn_trained_llm", first_turn), "trained"))
        if chosen_tn is not None:
            results.append(("learned_stop_untrained_llm", _auction_at_turn("learned_stop_untrained_llm", chosen_tn), "untrained"))
        else:
            results.append(("learned_stop_untrained_llm", {"row": _no_trigger_row("learned_stop_untrained_llm", dialogue), "render_item": None}, "untrained"))

    return results


def batch_evaluate(
    test_dialogues, pctr_model, stop_policy,
    trained_renderer, untrained_renderer, judge,
    lambda_gamma, gamma, seed, force_auction_last_turn=True,
    include_ablation=False, eval_batch_size=8,
    vllm_trained_model=None, vllm_untrained_model=None, vllm_max_new_tokens=128,
):
    """Two-phase evaluation: resolve auctions first, then batch-render LLM responses."""
    from tqdm import tqdm

    # Phase 1: collect all auction decisions (fast, no LLM)
    all_tasks = []
    # Deterministic per-session random-turn schedule keyed by session_id, so the
    # outcome evaluation and the envelope replay select the same turn for every
    # session regardless of iteration order.
    from src.eval.envelope_numerics import build_random_turn_map
    random_turn_map = build_random_turn_map(test_dialogues, seed)
    for d in tqdm(test_dialogues, desc="phase1: auction decisions"):
        policy_results = _resolve_auction_for_policy(
            "batch", d, pctr_model, stop_policy, lambda_gamma, gamma,
            random_turn=random_turn_map[d["session_id"]],
            force_auction_last_turn=force_auction_last_turn,
            include_ablation=include_ablation,
        )
        all_tasks.extend(policy_results)

    # Phase 2: batch render 鈥?free pCTR/stop_policy GPU memory first
    import gc, torch
    del pctr_model, stop_policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trained_items = []
    untrained_items = []
    trained_indices = []
    untrained_indices = []
    for i, (pname, task, renderer_type) in enumerate(all_tasks):
        ri = task.get("render_item")
        if ri is not None:
            if renderer_type == "trained":
                trained_items.append(ri)
                trained_indices.append(i)
            else:
                untrained_items.append(ri)
                untrained_indices.append(i)

    trained_responses = {}
    untrained_responses = {}

    # vLLM sequential mode: load one model at a time
    is_vllm = (trained_renderer == "vllm_deferred_trained")
    if is_vllm:
        from src.generation.base_llm_renderer import VLLMRenderer
        # First: untrained model
        if untrained_items:
            print(f"  [vLLM] Loading untrained model ({vllm_untrained_model}) for {len(untrained_items)} items...")
            renderer_u = VLLMRenderer(
                base_model=vllm_untrained_model,
                max_new_tokens=vllm_max_new_tokens,
                temperature=0.2, top_p=0.9,
                gpu_memory_utilization=0.45,
            )
            rendered = renderer_u.render_batch(untrained_items)
            for idx, (text, tag) in zip(untrained_indices, rendered):
                untrained_responses[idx] = text
            del renderer_u
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Second: trained model
        if trained_items:
            print(f"  [vLLM] Loading trained model ({vllm_trained_model}) for {len(trained_items)} items...")
            renderer_t = VLLMRenderer(
                base_model=vllm_trained_model,
                max_new_tokens=vllm_max_new_tokens,
                temperature=0.2, top_p=0.9,
                gpu_memory_utilization=0.45,
            )
            rendered = renderer_t.render_batch(trained_items)
            for idx, (text, tag) in zip(trained_indices, rendered):
                trained_responses[idx] = text
            del renderer_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        # HF generate mode (original)
        if trained_items and hasattr(trained_renderer, "render_batch"):
            rendered = trained_renderer.render_batch(trained_items, batch_size=eval_batch_size)
            for idx, (text, tag) in zip(trained_indices, rendered):
                trained_responses[idx] = text
        elif trained_items:
            for idx, (d, tn, ad) in zip(trained_indices, trained_items):
                text, tag = trained_renderer.render(d, tn, ad)
                trained_responses[idx] = text

        if untrained_items and hasattr(untrained_renderer, "render_batch"):
            rendered = untrained_renderer.render_batch(untrained_items, batch_size=eval_batch_size)
            for idx, (text, tag) in zip(untrained_indices, rendered):
                untrained_responses[idx] = text
        elif untrained_items:
            for idx, (d, tn, ad) in zip(untrained_indices, untrained_items):
                text, tag = untrained_renderer.render(d, tn, ad)
                untrained_responses[idx] = text

    all_responses = {**trained_responses, **untrained_responses}

    # Phase 3: settle episodes with pre-rendered responses
    rows = []
    for i, (pname, task, renderer_type) in enumerate(all_tasks):
        if "row" in task:
            rows.append(task["row"])
        else:
            p = task["row_partial"]
            renderer = trained_renderer if renderer_type == "trained" else untrained_renderer
            pre_response = all_responses.get(i)
            row = settle_episode(
                p["policy"], p["dialogue"], p["turn"],
                p["result"], p["meta"], renderer, judge,
                p["lambda_gamma"], gamma=p.get("gamma", gamma), pre_rendered_response=pre_response,
            )
            row["q_wait"] = p.get("q_wait", np.nan)
            row["q_auction"] = p.get("q_auction", np.nan)
            rows.append(row)

    return rows
