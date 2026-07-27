from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np


@dataclass
class AuctionResult:
    winner_index: int
    winner_ad: Dict[str, Any]
    pctr: float
    score: float
    second_score: float
    payment_per_click: float
    all_scores: List[float]
    all_pctrs: List[float]


def bid_score_transform(bid: float, mode: str = "virtual", theta_max: float = 4.1) -> float:
    """Monotone psi(b) used by the LLM-OSDA mechanism.

    The paper mechanism allows psi(b)=b as a base version, but this project
    uses the Myerson-style virtual value in the final mechanism path.
    """
    if mode == "identity":
        return float(bid)
    if mode == "virtual":
        return virtual_value(float(bid), theta_max)
    raise ValueError(f"Unknown psi mode: {mode}")


def run_quality_weighted_allocation(
    candidate_ads: List[Dict[str, Any]],
    pctrs: List[float],
    psi_mode: str = "virtual",
    theta_max: float = 4.1,
) -> AuctionResult | None:
    """Allocation rule i* = argmax_i psi(b_i) q_i(h_t).

    This is the allocation component of the final LLM-OSDA mechanism. Payment
    is intentionally set outside this function by the global critical CPC
    search over the complete stopping+allocation mechanism.
    """
    if len(candidate_ads) == 0:
        return None
    pctrs_arr = np.asarray(pctrs, dtype=np.float64)
    pctrs_arr = np.clip(pctrs_arr, 1e-6, 1.0)
    bids = np.asarray([float(a.get("bid", 0.0)) for a in candidate_ads], dtype=np.float64)
    psi = np.asarray([bid_score_transform(b, mode=psi_mode, theta_max=theta_max) for b in bids], dtype=np.float64)
    scores = psi * pctrs_arr
    winner = int(np.argmax(scores))
    max_score = float(scores[winner])
    if max_score <= 0.0:
        return None
    sorted_scores = np.sort(scores)[::-1]
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    return AuctionResult(
        winner_index=winner,
        winner_ad=candidate_ads[winner],
        pctr=float(pctrs_arr[winner]),
        score=max_score,
        second_score=second_score,
        payment_per_click=0.0,
        all_scores=scores.astype(float).tolist(),
        all_pctrs=pctrs_arr.astype(float).tolist(),
    )


def run_quality_adjusted_second_price(
    candidate_ads: List[Dict[str, Any]],
    pctrs: List[float],
    reserve_price: float = 0.0,
) -> AuctionResult:
    """Quality-adjusted second-price auction.

    score_i = bid_i * pCTR_i.
    Winner maximizes score_i.
    Per-click payment = second_score / pCTR_winner, capped by winner bid.
    """
    if len(candidate_ads) == 0:
        raise ValueError("candidate_ads cannot be empty")
    pctrs_arr = np.asarray(pctrs, dtype=np.float64)
    pctrs_arr = np.clip(pctrs_arr, 1e-6, 1.0)
    bids = np.asarray([float(a.get("bid", 0.0)) for a in candidate_ads], dtype=np.float64)
    scores = bids * pctrs_arr
    winner = int(np.argmax(scores))
    sorted_scores = np.sort(scores)[::-1]
    second_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    second_score = max(second_score, float(reserve_price) * float(pctrs_arr[winner]))
    raw_payment = second_score / float(pctrs_arr[winner])
    payment = min(float(bids[winner]), max(float(reserve_price), float(raw_payment)))
    return AuctionResult(
        winner_index=winner,
        winner_ad=candidate_ads[winner],
        pctr=float(pctrs_arr[winner]),
        score=float(scores[winner]),
        second_score=float(second_score),
        payment_per_click=float(payment),
        all_scores=scores.astype(float).tolist(),
        all_pctrs=pctrs_arr.astype(float).tolist(),
    )


# ---------------------------------------------------------------------------
# Myerson Virtual-Value Mechanism (Paper §4)
# ---------------------------------------------------------------------------

def virtual_value(bid: float, theta_max: float) -> float:
    """ψ(b) = 2b - θ_max for Uniform[θ_min, θ_max] prior."""
    return 2.0 * bid - theta_max


def virtual_value_inv(y: float, theta_max: float) -> float:
    """ψ⁻¹(y) = (y + θ_max) / 2."""
    return (y + theta_max) / 2.0


@dataclass
class VirtualValueAuctionResult:
    winner_index: int
    winner_ad: Dict[str, Any]
    pctr: float
    virtual_score: float
    theta_threshold: float
    payment_per_click: float
    cv_minus_winner: float
    all_virtual_scores: List[float]
    all_pctrs: List[float]


def run_virtual_value_auction(
    candidate_ads: List[Dict[str, Any]],
    pctrs: List[float],
    theta_min: float = 0.1,
    theta_max: float = 4.1,
    gamma: float = 0.95,
    cv_minus_winner: float = 0.0,
) -> VirtualValueAuctionResult | None:
    """Myerson virtual-value auction with inter-temporal externality (Paper §3-§4).

    Allocation: i* = argmax_i ψ(b_i) * q_i, subject to ψ(b_i) >= 0 (reserve).
    Pricing: P = ψ⁻¹(Θ / q_{i*}), where
        Θ = max(0, max_{j≠i*} ψ(b_j)·q_j, γ·CV^{-i*})

    Returns None if no ad passes the virtual-value reserve (all ψ(b_i) < 0).
    """
    if len(candidate_ads) == 0:
        return None

    pctrs_arr = np.asarray(pctrs, dtype=np.float64)
    pctrs_arr = np.clip(pctrs_arr, 1e-6, 1.0)
    bids = np.asarray([float(a.get("bid", 0.0)) for a in candidate_ads], dtype=np.float64)

    # Virtual values
    vv = np.array([virtual_value(b, theta_max) for b in bids], dtype=np.float64)
    virtual_scores = vv * pctrs_arr

    # Reserve filter: only ads with ψ(b) >= 0, i.e. b >= θ_max/2
    eligible = vv >= 0.0
    if not np.any(eligible):
        return None

    # Winner among eligible
    masked_scores = np.where(eligible, virtual_scores, -np.inf)
    winner = int(np.argmax(masked_scores))

    # Composite threshold Θ
    lateral_scores = np.copy(virtual_scores)
    lateral_scores[winner] = -np.inf
    lateral_max = float(np.max(lateral_scores)) if len(lateral_scores) > 1 else 0.0
    lateral_max = max(0.0, lateral_max)

    inter_temporal = float(gamma) * float(cv_minus_winner)
    theta_threshold = max(0.0, lateral_max, inter_temporal)

    # Critical-value pricing: P = ψ⁻¹(Θ / q_{i*})
    q_winner = float(pctrs_arr[winner])
    raw_price = virtual_value_inv(theta_threshold / q_winner, theta_max)
    payment = np.clip(raw_price, theta_min, float(bids[winner]))

    return VirtualValueAuctionResult(
        winner_index=winner,
        winner_ad=candidate_ads[winner],
        pctr=q_winner,
        virtual_score=float(virtual_scores[winner]),
        theta_threshold=float(theta_threshold),
        payment_per_click=float(payment),
        cv_minus_winner=float(cv_minus_winner),
        all_virtual_scores=virtual_scores.astype(float).tolist(),
        all_pctrs=pctrs_arr.astype(float).tolist(),
    )
