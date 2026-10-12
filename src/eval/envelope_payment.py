"""
Route B+ Envelope Payment Computation.

Given frozen models (Q_eta, V_phi), recompute payment via the Myerson envelope
formula instead of the Global Dynamic Critical CPC (GDC).

For each winner in each session:
1. Construct a bid grid z ∈ [θ_min, θ_max]
2. For each z, replay the full mechanism (stopping + allocation) to get:
   x_hat_i(z) = gamma^(tau(z)-1) * q_i(h_{tau(z)}) * 1{i*(z) = i}
3. Numerically integrate the step allocation to get envelope payment:
   m_i = b_i * x_hat_i(b_i) - ∫_{θ_min}^{b_i} x_hat_i(z) dz
   P_i^ENV = m_i / x_hat_i(b_i)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.eval.closed_loop_eval import (
    _run_policy_allocation,
    _ads_with_replaced_bid,
    compute_pctrs_for_state,
)
from src.eval.envelope_numerics import (
    build_bid_grid,
    check_monotonicity,
    compute_empirical_regret,
    compute_envelope_payment,
)
from src.models.text_mlp import PCTRModel
from src.policy.stop_policy import StopPolicy


DEFAULT_THETA_MIN = 0.1
DEFAULT_THETA_MAX = 4.1
DEFAULT_N_GRID = 64


@dataclass
class EnvelopeResult:
    """Per-session envelope payment result."""
    session_id: str
    policy: str
    winner_index: int
    true_bid: float
    # Allocation curve
    bid_grid: List[float]
    x_hat_values: List[float]
    # Envelope payment
    x_hat_at_bid: float
    envelope_transfer: float  # m_i
    envelope_cpc: float       # P_i^ENV
    gdc_cpc: float            # original GDC payment for comparison
    # Diagnostics
    is_monotone: bool
    n_violations: int
    max_regret: float
    ir_violation: bool
    negative_payment: bool


def estimate_expected_click_allocation(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    target_idx: int,
    bid_grid: np.ndarray,
    gamma: float = 0.95,
    theta_max: float = DEFAULT_THETA_MAX,
    force_auction_last_turn: bool = True,
    pctr_cache: Optional[Dict[int, List[float]]] = None,
    fixed_turn: Optional[int] = None,
) -> List[float]:
    """Estimate x_hat_i(z) for each z in bid_grid.

    x_hat_i(z) = gamma^(tau(z)-1) * q_i(h_{tau(z)}) * 1{i*(z) = i}

    This replays the full committed mechanism (stopping + allocation) for each
    counterfactual bid z, holding b_{-i} and the bid-blind path fixed.
    """
    x_hat_values = []

    for z in bid_grid:
        # Replace target's bid with z
        ads = _ads_with_replaced_bid(dialogue, target_idx, float(z))

        # Run full mechanism: stopping + allocation
        chosen_turn, result, meta, _, _ = _run_policy_allocation(
            policy_name, dialogue, pctr_model, stop_policy,
            candidate_ads=ads,
            fixed_turn=fixed_turn,
            force_auction_last_turn=force_auction_last_turn,
            psi_mode="identity",
            theta_max=theta_max,
            pctr_cache=pctr_cache,
            gamma=gamma,
        )

        if chosen_turn is None or result is None:
            # Mechanism didn't trigger — no ad inserted
            x_hat_values.append(0.0)
        elif int(result.winner_index) != int(target_idx):
            # Target didn't win
            x_hat_values.append(0.0)
        else:
            # Target wins at turn chosen_turn
            # x_hat = gamma^(tau-1) * q_i(h_tau)
            time_discount = gamma ** max(0, int(chosen_turn) - 1)
            q_i = float(meta.get("winner_pctr", result.pctr))
            x_hat = time_discount * q_i
            x_hat_values.append(x_hat)

    return x_hat_values


def run_envelope_for_session(
    policy_name: str,
    dialogue: Dict[str, Any],
    pctr_model: PCTRModel,
    stop_policy: StopPolicy,
    winner_index: int,
    true_bid: float,
    gdc_cpc: float,
    gamma: float = 0.95,
    theta_min: float = DEFAULT_THETA_MIN,
    theta_max: float = DEFAULT_THETA_MAX,
    n_grid: int = DEFAULT_N_GRID,
    force_auction_last_turn: bool = True,
    fixed_turn: Optional[int] = None,
) -> EnvelopeResult:
    """Full envelope payment pipeline for one session.

    1. Build bid grid
    2. Estimate x_hat_i(z) for each z
    3. Compute envelope payment
    4. Check monotonicity
    5. Compute empirical regret
    """
    # Build pctr cache for efficiency
    turns = dialogue.get("turns", [])
    pctr_cache: Dict[int, List[float]] = {}
    for idx, turn in enumerate(turns):
        tn = int(turn.get("turn", idx + 1))
        pctr_cache[tn] = compute_pctrs_for_state(pctr_model, dialogue, tn)

    # Step 1: Build bid grid
    bid_grid = build_bid_grid(
        true_bid=true_bid,
        theta_min=theta_min,
        theta_max=theta_max,
        n_grid=n_grid,
        extra_points=[gdc_cpc],
    )

    # Step 2: Estimate x_hat
    x_hat_values = estimate_expected_click_allocation(
        policy_name, dialogue, pctr_model, stop_policy,
        target_idx=winner_index,
        bid_grid=bid_grid,
        gamma=gamma,
        theta_max=theta_max,
        force_auction_last_turn=force_auction_last_turn,
        pctr_cache=pctr_cache,
        fixed_turn=fixed_turn,
    )

    # Step 3: Envelope payment
    env_result = compute_envelope_payment(bid_grid, x_hat_values, true_bid)

    # Step 4: Monotonicity check
    is_monotone, n_violations, _ = check_monotonicity(bid_grid, x_hat_values)

    # Step 5: Empirical regret
    regret_result = compute_empirical_regret(bid_grid, x_hat_values, true_bid)

    return EnvelopeResult(
        session_id=dialogue["session_id"],
        policy=policy_name,
        winner_index=winner_index,
        true_bid=true_bid,
        bid_grid=bid_grid.tolist(),
        x_hat_values=x_hat_values,
        x_hat_at_bid=env_result["x_hat_at_bid"],
        envelope_transfer=env_result["envelope_transfer"],
        envelope_cpc=env_result["envelope_cpc"],
        gdc_cpc=gdc_cpc,
        is_monotone=is_monotone,
        n_violations=n_violations,
        max_regret=regret_result["regret"],
        ir_violation=bool(env_result["envelope_cpc"] > true_bid + 1e-9),
        negative_payment=bool(env_result["envelope_cpc"] < -1e-9),
    )
