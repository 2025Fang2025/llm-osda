"""Framework-independent numerical helpers for envelope evaluation."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def build_bid_grid(
    true_bid: float,
    theta_min: float,
    theta_max: float,
    n_grid: int,
    extra_points: Optional[List[float]] = None,
) -> np.ndarray:
    """Build a bid grid containing the truthful bid and known thresholds."""
    grid = np.linspace(theta_min, theta_max, n_grid)
    points = [true_bid]
    if extra_points:
        points.extend(extra_points)
    points = [point for point in points if theta_min <= point <= theta_max]
    return np.unique(np.append(grid, points))


def piecewise_constant_integral(
    bid_grid: np.ndarray,
    x_hat_values: List[float] | np.ndarray,
) -> float:
    """Integrate a right-continuous step allocation on an ordered grid."""
    grid = np.asarray(bid_grid, dtype=float)
    x_hat = np.asarray(x_hat_values, dtype=float)
    if len(grid) < 2:
        return 0.0
    if len(grid) != len(x_hat):
        raise ValueError("bid_grid and x_hat_values must have the same length")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("bid_grid must be strictly increasing")
    return float(np.sum(np.diff(grid) * x_hat[:-1]))


def compute_envelope_payment(
    bid_grid: np.ndarray,
    x_hat_values: List[float],
    true_bid: float,
) -> Dict[str, float]:
    """Compute envelope payment from a sampled step allocation."""
    grid = np.asarray(bid_grid)
    x_hat = np.asarray(x_hat_values)
    bid_idx = np.argmin(np.abs(grid - true_bid))
    if np.abs(grid[bid_idx] - true_bid) < 1e-6:
        x_hat_at_bid = float(x_hat[bid_idx])
    else:
        x_hat_at_bid = float(np.interp(true_bid, grid, x_hat))

    if x_hat_at_bid <= 1e-10:
        return {
            "envelope_transfer": 0.0,
            "envelope_cpc": 0.0,
            "x_hat_at_bid": 0.0,
            "integral": 0.0,
        }

    mask = grid <= true_bid + 1e-9
    grid_sub = grid[mask]
    x_sub = x_hat[mask]
    if grid_sub[-1] < true_bid - 1e-9:
        grid_sub = np.append(grid_sub, true_bid)
        x_sub = np.append(x_sub, np.interp(true_bid, grid, x_hat))

    integral = piecewise_constant_integral(grid_sub, x_sub)
    envelope_transfer = true_bid * x_hat_at_bid - integral
    return {
        "envelope_transfer": float(envelope_transfer),
        "envelope_cpc": float(envelope_transfer / x_hat_at_bid),
        "x_hat_at_bid": x_hat_at_bid,
        "integral": integral,
    }


def check_monotonicity(
    bid_grid: np.ndarray,
    x_hat_values: List[float],
    tolerance: float = 1e-6,
) -> Tuple[bool, int, List[int]]:
    """Return allocation monotonicity and violation locations."""
    del bid_grid
    diffs = np.diff(np.asarray(x_hat_values))
    violations = np.where(diffs < -tolerance)[0]
    return len(violations) == 0, int(len(violations)), violations.tolist()


def compute_empirical_regret(
    bid_grid: np.ndarray,
    x_hat_values: List[float],
    true_bid: float,
) -> Dict[str, float]:
    """Compute winner-conditioned empirical regret on the evaluated grid."""
    grid = np.asarray(bid_grid)
    x_hat = np.asarray(x_hat_values)
    utilities = []
    for index, report in enumerate(grid):
        integral = piecewise_constant_integral(
            grid[: index + 1], x_hat[: index + 1]
        )
        transfer = report * float(x_hat[index]) - integral
        utilities.append(true_bid * float(x_hat[index]) - transfer)

    utilities = np.asarray(utilities)
    truthful_idx = np.argmin(np.abs(grid - true_bid))
    best_idx = int(np.argmax(utilities))
    return {
        "regret": float(max(0.0, utilities[best_idx] - utilities[truthful_idx])),
        "u_truthful": float(utilities[truthful_idx]),
        "max_u": float(utilities[best_idx]),
        "best_deviation_bid": float(grid[best_idx]),
    }


def build_random_turn_map(
    dialogues: List[Dict[str, Any]],
    seed: int,
) -> Dict[str, int]:
    """Choose one reproducible random turn per session."""
    rng = np.random.default_rng(seed)
    fixed_turns: Dict[str, int] = {}
    for dialogue in dialogues:
        turns = dialogue.get("turns", []) or [{"turn": 1}]
        chosen_idx = int(rng.integers(0, len(turns)))
        fixed_turns[dialogue["session_id"]] = int(
            turns[chosen_idx].get("turn", chosen_idx + 1)
        )
    return fixed_turns


def random_turn_map_from_outcomes(
    outcome_sessions: "Any",
    policy: str = "random_turn",
) -> Dict[str, int]:
    """Read the random-turn schedule from a frozen outcome-sessions table.

    Ensures the envelope replay reuses exactly the turn each session was
    evaluated at, so it aligns with the frozen LLM outcomes rather than
    drawing a fresh schedule.
    """
    rows = outcome_sessions[outcome_sessions["policy"] == policy]
    rows = rows[rows["triggered"] == 1] if "triggered" in rows.columns else rows
    schedule: Dict[str, int] = {}
    for session_id, turn in zip(rows["session_id"], rows["trigger_turn"]):
        if turn is None or (isinstance(turn, float) and np.isnan(turn)):
            continue
        schedule[str(session_id)] = int(turn)
    if not schedule:
        raise ValueError(
            f"No triggered '{policy}' rows found in the outcome table"
        )
    return schedule
