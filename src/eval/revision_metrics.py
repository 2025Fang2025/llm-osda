"""Paired session-bootstrap reporting for the fixed-horizon revision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

FIXED_REFINED = ["first_turn_trained_llm", "second_turn_trained_llm", "last_turn_trained_llm"]


def fixed_refined_for_horizon(horizon=None):
    """Fixed-timing refined-renderer controls that exist inside one horizon stratum.

    second_turn_* is a distinct arm only when turn 2 is INTERIOR.  In the T=2
    stratum it is the same decision as last_turn_*, and _resolve_auction_for_policy
    does not emit it; in the T=1 stratum turn 2 does not exist.  Keep this in sync
    with the HORIZON gate in closed_loop_eval._resolve_auction_for_policy.
    """
    if horizon is None:
        horizon = int(os.environ.get("LLM_OSDA_HORIZON", "3"))
    horizon = int(horizon)
    if horizon > 2:
        return list(FIXED_REFINED)
    if horizon == 2:
        return ["first_turn_trained_llm", "last_turn_trained_llm"]
    return ["first_turn_trained_llm"]


def mean_ci(values, n_bootstrap=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("CI requires nonempty, finite session outcomes")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean()
             for _ in range(n_bootstrap)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def build_headline_numbers(outcomes, n_bootstrap=2000, seed=42, horizon=None,
                           fixed_override=None, stratum_label=None):
    """fixed_override is for the POOLED table across horizon strata.

    After pooling T=1/2/3, second_turn_* exists only in the T=3 stratum, so it
    cannot enter a paired pivot over all 2,741 sessions.  last_turn_* is still
    well defined everywhere -- inside each stratum it lands on that stratum's
    final turn -- so pooling it reproduces exactly the original paper's
    Always-Last-Round baseline (mean realized T = 2.1824 vs the reported 2.19).
    Pass fixed_override=["first_turn_trained_llm", "last_turn_trained_llm"].
    """
    fixed = list(fixed_override) if fixed_override else fixed_refined_for_horizon(horizon)
    n_fixed = len(fixed)
    policies = fixed + ["learned_stop", "hindsight_bellman"]
    selected = outcomes[outcomes.policy.isin(policies)]
    if selected.duplicated(["session_id", "policy"]).any():
        raise ValueError("Duplicate session/policy outcomes")
    if not set(policies).issubset(set(selected.policy)):
        raise ValueError("Missing refined timing controls or hindsight outcomes")
    if "renderer_group" not in selected or not (selected.renderer_group == "trained").all():
        raise ValueError("Headline comparisons require the same trained renderer")
    table = selected.pivot(index="session_id", columns="policy", values="model_welfare")[policies]
    if table.empty or not np.isfinite(table.to_numpy()).all():
        raise ValueError("All policies must cover the same sessions with finite model welfare")
    values = table.to_numpy()
    if np.any(values[:, -1:] + 1e-7 < values[:, :-1]):
        raise ValueError("Hindsight does not dominate model welfare; check horizon/discount/replay")
    means = values.mean(axis=0)
    best = int(np.argmax(means[:n_fixed]))
    gap = float(means[-1] - means[best])
    gain = float(means[-2] - means[best])
    rng = np.random.default_rng(seed)
    samples = []
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    for _ in range(n_bootstrap):
        m = values[rng.integers(len(values), size=len(values))].mean(axis=0)
        # Re-select the best fixed policy in each paired resample.
        b = m[:n_fixed].max()
        samples.append([m[-1] - b, m[-2] - b])
    ci = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "metric": "discounted_model_welfare = gamma^(turn-1) * bid * predicted_click",
        "n_sessions": len(table), "n_bootstrap": n_bootstrap,
        "fixed_baselines": dict(zip(fixed, map(float, means[:n_fixed]))),
        "horizon": stratum_label if stratum_label is not None else (
            int(os.environ.get("LLM_OSDA_HORIZON", "3")) if horizon is None else int(horizon)),
        "best_fixed_policy": fixed[best], "best_fixed": float(means[best]),
        "hindsight_welfare_upper_bound": float(means[-1]),
        "hindsight_gap_over_best_fixed": gap,
        "hindsight_gap_ci95": ci[:, 0].tolist(),
        "learned_stop": float(means[-2]),
        "learned_gain_over_best_fixed": gain,
        "learned_gain_ci95": ci[:, 1].tolist(),
        "stopnet_realization_rate": gain / gap if gap > 1e-12 else None,
        "interpretation": "Future-information potential in predicted welfare; not an attainable causal gain, revenue bound, or DSIC certificate. Best fixed is selected on this evaluation set.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome_sessions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=None,
                        help="horizon stratum; defaults to $LLM_OSDA_HORIZON")
    args = parser.parse_args()
    result = build_headline_numbers(pd.read_csv(args.outcome_sessions), args.n_bootstrap,
                                    args.seed, horizon=args.horizon)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
