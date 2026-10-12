"""Merge frozen Round-2 outcomes with replayed envelope payments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from src.eval.revision_metrics import mean_ci


def paper_rows(horizon=None):
    """(outcome_policy, envelope_policy, label) rows that exist in one stratum.

    Two horizon-dependent facts drive this list:
      * second_turn_* is emitted only when turn 2 is INTERIOR (HORIZON > 2).  In
        the T=2 stratum it is the same decision as last_turn_*, so including it
        would either crash on an empty merge or duplicate a row.
      * last_turn_* always lands on turn HORIZON inside a complete-horizon
        stratum, so the label must say Round-HORIZON, not a hard-coded Round-3.
    """
    if horizon is None:
        horizon = int(os.environ.get("LLM_OSDA_HORIZON", "3"))
    horizon = int(horizon)
    rows = [
        ("first_turn", "first_turn", "Always-Round-1 (base)"),
        ("first_turn_trained_llm", "first_turn", "Always-Round-1 (refined)"),
    ]
    if horizon > 2:
        rows += [
            ("second_turn", "second_turn", "Always-Round-2 (base)"),
            ("second_turn_trained_llm", "second_turn", "Always-Round-2 (refined)"),
        ]
    if horizon > 1:  # at T=1 last_turn is literally the first_turn row above
        rows += [
            ("last_turn", "last_turn", f"Always-Round-{horizon} (base)"),
            ("last_turn_trained_llm", "last_turn", f"Always-Round-{horizon} (refined)"),
            ("random_turn", "random_turn", "Random-Round (base)"),
            ("random_turn_trained_llm", "random_turn", "Random-Round (refined)"),
        ]
    rows += [
        ("learned_stop_untrained_llm", "learned_stop", "LLM-OSDA (base)"),
        ("learned_stop", "learned_stop", "LLM-OSDA (refined)"),
        ("hindsight_bellman", "hindsight_bellman", "Hindsight (welfare bound; refined)"),
    ]
    return rows


PAPER_ROWS = paper_rows(3)  # retained for callers that pin the T=3 table


def _validate_merge(merged: pd.DataFrame, outcome_policy: str) -> None:
    if merged.empty:
        raise ValueError(f"No matched sessions for {outcome_policy}")
    if not np.allclose(merged["winner_bid"], merged["true_bid"], atol=1e-9):
        raise ValueError(f"Winner mismatch for {outcome_policy}")
    if "winner_ad_id_env" in merged and not (
        merged["winner_ad_id"] == merged["winner_ad_id_env"]
    ).all():
        raise ValueError(f"Winner ID mismatch for {outcome_policy}")
    if not np.allclose(merged["trigger_turn"], merged["chosen_turn"], atol=0):
        raise ValueError(
            f"Trigger mismatch for {outcome_policy}; rerun the frozen envelope replay"
        )


def build_paper_metrics(
    outcome_sessions: pd.DataFrame,
    envelope_results: pd.DataFrame,
    lambda_gamma: float = 0.25,
    require_revision: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
    horizon=None,
) -> pd.DataFrame:
    """Build the paper table without retraining or regenerating responses."""
    rows = []
    envelope_columns = [
        "session_id",
        "policy",
        "true_bid",
        "winner_ad_id",
        "chosen_turn",
        "envelope_cpc",
        "is_monotone",
        "max_regret",
    ]

    reference_sessions = None
    for outcome_policy, envelope_policy, label in paper_rows(horizon):
        outcomes = outcome_sessions[outcome_sessions["policy"] == outcome_policy].copy()
        if outcomes.empty:
            if require_revision:
                raise ValueError(f"Missing revision policy: {outcome_policy}")
            continue
        session_ids = set(outcomes.session_id)
        if reference_sessions is None:
            reference_sessions = session_ids
        if session_ids != reference_sessions:
            raise ValueError(f"Session coverage differs for {outcome_policy}")
        if "triggered" not in outcomes:
            outcomes["triggered"] = 1
        payments = envelope_results[
            envelope_results["policy"] == envelope_policy
        ][envelope_columns].rename(columns={
            "policy": "envelope_policy",
            "winner_ad_id": "winner_ad_id_env",
        })
        merged = outcomes.merge(payments, on="session_id", how="left", validate="one_to_one", indicator=True)
        active = merged["triggered"] == 1
        if (merged.loc[active, "_merge"] != "both").any():
            raise ValueError(f"Missing envelope payments for {outcome_policy}")
        if active.any():
            _validate_merge(merged.loc[active], outcome_policy)
        merged.loc[~active, "envelope_cpc"] = 0.0

        envelope_revenue = merged["envelope_cpc"] * merged["expected_click"]
        envelope_rent = (
            merged["winner_bid"] - merged["envelope_cpc"]
        ) * merged["expected_click"]
        envelope_reward = envelope_revenue - lambda_gamma * (
            1 - merged["gamma_score"]
        )
        revenue_lo, revenue_hi = mean_ci(envelope_revenue, n_bootstrap, seed)
        reward_lo, reward_hi = mean_ci(envelope_reward, n_bootstrap, seed)
        rows.append({
            "method": label,
            "outcome_policy": outcome_policy,
            "envelope_policy": envelope_policy,
            "n_sessions": len(merged),
            "ctr": merged["click_value"].mean(),
            "trigger": merged["trigger_turn"].mean(),
            "bid": merged["winner_bid"].mean(),
            "pay": merged["envelope_cpc"].mean(),
            "net_revenue": envelope_revenue.mean(),
            "net_revenue_ci95_low": revenue_lo, "net_revenue_ci95_high": revenue_hi,
            "reward_ci95_low": reward_lo, "reward_ci95_high": reward_hi,
            "model_welfare": merged["model_welfare"].mean() if "model_welfare" in merged else np.nan,
            "trigger_rate": merged["triggered"].mean(),
            "info_rent": envelope_rent.mean(),
            "gross_revenue": merged["gross_revenue"].mean(),
            "reward": envelope_reward.mean(),
            "retention": merged["gamma_score"].mean(),
        })
    return pd.DataFrame(rows)


def write_latex_table(metrics: pd.DataFrame, output_path: Path) -> None:
    lines = []
    for row in metrics.itertuples(index=False):
        lines.append(
            f"{row.method} & {row.ctr:.3f} & {row.trigger:.2f} & "
            f"{row.bid:.3f} & {row.pay:.3f} & {row.net_revenue:.3f} & "
            f"{row.info_rent:.3f} & {row.gross_revenue:.3f} & "
            f"{row.reward:.3f} \\\\"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome_sessions", required=True)
    parser.add_argument("--envelope_results", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lambda_gamma", type=float, default=0.25)
    parser.add_argument("--allow_legacy", action="store_true", help="Allow an older incomplete policy table")
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--horizon", type=int, default=None,
                        help="horizon stratum; defaults to $LLM_OSDA_HORIZON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_paper_metrics(
        pd.read_csv(args.outcome_sessions),
        pd.read_csv(args.envelope_results),
        lambda_gamma=args.lambda_gamma,
        require_revision=not args.allow_legacy, n_bootstrap=args.n_bootstrap,
        horizon=args.horizon,
    )
    metrics.to_csv(output_dir / "paper_metrics.csv", index=False)
    write_latex_table(metrics, output_dir / "paper_table_rows.tex")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
