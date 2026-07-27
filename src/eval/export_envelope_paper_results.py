"""Merge frozen Round-2 outcomes with replayed envelope payments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PAPER_ROWS = [
    ("first_turn", "first_turn", "Always-Round-1 (base)"),
    ("first_turn_trained_llm", "first_turn", "Always-Round-1 (refined)"),
    ("last_turn", "last_turn", "Always-Last-Round"),
    ("random_turn", "random_turn", "Random-Round"),
    ("learned_stop_untrained_llm", "learned_stop", "LLM-OSDA (base)"),
    ("learned_stop", "learned_stop", "LLM-OSDA (refined)"),
]


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

    for outcome_policy, envelope_policy, label in PAPER_ROWS:
        outcomes = outcome_sessions[outcome_sessions["policy"] == outcome_policy]
        payments = envelope_results[
            envelope_results["policy"] == envelope_policy
        ][envelope_columns].rename(columns={
            "policy": "envelope_policy",
            "winner_ad_id": "winner_ad_id_env",
        })
        merged = outcomes.merge(payments, on="session_id", how="inner")
        _validate_merge(merged, outcome_policy)

        envelope_revenue = merged["envelope_cpc"] * merged["expected_click"]
        envelope_rent = (
            merged["winner_bid"] - merged["envelope_cpc"]
        ) * merged["expected_click"]
        envelope_reward = envelope_revenue - lambda_gamma * (
            1 - merged["gamma_score"]
        )
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_paper_metrics(
        pd.read_csv(args.outcome_sessions),
        pd.read_csv(args.envelope_results),
        lambda_gamma=args.lambda_gamma,
    )
    metrics.to_csv(output_dir / "paper_metrics.csv", index=False)
    write_latex_table(metrics, output_dir / "paper_table_rows.tex")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
