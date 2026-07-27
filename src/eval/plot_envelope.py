"""
Plotting utilities for Route B+ envelope payment revision.

Generates figures for the paper:
1. Allocation monotonicity curves (x_hat vs bid for sample sessions)
2. Envelope CPC vs GDC scatter/histogram
3. Empirical regret distribution
4. Updated IC verification with envelope payment
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paper style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_allocation_monotonicity(
    curves_df: pd.DataFrame,
    output_path: Path,
    policy: str = "learned_stop",
    n_samples: int = 8,
    seed: int = 42,
):
    """Plot x_hat_i(z) curves for a sample of sessions.

    Shows that expected-click allocation is approximately non-decreasing in bid.
    """
    df = curves_df[curves_df["policy"] == policy].copy()
    sessions = df["session_id"].unique()

    rng = np.random.default_rng(seed)
    if len(sessions) > n_samples:
        # Pick sessions that actually have some variation (not all-zero)
        session_max_x = df.groupby("session_id")["x_hat"].max()
        nonzero_sessions = session_max_x[session_max_x > 0.01].index.tolist()
        if len(nonzero_sessions) >= n_samples:
            selected = rng.choice(nonzero_sessions, n_samples, replace=False)
        else:
            selected = rng.choice(sessions, n_samples, replace=False)
    else:
        selected = sessions

    n_cols = 4
    n_rows = int(np.ceil(len(selected) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows), sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, sid in enumerate(selected):
        ax = axes[idx // n_cols, idx % n_cols]
        sess_data = df[df["session_id"] == sid].sort_values("bid_probe")
        true_bid = sess_data["true_bid"].iloc[0]

        ax.plot(sess_data["bid_probe"], sess_data["x_hat"], "b.-", markersize=3, linewidth=1.2)
        ax.axvline(true_bid, color="red", linestyle="--", alpha=0.7, linewidth=0.8)
        ax.set_title(f"{sid[-8:]}", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Bid $z$")
        if idx % n_cols == 0:
            ax.set_ylabel(r"$\hat{x}_i(z)$")

    # Hide unused axes
    for idx in range(len(selected), n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    fig.suptitle(f"Expected-Click Allocation $\\hat{{x}}_i(z)$ vs. Bid (policy={policy})", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_envelope_vs_gdc(
    results_df: pd.DataFrame,
    output_path: Path,
    policy: str = "learned_stop",
):
    """Scatter plot of Envelope CPC vs GDC CPC, plus gap histogram."""
    df = results_df[results_df["policy"] == policy].copy()
    df = df[df["envelope_cpc"] > 0]  # only triggered sessions where target wins

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): Scatter
    ax = axes[0]
    max_val = max(df["envelope_cpc"].max(), df["gdc_cpc"].max()) * 1.05
    ax.scatter(df["gdc_cpc"], df["envelope_cpc"], s=8, alpha=0.4, edgecolors="none")
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("GDC CPC")
    ax.set_ylabel("Envelope CPC")
    ax.set_title("(a) Envelope vs. GDC Payment")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")

    # Panel (b): Gap histogram
    ax = axes[1]
    gap = df["envelope_cpc"] - df["gdc_cpc"]
    ax.hist(gap, bins=50, edgecolor="black", linewidth=0.3, alpha=0.7)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("$P^{ENV} - P^{GDC}$")
    ax.set_ylabel("Count")
    ax.set_title(f"(b) CPC Gap Distribution (mean={gap.mean():.4f})")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_regret_distribution(
    results_df: pd.DataFrame,
    output_path: Path,
    policy: str = "learned_stop",
):
    """Histogram of empirical regret under envelope payment."""
    df = results_df[results_df["policy"] == policy].copy()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    regrets = df["max_regret"]

    ax.hist(regrets, bins=50, edgecolor="black", linewidth=0.3, alpha=0.7, color="steelblue")
    ax.axvline(regrets.mean(), color="red", linestyle="--", linewidth=1, label=f"Mean={regrets.mean():.4f}")
    ax.axvline(regrets.median(), color="orange", linestyle="-.", linewidth=1, label=f"Median={regrets.median():.4f}")
    p95 = regrets.quantile(0.95)
    ax.axvline(p95, color="green", linestyle=":", linewidth=1, label=f"95th={p95:.4f}")

    ax.set_xlabel("Empirical Regret (max gain from misreporting)")
    ax.set_ylabel("Count")
    ax.set_title(f"Empirical Regret Distribution ({policy})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ic_verification_envelope(
    results_df: pd.DataFrame,
    output_path: Path,
    policy: str = "learned_stop",
):
    """Updated IC verification: utility as a function of bid multiplier.

    Uses envelope payment to show that u(theta, theta) >= u(theta, z) for most z.
    """
    df = results_df[results_df["policy"] == policy].copy()
    df = df[df["x_hat_at_bid"] > 0]

    # For each session, compute utility at true_bid and compare with regret
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel (a): utility at true bid vs regret
    ax = axes[0]
    u_truthful = (df["true_bid"] - df["envelope_cpc"]) * df["x_hat_at_bid"]
    ax.hist(u_truthful, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3, color="steelblue")
    ax.set_xlabel("Truthful Utility $u(\\theta, \\theta)$")
    ax.set_ylabel("Count")
    ax.set_title("(a) Utility at Truthful Report")
    ax.axvline(u_truthful.mean(), color="red", linestyle="--", label=f"Mean={u_truthful.mean():.4f}")
    ax.legend()

    # Panel (b): fraction with zero/negligible regret
    ax = axes[1]
    regrets = df["max_regret"]
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    fracs = [float((regrets <= t).mean()) for t in thresholds]
    ax.bar(range(len(thresholds)), fracs, tick_label=[f"≤{t}" for t in thresholds],
           color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Regret Threshold")
    ax.set_ylabel("Fraction of Sessions")
    ax.set_title("(b) Fraction with Regret ≤ Threshold")
    ax.set_ylim(0, 1.05)
    for i, f in enumerate(fracs):
        ax.text(i, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_updated_table1_comparison(
    results_df: pd.DataFrame,
    output_path: Path,
):
    """Bar chart comparing GDC vs Envelope metrics across policies."""
    policies_order = ["first_turn", "learned_stop", "last_turn", "random_turn"]
    policies_present = [p for p in policies_order if p in results_df["policy"].unique()]

    metrics = ["avg_payment", "avg_net_revenue", "avg_info_rent"]
    labels = ["Payment (CPC)", "Net Revenue", "Info Rent"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(policies_present))
    width = 0.35

    for ax, metric, label in zip(axes, metrics, labels):
        gdc_vals = []
        env_vals = []
        for p in policies_present:
            g = results_df[results_df["policy"] == p]
            if metric == "avg_payment":
                gdc_vals.append(g["gdc_cpc"].mean())
                env_vals.append(g["envelope_cpc"].mean())
            elif metric == "avg_net_revenue":
                gdc_vals.append(g["gdc_net_revenue"].mean())
                env_vals.append(g["env_net_revenue"].mean())
            elif metric == "avg_info_rent":
                gdc_vals.append(g["gdc_info_rent"].mean())
                env_vals.append(g["env_info_rent"].mean())

        ax.bar(x - width / 2, gdc_vals, width, label="GDC", color="lightcoral", edgecolor="black", linewidth=0.3)
        ax.bar(x + width / 2, env_vals, width, label="Envelope", color="steelblue", edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in policies_present], fontsize=8)
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    fig.suptitle("GDC vs. Envelope Payment: Key Metrics by Policy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def generate_all_figures(output_dir: str | Path):
    """Load saved results and generate all figures."""
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(output_dir / "envelope_results.csv")
    curves_df = pd.read_csv(output_dir / "allocation_curves.csv")

    print(f"Loaded: {len(results_df)} session results, {len(curves_df)} curve points")

    plot_allocation_monotonicity(curves_df, fig_dir / "allocation_monotonicity.pdf")
    plot_envelope_vs_gdc(results_df, fig_dir / "envelope_vs_gdc.pdf")
    plot_regret_distribution(results_df, fig_dir / "regret_distribution.pdf")
    plot_ic_verification_envelope(results_df, fig_dir / "ic_verification_envelope.pdf")
    plot_updated_table1_comparison(results_df, fig_dir / "table1_comparison.pdf")

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="outputs/envelope_revision")
    args = parser.parse_args()
    generate_all_figures(args.output_dir)
