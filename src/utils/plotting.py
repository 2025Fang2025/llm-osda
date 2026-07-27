"""
Training curve plotting utilities.
Saves loss/AUC curves at both step-level and epoch-level.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _plot_step_loss(ax, step_history: List[Dict], title: str):
    if not step_history:
        return
    steps = [h["step"] for h in step_history]
    losses = [h["loss"] for h in step_history]
    ax.plot(steps, losses, alpha=0.4, linewidth=0.5, color="blue", label="step loss")
    window = max(1, len(losses) // 50)
    if window > 1:
        import numpy as np
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color="red", linewidth=1.5, label=f"smooth (w={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_pctr_curves(metrics: Dict[str, Any], output_dir: str | Path) -> None:
    if not HAS_MPL:
        return
    history = metrics.get("epoch_history", [])
    step_history = metrics.get("step_history", [])
    if not history and not step_history:
        return
    output_dir = Path(output_dir)

    n_cols = 2 + (1 if step_history else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if step_history:
        _plot_step_loss(axes[col], step_history, "pCTR Step Loss")
        col += 1

    if history:
        epochs = [h["epoch"] for h in history]
        axes[col].plot(epochs, [h["train_loss"] for h in history], label="Train Loss (epoch avg)")
        axes[col].plot(epochs, [h["val_log_loss"] for h in history], label="Val Log Loss")
        axes[col].set_xlabel("Epoch")
        axes[col].set_ylabel("Loss")
        axes[col].set_title("pCTR Epoch Loss")
        axes[col].legend()
        axes[col].grid(True, alpha=0.3)
        col += 1

        auc_values = [h["val_auc"] for h in history if h["val_auc"] is not None]
        if auc_values:
            auc_epochs = [h["epoch"] for h in history if h["val_auc"] is not None]
            axes[col].plot(auc_epochs, auc_values, color="green", marker=".")
            axes[col].set_xlabel("Epoch")
            axes[col].set_ylabel("AUC")
            axes[col].set_title(f"pCTR Val AUC (best={max(auc_values):.4f})")
            axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "pctr_training_curves.png", dpi=150)
    plt.close()


def plot_judge_curves(metrics: Dict[str, Any], output_dir: str | Path) -> None:
    if not HAS_MPL:
        return
    history = metrics.get("epoch_history", [])
    step_history = metrics.get("step_history", [])
    if not history and not step_history:
        return
    output_dir = Path(output_dir)

    n_cols = 3 + (1 if step_history else 0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    col = 0
    if step_history:
        _plot_step_loss(axes[col], step_history, "Judge Step Loss")
        col += 1

    if history:
        epochs = [h["epoch"] for h in history]
        axes[col].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
        axes[col].plot(epochs, [h["val_click_log_loss"] for h in history], label="Val Click LogLoss")
        axes[col].set_xlabel("Epoch")
        axes[col].set_ylabel("Loss")
        axes[col].set_title("Judge Epoch Loss")
        axes[col].legend()
        axes[col].grid(True, alpha=0.3)
        col += 1

        axes[col].plot(epochs, [h["val_gamma_mae"] for h in history], color="orange")
        axes[col].set_xlabel("Epoch")
        axes[col].set_ylabel("MAE")
        axes[col].set_title(f"Judge Gamma MAE (best={min(h['val_gamma_mae'] for h in history):.4f})")
        axes[col].grid(True, alpha=0.3)
        col += 1

        auc_values = [h["val_click_auc"] for h in history if h["val_click_auc"] is not None]
        if auc_values:
            auc_epochs = [h["epoch"] for h in history if h["val_click_auc"] is not None]
            axes[col].plot(auc_epochs, auc_values, color="green", marker=".")
            axes[col].set_xlabel("Epoch")
            axes[col].set_ylabel("AUC")
            axes[col].set_title(f"Judge Click AUC (best={max(auc_values):.4f})")
            axes[col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "judge_training_curves.png", dpi=150)
    plt.close()


def plot_stop_policy_curves(metrics: Dict[str, Any], output_dir: str | Path) -> None:
    if not HAS_MPL:
        return
    history = metrics.get("epoch_history", [])
    if not history:
        return
    output_dir = Path(output_dir)
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss (batch)")
    axes[0].plot(epochs, [h["q_mse"] for h in history], label="Full Q MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Stop Policy Q-value MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [h["action_match_acc"] for h in history], color="purple", marker=".")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Action Match Acc (final={history[-1]['action_match_acc']:.4f})")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "stop_policy_training_curves.png", dpi=150)
    plt.close()


def plot_all_training_curves(
    pctr_metrics: Dict[str, Any],
    judge_metrics: Dict[str, Any],
    stop_metrics: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    if not HAS_MPL:
        print("  [warning] matplotlib not available, skipping training curve plots.")
        return
    output_dir = Path(output_dir)
    plot_pctr_curves(pctr_metrics, output_dir)
    plot_judge_curves(judge_metrics, output_dir)
    plot_stop_policy_curves(stop_metrics, output_dir)
    print(f"  Training curves saved to {output_dir}/*.png")


def plot_unified_user_model_curves(metrics: Dict[str, Any], output_dir: str | Path) -> None:
    """Plot training curves for the unified user model (pCTR + Judge combined)."""
    if not HAS_MPL:
        print("  [warning] matplotlib not available, skipping training curve plots.")
        return
    history = metrics.get("epoch_history", [])
    step_history = metrics.get("step_history", [])
    if not history and not step_history:
        return
    output_dir = Path(output_dir)
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Step loss, Train/Val loss, pCTR AUC
    if step_history:
        _plot_step_loss(axes[0][0], step_history, "Step Loss (unified)")
    else:
        axes[0][0].set_visible(False)

    if history:
        epochs = [h["epoch"] for h in history]

        # Train loss
        axes[0][1].plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="blue")
        axes[0][1].plot(epochs, [h["pctr_val_loss"] for h in history], label="pCTR Val Loss", color="orange")
        axes[0][1].plot(epochs, [h["judge_val_loss"] for h in history], label="Judge Val Loss", color="green")
        axes[0][1].set_xlabel("Epoch")
        axes[0][1].set_ylabel("Loss")
        axes[0][1].set_title("Train & Val Loss")
        axes[0][1].legend()
        axes[0][1].grid(True, alpha=0.3)

        # pCTR AUC
        auc_vals = [h["pctr_val_auc"] for h in history if h.get("pctr_val_auc") is not None]
        if auc_vals:
            auc_eps = [h["epoch"] for h in history if h.get("pctr_val_auc") is not None]
            axes[0][2].plot(auc_eps, auc_vals, color="green", marker=".")
            axes[0][2].set_xlabel("Epoch")
            axes[0][2].set_ylabel("AUC")
            axes[0][2].set_title(f"pCTR Val AUC (best={max(auc_vals):.4f})")
            axes[0][2].grid(True, alpha=0.3)

        # Row 2: Judge AUC, Gamma MAE, Combined val loss
        judge_auc = [h["judge_val_auc"] for h in history if h.get("judge_val_auc") is not None]
        if judge_auc:
            judge_eps = [h["epoch"] for h in history if h.get("judge_val_auc") is not None]
            axes[1][0].plot(judge_eps, judge_auc, color="green", marker=".")
            axes[1][0].set_xlabel("Epoch")
            axes[1][0].set_ylabel("AUC")
            axes[1][0].set_title(f"Judge Val AUC (best={max(judge_auc):.4f})")
            axes[1][0].grid(True, alpha=0.3)

        gamma_mae = [h.get("gamma_mae", 0) for h in history]
        if any(g > 0 for g in gamma_mae):
            axes[1][1].plot(epochs, gamma_mae, color="orange", marker=".")
            axes[1][1].set_xlabel("Epoch")
            axes[1][1].set_ylabel("MAE")
            axes[1][1].set_title(f"Gamma MAE (best={min(gamma_mae):.4f})")
            axes[1][1].grid(True, alpha=0.3)

        # Combined val loss trend
        combined = [h["pctr_val_loss"] + h["judge_val_loss"] + h.get("gamma_mae", 0) for h in history]
        axes[1][2].plot(epochs, combined, color="red", marker=".")
        axes[1][2].set_xlabel("Epoch")
        axes[1][2].set_ylabel("Combined Val Loss")
        axes[1][2].set_title("pCTR + Judge + Gamma (early stop metric)")
        axes[1][2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "unified_user_model_curves.png", dpi=150)
    plt.close()
    print(f"  Unified user model curves saved to {output_dir}/unified_user_model_curves.png")
