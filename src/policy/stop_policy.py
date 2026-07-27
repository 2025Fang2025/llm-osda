from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class StopQNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 0 = WAIT, 1 = AUCTION
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class StopState:
    session_id: str
    turn: int
    n_turns: int
    features: List[float]
    immediate_reward: float
    auction_meta: Dict[str, Any]


def stop_features_from_auction(turn_number: int, n_turns: int, auction_meta: Dict[str, Any]) -> List[float]:
    scores = np.asarray(auction_meta["scores"], dtype=np.float64)
    pctrs = np.asarray(auction_meta["pctrs"], dtype=np.float64)
    bids = np.asarray(auction_meta["bids"], dtype=np.float64)
    order = np.argsort(scores)[::-1]
    max_score = float(scores[order[0]])
    second_score = float(scores[order[1]]) if len(scores) > 1 else 0.0
    gap = max_score - second_score
    winner = int(order[0])
    return [
        float(turn_number) / 5.0,
        float(turn_number) / float(max(1, n_turns)),
        float(n_turns) / 5.0,
        max_score,
        second_score,
        gap,
        float(pctrs[winner]),
        float(np.max(pctrs)),
        float(np.mean(pctrs)),
        float(np.std(pctrs)),
        float(bids[winner]),
        float(np.max(bids)),
        float(np.mean(bids)),
        float(auction_meta["payment"]),
    ]


class StopPolicy:
    def __init__(self, scaler: StandardScaler, model: StopQNetwork, device: torch.device):
        self.scaler = scaler
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def q_values(self, features: List[float]) -> np.ndarray:
        X = self.scaler.transform(np.asarray([features], dtype=np.float32))
        with torch.no_grad():
            q = self.model(torch.tensor(X, dtype=torch.float32, device=self.device)).detach().cpu().numpy()[0]
        return q

    def should_auction(self, features: List[float]) -> bool:
        q = self.q_values(features)
        return bool(q[1] >= q[0])

    def save(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        torch.save({
            "state_dict": self.model.state_dict(),
            "input_dim": self.model.net[0].in_features,
        }, output_dir / "stop_policy.pt")
        joblib.dump(self.scaler, output_dir / "stop_scaler.joblib")

    @staticmethod
    def load(output_dir: str | Path, device: torch.device, hidden_dim: int = 128) -> "StopPolicy":
        output_dir = Path(output_dir)
        scaler = joblib.load(output_dir / "stop_scaler.joblib")
        ckpt = torch.load(output_dir / "stop_policy.pt", map_location=device)
        model = StopQNetwork(ckpt["input_dim"], hidden_dim=hidden_dim)
        model.load_state_dict(ckpt["state_dict"])
        return StopPolicy(scaler, model, device)


def compute_bellman_targets(states_by_session: Dict[str, List[StopState]], discount: float = 0.95) -> Tuple[np.ndarray, np.ndarray, List[StopState]]:
    """Exact optimal-stopping Bellman targets on replayed trajectories.

    Q(s, AUCTION) = immediate_reward(s)
    Q(s, WAIT) = discount * V(next_state), or 0 at the end.
    V(s) = max(Q(s, AUCTION), Q(s, WAIT))
    """
    X, Y, flat = [], [], []
    for sid, states in states_by_session.items():
        V_next = 0.0
        q_targets_reversed = []
        for st in reversed(states):
            q_auction = float(st.immediate_reward)
            q_wait = float(discount * V_next)
            V = max(q_auction, q_wait)
            q_targets_reversed.append([q_wait, q_auction])
            V_next = V
        q_targets = list(reversed(q_targets_reversed))
        for st, y in zip(states, q_targets):
            X.append(st.features)
            Y.append(y)
            flat.append(st)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32), flat


def train_stop_policy(
    states_by_session: Dict[str, List[StopState]],
    device: torch.device,
    hidden_dim: int = 128,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 1e-3,
    discount: float = 0.95,
    patience: int = 10,
) -> Tuple[StopPolicy, Dict[str, Any]]:
    X, Y, flat_states = compute_bellman_targets(states_by_session, discount=discount)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    model = StopQNetwork(Xs.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.tensor(Xs), torch.tensor(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    target_action = np.argmax(Y, axis=1)
    epoch_history = []
    best_mse = float("inf")
    best_state = None
    no_improve = 0
    for epoch in tqdm(range(epochs), desc="train stop policy", leave=False):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in dl:
            xb = xb.to(device).float(); yb = yb.to(device).float()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            pred_all = model(torch.tensor(Xs, dtype=torch.float32, device=device)).detach().cpu().numpy()
        mse_val = float(((pred_all - Y) ** 2).mean())
        pred_action = np.argmax(pred_all, axis=1)
        acc = float((target_action == pred_action).mean())

        if mse_val < best_mse:
            best_mse = mse_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        epoch_history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "q_mse": mse_val,
            "action_match_acc": acc,
        })
        if patience > 0 and no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xs, dtype=torch.float32, device=device)).detach().cpu().numpy()
    mse = float(((pred - Y) ** 2).mean())
    pred_action = np.argmax(pred, axis=1)
    acc = float((target_action == pred_action).mean())
    metrics = {
        "train_q_mse": mse,
        "train_action_match_acc": acc,
        "n_states": int(len(X)),
        "discount": float(discount),
        "target_auction_rate": float((target_action == 1).mean()),
        "pred_auction_rate": float((pred_action == 1).mean()),
        "epoch_history": epoch_history,
    }
    return StopPolicy(scaler, model, device), metrics
