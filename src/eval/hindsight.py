"""Hindsight (information-relaxation) Bellman stopping.  (NEW FILE: src/eval/hindsight.py)

What this computes
------------------
Along the REALIZED dialogue path, with W_t = max_i psi(b_i) q_i(h_t):

    V_{T+1} = 0
    V_t     = max(W_t, gamma * V_{t+1})
    tau     = inf{ t : W_t > 0  and  W_t >= gamma * V_{t+1} }

IMPORTANT -- how this must be reported
--------------------------------------
This policy sees the whole realized path before choosing, so it is an
INFORMATION-RELAXATION UPPER BOUND on any causal policy (Brown-Smith-Sun 2010:
it evaluates E[max] rather than max E).  In the paper it MUST be labelled
"hindsight upper bound" / "information relaxation bound".  It MUST NOT be
called "optimal", and the gap to it MUST NOT be called "regret".

The bound applies only to discounted model welfare, not judged revenue or
retention-adjusted reward. Its gap over the best fixed policy measures potential
under future information; it does not prove an attainable causal improvement.
Pathwise IC diagnostics on this relaxed allocation do not certify the causal
ideal policy or the learned StopNet.
"""
from __future__ import annotations

import os
from src.data.horizon import HORIZON, validate_dialogue
from typing import Any, Dict, List, Tuple

HINDSIGHT_GAMMA = float(os.environ.get("HINDSIGHT_GAMMA", "0.95"))
# Fixed horizon T.  The data generator used a single global max_turns=3 for all
# 14,918 sessions, so T is an environment constant, never a per-session value.


def hindsight_bellman_turn(
    dialogue: Dict[str, Any],
    pctr_model,
    candidate_ads: List[Dict[str, Any]] | None = None,
    gamma: float | None = None,
    horizon: int | None = None,
    psi_mode: str = "identity",
    theta_max: float = 4.1,
    pctr_cache: Dict[int, List[float]] | None = None,
) -> Tuple[int | None, Dict[str, Any]]:
    """Return (tau, diagnostics).  tau is None if no turn has W_t > 0."""
    from src.eval.closed_loop_eval import mechanism_meta_at_turn

    gamma = HINDSIGHT_GAMMA if gamma is None else float(gamma)
    horizon = HORIZON if horizon is None else int(horizon)

    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    validate_dialogue(dialogue, horizon)
    tns = list(range(1, horizon + 1))

    W: Dict[int, float] = {}
    for tn in tns:
        result, _meta = mechanism_meta_at_turn(
            dialogue, tn, pctr_model, candidate_ads=candidate_ads,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        W[tn] = float(result.score) if result is not None else 0.0

    # Backward induction.  V_next[t] holds V_{t+1}; V_{T+1} = 0.
    V_next: Dict[int, float] = {}
    v = 0.0
    for tn in reversed(tns):
        V_next[tn] = v
        v = max(W[tn], gamma * v)
    V_1 = v

    diag = {"W": dict(W), "V_next": dict(V_next), "V_1": V_1, "gamma": gamma}
    for tn in tns:
        if W[tn] > 0.0 and W[tn] >= gamma * V_next[tn]:
            diag["tau"] = tn
            return tn, diag
    diag["tau"] = None
    return None, diag


def cross_turn_cv(dialogue, pctr_model, candidate_ads=None,
                  horizon: int | None = None, psi_mode: str = "identity",
                  theta_max: float = 4.1, pctr_cache=None) -> float:
    """Coefficient of variation of W_t across turns, within one session.

    Diagnostic for the encoder question: StopNet's ONLY timing signal is how
    much W_t varies across t.  If an encoder produces a flat W_t profile, no
    stopping rule can beat a fixed turn.  Pure forward pass, zero LLM calls.
    """
    import numpy as np
    from src.eval.closed_loop_eval import mechanism_meta_at_turn

    horizon = HORIZON if horizon is None else int(horizon)
    turns = dialogue.get("turns", [])
    tns = [int(t.get("turn", i + 1)) for i, t in enumerate(turns)]
    tns = [t for t in tns if t <= horizon]
    if len(tns) < 2:
        return float("nan")
    ws = []
    for tn in tns:
        result, _ = mechanism_meta_at_turn(
            dialogue, tn, pctr_model, candidate_ads=candidate_ads,
            psi_mode=psi_mode, theta_max=theta_max, pctr_cache=pctr_cache,
        )
        ws.append(float(result.score) if result is not None else 0.0)
    ws = np.asarray(ws, dtype=float)
    m = float(ws.mean())
    if m <= 0:
        return float("nan")
    return float(ws.std() / m)
