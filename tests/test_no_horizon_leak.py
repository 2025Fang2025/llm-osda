"""Guard against re-introducing the horizon (n_turns) leak.

   (NEW FILE: tests/test_no_horizon_leak.py)

Background
----------
In this dataset the realized horizon equals the realized auction turn for
14,207/14,207 sessions, so dims 1 (turn/n_turns) and 2 (n_turns/5) of the old
14-dim feature map handed StopNet its own label.  These tests fail on the
pre-fix code, which is the point.

The static-scan tests at the bottom need neither torch nor a GPU, so they also
run on a laptop.
"""
from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# static scans: no torch required
# --------------------------------------------------------------------------

def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _strip_comments(code: str) -> str:
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))


def test_no_live_14dim_call_sites_remain():
    """Every evaluated path must use stop_features_v2."""
    offenders = []
    for p in ROOT.joinpath("src").rglob("*.py"):
        if p.name == "stop_policy.py":        # the definition itself may stay
            continue
        if "stop_features_from_auction(" in _strip_comments(p.read_text(encoding="utf-8")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"these still call the 14-dim leaky map: {offenders}"


def test_v2_code_never_references_n_turns():
    """AST-level check, so a docstring that *mentions* n_turns is fine but any
    real read of it is not."""
    import ast as _ast
    tree = _ast.parse(_src("src/policy/stop_policy.py"))
    fn = next(n for n in _ast.walk(tree)
              if isinstance(n, _ast.FunctionDef) and n.name == "stop_features_v2")
    names = {n.id for n in _ast.walk(fn) if isinstance(n, _ast.Name)}
    names |= {a.arg for a in fn.args.args}
    names |= {n.attr for n in _ast.walk(fn) if isinstance(n, _ast.Attribute)}
    consts = {n.value for n in _ast.walk(fn)
              if isinstance(n, _ast.Constant) and isinstance(n.value, str)}
    subscripts = {c for c in consts if c in ("n_turns", "turn_number/n_turns")}
    assert "n_turns" not in names, f"stop_features_v2 reads the realized horizon: {names}"
    assert not subscripts, f"stop_features_v2 indexes a horizon key: {subscripts}"


def test_learned_stop_uses_horizon_not_len_turns():
    """len(turns) is oracle info: a session is short *because* the ad already ran."""
    code = _src("src/eval/closed_loop_eval.py")
    body = code.split("def _choose_learned_stop_turn(")[1].split("\ndef ")[0]
    assert "is_last = (tn == HORIZON)" in body, "stopping must key on the horizon constant"
    assert "len(turns) - 1" not in body, "len(turns) still leaks into the stopping rule"


def test_subset_tool_enforces_fixed_horizon():
    code = _src("tools/make_t3_subset.py")
    assert 'd["n_turns"] = args.horizon' in code


# --------------------------------------------------------------------------
# behavioural tests: need torch
# --------------------------------------------------------------------------

def _meta(n_ads: int = 3):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    pctrs = rng.uniform(0.1, 0.9, n_ads)
    bids = rng.uniform(0.5, 3.0, n_ads)
    return {"scores": (pctrs * bids).tolist(), "pctrs": pctrs.tolist(),
            "bids": bids.tolist(), "payment": 0.7}


def test_v2_signature_has_no_n_turns():
    pytest.importorskip("torch")
    from src.policy import stop_policy as sp
    params = list(inspect.signature(sp.stop_features_v2).parameters)
    assert "n_turns" not in params, f"n_turns leaked back into the signature: {params}"


def test_v2_is_12_dim_and_turn_scaled():
    pytest.importorskip("torch")
    from src.policy import stop_policy as sp
    f = sp.stop_features_v2(2, _meta())
    assert len(f) == 12 == sp.STOP_FEATURE_DIM_V2
    assert f[0] == pytest.approx(2.0 / 5.0)
    assert all(v == v for v in f)          # no NaN


def test_legacy_14dim_map_raises_when_guard_is_armed():
    """THE core regression guard: run_revision.sh always exports LLM_OSDA_HORIZON
    (default 3, overridable per stratum), so any call site we forgot to migrate
    dies loudly instead of silently leaking the realized horizon."""
    pytest.importorskip("torch")
    from src.policy import stop_policy as sp
    old = os.environ.get("LLM_OSDA_HORIZON")
    os.environ["LLM_OSDA_HORIZON"] = "3"
    try:
        with pytest.raises(RuntimeError, match="stop_features_v2"):
            sp.stop_features_from_auction(2, 3, _meta())
    finally:
        if old is None:
            os.environ.pop("LLM_OSDA_HORIZON", None)
        else:
            os.environ["LLM_OSDA_HORIZON"] = old


def test_stopnet_dim_is_inferred_from_features_not_hardcoded():
    pytest.importorskip("torch")
    import torch
    from src.policy import stop_policy as sp
    net = sp.StopQNetwork(input_dim=sp.STOP_FEATURE_DIM_V2)
    assert net(torch.zeros(1, 12)).shape == (1, 2)
    with pytest.raises(RuntimeError):
        net(torch.zeros(1, 14))


# --------------------------------------------------------------------------
# hindsight bound: exercises the REAL function via a stubbed mechanism
# --------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, score):
        self.score = score


def _run_hindsight(monkeypatch, w_by_turn):
    """Drive the real hindsight_bellman_turn with a prescribed W_t path."""
    pytest.importorskip("torch")
    import src.eval.closed_loop_eval as cle
    from src.eval import hindsight as hs

    def fake_meta(dialogue, tn, pctr_model, **kw):
        w = w_by_turn.get(tn)
        return (None, {}) if w is None else (_FakeResult(w), {})

    monkeypatch.setattr(cle, "mechanism_meta_at_turn", fake_meta)
    dialogue = {"session_id": "s", "turns": [{"turn": t} for t in sorted(w_by_turn)]}
    return hs.hindsight_bellman_turn(dialogue, pctr_model=None, gamma=0.95, horizon=3)


def test_hindsight_stops_immediately_on_a_flat_path(monkeypatch):
    """Flat W and gamma<1 => waiting strictly loses => stop at t=1."""
    tau, diag = _run_hindsight(monkeypatch, {1: 1.0, 2: 1.0, 3: 1.0})
    assert tau == 1
    assert diag["V_1"] == pytest.approx(1.0)


def test_hindsight_waits_for_a_later_peak(monkeypatch):
    """W_3 large enough to beat discounted waiting => the bound stops at t=3."""
    tau, diag = _run_hindsight(monkeypatch, {1: 0.5, 2: 0.5, 3: 1.0})
    assert tau == 3
    assert diag["V_1"] == pytest.approx(0.95 * 0.95 * 1.0)


def test_hindsight_dominates_every_fixed_turn(monkeypatch):
    """The defining property of the bound: it is >= any fixed-turn payoff."""
    W = {1: 0.4, 2: 0.9, 3: 0.3}
    tau, diag = _run_hindsight(monkeypatch, W)
    gamma = 0.95
    assert tau == 2
    best_fixed = max(gamma ** (t - 1) * w for t, w in W.items())
    assert diag["V_1"] + 1e-12 >= best_fixed


def test_hindsight_returns_none_when_no_turn_has_value(monkeypatch):
    tau, _ = _run_hindsight(monkeypatch, {1: 0.0, 2: 0.0, 3: 0.0})
    assert tau is None


def test_hindsight_rejects_turns_beyond_the_horizon(monkeypatch):
    """An out-of-protocol path must not be silently truncated."""
    with pytest.raises(ValueError, match="complete horizon"):
        _run_hindsight(monkeypatch, {1: 0.1, 2: 0.1, 3: 0.1, 4: 99.0})
