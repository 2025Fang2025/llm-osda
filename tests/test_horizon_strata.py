"""Horizon-STRATUM guards.  (NEW FILE: tests/test_horizon_strata.py)

A stratum is the set of sessions whose dialogue really has T turns.  Stratified
evaluation must never change the test set: every stratum's test split is carved
out of the ORIGINAL test_sessions list, so the union of the strata is a subset of
the original test set and no training/validation session can leak in.

These tests pin down two things that are easy to break:
  1. the arm list must not contain a duplicate or non-existent fixed-turn arm
     (second_turn == last_turn at T=2; turn 2 does not exist at T=1);
  2. the subset tool must inherit split membership rather than re-draw it.
"""
import ast
import json
import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _reload_with_horizon(module_name, horizon):
    """Import a module fresh under a given LLM_OSDA_HORIZON."""
    import importlib
    import sys
    old = os.environ.get("LLM_OSDA_HORIZON")
    os.environ["LLM_OSDA_HORIZON"] = str(horizon)
    try:
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)
    finally:
        if old is None:
            os.environ.pop("LLM_OSDA_HORIZON", None)
        else:
            os.environ["LLM_OSDA_HORIZON"] = old


# ---------------------------------------------------------------- arm lists

@pytest.mark.parametrize("horizon,expected", [
    (3, ["first_turn_trained_llm", "second_turn_trained_llm", "last_turn_trained_llm"]),
    (2, ["first_turn_trained_llm", "last_turn_trained_llm"]),
    (1, ["first_turn_trained_llm"]),
])
def test_fixed_refined_controls_track_the_horizon(horizon, expected):
    from src.eval.revision_metrics import fixed_refined_for_horizon
    assert fixed_refined_for_horizon(horizon) == expected


def test_second_turn_is_dropped_whenever_it_duplicates_last_turn():
    """At T=2, second_turn and last_turn are the SAME decision.  Emitting both
    would put two identical rows in the main table under different names."""
    from src.eval.export_envelope_paper_results import paper_rows
    for horizon in (1, 2):
        names = [r[0] for r in paper_rows(horizon)]
        assert "second_turn" not in names, horizon
        assert "second_turn_trained_llm" not in names, horizon
    names3 = [r[0] for r in paper_rows(3)]
    assert "second_turn" in names3 and "second_turn_trained_llm" in names3


def test_last_turn_row_is_labelled_with_the_actual_horizon():
    from src.eval.export_envelope_paper_results import paper_rows
    labels = {r[0]: r[2] for r in paper_rows(2)}
    assert labels["last_turn"] == "Always-Round-2 (base)"
    labels3 = {r[0]: r[2] for r in paper_rows(3)}
    assert labels3["last_turn"] == "Always-Round-3 (base)"


def test_paper_rows_have_no_duplicate_outcome_policy():
    from src.eval.export_envelope_paper_results import paper_rows
    for horizon in (1, 2, 3):
        names = [r[0] for r in paper_rows(horizon)]
        assert len(names) == len(set(names)), (horizon, names)


def test_eval_gates_the_fixed_turn_two_arm_on_the_horizon():
    """Source-level guard: the hard-coded turn 2 must sit behind a HORIZON test."""
    src = (ROOT / "src/eval/closed_loop_eval.py").read_text()
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond = ast.dump(node.test)
            if "HORIZON" in cond:
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                if "second_turn" in body:
                    hits.append(cond)
    assert hits, "the fixed t=2 arm is no longer gated on HORIZON"


# ------------------------------------------------------- test-set invariance

def _splits(train, val, test):
    return {"train_sessions": list(train), "val_sessions": list(val),
            "test_sessions": list(test)}


def test_subset_tool_inherits_split_membership(tmp_path):
    """The stratum must be CARVED OUT of the original splits, never re-drawn."""
    import subprocess
    import sys
    dialogues = tmp_path / "d.jsonl"
    rows = []
    for i in range(6):
        n = 2 if i % 2 else 3
        rows.append({
            "session_id": f"s{i}", "auction_turn": n, "n_turns": n,
            "turns": [{"turn": t, "assistant": "x", "ads_inserted": []}
                      for t in range(1, n + 1)],
        })
    dialogues.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    orig = tmp_path / "splits.json"
    orig.write_text(json.dumps(_splits(["s0", "s1"], ["s2", "s3"], ["s4", "s5"])))

    for horizon in (2, 3):
        out = tmp_path / f"t{horizon}.jsonl"
        new = tmp_path / f"splits_t{horizon}.json"
        subprocess.run([sys.executable, str(ROOT / "tools/make_t3_subset.py"),
                        "--dialogues", str(dialogues), "--out", str(out),
                        "--splits", str(orig), "--out-splits", str(new),
                        "--horizon", str(horizon)], check=True, cwd=ROOT)
        a = json.loads(orig.read_text())
        b = json.loads(new.read_text())
        for name in ("train_sessions", "val_sessions", "test_sessions"):
            assert set(b[name]).issubset(a[name]), (horizon, name)
        assert not set(b["test_sessions"]) & (set(a["train_sessions"]) | set(a["val_sessions"]))


def test_strata_test_splits_are_disjoint_and_cover_no_extra_sessions(tmp_path):
    """Union over strata stays inside the original test set, with no overlap."""
    import subprocess
    import sys
    dialogues = tmp_path / "d.jsonl"
    rows = []
    for i in range(9):
        n = (i % 3) + 1
        rows.append({
            "session_id": f"s{i}", "auction_turn": n, "n_turns": n,
            "turns": [{"turn": t, "assistant": "x", "ads_inserted": []}
                      for t in range(1, n + 1)],
        })
    dialogues.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    orig = tmp_path / "splits.json"
    orig.write_text(json.dumps(_splits(["s0", "s1", "s2"], ["s3", "s4"],
                                       ["s5", "s6", "s7", "s8"])))
    seen = set()
    for horizon in (1, 2, 3):
        out = tmp_path / f"t{horizon}.jsonl"
        new = tmp_path / f"splits_t{horizon}.json"
        subprocess.run([sys.executable, str(ROOT / "tools/make_t3_subset.py"),
                        "--dialogues", str(dialogues), "--out", str(out),
                        "--splits", str(orig), "--out-splits", str(new),
                        "--horizon", str(horizon)], check=True, cwd=ROOT)
        test = set(json.loads(new.read_text())["test_sessions"])
        assert not (test & seen), f"strata overlap at T={horizon}"
        seen |= test
    assert seen.issubset(set(json.loads(orig.read_text())["test_sessions"]))


# ------------------------------------------------- mixed-horizon training

def _traj(turns, n_turns, reward=1.0):
    from src.policy.stop_policy import StopState
    return [StopState("s", t, n_turns, [0.0] * 12, reward, {}) for t in turns]


def test_mixed_horizon_targets_terminate_each_trajectory_at_its_own_T():
    """Training MAY mix strata: the realized T builds the Bellman target, it is
    never an input feature.  Each trajectory must get V_{T_i+1} = 0 at its own
    end, so the last WAIT target is 0 for every horizon."""
    pytest.importorskip("torch")
    from src.policy.stop_policy import compute_bellman_targets
    for turns in ([1], [1, 2], [1, 2, 3]):
        _, Y, _ = compute_bellman_targets({"s": _traj(turns, len(turns))}, horizon=None)
        assert Y[-1][0] == pytest.approx(0.0), turns
        assert len(Y) == len(turns)
        for row in Y[:-1]:
            assert row[0] == pytest.approx(0.95)


def test_mixed_horizon_rejects_truncation_posing_as_termination():
    """The guard that makes mixed training safe: a replay that stopped early must
    not receive WAIT = 0, which would teach the model that waiting is worthless."""
    pytest.importorskip("torch")
    from src.policy.stop_policy import compute_bellman_targets
    with pytest.raises(ValueError, match="does not reach its recorded horizon"):
        compute_bellman_targets({"s": _traj([1, 2], 3)}, horizon=None)
    with pytest.raises(ValueError, match="no gap"):
        compute_bellman_targets({"s": _traj([1, 3], 3)}, horizon=None)
    with pytest.raises(ValueError, match="no gap"):
        compute_bellman_targets({"s": _traj([2], 2)}, horizon=None)


def test_default_global_horizon_mode_is_unchanged():
    """Regression: the completed T=3 run must reproduce bit-for-bit."""
    pytest.importorskip("torch")
    import numpy as np
    from src.policy.stop_policy import compute_bellman_targets
    _, Y, _ = compute_bellman_targets({"s": _traj([1, 2, 3], 3)})
    np.testing.assert_allclose(Y[:, 0], [0.95, 0.95, 0.0])
    for bad in ([1], [1, 2], [1, 3], [3, 2, 1]):
        with pytest.raises(ValueError, match="complete horizon"):
            compute_bellman_targets({"s": _traj(bad, len(bad))})


def test_mixed_horizon_evaluation_is_refused_not_silently_wrong():
    """Evaluation must NOT mix strata.  force_auction_last_turn keys on
    is_last == (t == HORIZON); in a mixed set a T=2 session would never reach the
    global HORIZON, so the末轮 guarantee would silently vanish.  validate_dialogue
    has to reject it instead."""
    from src.data.horizon import validate_dialogue
    short = {"session_id": "s", "turns": [{"turn": 1}, {"turn": 2}]}
    with pytest.raises(ValueError, match="complete horizon"):
        validate_dialogue(short, horizon=3)
