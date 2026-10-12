"""Guards for the stratified pipeline: mixed-horizon TRAIN, per-stratum EVAL, pooled report.
(NEW FILE: tests/test_strata_pipeline.py)
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------- fixtures ---
def _session(sid, n, *, auction_turn=1, empty_early=False, pre_ad=False, gap=False):
    turns = []
    for i in range(n):
        t = i + 1
        if gap and t == n and n > 1:
            t = n + 1
        turns.append({"turn": t,
                      "user": f"u{t}",
                      "assistant": "" if (empty_early and i < n - 1) else f"a{t}",
                      "ads_inserted": [{"ad_id": "x"}] if (pre_ad and i < n - 1) else []})
    return {"session_id": sid, "n_turns": n, "auction_turn": auction_turn,
            "turns": turns, "candidate_ads": [{"ad_id": "a1", "bid": 1.0}]}


@pytest.fixture
def corpus(tmp_path):
    """3 x T=1, 4 x T=2, 5 x T=3 clean + 4 dirty sessions, split 'original'-style."""
    rows, clean = [], []
    for i in range(3):
        rows.append(_session(f"t1_{i}", 1)); clean.append(f"t1_{i}")
    for i in range(4):
        rows.append(_session(f"t2_{i}", 2, auction_turn=2)); clean.append(f"t2_{i}")
    for i in range(5):
        rows.append(_session(f"t3_{i}", 3, auction_turn=3)); clean.append(f"t3_{i}")
    dirty = {"d_null": _session("d_null", 3, auction_turn=None),
             "d_empty": _session("d_empty", 3, empty_early=True),
             "d_pread": _session("d_pread", 3, pre_ad=True),
             "d_gap": _session("d_gap", 3, gap=True)}
    rows.extend(dirty.values())
    p = tmp_path / "dialogues.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    allids = [r["session_id"] for r in rows]
    splits = tmp_path / "orig_splits.json"
    splits.write_text(json.dumps({
        "train_sessions": [s for s in allids if s.endswith(("_0", "_1"))],
        "val_sessions": [s for s in allids if s.endswith("_2")],
        "test_sessions": [s for s in allids if not s.endswith(("_0", "_1", "_2"))] +
                         ["d_null", "d_empty", "d_pread", "d_gap"],
    }))
    return p, splits, set(clean)


def _run_subset(tmp_path, corpus_path, splits_path):
    out = tmp_path / "work"
    r = subprocess.run([sys.executable, str(ROOT / "tools/make_strata_subset.py"),
                        "--dialogues", str(corpus_path), "--out", str(out / "clean.jsonl"),
                        "--splits", str(splits_path),
                        "--out-train-splits", str(out / "main/data_splits.json"),
                        "--out-eval-splits-dir", str(out),
                        "--report", str(out / "report.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


# ------------------------------------------------------- make_strata_subset --
def test_subset_keeps_only_clean_sessions_and_real_n_turns(tmp_path, corpus):
    path, splits, clean = corpus
    out = _run_subset(tmp_path, path, splits)
    kept = [json.loads(l) for l in (out / "clean.jsonl").read_text().splitlines() if l.strip()]
    assert {d["session_id"] for d in kept} == clean
    # n_turns must stay REAL; make_t3_subset.py overwrote it with the constant.
    for d in kept:
        assert d["n_turns"] == len(d["turns"]) == int(d["session_id"][1])


def test_subset_never_redraws_the_original_split(tmp_path, corpus):
    path, splits, clean = corpus
    out = _run_subset(tmp_path, path, splits)
    orig = json.loads(splits.read_text())
    new = json.loads((out / "main/data_splits.json").read_text())
    for k in ("train_sessions", "val_sessions", "test_sessions"):
        assert set(new[k]) <= set(orig[k])
        assert set(new[k]) == set(orig[k]) & clean
    assert not set(new["test_sessions"]) & (set(orig["train_sessions"]) | set(orig["val_sessions"]))


def test_strata_partition_the_clean_test_set_and_share_train_val(tmp_path, corpus):
    path, splits, _ = corpus
    out = _run_subset(tmp_path, path, splits)
    main = json.loads((out / "main/data_splits.json").read_text())
    seen = set()
    for h in (1, 2, 3):
        sp = json.loads((out / f"eval_T{h}/data_splits.json").read_text())
        assert sp["stratum_horizon"] == h
        # train/val identical everywhere: proxy-renderer pool must not drift.
        assert sp["train_sessions"] == main["train_sessions"]
        assert sp["val_sessions"] == main["val_sessions"]
        ids = set(sp["test_sessions"])
        assert all(s.startswith(f"t{h}_") for s in ids)
        assert not ids & seen
        seen |= ids
    assert seen == set(main["test_sessions"])


def test_weights_sum_to_one(tmp_path, corpus):
    path, splits, _ = corpus
    out = _run_subset(tmp_path, path, splits)
    rep = json.loads((out / "report.json").read_text())
    assert abs(sum(rep["weights"].values()) - 1.0) < 1e-12
    assert rep["n_test_clean"] <= rep["n_test_original"]


# ------------------------------------------------------- mixed-horizon check --
def test_validate_dialogue_mixed_accepts_any_T_but_rejects_truncation():
    from src.data.horizon import validate_dialogue_mixed, validate_dialogue
    for n in (1, 2, 3):
        validate_dialogue_mixed(_session("ok", n))
    with pytest.raises(ValueError, match="does not reach"):
        validate_dialogue_mixed({"session_id": "x", "n_turns": 3,
                                 "turns": [{"turn": 1}, {"turn": 2}]})
    with pytest.raises(ValueError, match="no gap"):
        validate_dialogue_mixed({"session_id": "x", "n_turns": 3,
                                 "turns": [{"turn": 1}, {"turn": 3}]})
    # The strict evaluation-side check must stay strict.
    with pytest.raises(ValueError):
        validate_dialogue(_session("x", 2))


def test_mixed_training_is_refused_outside_mode_train():
    """--mixed_horizon_train must never reach the evaluation path."""
    src = (ROOT / "src/run_final_experiment.py").read_text()
    assert '--mixed_horizon_train requires --mode train' in src
    tree = ast.parse(src)
    # the guard raises before rounds are computed
    assert src.index('--mixed_horizon_train requires --mode train') < src.index('n_train = args.train_rounds')
    # both StopNet call sites forward the flag
    assert src.count('mixed_horizon=args.mixed_horizon_train') == 2
    assert src.count('horizon=None if args.mixed_horizon_train else HORIZON') == 2
    assert tree is not None


# ------------------------------------------------------------- pool_strata ---
def _fake_stratum(d, h, ids, policies):
    rows = []
    for p in policies:
        for i, s in enumerate(ids):
            rows.append({"policy": p, "session_id": s, "renderer_group": "trained",
                         "model_welfare": (9.0 if p == "hindsight_bellman" else 1.0 + i * 0.1),
                         "revenue": 0.5 + i * 0.01, "gross_revenue": 0.7, "info_rent": 0.2,
                         "gamma_score": 0.8, "trigger_turn": float(h), "expected_click": 0.3,
                         "clicked_binary": 1.0, "reward": 0.4})
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "final_evaluation_sessions.csv", index=False)
    (d / "data_splits.json").write_text(json.dumps(
        {"train_sessions": [], "val_sessions": [], "test_sessions": list(ids)}))


BASE_POL = ["first_turn_trained_llm", "last_turn_trained_llm", "learned_stop", "hindsight_bellman"]


@pytest.fixture
def pooled_root(tmp_path):
    root = tmp_path / "run"
    for h, n in ((1, 2), (2, 5), (3, 3)):
        pol = BASE_POL + (["second_turn_trained_llm"] if h > 2 else [])
        _fake_stratum(root / f"eval_T{h}", h, [f"T{h}s{i}" for i in range(n)], pol)
    return root


def test_pooled_mean_equals_stratum_weighted_mean(pooled_root):
    from tools.pool_strata import load_strata, check_weighted_equals_pooled
    pooled = load_strata({h: pooled_root / f"eval_T{h}/final_evaluation_sessions.csv" for h in (1, 2, 3)})
    assert pooled.session_id.nunique() == 10
    assert check_weighted_equals_pooled(pooled)["max_abs_diff"] < 1e-12
    # trigger_turn pools to the session-weighted mean realized round
    lt = pooled[pooled.policy == "last_turn_trained_llm"]
    assert abs(lt.trigger_turn.mean() - (2 * 1 + 5 * 2 + 3 * 3) / 10) < 1e-12


def test_pool_rejects_sessions_shared_between_strata(tmp_path):
    from tools.pool_strata import load_strata
    root = tmp_path / "bad"
    _fake_stratum(root / "eval_T1", 1, ["dup"], BASE_POL)
    _fake_stratum(root / "eval_T2", 2, ["dup"], BASE_POL)
    with pytest.raises(ValueError, match="more than one stratum"):
        load_strata({h: root / f"eval_T{h}/final_evaluation_sessions.csv" for h in (1, 2)})


def test_pool_rejects_stratum_that_skipped_sessions(tmp_path):
    from tools.pool_strata import load_strata
    root = tmp_path / "short"
    _fake_stratum(root / "eval_T3", 3, ["a", "b"], BASE_POL)
    (root / "eval_T3/data_splits.json").write_text(json.dumps(
        {"train_sessions": [], "val_sessions": [], "test_sessions": ["a", "b", "c"]}))
    with pytest.raises(ValueError, match="split lists"):
        load_strata({3: root / "eval_T3/final_evaluation_sessions.csv"},
                    {3: root / "eval_T3/data_splits.json"})


def test_second_turn_is_excluded_from_the_pooled_pairing(pooled_root):
    from tools.pool_strata import load_strata, STRATUM_ONLY, POOLED_FIXED
    from src.eval.revision_metrics import build_headline_numbers
    pooled = load_strata({h: pooled_root / f"eval_T{h}/final_evaluation_sessions.csv" for h in (1, 2, 3)})
    # Pooling it would break the paired pivot: it covers 3 of 10 sessions.
    with pytest.raises(ValueError):
        build_headline_numbers(pooled, n_bootstrap=20,
                               fixed_override=POOLED_FIXED + ["second_turn_trained_llm"])
    r = build_headline_numbers(pooled[~pooled.policy.isin(STRATUM_ONLY)],
                               n_bootstrap=20, fixed_override=POOLED_FIXED,
                               stratum_label="pooled_T123")
    assert r["n_sessions"] == 10 and r["horizon"] == "pooled_T123"
    assert set(r["fixed_baselines"]) == set(POOLED_FIXED)


def test_single_stratum_headline_behaviour_is_unchanged(tmp_path):
    """fixed_override defaults to None: a single-stratum T=3 report still uses 3 arms.

    Built on a synthetic stratum rather than a completed run, so the guard keeps
    working after outputs/ is cleared (StopNet and the evaluation set changed,
    so the old 708-session artefacts were deleted on purpose).
    """
    from src.eval.revision_metrics import build_headline_numbers, fixed_refined_for_horizon
    d = tmp_path / "eval_T3"
    ids = [f"s{i}" for i in range(6)]
    _fake_stratum(d, 3, ids, BASE_POL + ["second_turn_trained_llm"])
    df = pd.read_csv(d / "final_evaluation_sessions.csv")
    r = build_headline_numbers(df, n_bootstrap=50, horizon=3)
    assert r["horizon"] == 3 and r["n_sessions"] == len(ids)
    assert set(r["fixed_baselines"]) == set(fixed_refined_for_horizon(3))
    assert "second_turn_trained_llm" in r["fixed_baselines"]


def test_run_strata_shell_contract():
    sh = (ROOT / "run_strata.sh").read_text()
    code = "\n".join(l for l in sh.splitlines() if not l.lstrip().startswith("#"))
    assert "--mode train --mixed_horizon_train" in code
    assert "--mode eval --eval_rounds 1 --model_dir" in code
    assert "--mode all" not in code        # the whole point of the split
    assert 'LLM_OSDA_HORIZON="$h"' in code  # one process per stratum
    assert "tools/pool_strata.py" in code
    # Every old run_revision.sh stage must have a successor, or a guard silently
    # stops running: S1->D, S2->TEST, S3->AB, S4->TRAIN+EVAL, S5->ENV, S6->POOL.
    for stage in ("D", "TEST", "AB", "TRAIN", "EVAL", "ENV", "POOL"):
        assert f'if want {stage}; then' in code, f"stage {stage} missing"
    assert "STAGES=(D TEST TRAIN AB EVAL ENV POOL)" in code
    # AB must come after TRAIN so it reuses that StopNet instead of fitting one
    # on the 9,527-session mixed pool.
    assert code.index("if want TRAIN; then") < code.index("if want AB; then")
    assert '--model_dir "$AB_MODEL"' in code
    assert "-m pytest tests -q" in code                  # S2 successor
    assert 'LLM_OSDA_DEDUP="$DEDUP"' in code             # S3 successor
    assert "assert_frame_equal" in code                  # S3 paired check
    assert "export_envelope_paper_results" in code       # S5 successor
    assert "src.eval.revision_metrics" in code           # S6 successor
    # run_revision.sh must remain a usable fallback
    assert (ROOT / "run_revision.sh").is_file()
