"""Build ONE clean mixed-horizon corpus plus PER-STRATUM evaluation splits.
(NEW FILE: tools/make_strata_subset.py -- supersedes tools/make_t3_subset.py)

Why this exists
---------------
Training and evaluation need opposite things from the same corpus:

  * TRAINING must mix T=1/2/3.  Restricting StopNet to the T=3 slice throws
    away 75.6% of the clean pool, and a model trained on 2,324 sessions is not
    comparable with the original run's 10,444.  Mixing is sound: the realized T
    builds the Bellman target, it is not an input feature (12-dim
    stop_features_v2 carries no n_turns).

  * EVALUATION must stay inside ONE stratum.  The forced last-turn auction has
    to decide "is this the last turn?".  On a mixed corpus the only options are
    `t == HORIZON` (a T=2 session never reaches 3, so the guarantee silently
    vanishes) or `t == len(turns)` (that is the label: n_turns == auction_turn
    for 14,207/14,207 sessions).  Inside a stratum the two coincide and the
    constant is an experimental setting, not a per-session peek.

The original split is NEVER re-drawn
------------------------------------
`--splits` is filtered, not resampled.  train/val keep the FULL clean mixed
membership in every stratum file, so the proxy-renderer pool and the
non-test SFT rows are byte-identical across the three evaluation runs; only
`test_sessions` is partitioned by realized T.  The three test lists are
mutually disjoint and their union is the clean part of the original test set,
so the stratum-weighted mean equals the pooled mean exactly.

Filters (reported separately so the paper can disclose exact counts):
  F1  turns are 1..n with no gap and len(turns) == n_turns
  F2  auction_turn is not None            -- drops the corrupted records
  F3  turns 1..n-1 have non-empty assistant text
  F4  turns 1..n-1 carry no pre-inserted ad
  F5  realized T is within --max-horizon

The generator censors paths by its own content decisions, so conditioning on a
complete clean path is a selection.  Report the selected population; do not
extrapolate to all sessions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def iter_jsonl(p):
    """Stream: dialogues_all.jsonl is ~100 MB, never load it all at once."""
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def classify(d, max_horizon, drop_null_auction_turn):
    """Return (realized_T, None) if the session is clean, else (None, reason)."""
    turns = d.get("turns", [])
    n = len(turns)
    if n == 0:
        return None, "drop_F1_empty"
    if [int(t.get("turn", i + 1)) for i, t in enumerate(turns)] != list(range(1, n + 1)):
        return None, "drop_F1_turn_sequence"
    declared = d.get("n_turns", n)
    if int(declared) != n:
        return None, "drop_F1_n_turns_mismatch"
    if n > max_horizon:
        return None, "drop_F5_over_max_horizon"
    if drop_null_auction_turn and d.get("auction_turn") is None:
        return None, "drop_F2_null_auction_turn"
    early = turns[: n - 1]
    if any(not str(t.get("assistant") or "").strip() for t in early):
        return None, "drop_F3_empty_assistant"
    if any(t.get("ads_inserted") for t in early):
        return None, "drop_F4_pre_inserted_ad"
    return n, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", required=True, help="clean mixed-horizon corpus (jsonl)")
    ap.add_argument("--splits", required=True, help="ORIGINAL data_splits.json; filtered, never re-drawn")
    ap.add_argument("--out-train-splits", required=True,
                    help="splits for --mode train (test_sessions = all clean strata)")
    ap.add_argument("--out-eval-splits-dir", required=True,
                    help="writes <dir>/eval_T{h}/data_splits.json for each stratum")
    ap.add_argument("--max-horizon", type=int, default=3)
    ap.add_argument("--drop-null-auction-turn", action="store_true", default=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if not Path(args.splits).is_file():
        raise FileNotFoundError(args.splits)
    sp = json.loads(Path(args.splits).read_text())
    names = ("train_sessions", "val_sessions", "test_sessions")
    parts = {name: set(sp[name]) for name in names}
    for a in names:
        for b in names:
            if a < b and parts[a] & parts[b]:
                raise ValueError(f"Original splits overlap: {a} & {b}")
    all_original = set().union(*parts.values())

    stats = Counter()
    horizon_of = {}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out_f:
        for d in iter_jsonl(args.dialogues):
            stats["total"] += 1
            T, reason = classify(d, args.max_horizon, args.drop_null_auction_turn)
            if reason:
                stats[reason] += 1
                continue
            sid = d["session_id"]
            if sid in horizon_of:
                raise ValueError(f"Duplicate session_id: {sid}")
            horizon_of[sid] = T
            # n_turns is left at its REAL value on purpose: make_t3_subset.py
            # overwrote it with the constant, which is wrong for a mixed corpus.
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats[f"kept_T{T}"] += 1

    kept = set(horizon_of)
    if not kept.issubset(all_original):
        raise ValueError(f"{len(kept - all_original)} kept sessions are absent from the original splits")

    # --- splits for --mode train: filter only, never re-draw ---
    train_sp = {}
    for k, v in sp.items():
        train_sp[k] = [s for s in v if s in kept] if isinstance(v, list) else v
    for name in names:
        stats[f"split_{name}"] = len(train_sp[name])
    Path(args.out_train_splits).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_train_splits).write_text(json.dumps(train_sp, indent=2))
    print(f"[strata] train splits -> {args.out_train_splits} "
          f"(train={len(train_sp['train_sessions'])}, val={len(train_sp['val_sessions'])}, "
          f"test={len(train_sp['test_sessions'])})")

    # --- per-stratum evaluation splits: ONLY test_sessions is partitioned ---
    by_T = defaultdict(list)
    for s in train_sp["test_sessions"]:
        by_T[horizon_of[s]].append(s)
    covered = []
    strata = {}
    for h in range(1, args.max_horizon + 1):
        test_h = sorted(by_T.get(h, []))
        covered.extend(test_h)
        eval_sp = dict(train_sp)
        eval_sp["test_sessions"] = test_h
        eval_sp["stratum_horizon"] = h
        out_p = Path(args.out_eval_splits_dir) / f"eval_T{h}" / "data_splits.json"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(eval_sp, indent=2))
        strata[f"T{h}"] = {"n_test": len(test_h), "path": str(out_p)}
        stats[f"eval_test_T{h}"] = len(test_h)
        print(f"[strata] T={h}: n_test={len(test_h)} -> {out_p}")

    # The three strata must partition the clean test set exactly, otherwise the
    # weighted recombination is not equal to the pooled mean.
    if len(covered) != len(set(covered)):
        raise ValueError("Strata test sets overlap")
    if set(covered) != set(train_sp["test_sessions"]):
        raise ValueError("Strata test sets do not cover the clean test set")
    if set(train_sp["test_sessions"]) & (parts["train_sessions"] | parts["val_sessions"]):
        raise ValueError("New test intersects the original train/validation sets")

    weights = {k: v["n_test"] / len(covered) for k, v in strata.items()} if covered else {}
    report = {"stats": dict(stats), "strata": strata, "weights": weights,
              "n_test_clean": len(covered), "n_test_original": len(parts["test_sessions"])}
    print(f"[strata] {dict(stats)}")
    print(f"[strata] weights {weights}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
