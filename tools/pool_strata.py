"""Recombine the three horizon strata into ONE table over the clean test set.
(NEW FILE: tools/pool_strata.py)

Evaluation runs once per stratum (T=1 / T=2 / T=3) because the forced
last-turn auction needs a horizon constant.  Reporting, however, must be on the
original test population.  This script concatenates the per-session CSVs and
recomputes every statistic on the pooled rows.

Two facts that drive the implementation:

1. A stratum-weighted mean IS the pooled mean, exactly:
       sum_l n_l * mean_l / sum_l n_l == pooled mean      (verified, diff = 0)
   so the point estimates could be obtained either way.
2. A confidence interval CANNOT be weighted.  Bootstrap must resample the
   pooled session rows, otherwise the paired structure across policies and the
   between-stratum variance are both lost.  Hence we concatenate first and
   bootstrap once.

`second_turn*` is emitted only where turn 2 is INTERIOR, i.e. the T=3 stratum
(708 of 2,741 sessions).  It is reported separately with its own n and must
never be pooled into a paired comparison; the pooled fixed-timing controls are
first_turn_trained_llm (always round 1) and last_turn_trained_llm (each
stratum's own final round == the paper's Always-Last-Round).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# tools/ is a sibling of src/; allow `python3 tools/pool_strata.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Columns reported as per-session means in the main table.
MAIN_METRICS = ["model_welfare", "gross_revenue", "info_rent", "revenue",
                "gamma_score", "trigger_turn", "expected_click", "clicked_binary", "reward"]
ENV_METRICS = ["x_hat_at_bid", "envelope_cpc", "gdc_cpc", "env_net_revenue",
               "gdc_net_revenue", "env_info_rent", "gdc_info_rent",
               "is_monotone", "max_regret", "negative_payment"]
# Arms that exist in every stratum, so they can be compared pairwise on the pool.
POOLED_FIXED = ["first_turn_trained_llm", "last_turn_trained_llm"]
STRATUM_ONLY = ("second_turn", "second_turn_trained_llm")


def load_strata(paths_by_h, splits_by_h=None):
    frames = []
    for h in sorted(paths_by_h):
        p = Path(paths_by_h[h])
        if not p.is_file():
            raise FileNotFoundError(f"stratum T={h}: {p}")
        df = pd.read_csv(p)
        df["stratum"] = h
        if splits_by_h and h in splits_by_h:
            expected = set(json.loads(Path(splits_by_h[h]).read_text())["test_sessions"])
            got = set(df.session_id.unique())
            if got != expected:
                raise ValueError(
                    f"stratum T={h}: evaluated {len(got)} sessions but its split lists "
                    f"{len(expected)}; missing={len(expected - got)} extra={len(got - expected)}")
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)
    # The strata must partition the sessions, or the pooled mean double counts.
    owner = pooled.groupby("session_id")["stratum"].nunique()
    if (owner > 1).any():
        raise ValueError(f"{int((owner > 1).sum())} sessions appear in more than one stratum")
    if pooled.duplicated(["session_id", "policy"]).any():
        raise ValueError("duplicate session/policy rows after pooling")
    return pooled


def per_policy_table(pooled):
    rows = []
    for policy, g in pooled.groupby("policy"):
        row = {"policy": policy, "n_sessions": int(len(g)),
               "strata": "+".join(f"T{h}" for h in sorted(g.stratum.unique()))}
        for m in MAIN_METRICS:
            if m in g:
                row[m] = float(g[m].mean())
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
    return out


def check_weighted_equals_pooled(pooled, metric="model_welfare", tol=1e-9):
    """Numerically confirm the identity used to justify stratified evaluation."""
    diffs = {}
    for policy, g in pooled.groupby("policy"):
        if metric not in g:
            continue
        by = g.groupby("stratum")[metric].agg(["size", "mean"])
        weighted = float((by["size"] * by["mean"]).sum() / by["size"].sum())
        diffs[policy] = abs(weighted - float(g[metric].mean()))
    worst = max(diffs.values()) if diffs else 0.0
    if worst > tol:
        raise ValueError(f"weighted != pooled, worst |diff| = {worst:.3e}")
    return {"metric": metric, "max_abs_diff": worst, "n_policies": len(diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True,
                    help="directory holding eval_T1/ eval_T2/ eval_T3/")
    ap.add_argument("--sessions_name", default="final_evaluation_sessions.csv")
    ap.add_argument("--envelope_name", default="envelope/envelope_results.csv")
    ap.add_argument("--strata", default="1,2,3")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.run_root)
    hs = [int(x) for x in args.strata.split(",") if x.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sess_paths = {h: root / f"eval_T{h}" / args.sessions_name for h in hs}
    split_paths = {h: root / f"eval_T{h}" / "data_splits.json" for h in hs}
    split_paths = {h: p for h, p in split_paths.items() if p.is_file()}
    pooled = load_strata(sess_paths, split_paths)
    pooled.to_csv(out / "pooled_sessions.csv", index=False)

    n_by_h = pooled.groupby("stratum")["session_id"].nunique().to_dict()
    n_total = int(sum(n_by_h.values()))
    print(f"[pool] sessions per stratum {n_by_h}  total={n_total}")

    identity = check_weighted_equals_pooled(pooled)
    print(f"[pool] weighted == pooled: max|diff| = {identity['max_abs_diff']:.3e} "
          f"over {identity['n_policies']} policies")

    table = per_policy_table(pooled)
    table.to_csv(out / "pooled_policy_table.csv", index=False)
    print(table.to_string(index=False))

    strat_rows = []
    for (policy, h), g in pooled.groupby(["policy", "stratum"]):
        r = {"policy": policy, "stratum": f"T{h}", "n_sessions": int(len(g))}
        r.update({m: float(g[m].mean()) for m in MAIN_METRICS if m in g})
        strat_rows.append(r)
    pd.DataFrame(strat_rows).sort_values(["policy", "stratum"]).to_csv(
        out / "stratified_policy_table.csv", index=False)

    # Headline numbers on the pooled rows.  second_turn* is excluded by
    # construction because it exists in one stratum only.
    from src.eval.revision_metrics import build_headline_numbers
    comparable = pooled[~pooled.policy.isin(STRATUM_ONLY)]
    headline = build_headline_numbers(
        comparable, n_bootstrap=args.n_bootstrap, seed=args.seed,
        fixed_override=POOLED_FIXED, stratum_label="pooled_T" + "".join(map(str, hs)))
    headline["sessions_per_stratum"] = {f"T{h}": int(v) for h, v in n_by_h.items()}
    headline["stratum_weights"] = {f"T{h}": v / n_total for h, v in n_by_h.items()}
    headline["weighted_equals_pooled"] = identity
    headline["excluded_from_pairing"] = {
        p: int(pooled[pooled.policy == p].session_id.nunique())
        for p in STRATUM_ONLY if p in set(pooled.policy)}
    headline["note_last_turn"] = (
        "last_turn_trained_llm lands on each stratum's own final round, so the pooled "
        "arm reproduces the paper's Always-Last-Round baseline; pooled mean trigger_turn "
        "is reported in pooled_policy_table.csv.")
    (out / "pooled_headline_numbers.json").write_text(json.dumps(headline, indent=2, allow_nan=False))
    print(json.dumps({k: headline[k] for k in
                      ("n_sessions", "best_fixed_policy", "best_fixed", "learned_stop",
                       "learned_gain_over_best_fixed", "learned_gain_ci95",
                       "hindsight_welfare_upper_bound", "stopnet_realization_rate",
                       "sessions_per_stratum", "excluded_from_pairing")}, indent=2))

    env_paths = {h: root / f"eval_T{h}" / args.envelope_name for h in hs}
    if all(p.is_file() for p in env_paths.values()):
        env = load_strata(env_paths)
        env.to_csv(out / "pooled_envelope_results.csv", index=False)
        rows = []
        for policy, g in env.groupby("policy"):
            r = {"policy": policy, "n_sessions": int(len(g)),
                 "strata": "+".join(f"T{h}" for h in sorted(g.stratum.unique()))}
            r.update({m: float(g[m].mean()) for m in ENV_METRICS if m in g})
            if "max_regret" in g:
                r["regret_p95"] = float(np.quantile(g["max_regret"], 0.95))
                r["zero_regret_rate"] = float((g["max_regret"] <= 1e-9).mean())
            rows.append(r)
        pd.DataFrame(rows).sort_values("policy").to_csv(out / "pooled_envelope_table.csv", index=False)
        print(f"[pool] pooled envelope table -> {out / 'pooled_envelope_table.csv'}")
    else:
        missing = [str(p) for p in env_paths.values() if not p.is_file()]
        print(f"[pool] envelope CSVs incomplete, skipping: {missing}")

    print(f"[pool] wrote {out}")


if __name__ == "__main__":
    main()
