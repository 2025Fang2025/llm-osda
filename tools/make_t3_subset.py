"""Export the clean fixed-horizon (T=3) subset.  (NEW FILE: tools/make_t3_subset.py)

Keeps only sessions where the horizon is actually observed and the context is
uncontaminated, so that:
  * n_turns == 3 for every session  => the n_turns feature carries no label
  * is_last == (t == 3)             => a true horizon boundary, not len(turns)
  * V_4 = 0 is legitimate
  * Random-Round samples from {1,2,3} automatically

Filters (each reported separately so the paper can disclose exact counts):
  F1  len(turns) == 3
  F2  auction_turn is not None      -- drops the 711 corrupted records
  F3  turns 1..2 have non-empty assistant text
  F4  turns 1..2 carry no pre-inserted ad

The generator censors paths by its own content decisions. Conditioning on a
complete clean path introduces selection bias even though all policy rows share
the same subset. Report the selected population and do not extrapolate to all sessions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_jsonl(p):
    """Stream: dialogues_all.jsonl is ~100 MB, never load it all at once."""
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="", help="original data_splits.json; preserves train/test membership")
    ap.add_argument("--out-splits", default="")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--drop-null-auction-turn", action="store_true", default=True)
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    if args.splits and not Path(args.splits).is_file():
        raise FileNotFoundError(args.splits)
    stats = Counter()
    keep_ids = set()
    auction_turn_hist = Counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as out_f:
        for d in iter_jsonl(args.dialogues):
            stats["total"] += 1
            turns = d.get("turns", [])
            if len(turns) != args.horizon:
                stats["drop_F1_len"] += 1
                continue
            if [int(t.get("turn", i + 1)) for i, t in enumerate(turns)] != list(range(1, args.horizon + 1)):
                stats["drop_invalid_turn_sequence"] += 1
                continue
            if args.drop_null_auction_turn and d.get("auction_turn") is None:
                stats["drop_F2_null_auction_turn"] += 1
                continue
            early = turns[: args.horizon - 1]
            if any(not str(t.get("assistant") or "").strip() for t in early):
                stats["drop_F3_empty_assistant"] += 1
                continue
            if any(t.get("ads_inserted") for t in early):
                stats["drop_F4_pre_inserted_ad"] += 1
                continue
            d["n_turns"] = args.horizon  # make the constant explicit in the record
            if d["session_id"] in keep_ids:
                raise ValueError(f"Duplicate session_id: {d['session_id']}")
            keep_ids.add(d["session_id"])
            auction_turn_hist[str(d.get("auction_turn"))] += 1
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats["kept"] = len(keep_ids)
    stats_auction = {f"kept_auction_turn_{k}": v for k, v in sorted(auction_turn_hist.items())}
    stats.update(stats_auction)
    kept = keep_ids

    if args.splits and Path(args.splits).exists():
        with open(args.splits) as f:
            sp = json.load(f)
        names = ("train_sessions", "val_sessions", "test_sessions")
        parts = [set(sp[name]) for name in names]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i)):
            raise ValueError("Original train/val/test splits overlap")
        if not keep_ids.issubset(set.union(*parts)):
            raise ValueError("Kept sessions missing from original splits")
        new_sp = {}
        for k, v in sp.items():
            if isinstance(v, list):
                new_sp[k] = [s for s in v if s in keep_ids]
                stats[f"split_{k}"] = len(new_sp[k])
            else:
                new_sp[k] = v
        out_sp = args.out_splits or str(Path(args.out).parent / "data_splits.json")
        Path(out_sp).parent.mkdir(parents=True, exist_ok=True)
        with open(out_sp, "w") as f:
            json.dump(new_sp, f, indent=2)
        print(f"[subset] wrote splits -> {out_sp}")

    print(f"[subset] {dict(stats)}")
    print(f"[subset] wrote {len(kept)} sessions -> {args.out}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump(dict(stats), f, indent=2)


if __name__ == "__main__":
    main()
