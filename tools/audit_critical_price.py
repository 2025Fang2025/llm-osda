#!/usr/bin/env python3
"""E4: audit the DEPLOYED critical-price rule against the grid allocation curves.

Zero new experiments: reads only outputs/strata/eval_T*/envelope/{allocation_curves,
envelope_results}.csv produced by run_strata.sh.

Three questions:
  Q1  Is the deployed price kappa = gdc_cpc equal to the true grid critical price
      min{z : xhat(z) > 0}?  The shipped bisection assumes monotonicity.
  Q2  Under the deployed rule the price does NOT depend on the winner's own report,
      so u(z) = (v - kappa) * xhat(z) and the best response is argmax_z xhat(z).
      How large is that regret, and what happens to platform revenue?
  Q3  Is xhat constant on the winning region (as the ideal mechanism implies)?
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TOL = 1e-9


def audit(root: Path, strata=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for h in strata:
        ed = root / f"eval_T{h}" / "envelope"
        curves = pd.read_csv(ed / "allocation_curves.csv")
        res = pd.read_csv(ed / "envelope_results.csv").set_index(["session_id", "policy"])
        for (sid, pol), g in curves.groupby(["session_id", "policy"], sort=False):
            if (sid, pol) not in res.index:
                continue
            r = res.loc[(sid, pol)]
            g = g.sort_values("bid_probe")
            z = g.bid_probe.to_numpy(float)
            x = g.x_hat.to_numpy(float)
            v = float(r.true_bid)
            kappa = float(r.gdc_cpc)
            pos = x > TOL
            kappa_grid = float(z[pos].min()) if pos.any() else float("nan")
            # truthful allocation: grid point closest to v
            i_v = int(np.argmin(np.abs(z - v)))
            x_v = float(x[i_v])
            # best response under the deployed (own-report-independent) price
            i_best = int(np.argmax(x))
            x_best = float(x[i_best])
            # winning region: is xhat constant where positive?
            xpos = x[pos]
            rows.append(dict(
                stratum=h, session_id=sid, policy=pol, true_bid=v,
                kappa_deployed=kappa, kappa_grid=kappa_grid,
                kappa_err=kappa - kappa_grid,
                is_monotone=int(r.is_monotone), n_violations=int(r.n_violations),
                envelope_cpc=float(r.envelope_cpc),
                max_regret_envelope=float(r.max_regret),
                x_at_bid=float(r.x_hat_at_bid), x_truthful=x_v, x_best=x_best,
                best_report=float(z[i_best]),
                regret_deployed=max(0.0, (v - kappa) * (x_best - x_v)),
                rev_truthful=kappa * x_v, rev_strategic=kappa * x_best,
                x_pos_spread=float(xpos.max() - xpos.min()) if pos.any() else 0.0,
                x_pos_rel_spread=float((xpos.max() - xpos.min()) / xpos.max()) if pos.any() and xpos.max() > 0 else 0.0,
                n_pos=int(pos.sum()), n_grid=len(z),
            ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/strata")
    ap.add_argument("--out", default="outputs/strata/pooled/critical_price_audit.csv")
    ap.add_argument("--json_out", default="outputs/strata/pooled/critical_price_audit.json")
    a = ap.parse_args()
    root = Path(a.root)
    df = audit(root)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    summary = {}
    for pol, g in df.groupby("policy"):
        mono = g[g.is_monotone == 1]
        nonmono = g[g.is_monotone == 0]
        d = dict(
            n=len(g), monotonicity_rate=float(g.is_monotone.mean()),
            # Q1
            kappa_exact_rate=float((g.kappa_err.abs() < 1e-6).mean()),
            kappa_err_mean=float(g.kappa_err.mean()),
            kappa_err_absmax=float(g.kappa_err.abs().max()),
            kappa_err_absmax_monotone=float(mono.kappa_err.abs().max()) if len(mono) else None,
            kappa_err_absmax_nonmono=float(nonmono.kappa_err.abs().max()) if len(nonmono) else None,
            kappa_overcharge_rate=float((g.kappa_err > 1e-6).mean()),
            # Q2
            regret_deployed_mean=float(g.regret_deployed.mean()),
            regret_deployed_p95=float(g.regret_deployed.quantile(0.95)),
            regret_deployed_max=float(g.regret_deployed.max()),
            zero_regret_rate_deployed=float((g.regret_deployed <= 1e-9).mean()),
            regret_envelope_mean=float(g.max_regret_envelope.mean()),
            regret_envelope_p95=float(g.max_regret_envelope.quantile(0.95)),
            rev_truthful_mean=float(g.rev_truthful.mean()),
            rev_strategic_mean=float(g.rev_strategic.mean()),
            rev_change_pct=float((g.rev_strategic.mean() / g.rev_truthful.mean() - 1) * 100)
            if g.rev_truthful.mean() else None,
            # Q3
            x_const_on_win_rate=float((g.x_pos_rel_spread < 1e-6).mean()),
            x_pos_rel_spread_mean=float(g.x_pos_rel_spread.mean()),
        )
        summary[pol] = d
    Path(a.json_out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
