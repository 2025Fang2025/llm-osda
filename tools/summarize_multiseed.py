"""Aggregate session means across StopNet seeds (sample SD, not session SD)."""
import argparse,csv,json,math,statistics
from pathlib import Path

POLICIES={
 'first_turn':'Always-Round-1 (base)',
 'first_turn_trained_llm':'Always-Round-1 (refined)',
 'last_turn':'Always-Last-Round (base)',
 'last_turn_trained_llm':'Always-Last-Round (refined)',
 'random_turn':'Random-Round (base)',
 'random_turn_trained_llm':'Random-Round (refined)',
 'learned_stop_untrained_llm':'LLM-OSDA (base)',
 'learned_stop':'LLM-OSDA (refined)',
 'hindsight_bellman':'Hindsight reference (refined)',
}
METRICS={'Net Rev.':'revenue','Reward':'reward','CTR':'click_value','Trigger':'trigger_turn',
 'Bid':'winner_bid','Pay':'payment','Info. Rent':'info_rent','Bid Value':'gross_revenue'}

def summarize(root,seeds):
 if len(seeds)<2 or len(set(seeds))!=len(seeds):raise ValueError('Need distinct training seeds')
 runs=[];common=None;reference=None;baseline=None;warnings=[]
 for seed in seeds:
  folder=root/f'seed_{seed}'
  manifest=json.loads((folder/'multiseed_manifest.json').read_text())
  if manifest.pop('train_seed')!=seed:raise ValueError('Seed manifest mismatch')
  if common is not None and manifest!=common:raise ValueError('Frozen inputs or evaluation seed differ')
  common=manifest
  with (folder/'pooled/pooled_sessions.csv').open() as f:rows=list(csv.DictReader(f))
  by={p:{} for p in POLICIES}
  for r in rows:
   p=r['policy']
   if p not in by:continue
   sid=r['session_id']
   if sid in by[p]:raise ValueError(f'Duplicate {p}/{sid}')
   if int(float(r['triggered']))!=1:raise ValueError('Nontriggered record requires an explicit metric convention')
   by[p][sid]=r
  ids=set(by['learned_stop'])
  if not ids or any(set(g)!=ids for g in by.values()):raise ValueError('Policy session membership differs')
  split_ids=set()
  for h in (1,2,3):
   hs=set(json.loads((folder/f'eval_T{h}/data_splits.json').read_text())['test_sessions'])
   if split_ids & hs:raise ValueError('Overlapping strata')
   split_ids |= hs
  if ids!=split_ids:raise ValueError('Results differ from test splits')
  if reference is not None and ids!=reference:raise ValueError('Seeds evaluated different sessions')
  reference=ids
  if baseline is not None:
   for p in POLICIES:
    if p.startswith('learned_stop'):continue
    largest=max(abs(float(by[p][i][m])-float(baseline[p][i][m])) for i in ids for m in METRICS.values())
    if largest>1e-7:warnings.append(f'{p}: seed {seed} differs from first seed by up to {largest:g}; inspect evaluation nondeterminism')
  else:baseline=by
  for p,g in by.items():
   r={'seed':seed,'policy':p,'n_sessions':len(ids)}
   for label,m in METRICS.items():
    vals=[float(x[m]) for x in g.values()]
    if not all(math.isfinite(v) for v in vals):raise ValueError(f'Nonfinite {p}/{m}')
    r[m]=statistics.mean(vals)
   runs.append(r)
 out=root/'summary';out.mkdir(exist_ok=True)
 def save(name,rows):
  with (out/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save('per_seed_metrics.csv',runs)
 stats=[];tex=['% Mean and sample SD across StopNet training seeds. Fixed evaluation seed.',
 '% CTR is undiscounted; Bid Value is discounted bid-weighted proxy-click value.']
 for p,title in POLICIES.items():
  selected=[r for r in runs if r['policy']==p];cells=[]
  for label,m in METRICS.items():
   vals=[r[m] for r in selected];avg=statistics.mean(vals);sd=statistics.stdev(vals)
   stats.append(dict(policy=p,metric=m,n_seeds=len(seeds),n_sessions=len(reference),mean=avg,std=sd))
   cells.append(f'${avg:.3f} \\pm {sd:.3f}$')
  tex.append(title+' & '+' & '.join(cells)+r' \\')
 save('main_table_mean_std.csv',stats)
 (out/'main_table_rows.tex').write_text('\n'.join(tex)+'\n')
 gains=[]
 for seed in seeds:
  d={r['policy']:r for r in runs if r['seed']==seed}
  for comparator in ['first_turn_trained_llm','last_turn_trained_llm','first_turn']:
   base=d[comparator]['revenue'];v=d['learned_stop']['revenue']
   if base==0:raise ValueError('Zero baseline revenue')
   gains.append(dict(seed=seed,baseline=comparator,revenue_difference=v-base,relative_gain_percent=100*(v/base-1)))
 save('per_seed_gains.csv',gains)
 gs=[]
 for b in dict.fromkeys(r['baseline'] for r in gains):
  for m in ['revenue_difference','relative_gain_percent']:
   v=[r[m] for r in gains if r['baseline']==b]
   gs.append(dict(baseline=b,metric=m,mean=statistics.mean(v),std=statistics.stdev(v)))
 save('gain_mean_std.csv',gs)
 (out/'summary.json').write_text(json.dumps(dict(seeds=seeds,n_sessions=len(reference),sd_ddof=1,
  scope='StopNet training seeds; frozen quality/judge/renderer, target bid seed and evaluation seed',
  warnings=warnings),indent=2))
 for w in warnings:print('WARNING:',w)
 print('Wrote',out)
 return stats

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True)
 ap.add_argument('--seeds',type=int,nargs='+',required=True);a=ap.parse_args();summarize(a.root,a.seeds)
