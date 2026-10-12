import importlib.util
import csv,json,math
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('multiseed_summary',Path(__file__).resolve().parents[1]/'tools/summarize_multiseed.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(root):
 for seed in [1,2,3]:
  d=root/f'seed_{seed}';(d/'pooled').mkdir(parents=True)
  (d/'multiseed_manifest.json').write_text(json.dumps(dict(train_seed=seed,eval_seed=42,hashes={'data':'fixed'})))
  for h in [1,2,3]:
   (d/f'eval_T{h}').mkdir();(d/f'eval_T{h}/data_splits.json').write_text(json.dumps({'test_sessions':[f's{h}']}))
  rows=[]
  for p in m.POLICIES:
   for h in [1,2,3]:
    r=dict(policy=p,session_id=f's{h}',triggered=1,**{v:1 for v in m.METRICS.values()})
    r.update(click_value=.8,expected_click=.5)
    if p.startswith('learned_stop'):r['revenue']=seed+h
    rows.append(r)
  with (d/'pooled/pooled_sessions.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def test_seed_std_and_ctr(tmp_path):
 fixture(tmp_path);stats=m.summarize(tmp_path,[1,2,3])
 r=next(r for r in stats if r['policy']=='learned_stop' and r['metric']=='revenue')
 assert r['mean']==4 and r['std']==1
 ctr=next(r for r in stats if r['policy']=='learned_stop' and r['metric']=='click_value')
 assert math.isclose(ctr['mean'],.8)
 assert len(stats)==9*8

def test_mismatched_manifest(tmp_path):
 fixture(tmp_path);p=tmp_path/'seed_2/multiseed_manifest.json';d=json.loads(p.read_text());d['eval_seed']=9;p.write_text(json.dumps(d))
 with pytest.raises(ValueError,match='Frozen inputs'):m.summarize(tmp_path,[1,2,3])

def test_incomplete_results(tmp_path):
 fixture(tmp_path);p=tmp_path/'seed_2/pooled/pooled_sessions.csv';lines=p.read_text().splitlines();p.write_text('\n'.join(lines[:-1])+'\n')
 with pytest.raises(ValueError,match='membership'):m.summarize(tmp_path,[1,2,3])
