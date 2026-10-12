import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import dedup
from src.eval.revision_metrics import build_headline_numbers, FIXED_REFINED


@pytest.fixture(autouse=True)
def clear_cache():
    dedup._RENDER_CACHE.clear()
    dedup._JUDGE_CACHE.clear()
    for key in dedup._STATS:
        dedup._STATS[key] = 0
    yield
    dedup._CACHE_PATH = None


def test_cache_survives_a_fresh_process(tmp_path):
    d, ad = {"session_id": "s", "turns": [{"turn": 2, "user": "question"}]}, {"ad_id": "a"}
    rk = dedup._render_key('frozen-renderer', d, 2, ad)
    jk = dedup._judge_key('frozen-judge', d, 2, ad, 'reply')
    dedup._RENDER_CACHE[rk] = ('reply', 'tag')
    dedup._JUDGE_CACHE[jk] = {'click_value': .7}
    path = tmp_path/'cache.json'
    dedup.save_cache(path)
    code = '''
import sys
from src.eval import dedup
p = sys.argv[1]
dedup.load_cache(p)
d = {"session_id": "s", "turns": [{"turn": 2, "user": "question"}]}
ad = {"ad_id": "a"}
assert dedup._RENDER_CACHE[dedup._render_key('frozen-renderer', d, 2, ad)] == ('reply', 'tag')
assert dedup._JUDGE_CACHE[dedup._judge_key('frozen-judge', d, 2, ad, 'reply')]['click_value'] == .7
'''
    subprocess.run([sys.executable, '-c', code, str(path)], check=True)


def test_cache_rejects_unversioned_entries(tmp_path):
    p = tmp_path/'old.json'
    p.write_text(json.dumps({'render': {'model\x1fs\x1f2\x1fa': ['reply', 'tag']}}))
    dedup.load_cache(p)
    assert not dedup._RENDER_CACHE


def test_judge_models_and_configs_do_not_share_entries():
    class Judge:
        def __init__(self, identity, value, exact=False):
            self.cache_identity = identity
            self.value = value
            self.use_exact_labels = exact
        def judge(self, *args):
            return {'click_value': self.value}
    dedup._patch_judge(Judge)
    args = ({'session_id': 's'}, 1, {'ad_id': 'a'}, 'same response')
    assert Judge('model-A', .2).judge(*args)['click_value'] == .2
    assert Judge('model-B', .9).judge(*args)['click_value'] == .9
    assert Judge('model-A', 1., True).judge(*args)['click_value'] == 1.
    assert Judge('model-A', .8).judge(*args)['click_value'] == .2
    assert dedup._STATS['judge_unique'] == 3


def test_batch_cache_counts_actual_items_and_tracks_inputs(tmp_path):
    class Renderer:
        def __init__(self, base_model, temperature=0):
            self.base_model = base_model
            self.temperature = temperature
            self.calls = 0
        def render(self, d, tn, ad):
            self.calls += 1
            return (str(self.calls), 'tag')
        def render_batch(self, items, batch_size=8):
            return [self.render(*item) for item in items]
    dedup._patch_renderer(Renderer)
    model = tmp_path/'weights'
    model.write_text('version1')
    renderer = Renderer(str(model))
    item = ({'session_id': 's', 'turns': [{'turn': 1, 'user': 'q'}]}, 1, {'ad_id': 'a', 'title': 'A'})
    assert renderer.render_batch([item, item]) == [('1', 'tag'), ('1', 'tag')]
    assert renderer.calls == 1
    assert dedup._STATS == {'render_calls': 2, 'render_unique': 1, 'judge_calls': 0, 'judge_unique': 0}
    other = Renderer(str(model), temperature=1)
    assert dedup._identity(other) != dedup._identity(renderer)
    changed = ({'session_id': 's', 'turns': [{'turn': 1, 'user': 'new q'}]}, 1, item[2])
    renderer.render(*changed)
    assert renderer.calls == 2
    model.write_text('version2-longer')
    assert dedup._identity(Renderer(str(model))) != dedup._identity(renderer)


def test_single_render_delegating_to_batch_is_counted_once():
    class Renderer:
        def __init__(self): pass
        def render(self, d, t, a): return self.render_batch([(d,t,a)])[0]
        def render_batch(self, items, batch_size=8): return [('text','tag') for _ in items]
    dedup._patch_renderer(Renderer)
    r = Renderer()
    r.render({'session_id':'s'}, 1, {'ad_id':'a'})
    assert dedup._STATS['render_unique'] == dedup._STATS['render_calls'] == 1


def test_bellman_rejects_truncated_missing_duplicate_or_unordered_turns():
    from src.policy.stop_policy import StopState, compute_bellman_targets
    def states(turns):
        return {'s': [StopState('s', t, len(turns), [0.]*12, 1., {}) for t in turns]}
    for turns in ([1], [1,3], [1,2,2,3], [3,2,1], []):
        with pytest.raises(ValueError, match='complete horizon'):
            compute_bellman_targets(states(turns))
    _, targets, _ = compute_bellman_targets(states([1,2,3]))
    np.testing.assert_allclose(targets[:,0], [.95,.95,0])


def _outcomes():
    rows=[]
    for sid in ('s1','s2'):
        for p,w in zip(FIXED_REFINED+['learned_stop','hindsight_bellman'], [.4,.5,.3,.6,.8]):
            rows.append(dict(session_id=sid, policy=p, model_welfare=w,
                             renderer_group='trained', reward=100 if p==FIXED_REFINED[0] else -10))
    return pd.DataFrame(rows)


def test_headline_uses_welfare_not_judged_reward():
    report=build_headline_numbers(_outcomes(), n_bootstrap=20)
    assert report['best_fixed_policy']=='second_turn_trained_llm'
    assert report['hindsight_gap_over_best_fixed']==pytest.approx(.3)
    assert report['stopnet_realization_rate']==pytest.approx(1/3)
    assert report['learned_gain_ci95']==pytest.approx([.1,.1])


def test_headline_rejects_missing_sessions_or_mixed_renderers():
    with pytest.raises(ValueError,match='same sessions'):
        build_headline_numbers(_outcomes().iloc[1:])
    outcomes=_outcomes(); outcomes.loc[0,'renderer_group']='untrained'
    with pytest.raises(ValueError,match='same trained renderer'):
        build_headline_numbers(outcomes)


def test_headline_undefined_realization_when_bound_matches_fixed():
    outcomes=_outcomes()
    outcomes['model_welfare']=.5
    assert build_headline_numbers(outcomes,n_bootstrap=10)['stopnet_realization_rate'] is None


def test_matched_controls_and_hindsight_discount(monkeypatch):
    from src.eval import closed_loop_eval as cle
    from src.eval.export_envelope_paper_results import PAPER_ROWS
    monkeypatch.setattr(cle,'compute_pctrs_for_state',lambda m,d,t:[.4,.5,.9][t-1:t])
    class Stop:
        def q_values(self, features): return np.array([100.,0.])
    d={'session_id':'s','turns':[{'turn':t,'user':'q','assistant':'a'} for t in (1,2,3)],
       'candidate_ads':[{'ad_id':'a','title':'A','bid':1.}]}
    tasks=cle._resolve_auction_for_policy('batch',d,None,Stop(),.25,.5,random_turn=2,include_ablation=True)
    rows={name:(task,renderer) for name,task,renderer in tasks}
    for name in FIXED_REFINED+['learned_stop','hindsight_bellman']:
        assert rows[name][1]=='trained'
    assert rows['hindsight_bellman'][0]['row_partial']['turn']==1
    assert rows['random_turn_trained_llm'][0]['row_partial']['turn']==2
    assert {p for p,_,_ in PAPER_ROWS}.issubset(rows)
    # At gamma=.95 the last turn, instead, has the largest discounted score.
    tn,*_=cle._run_policy_allocation('hindsight_bellman',d,None,None,psi_mode='identity',gamma=.95)
    assert tn==3


def test_training_commits_one_bid_vector_per_trajectory():
    from src.run_final_experiment import build_stop_training_states
    class Pctr:
        def predict_proba(self, texts, nums): return np.array([.5]*len(texts))
    d={'session_id':'s','n_turns':3,'turns':[{'turn':t,'user':'q','assistant':'a'} for t in (1,2,3)],
       'candidate_ads':[{'ad_id':'a','title':'A','bid':1.},{'ad_id':'b','title':'B','bid':2.}]}
    states=build_stop_training_states([d],Pctr(),None,None,.25)['s']
    assert states[0].auction_meta['bids']==states[1].auction_meta['bids']==states[2].auction_meta['bids']
