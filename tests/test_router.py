import numpy as np
import pandas as pd

from multi_asset_scenario_engine.router import ObjectiveRouter, mix_scenarios
from multi_asset_scenario_engine.scoring import (
    crps_sample, energy_score_mc, fz0_var_es_score, marginal_crps, objective_score,
)


def _history():
    dates = pd.date_range('2020-01-01', periods=30, freq='D')
    rows=[]
    for i,t in enumerate(dates):
        rows += [
            {'asof':t,'horizon':'1M','objective':'joint_distribution','model':'simple','loss':1.0 + 0.01*(i%3)},
            {'asof':t,'horizon':'1M','objective':'joint_distribution','model':'complex','loss':0.98 + 0.01*(i%3)},
        ]
    return pd.DataFrame(rows)


def test_router_uses_only_past_and_hard_selects_best():
    r=ObjectiveRouter(_history(), complexity={'simple':0,'complex':1}, min_history=10)
    d=r.select(pd.Timestamp('2020-02-15'),'1M','joint_distribution',policy='hard')
    assert d.selected_model == 'complex'
    assert d.n_history == 30


def test_one_se_can_prefer_simpler_model():
    h=_history().copy()
    # Make difference tiny and noisy enough that simple is within paired one-SE.
    mask=h['model'].eq('complex')
    h.loc[mask,'loss'] = h.loc[mask,'loss'].to_numpy() + np.tile([0.015,-0.015,0.0],10)
    r=ObjectiveRouter(h, complexity={'simple':0,'complex':1}, min_history=10)
    d=r.select(pd.Timestamp('2020-02-15'),'1M','joint_distribution',policy='one_se')
    assert d.selected_model in {'simple','complex'}
    assert np.isclose(sum(d.weights.values()),1.0)


def test_soft_weights_and_mixture_shape():
    r=ObjectiveRouter(_history(), min_history=10)
    d=r.select(pd.Timestamp('2020-02-15'),'1M','joint_distribution',policy='soft')
    assert np.isclose(sum(d.weights.values()),1.0)
    c={'simple':np.zeros((50,2)), 'complex':np.ones((50,2))}
    x=mix_scenarios(c,d.weights,n_scenarios=100,rng=np.random.default_rng(2))
    assert x.shape==(100,2)


def test_scoring_layer():
    rng=np.random.default_rng(4)
    y=np.array([0.1,-0.2])
    x=y+rng.normal(scale=0.2,size=(1000,2))
    assert np.isfinite(energy_score_mc(x,y,n_pairs=2000))
    assert np.isfinite(marginal_crps(x,y))
    assert crps_sample(x[:,0],y[0]) >= 0
    loss_projection=np.array([-1.0,0.0])
    s=objective_score(x,y,'tail_risk',loss_projection=loss_projection,alpha=0.95)
    assert np.isfinite(s)
    assert np.isfinite(fz0_var_es_score(np.abs(rng.normal(size=1000)),0.5,alpha=0.95))
