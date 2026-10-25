import numpy as np
import pandas as pd
from multi_asset_scenario_engine.duration import precompute_duration_aggregates, pit_to_normal, rank_map_to_duration
from multi_asset_scenario_engine.scoring import energy_score, variogram_score
from multi_asset_scenario_engine.dependence import fit_hierarchical_dependence, sample_hierarchical_dependence

def test_duration_aggregates_are_additive():
    idx=pd.date_range("2020-01-01",periods=6,freq="D")
    x=pd.DataFrame({"a":[1,2,3,4,5,6],"b":[2,1,0,-1,-2,-3]},index=idx,dtype=float)
    agg=precompute_duration_aggregates(x,3)
    assert np.allclose(agg[3].iloc[0].to_numpy(),x.iloc[:3].sum().to_numpy())

def test_pit_and_rank_map_shapes():
    hist=np.column_stack([np.linspace(-2,2,101),np.linspace(-1,3,101)])
    z=pit_to_normal([[0.0,1.0]],np.sort(hist,axis=0))
    assert z.shape==(1,2)
    draws=np.array([[-1.0,0.5],[0.2,-0.1],[1.4,1.1]])
    mapped=rank_map_to_duration(draws,np.sort(hist,axis=0))
    assert mapped.shape==draws.shape and np.isfinite(mapped).all()

def test_proper_scores_are_finite():
    rng=np.random.default_rng(1)
    y=np.array([0.1,-0.2,0.05])
    close=y+rng.normal(scale=0.02,size=(500,3))
    far=y+2.0+rng.normal(scale=0.02,size=(500,3))
    assert energy_score(close,y)<energy_score(far,y)
    vs,_=variogram_score(close,y)
    assert np.isfinite(vs)

def test_hierarchy_fit_and_draw_shape():
    rng=np.random.default_rng(7)
    g=rng.normal(size=(250,1))
    x=np.hstack([g+0.3*rng.normal(size=(250,2)),-0.7*g+0.4*rng.normal(size=(250,2))])
    labels=np.array(["equity","equity","rates","rates"],dtype=object)
    fit=fit_hierarchical_dependence(x,labels,max_global=2)
    draws=sample_hierarchical_dependence(fit,100,rng=np.random.default_rng(8))
    assert draws.shape==(100,4)
    assert 1<=fit["k_global"]<=2
