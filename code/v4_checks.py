"""Artificial causal/configuration checks. Never consumes competition outcomes."""
import numpy as np
import pandas as pd
from v4_ensemble import learn_weights

def artificial_checks():
    cfg=dict(components=['LastValue','Prophet','SharedSeasonal'],prior=[1/3]*3,l1_penalty=.05,loss_scale_floor=1.)
    fixture=pd.DataFrame([dict(territory_id=i,category='fixture',prefix_n=t,horizon=1,target_index=t,actual=10.,
        LastValue=0.,Prophet=10.,SharedSeasonal=20.) for i in range(3) for t in range(8,16)])
    before,rec=learn_weights(fixture,{0,1},12,1,cfg)
    changed=fixture.copy();changed.loc[changed.target_index>=12,'actual']=999999.
    after,_=learn_weights(changed,{0,1},12,1,cfg)
    assert np.array_equal(before,after), 'Future target changed causal weights'
    # Non-symmetric forecast errors make penalty effects visible.
    fixture['SharedSeasonal']=40.
    low,_=learn_weights(fixture,{0,1},12,1,cfg)
    high,_=learn_weights(fixture,{0,1},12,1,dict(cfg,l1_penalty=100.))
    assert np.linalg.norm(low-high)>.1
    assert np.allclose(high,[1/3]*3)
    empty,_=learn_weights(fixture,{0,1},8,6,cfg)
    assert np.allclose(empty,[1/3]*3)
    outsider=fixture.copy();outsider.loc[outsider.territory_id==2,'actual']=999999.
    outsider_w,_=learn_weights(outsider,{0,1},12,1,cfg)
    assert np.array_equal(low,outsider_w), 'Evaluation labels changed training'
    return dict(status='ARTIFICIAL_CAUSAL_CONFIG_CHECKS_PASSED',future_invariance=True,evaluation_territory_invariance=True,
        penalty_changes_actual_weights=True,empty_history_uses_prior=True,competition_rows_used=0,actual_low_penalty_weights=low.tolist(),
        actual_high_penalty_weights=high.tolist())
if __name__=='__main__':
    import json
    print(json.dumps(artificial_checks()))
