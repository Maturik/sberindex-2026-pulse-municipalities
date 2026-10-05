"""Causal convex forecast combination, with configurable penalties and prior."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

KEY = ['territory_id','category','prefix_n','horizon','target_index']

def wide_cache(frame, components):
    """Reject unequal method pairs instead of silently intersecting them."""
    assert not frame.duplicated(KEY+['method']).any()
    methods = sorted(frame.method.unique())
    expected = frame[frame.method==methods[0]][KEY].sort_values(KEY).reset_index(drop=True)
    for method in methods:
        g = frame[frame.method==method].sort_values(KEY).reset_index(drop=True)
        assert g[KEY].equals(expected), 'Unequal source pairs '+method
    for col in ('actual','reference_scale','mase_scale'):
        assert frame.groupby(KEY)[col].nunique(dropna=False).max()==1
    metadata = frame[frame.method==methods[0]].drop(columns=['method','prediction','split']).set_index(KEY)
    predictions = frame.pivot(index=KEY,columns='method',values='prediction')
    result = metadata.join(predictions).reset_index().sort_values(KEY).reset_index(drop=True)
    assert set(components).issubset(result.columns)
    assert np.isfinite(result[components+['actual','reference_scale']].to_numpy()).all()
    assert (result[components].to_numpy()>=0).all()
    return result, methods

def learn_weights(table, training_ids, n, horizon, config):
    """A label is usable only after its target month has been observed."""
    components = config['components']
    prior = np.asarray(config['prior'],dtype=float)
    assert len(prior)==len(components) and (prior>=0).all() and abs(prior.sum()-1)<1e-12
    g = table[(table.territory_id.isin(training_ids)) & (table.horizon==horizon)
              & (table.target_index<n) & (table.prefix_n<n)].copy()
    record = dict(prefix_n=int(n),horizon=int(horizon),training_pairs=len(g),training_series=0,
                  training_territories=0,max_training_target=None,solver_status='PRIOR_NO_MATURE_LABELS')
    if g.empty:
        return prior, record
    assert (g.target_index<n).all()
    count = g.groupby(['territory_id','category']).actual.transform('size').to_numpy()
    series = g.groupby(['territory_id','category']).ngroups
    pair_weight = 1.0/(series*count)
    x,y = g[components].to_numpy(),g.actual.to_numpy()
    baseline = float(np.dot(pair_weight,np.abs(y-x@prior)))
    scale = max(baseline,float(config['loss_scale_floor']))
    m,k = x.shape
    zero = csr_matrix((m,k))
    design = csr_matrix(x)
    identity = eye(m,format='csr')
    # e bounds |y-Xw|; u bounds |w-prior|. All w,e,u are nonnegative.
    constraints = vstack([hstack([design,-identity,zero]),hstack([-design,-identity,zero]),
                          hstack([eye(k),csr_matrix((k,m)),-eye(k)]),
                          hstack([-eye(k),csr_matrix((k,m)),-eye(k)])],format='csr')
    upper = np.concatenate([y,-y,prior,-prior])
    objective = np.concatenate([np.zeros(k),pair_weight/scale,np.full(k,float(config['l1_penalty']))])
    equality = csr_matrix(([1.]*k,([0]*k,list(range(k)))),shape=(1,k+m+k))
    tolerance=float(config.get('feasibility_tolerance',1e-7))
    solved = linprog(objective,A_ub=constraints,b_ub=upper,A_eq=equality,b_eq=[1.],
                     bounds=(0,None),method='highs',options={'dual_feasibility_tolerance':tolerance,'primal_feasibility_tolerance':tolerance})
    if not solved.success:
        raise RuntimeError('Convex solver failed: '+solved.message)
    w = solved.x[:k]
    assert np.isfinite(w).all() and w.min()>=-tolerance and abs(w.sum()-1)<=tolerance
    w = np.maximum(w,0.)
    w /= w.sum()
    value = float(np.dot(pair_weight,np.abs(y-x@w))/scale+float(config['l1_penalty'])*np.abs(w-prior).sum())
    assert abs(value-solved.fun)<=1e-5
    record.update(training_series=int(series),training_territories=int(g.territory_id.nunique()),
                  max_training_target=int(g.target_index.max()),solver_status='OPTIMAL',
                  training_uniform_native_MAE=baseline,loss_scale=scale,objective=value)
    return w, record

def forecast_cache(table, training_ids, protocol):
    """Apply the same origin-specific weights to train and evaluation territories."""
    config = protocol['ensemble']
    predictions,records = [],[]
    for n in protocol['origins']:
        for h in protocol['horizons']:
            if n+h-1>23: continue
            w,record = learn_weights(table,training_ids,n,h,config)
            record.update(dict(zip(config['components'],w.tolist())))
            records.append(record)
            g=table[(table.prefix_n==n)&(table.horizon==h)].copy()
            if g.empty: continue
            g['OnlineHorizonBlend']=g[config['components']].to_numpy()@w
            g['ClippedEqualBlend']=g[config['components']].to_numpy().mean(axis=1)
            predictions.append(g)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(records)
