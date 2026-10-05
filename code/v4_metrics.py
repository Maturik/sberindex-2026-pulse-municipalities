"""Explicit equal-series/equal-horizon arithmetic and paired conditional intervals."""
import numpy as np
import pandas as pd

def metrics(frame):
    d=frame.copy()
    d['error']=d.prediction-d.actual
    d['loss']=d.error.abs()
    d['scaled']=d.loss/d.reference_scale
    records=[]
    for (period,method,h),g in d.groupby(['period','method','horizon']):
        records.append(dict(period=period,method=method,horizon=int(h),
            native_macro_MAE=float(g.groupby(['territory_id','category']).loss.mean().mean()),
            macro_scaled_MAE=float(g.groupby(['territory_id','category']).scaled.mean().mean()),
            bias=float(g.error.mean()),pairs=len(g),series=g.groupby(['territory_id','category']).ngroups,
            territories=int(g.territory_id.nunique()),origins=int(g.prefix_n.nunique())))
    return pd.DataFrame(records)

def paired_bootstrap(frame,comparators,config):
    results={}
    for scaled in (False,True):
        d=frame.copy()
        d['loss']=(d.prediction-d.actual).abs()/(d.reference_scale if scaled else 1.)
        table=d.groupby(['territory_id','category','horizon','method']).loss.mean().unstack('method')
        for comparator in comparators:
            delta=table[comparator]-table['OnlineHorizonBlend']
            cluster=delta.groupby(level=[0,2]).agg(['sum','count']).unstack(1).fillna(0.)
            sums=cluster['sum'][[1,3,6]].to_numpy();counts=cluster['count'][[1,3,6]].to_numpy()
            point=float((sums.sum(axis=0)/counts.sum(axis=0)).mean())
            rng=np.random.default_rng(config['seed'])
            draws=[]
            for _ in range(config['repetitions']):
                ix=rng.integers(0,len(sums),len(sums))
                draws.append(float((sums[ix].sum(axis=0)/counts[ix].sum(axis=0)).mean()))
            results[('scaled_' if scaled else 'native_')+comparator]=dict(gain_comparator_minus_candidate=point,
                CI95=np.quantile(draws,config['quantiles']).tolist(),territories=len(sums),repetitions=config['repetitions'],seed=config['seed'])
    return dict(comparisons=results,limitation=config['cluster'])
