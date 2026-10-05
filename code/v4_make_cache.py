"""Rebuild the causal source forecasts from the licensed raw consumption parquet.

Default configuration reconstructs the V3 recipe. Full regeneration is costly and
was not repeated for V4. Configuration changes require a new cache, never relabel
the verified old CSV with changed base-model settings.
"""
from pathlib import Path
import argparse,json,hashlib
from v2_runtime import configure

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--temp-dir',required=True);ap.add_argument('--project-root')
    a=ap.parse_args();configure(a.temp_dir,a.project_root)
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    df=pd.read_parquet(a.data)
    assert hashlib.sha256(Path(a.data).read_bytes()).hexdigest()==cfg['data_sha256']
    # Official schema: municipality ID, category, month and value; names are bound
    # in the historical reproducible preparation code and checked before use.
    columns=cfg['data_columns']
    for name in columns.values():assert name in df.columns
    df=df.rename(columns={v:k for k,v in columns.items()})
    df['date']=pd.to_datetime(df['date'])
    keys=sorted(set(cfg['panel']['training_ids'])|set(cfg['panel']['evaluation_ids']))
    index=pd.MultiIndex.from_product([keys,cfg['categories']],names=['territory_id','category'])
    months=pd.date_range('2023-01-01',periods=24,freq='MS')
    df['date']=df['date'].dt.to_period('M').dt.to_timestamp()
    values=df.pivot(index=['territory_id','category'],columns='date',values='value').reindex(index=index,columns=months)
    profile_rows=df[df.territory_id.isin(cfg['calendar_profile_ids']) & (df.date.dt.year==2023)]
    profile_index=pd.MultiIndex.from_product([cfg['calendar_profile_ids'],cfg['categories']],names=index.names)
    profile_y=profile_rows.pivot(index=index.names,columns='date',values='value').reindex(index=profile_index,columns=months[:12]).to_numpy()
    profiles={}
    for cat in cfg['categories']:
        y=profile_y[np.array([key[1]==cat for key in profile_index])]
        assert np.isfinite(y).all()
        logged=np.log1p(y);design=np.column_stack([np.ones(12),np.arange(12)])
        residual=logged-(design@np.linalg.lstsq(design,logged.T,rcond=None)[0]).T
        profile=np.median(residual,axis=0);profiles[cat]=profile-profile.mean()
    rows=[]
    model=cfg['base_model'];shared=model['shared']
    for (territory,cat),yy in values.iterrows():
        y=yy.to_numpy(dtype=float)
        if not np.isfinite(y[:12]).all():continue
        scale=max(float(np.median(abs(y[:6]))),1.)
        mase=float(np.mean(abs(np.diff(y[:12]))))
        for n in cfg['origins']:
            if not np.isfinite(y[:n]).all():continue
            known=y[:n];dates=pd.date_range('2023-01-31',periods=n,freq='ME')
            future=pd.date_range(dates[-1],periods=13,freq='ME')[1:]
            profile=profiles[cat]*shared['calendar_weight']
            adjusted=np.log1p(known)-profile[np.arange(n)%12]
            window=shared['local_window']
            slope=np.polyfit(np.arange(window),adjusted[-window:],1)[0]
            trend=np.cumsum(shared['damping']**np.arange(1,13))
            predictions={'LastValue':np.repeat(known[-1],12),
                         'SharedSeasonal':np.expm1(adjusted[-1]+slope*trend+profile[np.arange(n,n+12)%12])}
            for method in ('Prophet','ProphetYearly'):
                engine=Prophet(**model['prophet'])
                if method=='ProphetYearly':engine.add_seasonality('yearly',**model['prophet_yearly'])
                engine.fit(pd.DataFrame({'ds':dates,'y':known}))
                predictions[method]=engine.predict(pd.DataFrame({'ds':future})).yhat.to_numpy()
            predictions['EqualBlend']=(predictions['LastValue']+predictions['Prophet']+predictions['SharedSeasonal'])/3
            for h in cfg['horizons']:
                target=n+h-1
                if target>23 or not np.isfinite(y[target]):continue
                for method,prediction in predictions.items():
                    rows.append(dict(split='evaluation',method=method,territory_id=int(territory),category=cat,prefix_n=n,horizon=h,target_index=target,
                        actual=float(y[target]),prediction=max(float(prediction[h-1]),0.),reference_scale=scale,mase_scale=mase))
    output=Path(a.output)
    assert not output.exists()
    pd.DataFrame(rows).to_csv(output,index=False)
    output.with_suffix('.recipe.json').write_text(json.dumps(dict(base_model=cfg['base_model'],calendar_profile_ids=cfg['calendar_profile_ids'],
        origins=cfg['origins'],horizons=cfg['horizons'],categories=cfg['categories'],data_columns=cfg['data_columns'],
        cache_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_data_sha256=hashlib.sha256(Path(a.data).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
