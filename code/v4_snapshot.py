"""Apply the same V4 algorithm at the final December2024 snapshot.

This is deterministic deployment inference, not a new model comparison. All
months2025 are unknown to the supplied dataset; output is historical projection.
"""
from pathlib import Path
import argparse,hashlib,json,logging,time
from concurrent.futures import ThreadPoolExecutor
from v2_runtime import configure

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--output-dir',required=True);ap.add_argument('--temp-dir',required=True);ap.add_argument('--project-root')
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args();configure(args.temp_dir,args.project_root)
    import numpy as np
    import pandas as pd
    from prophet import Prophet
    from v4_ensemble import wide_cache,learn_weights
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert digest(args.data)==cfg['data_sha256'] and digest(args.cache)==cfg['source_cache']['sha256']
    output=Path(args.output_dir);assert not output.exists();output.mkdir()
    claim=dict(status='CLAIMED_ONE_APPLICATION',config_sha256=digest(args.config),code_sha256=digest(__file__),
        cache_sha256=digest(args.cache),data_sha256=digest(args.data),algorithm='Same frozen V4 ensemble at prefix_n24',
        evaluation_or_model_selection=False,origin='2024-12',horizons=cfg['horizons'])
    (output/'CLAIM.json').write_text(json.dumps(claim,indent=2),encoding='utf-8')
    start=time.monotonic()
    try:
        table,_=wide_cache(pd.read_csv(args.cache),cfg['ensemble']['components'])
        weights,records={},[]
        for h in cfg['horizons']:
            w,record=learn_weights(table,cfg['panel']['training_ids'],24,h,cfg['ensemble'])
            weights[h]=w;record.update(dict(zip(cfg['ensemble']['components'],w.tolist())));records.append(record)
        pd.DataFrame(records).to_csv(output/'WEIGHTS_ASOF_DEC2024.csv',index=False)
        df=pd.read_parquet(args.data)
        df['month']=pd.to_datetime(df.date).dt.to_period('M').dt.to_timestamp('M')
        months=pd.date_range('2023-01-31',periods=24,freq='ME')
        wide=df.pivot(index=['territory_id','category'],columns='month',values='value').reindex(columns=months)
        profile=wide[wide.index.get_level_values(0).isin(cfg['calendar_profile_ids'])].iloc[:,:12]
        y=profile.to_numpy(float);assert np.isfinite(y).all()
        design=np.column_stack([np.ones(12),np.arange(12)]);log=np.log1p(y)
        residual=log-(design@np.linalg.lstsq(design,log.T,rcond=None)[0]).T
        profiles={}
        for category in cfg['categories']:
            v=np.median(residual[profile.index.get_level_values(1)==category],axis=0);profiles[category]=v-v.mean()
        ready=wide[wide.notna().all(axis=1)]
        excluded=wide[~wide.notna().all(axis=1)]
        logging.getLogger('cmdstanpy').disabled=True;logging.getLogger('prophet').setLevel(logging.ERROR)
        settings=cfg['base_model'];shared=settings['shared'];future=pd.date_range('2025-01-31',periods=12,freq='ME')
        def one(item):
            key,row=item;known=row.to_numpy(float)
            engine=Prophet(**settings['prophet']);engine.fit(pd.DataFrame({'ds':months,'y':known}))
            base=np.maximum(engine.predict(pd.DataFrame({'ds':future})).yhat.to_numpy(),0.)
            calendar=profiles[key[1]]*shared['calendar_weight'];adjusted=np.log1p(known)-calendar[np.arange(24)%12]
            window=shared['local_window'];slope=np.polyfit(np.arange(window),adjusted[-window:],1)[0]
            seasonal=np.maximum(np.expm1(adjusted[-1]+slope*np.cumsum(shared['damping']**np.arange(1,13))+calendar),0.)
            records=[]
            for h in cfg['horizons']:
                components=np.asarray([known[-1],base[h-1],seasonal[h-1]])
                prediction=float(components@weights[h]);assert np.isfinite(prediction) and prediction>=0
                records.append(dict(territory_id=int(key[0]),category=key[1],origin='2024-12',horizon=h,
                    target=future[h-1].strftime('%Y-%m'),method='OnlineHorizonBlend',prediction=prediction,
                    units='native value; currency unconfirmed',forecast_kind='historical snapshot projection; not live2026 forecast'))
            return records
        rows=[]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i,result in enumerate(pool.map(one,ready.iterrows()),1):
                rows.extend(result)
                if i%1000==0:print(json.dumps(dict(stage='SNAPSHOT_PROJECTION',series=i,total=len(ready),seconds=time.monotonic()-start)),flush=True)
        pd.DataFrame(rows).to_csv(output/'FORECAST_2025_H1_H3_H6.csv',index=False)
        excluded.index.to_frame(index=False).to_csv(output/'EXCLUDED.csv',index=False)
        terminal=dict(status='SNAPSHOT_APPLICATION_EXECUTED',series=len(ready),territories=ready.index.get_level_values(0).nunique(),
            rows=len(rows),excluded_series=len(excluded),prophet_fits=len(ready),ensemble_fits=3,elapsed_seconds=time.monotonic()-start,
            forecast_sha256=digest(output/'FORECAST_2025_H1_H3_H6.csv'),origin='2024-12',targets=['2025-01','2025-03','2025-06'],
            evaluation_or_model_selection=False,independent_verification=False,Cloud='NOT_RUN')
        (output/'TERMINAL.json').write_text(json.dumps(terminal,indent=2),encoding='utf-8')
        print(json.dumps(terminal),flush=True)
    except Exception as e:
        if not (output/'TERMINAL.json').exists():(output/'TERMINAL.json').write_text(json.dumps(dict(status='FAILED_NO_RETRY',error=repr(e))),encoding='utf-8')
        raise
if __name__=='__main__':main()
