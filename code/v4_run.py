"""Execute one frozen causal ensemble check using a verified V3 forecast cache."""
from pathlib import Path
from datetime import datetime,timezone
import argparse, hashlib, json, sys, time, traceback, importlib.metadata
from v2_runtime import configure

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def save(p,obj):
    with Path(p).open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',required=True);ap.add_argument('--temp-dir',required=True);ap.add_argument('--project-root')
    a=ap.parse_args();configure(a.temp_dir,a.project_root)
    import numpy as np
    import pandas as pd
    from v4_ensemble import wide_cache,forecast_cache,KEY
    from v4_metrics import metrics,paired_bootstrap
    run=Path(a.run_dir).resolve();out=run/'output'
    frozen=json.loads((run/'FREEZE_MANIFEST.json').read_text(encoding='utf-8'))
    required={'PROTOCOL.json','input/FORECAST_CACHE.csv','input/DETECTOR_SCALE.json'}
    assert required.issubset(frozen['members']), 'Required freeze members missing'
    for name,digest in frozen['members'].items(): assert sha(run/name)==digest,'Freeze mismatch '+name
    for name in ('v4_run.py','v4_ensemble.py','v4_metrics.py','v2_runtime.py'):
        assert sha(Path(__file__).parent/name)==frozen['members']['code/'+name]
    p=json.loads((run/'PROTOCOL.json').read_text(encoding='utf-8'))
    versions={name:importlib.metadata.version(name) for name in p['package_versions']}
    assert versions==p['package_versions'], 'Runtime version mismatch'
    recipe=p['source_cache']['recipe']
    assert recipe['base_model']==p['base_model'] and recipe['calendar_profile_ids']==p['calendar_profile_ids']
    assert recipe['data_sha256']==p['data_sha256'] and recipe['data_columns']==p['data_columns']
    assert recipe['origins']==p['origins'] and recipe['horizons']==p['horizons'] and recipe['categories']==p['categories']
    assert p['horizons']==[1,3,6] and p['origins']==list(range(12,24)), 'Frozen evaluation design, not a tunable hyperparameter'
    assert sha(run/'input/DETECTOR_SCALE.json')==p['detector_scale_sha256']
    sigma=json.loads((run/'input/DETECTOR_SCALE.json').read_text(encoding='utf-8'))
    assert set(sigma)==set(p['categories']) and all(np.isfinite(v) and v>0 for v in sigma.values()), 'Invalid category sigma'
    assert sha(run/'input/FORECAST_CACHE.csv')==p['source_cache']['sha256']
    assert out.is_dir() and not any(out.iterdir())
    assert not (run/'receipts/BURNED.json').exists() and not (run/'receipts/TERMINAL.json').exists()
    train=set(p['panel']['training_ids']);evaluation=set(p['panel']['evaluation_ids'])
    assert len(train)==len(evaluation)==128 and not train&evaluation
    ranked=sorted(train|evaluation,key=lambda i:hashlib.sha256(('SBER_V4_20261005:'+str(i)).encode()).hexdigest())
    assert ranked[:128]==p['panel']['training_ids'] and ranked[128:]==p['panel']['evaluation_ids']
    assert not set(p['calendar_profile_ids'])&(train|evaluation)
    frame=pd.read_csv(run/'input/FORECAST_CACHE.csv')
    assert set(frame.territory_id).issubset(train|evaluation)
    assert (frame.target_index==frame.prefix_n+frame.horizon-1).all()
    assert frame.horizon.isin(p['horizons']).all()
    table,old_methods=wide_cache(frame,p['ensemble']['components'])
    start=time.monotonic()
    save(run/'receipts/BURNED.json',dict(at=now(),freeze_sha256=sha(run/'FREEZE_MANIFEST.json'),status='ONE_RUN_CLAIMED'))
    try:
        pred,weights=forecast_cache(table,train,p)
        weights.to_csv(out/'WEIGHTS_BY_ORIGIN.csv',index=False)
        methods=old_methods+['ClippedEqualBlend','OnlineHorizonBlend']
        assert np.isfinite(pred[methods].to_numpy()).all() and (pred[methods].to_numpy()>=0).all()
        long=pred[pred.territory_id.isin(evaluation)].melt(id_vars=KEY+['actual','reference_scale','mase_scale'],value_vars=methods,
            var_name='method',value_name='prediction')
        long.to_csv(out/'EVALUATION_PREDICTIONS.csv',index=False)
        periods=[]
        for label,indices in [('All2024',list(range(12,24))),('JanJun',list(range(12,18))),('JulDec',p['primary_target_indices'])]:
            g=long[long.target_index.isin(indices)].copy();g['period']=label;periods.append(g)
        period_frame=pd.concat(periods,ignore_index=True)
        mt=metrics(period_frame);mt.to_csv(out/'METRICS.csv',index=False)
        category=[]
        for cat in p['categories']:
            g=metrics(period_frame[period_frame.category==cat]);g['category']=cat;category.append(g)
        pd.concat(category).to_csv(out/'CATEGORY_METRICS.csv',index=False)
        primary=period_frame[period_frame.period=='JulDec']
        boot=paired_bootstrap(primary,p['comparisons'],p['bootstrap']);save(out/'CONDITIONAL_BOOTSTRAP.json',boot)
        score=mt[mt.period=='JulDec'].groupby('method').native_macro_MAE.mean().to_dict()
        scaled=mt[mt.period=='JulDec'].groupby('method').macro_scaled_MAE.mean().to_dict()
        h1=pred[pred.horizon==1].copy()
        h1['z']=(h1.actual-h1.OnlineHorizonBlend)/h1.reference_scale/h1.category.map(sigma)
        assert np.isfinite(h1.z).all(), 'Nonfinite standardized residual'
        dc=p['detector'];grid=dc['threshold_grid']
        candidates=np.unique(np.concatenate([grid['include'],np.geomspace(grid['min'],grid['max'],grid['count'])]))
        cal=h1[h1.territory_id.isin(train)&h1.target_index.isin(dc['calibration_target_indices'])]
        threshold=next(float(t) for t in candidates if float((cal.z.abs()>t).mean())<=dc['calibration_alarm_budget'])
        save(out/'DETECTOR_CALIBRATION.json',dict(threshold=threshold,alarm_rate=float((cal.z.abs()>threshold).mean()),pairs=len(cal),budget=dc['calibration_alarm_budget'],labels='NOT_FPR'))
        alarms=h1[h1.territory_id.isin(evaluation)&h1.target_index.isin(dc['evaluation_target_indices'])].copy()
        alarms['alarm']=(alarms.z.abs()>threshold).astype(int)
        alarms[KEY+['actual','OnlineHorizonBlend','z','alarm']].to_csv(out/'DETECTOR_ALARMS.csv',index=False)
        # Complete four-month residual windows; injected disturbances never alter forecasts.
        sc=p['synthetic'];synthetic=[]
        window=h1[h1.territory_id.isin(evaluation)&h1.target_index.isin(range(sc['start_month_index'],sc['start_month_index']+sc['length']))]
        for key,g in window.groupby(['territory_id','category']):
            g=g.sort_values('target_index')
            if len(g)!=sc['length']:continue
            z=g.z.to_numpy();base=abs(z)>threshold
            for kind in sc['kinds']:
                for sign in sc['signs']:
                    for magnitude in sc['magnitudes']:
                        disturbance=np.ones(sc['length'])
                        if kind=='ramp':disturbance=np.arange(1,sc['length']+1)/sc['length']
                        elif kind=='spike':disturbance[1:]=0.
                        changed=abs(z+sign*magnitude*disturbance)>threshold
                        additional=np.flatnonzero(changed&~base)
                        synthetic.append(dict(territory_id=key[0],category=key[1],kind=kind,sign=sign,magnitude=magnitude,
                            additional_hit=int(len(additional)>0),penalized_delay=int(additional[0]) if len(additional) else sc['length']))
        sf=pd.DataFrame(synthetic);sf.to_csv(out/'SYNTHETIC_CASES.csv',index=False)
        sf.groupby('kind').agg(additional_hit=('additional_hit','mean'),penalized_delay=('penalized_delay','mean'),cases=('kind','size')).reset_index().to_csv(out/'SYNTHETIC_SUMMARY.csv',index=False)
        success=all(score['OnlineHorizonBlend']<score[m] and boot['comparisons']['native_'+m]['CI95'][0]>0 for m in ('Prophet','ProphetYearly'))
        incremental=boot['comparisons']['native_ClippedEqualBlend']['CI95'][0]>0
        result=dict(by='Лея',at=now(),status='EXECUTED',candidate=p['candidate'],scientific_outcome='CONDITIONAL_REUSED_DATA_ADVANTAGE' if success else 'NO_CONFIRMED_ADVANTAGE',
            incremental_learning_advantage=bool(incremental),primary_native_scores=score,secondary_scaled_scores=scaled,
            coverage=dict(planned_evaluation_territories=len(evaluation),actual_evaluation_territories=int(long.territory_id.nunique()),
                forecast_pairs_per_method=int(len(long)/len(methods)),primary_pairs_per_method=int(len(primary)/len(methods)),
                h6_warmup='Five of six JulDec h6 forecasts use uniform prior; only December has mature h6 labels'),
            detector=dict(threshold=threshold,calibration_alarm_rate=float((cal.z.abs()>threshold).mean()),calibration_pairs=len(cal),
                evaluation_alarm_rate=float(alarms.alarm.mean()),alarms=int(alarms.alarm.sum()),evaluation_pairs=len(alarms),not_FPR=True),
            counters=dict(ensemble_LP_fits=int((weights.solver_status=='OPTIMAL').sum()),prophet_fits=0,core_fits=0,foundation_fits=0),
            runtime_versions=versions,elapsed_seconds=time.monotonic()-start,Cloud='NOT_RUN',CLAUDE_REVIEW='NOT_RUN',external_delivery_sends=0,
            decision='Keep preassigned OnlineHorizonBlend; no test-based rescue or selection; all2024 outcomes previously known.',
            stop_go='GO_AUTHORIZED_V4_PACKAGE_REVIEW',handoff='Arithmetic checks and actual final Claude review; build and seal local research package.')
        save(out/'RESULT.json',result)
        save(run/'receipts/TERMINAL.json',dict(**result,freeze_sha256=sha(run/'FREEZE_MANIFEST.json'),output_sha256={f.name:sha(f) for f in out.iterdir()}))
        print(json.dumps(result,ensure_ascii=True),flush=True)
    except Exception as error:
        if not (run/'receipts/TERMINAL.json').exists():
            save(run/'receipts/TERMINAL.json',dict(at=now(),status='FAILED_NO_RETRY',error=repr(error),traceback=traceback.format_exc(),freeze_sha256=sha(run/'FREEZE_MANIFEST.json')))
        raise
if __name__=='__main__':main()
