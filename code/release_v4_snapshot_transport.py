"""Prepare an independently regenerated cache for unchanged V4 snapshot code.

Strict raw-data and recipe checks remain. A new cache is a new transport/input
identity, never an assertion of numerical equality to the original cache.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys,subprocess
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for flag in ['config','cache','data','output-dir','temp-dir']:ap.add_argument('--'+flag,required=True)
    ap.add_argument('--project-root');ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args()
    from v2_runtime import configure
    configure(a.temp_dir,a.project_root)
    cfgpath=Path(a.config).resolve();cache=Path(a.cache).resolve();data=Path(a.data).resolve()
    cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    assert digest(data)==cfg['data_sha256'],'Different raw bytes: STOP; new provenance/input protocol required.'
    assert cfg['horizons']==[1,3,6],'No implicit h12 extension.'
    original=cfg['source_cache']['sha256'];cache_sha=digest(cache);recipe_sha=None
    if cache_sha!=original:
        recipefile=cache.with_suffix('.recipe.json')
        recipe=json.loads(recipefile.read_text(encoding='utf-8'));recipe_sha=digest(recipefile)
        assert recipe['cache_sha256']==cache_sha
        assert recipe['source_data_sha256']==cfg['data_sha256']
        for key in ['base_model','calendar_profile_ids','origins','horizons','categories','data_columns']:
            assert recipe[key]==cfg[key],f'Recipe mismatch: {key}'
        cfg['source_cache']['sha256']=cache_sha
        cfg['source_cache']['replication_cache']='Different recipe-backed cache bytes; independent regeneration and numerical equality NOT_ASSERTED by this transport.'
    else:
        expected={k:cfg[k] for k in ['base_model','calendar_profile_ids','data_sha256','data_columns','origins','horizons','categories']}
        assert cfg['source_cache']['recipe']==expected
    out=Path(a.output_dir).resolve();assert not out.exists();out.mkdir(parents=True)
    newcfg=out/'SNAPSHOT_INPUT_CONFIG.json';newcfg.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    receipt=dict(by='Лея',at=datetime.now(timezone.utc).isoformat(),status='TRANSPORT_PREPARED_NO_MODEL_FITS',
        original_config_sha256=digest(cfgpath),transport_config_sha256=digest(newcfg),data_sha256=digest(data),
        original_cache_sha256=original,supplied_cache_sha256=cache_sha,recipe_sha256=recipe_sha,
        different_cache_bytes=cache_sha!=original,independent_regeneration='NOT_VERIFIED_BY_TRANSPORT',exact_numerical_replication='NOT_ASSERTED',model_fits=0,
        native_snapshot_code_sha256=digest(Path(__file__).parent/'v4_snapshot.py'),Cloud='NOT_RUN')
    (out/'TRANSPORT_PREPARED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.prepare_only:
        print('TRANSPORT_PREPARED_NO_MODEL_FITS');return
    cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).parent/'v4_snapshot.py'),
         '--config',str(newcfg),'--cache',str(cache),'--data',str(data),'--output-dir',str(out/'projection'),
         '--temp-dir',a.temp_dir,'--workers',str(a.workers)]
    if a.project_root:cmd+=['--project-root',a.project_root]
    raise SystemExit(subprocess.run(cmd).returncode)
if __name__=='__main__':main()
