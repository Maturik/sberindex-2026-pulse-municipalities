"""Prepare a new independent V4 reproduction; never overwrite a burned run."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
from v2_runtime import configure

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--sigma',required=True)
    ap.add_argument('--run-dir',required=True);ap.add_argument('--temp-dir',required=True);ap.add_argument('--project-root')
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args();configure(args.temp_dir,args.project_root)
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    cache=Path(args.cache).resolve();sigma=Path(args.sigma).resolve()
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(cache)!=config['source_cache']['sha256']:
        recipe=json.loads(cache.with_suffix('.recipe.json').read_text(encoding='utf-8'))
        assert recipe['cache_sha256']==digest(cache)
        assert recipe['base_model']==config['base_model'] and recipe['calendar_profile_ids']==config['calendar_profile_ids']
        assert all(recipe[key]==config[key] for key in ('origins','horizons','categories','data_columns'))
        assert recipe['source_data_sha256']==config['data_sha256']
        config['source_cache']['sha256']=digest(cache)
        config['source_cache']['recipe']={key:config[key] for key in ('base_model','calendar_profile_ids','data_sha256','data_columns','origins','horizons','categories')}
        config['source_cache']['replication_cache']='Rebuilt independently; exact numerical/byte replication of original source cache not assumed.'
    else:
        assert config['source_cache']['recipe']=={key:config[key] for key in ('base_model','calendar_profile_ids','data_sha256','data_columns','origins','horizons','categories')}, 'Old cache cannot represent changed base recipe'
    assert digest(sigma)==config['detector_scale_sha256']
    run=Path(args.run_dir).resolve();assert not run.exists()
    run.mkdir()
    for name in ('code','input','output','receipts'):(run/name).mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copyfile(file,run/'code'/file.name)
    shutil.copyfile(cache,run/'input/FORECAST_CACHE.csv');shutil.copyfile(sigma,run/'input/DETECTOR_SCALE.json')
    (run/'PROTOCOL.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    members={str(p.relative_to(run)).replace('\\','/'):digest(p) for p in run.rglob('*') if p.is_file()}
    (run/'FREEZE_MANIFEST.json').write_text(json.dumps(dict(members=members,replication=True),indent=2),encoding='utf-8')
    if args.prepare_only:
        print('PREPARED_ONLY_NO_EVALUATION');return
    command=[sys.executable,'-X','utf8','-B',str(run/'code/v4_run.py'),'--run-dir',str(run),'--temp-dir',args.temp_dir]
    if args.project_root:command+=['--project-root',args.project_root]
    raise SystemExit(subprocess.run(command).returncode)
if __name__=='__main__':main()
