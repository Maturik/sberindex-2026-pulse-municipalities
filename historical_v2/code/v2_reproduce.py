"""Prepare a fresh portable run from official data and the public recipe.

The contest data must be obtained by the participant from the official source.
This program does not upload or redistribute it. It refuses existing runs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data",required=True,help="official consumption.parquet")
parser.add_argument("--protocol",required=True,help="published PROTOCOL.json")
parser.add_argument("--model-dir",required=True,help="local pinned amazon/chronos-2 files")
parser.add_argument("--run-dir",required=True,help="new output directory")
parser.add_argument("--temp-dir",required=True)
parser.add_argument("--project-root",help="optional local vendored dependencies")
options=parser.parse_args()
from v2_runtime import configure
configure(options.temp_dir,options.project_root)
import numpy as np
import pandas as pd
from v2_forecasting import calendar_profiles,reference_scale


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,obj):
    with Path(path).open("x",encoding="utf-8",newline="\n") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


protocol=json.loads(Path(options.protocol).read_text(encoding="utf-8"))
assert sha(options.data)==protocol["data_sha256"],"Official input differs from frozen source"
model_dir=Path(options.model_dir).resolve()
for item in protocol["chronos2"]["files"]:
    assert sha(model_dir/item["file"])==item["sha256"],"Pinned model file mismatch"
run=Path(options.run_dir).resolve()
run.mkdir()
for name in ("input","code","output","receipts"):
    (run/name).mkdir()
df=pd.read_parquet(options.data)
df["month"]=pd.to_datetime(df.date).dt.to_period("M").dt.to_timestamp("M")
assert not df.duplicated(["territory_id","category","month"]).any()
assert (df.value>=0).all()
months=pd.date_range("2023-01-31",periods=24,freq="ME")
wide=df.pivot(index=["territory_id","category"],columns="month",values="value").reindex(columns=months)
cats=protocol["categories"]
excluded=set(protocol["panel"]["excluded_ids"])
eligible=[]
for tid,group in wide.groupby(level=0):
    if tid not in excluded and set(group.index.get_level_values(1))==set(cats) and group.iloc[:,:12].notna().all().all():
        eligible.append(int(tid))
ranked=sorted(eligible,key=lambda tid:hashlib.sha256(f"SBER_V2_20261004:{tid}".encode()).hexdigest())
expected={"validation":ranked[:128],"holdout":ranked[128:384],"training":ranked[384:]}
for panel,ids in expected.items():
    assert ids==protocol["panel"][panel+"_ids"],"Panel reconstruction differs"
    part=wide.loc[wide.index.get_level_values(0).isin(ids)].sort_index()
    np.save(run/f"input/{panel}.npy",part.to_numpy(float))
    save(run/f"input/{panel}_keys.json",[[int(t),c] for t,c in part.index])
training=np.load(run/"input/training.npy")
keys=json.loads((run/"input/training_keys.json").read_text(encoding="utf-8"))
np.save(run/"input/calendar_profiles.npy",calendar_profiles(training,[k[1] for k in keys],cats))
scales=reference_scale(training)
diff=np.diff(training[:,:12],axis=1)/scales[:,None]
sigma={}
for c in cats:
    v=diff[np.array([k[1] for k in keys])==c].ravel()
    sigma[c]=max(float(1.4826*np.median(np.abs(v-np.median(v)))),.02)
save(run/"input/detector_scale.json",sigma)
protocol["source"]=str(Path(options.data).resolve())
protocol["chronos2"]["model_dir"]=str(model_dir)
save(run/"PROTOCOL.json",protocol)
for name in ("v2_run.py","v2_forecasting.py","v2_detectors.py","v2_metrics.py","v2_runtime.py","v2_tests.py"):
    shutil.copyfile(Path(__file__).parent/name,run/"code"/name)
members={str(p.relative_to(run)).replace("\\","/"):sha(p) for base in (run/"input",run/"code") for p in base.rglob("*") if p.is_file()}
members["PROTOCOL.json"]=sha(run/"PROTOCOL.json")
save(run/"FREEZE_MANIFEST.json",dict(members=members,recipe="same authorized frozen V2; paths adapted only"))
save(run/"receipts/REPRODUCTION_PREPARED.json",dict(status="PREPARED_NOT_EXECUTED",input_sha256=sha(options.data),
    protocol_sha256=sha(run/"PROTOCOL.json"),model_sha256={x["file"]:x["sha256"] for x in protocol["chronos2"]["files"]}))
print(str(run/"code/v2_run.py"))
