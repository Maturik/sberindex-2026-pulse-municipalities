"""Project the locked V2 winner from Dec2024 to Jan-Dec2025.

This is a historical snapshot, not a live forecast for2026. It does not score
new candidates or select a new winner. Official data stay in local storage.
"""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from v2_runtime import configure


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True, help="new local directory")
    parser.add_argument("--temp-dir", required=True)
    parser.add_argument("--project-root", help="optional vendored dependencies")
    args = parser.parse_args()
    configure(args.temp_dir, args.project_root)
    import numpy as np
    import pandas as pd
    from v2_forecasting import (reference_scale, local_candidates, pooled_models,
                                pooled_predict, foundation_predict,
                                clip_predictions)

    run = Path(args.run_dir).resolve()
    terminal = json.loads((run/"receipts/TERMINAL.json").read_text(encoding="utf-8"))
    assert terminal["status"] == "EXECUTED", "Scientific run must have completed"
    freeze = json.loads((run/"FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for member, digest in freeze["members"].items():
        assert sha(run/member) == digest, "Frozen input/code changed: "+member
    # This script lives next to an identical copy of the frozen forecasters.
    assert sha(Path(__file__).with_name("v2_forecasting.py")) == freeze["members"]["code/v2_forecasting.py"]
    protocol = json.loads((run/"PROTOCOL.json").read_text(encoding="utf-8"))
    assert sha(protocol["source"]) == protocol["data_sha256"]
    winner = terminal["winner"]
    source = pd.read_parquet(protocol["source"])
    source["month"] = pd.to_datetime(source.date).dt.to_period("M").dt.to_timestamp("M")
    months = pd.date_range("2023-01-31", periods=24, freq="ME")
    assert not source.duplicated(["territory_id", "category", "month"]).any()
    wide = source.pivot(index=["territory_id", "category"], columns="month", values="value").reindex(columns=months).sort_index()
    complete = wide.notna().all(axis=1)
    included = wide.loc[complete]
    values = included.to_numpy(float)
    keys = list(included.index)
    cats = protocol["categories"]
    profiles = np.load(run/"input/calendar_profiles.npy")
    scales = reference_scale(values)
    future = pd.date_range("2025-01-31", periods=12, freq="ME")
    out = Path(args.output_dir).resolve()
    out.mkdir()
    save(out/"PARAMS.json", dict(method=winner, origin="2024-12-31", targets="2025-01..2025-12",
        scientific_freeze_sha256=sha(run/"FREEZE_MANIFEST.json"), source_sha256=protocol["data_sha256"],
        kind="historical_snapshot_projection", no_future_actuals=True,
        missing_policy="Exclude incomplete24-month series; no imputation or substitution",
        model_selection="Unchanged validation-selected V2 winner", new_hypotheses=0))
    counters = dict(local_curves=0, global_fits=0, chronos_calls=0, core_fits=0)
    if winner == "GlobalLightGBM":
        training = np.load(run/"input/training.npy")
        train_keys = json.loads((run/"input/training_keys.json").read_text(encoding="utf-8"))
        models, _ = pooled_models(training, reference_scale(training), [k[1] for k in train_keys], cats,24)
        counters["global_fits"] = sum(m is not None for m in models.values())
        predictions = pooled_predict(models, values, scales, [k[1] for k in keys], cats)
    elif winner.startswith("Chronos2"):
        import torch
        from chronos import Chronos2Pipeline
        for member in protocol["chronos2"]["files"]:
            assert sha(Path(protocol["chronos2"]["model_dir"])/member["file"]) == member["sha256"]
        torch.set_num_threads(2)
        pipeline = Chronos2Pipeline.from_pretrained(protocol["chronos2"]["model_dir"],
            device_map="cpu", dtype=torch.float32, local_files_only=True)
        calendar = winner == "Chronos2CalendarCross"
        groups = []
        if calendar:
            for cat in cats:
                indices = [i for i, key in enumerate(keys) if key[1] == cat]
                groups.extend(indices[k:k+16] for k in range(0,len(indices),16))
        else:
            groups = [list(range(k,min(k+128,len(keys)))) for k in range(0,len(keys),128)]
        predictions = np.empty((len(keys),12))
        for group in groups:
            predictions[group], _ = foundation_predict(pipeline, values[group], scales[group], calendar)
            counters["chronos_calls"] += 1
    else:
        curves = []
        for key, y in zip(keys,values):
            curves.append(local_candidates(y, profiles[cats.index(key[1])], months, future, [winner])[winner])
        counters["local_curves"] = len(curves)
        predictions = np.asarray(curves)
    predictions = clip_predictions(predictions)
    assert predictions.shape == (len(keys),12) and np.isfinite(predictions).all()
    rows = [dict(territory_id=int(key[0]), category=key[1], origin="2024-12-31",
                 target=str(date.date()), horizon=h, method=winner, prediction=float(p),
                 kind="historical_snapshot_projection", units="native_value")
            for key, curve in zip(keys,predictions) for h,(date,p) in enumerate(zip(future,curve),1)]
    pd.DataFrame(rows).to_csv(out/"FORECAST_2025.csv", index=False)
    excluded = wide.loc[~complete].copy()
    exclusions = [dict(territory_id=int(key[0]), category=key[1],
                        missing_months=int(row.isna().sum()), reason="incomplete24_month_history")
                   for key,row in excluded.iterrows()]
    pd.DataFrame(exclusions, columns=["territory_id","category","missing_months","reason"]).to_csv(out/"EXCLUDED_SERIES.csv", index=False)
    result = dict(at=datetime.now(timezone.utc).isoformat(), by="Лея", status="PROJECTION_EXECUTED",
        method=winner, series=len(keys), territories=len(set(k[0] for k in keys)), rows=len(rows),
        excluded_series=len(exclusions), negative_predictions=int((predictions<0).sum()),
        counters=counters, origin="2024-12-31", targets="2025-01..2025-12",
        private_paths=[str(out/"FORECAST_2025.csv"),str(out/"EXCLUDED_SERIES.csv")],
        file_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        decision="Selected model unchanged; archival full projection is not validation of future2026",
        external_sends=0, Cloud="NOT_RUN")
    save(out/"RESULT.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
