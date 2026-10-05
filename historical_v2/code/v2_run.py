"""Frozen single-shot V2 runner. Selection receipts precede held evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import logging
import time
from v2_runtime import configure


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--run-dir", required=True)
    args.add_argument("--temp-dir", required=True)
    args.add_argument("--project-root", help="optional local vendored dependencies")
    options = args.parse_args()
    configure(options.temp_dir, options.project_root)
    import numpy as np
    import pandas as pd
    import torch
    from chronos import Chronos2Pipeline
    from v2_forecasting import (METHODS, reference_scale, local_candidates, pooled_models,
                               pooled_predict, foundation_predict, clip_predictions)
    from v2_metrics import metric_table, rank_methods, conditional_bootstrap
    from v2_detectors import DETECTORS, trace, calibrate, synthetic_cases
    torch.set_num_threads(2)
    logging.getLogger("cmdstanpy").disabled = True
    logging.getLogger("prophet").setLevel(logging.ERROR)
    run = Path(options.run_dir).resolve()
    output = run/"output"
    freeze = json.loads((run/"FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in freeze["members"].items():
        assert sha(run/name) == digest, "freeze mismatch: "+name
    start = time.monotonic()
    protocol = json.loads((run/"PROTOCOL.json").read_text(encoding="utf-8"))
    cats = protocol["categories"]
    training = np.load(run/"input/training.npy")
    training_keys = json.loads((run/"input/training_keys.json").read_text(encoding="utf-8"))
    train_scale = reference_scale(training)
    train_cat = [key[1] for key in training_keys]
    profiles = np.load(run/"input/calendar_profiles.npy")
    sigma = json.loads((run/"input/detector_scale.json").read_text(encoding="utf-8"))
    data, keys, scales = {}, {}, {}
    for panel in ("validation", "holdout"):
        data[panel] = np.load(run/f"input/{panel}.npy")
        keys[panel] = json.loads((run/f"input/{panel}_keys.json").read_text(encoding="utf-8"))
        scales[panel] = reference_scale(data[panel])
    for file in protocol["chronos2"]["files"]:
        assert sha(Path(protocol["chronos2"]["model_dir"])/file["file"]) == file["sha256"], "model hash mismatch"
    pipeline = Chronos2Pipeline.from_pretrained(protocol["chronos2"]["model_dir"],
        device_map="cpu", dtype=torch.float32, local_files_only=True)
    counters = dict(prophet_fits=0, global_fits=0, chronos_calls=0, core_fits=0, foundation_fits=0)
    fit_metadata = []
    cache = {}

    def forecasts(panel, n, selected=None):
        """Same eligible prefix mask across methods; compute once and cache."""
        cache_key = (panel, n, selected)
        if cache_key in cache:
            return cache[cache_key]
        if selected is not None and (panel, n, None) in cache:
            pp, qq, mask = cache[(panel, n, None)]
            return ({selected: pp[selected]}, {selected: qq[selected]} if selected in qq else {}, mask)
        y = data[panel]
        eligible = np.isfinite(y[:, :n]).all(axis=1)
        wanted = METHODS if selected is None else [selected]
        pred = {name: np.full((len(y), 12), np.nan) for name in wanted}
        bounds = {}
        ids = np.flatnonzero(eligible)
        dates = pd.date_range("2023-01-31", periods=n, freq="ME")
        future = pd.date_range(dates[-1], periods=13, freq="ME")[1:]
        local = [m for m in wanted if m not in ("GlobalLightGBM", "Chronos2Plain", "Chronos2CalendarCross")]
        if local:
            for i in ids:
                values = local_candidates(y[i, :n], profiles[cats.index(keys[panel][i][1])], dates, future, local)
                counters["prophet_fits"] += int("Prophet" in local or "EqualBlend" in local)+int("ProphetYearly" in local)
                for method in local:
                    pred[method][i] = clip_predictions(values[method])
        if "GlobalLightGBM" in wanted:
            models, metadata = pooled_models(training, train_scale, train_cat, cats, n)
            counters["global_fits"] += sum(m is not None for m in models.values())
            fit_metadata.append(dict(panel=panel, prefix_n=n, models=metadata))
            if len(ids):
                pred["GlobalLightGBM"][ids] = clip_predictions(pooled_predict(models,
                    [y[i, :n] for i in ids], scales[panel][ids], [keys[panel][i][1] for i in ids], cats))
        for method in ("Chronos2Plain", "Chronos2CalendarCross"):
            if method not in wanted:
                continue
            calendar = method == "Chronos2CalendarCross"
            bounds[method] = np.full((len(y), 12, 3), np.nan)
            if calendar:
                groups = []
                for c in cats:
                    all_category = [i for i, key in enumerate(keys[panel]) if key[1] == c]
                    for offset in range(0, len(all_category), 16):
                        group = [i for i in all_category[offset:offset+16] if eligible[i]]
                        if group:
                            groups.append(group)
            else:
                groups = [ids[o:o+128].tolist() for o in range(0, len(ids), 128)]
            for group in groups:
                pp, qq = foundation_predict(pipeline, [y[i, :n] for i in group], scales[panel][group], calendar)
                counters["chronos_calls"] += 1
                pred[method][group] = clip_predictions(pp)
                bounds[method][group] = clip_predictions(qq)
        for method, pp in pred.items():
            assert np.isfinite(pp[eligible]).all(), "candidate failure: "+method
        cache[cache_key] = (pred, bounds, eligible)
        print(json.dumps(dict(at=now(), panel=panel, prefix_n=n, selected=selected,
                              eligible_series=int(eligible.sum()), elapsed_seconds=time.monotonic()-start)), flush=True)
        return cache[cache_key]

    def forecast_rows(panel, n, horizons, target_stop, selected=None):
        pred, bounds, eligible = forecasts(panel, n, selected)
        rows = []
        y = data[panel]
        for h in horizons:
            target = n+h-1
            if target > target_stop:
                continue
            common = np.flatnonzero(eligible & np.isfinite(y[:, target]))
            for method, pp in pred.items():
                for i in common:
                    q = bounds.get(method)
                    rows.append(dict(split=panel, method=method, territory_id=keys[panel][i][0], category=keys[panel][i][1],
                        prefix_n=n, horizon=h, target_index=target, actual=float(y[i,target]),
                        prediction=float(pp[i,h-1]), reference_scale=float(scales[panel][i]),
                        mase_scale=float(np.mean(np.abs(np.diff(y[i,:12])))),
                        lower10=float(q[i,h-1,0]) if q is not None else None,
                        upper90=float(q[i,h-1,2]) if q is not None else None))
        return rows

    def detector_stream(panel, winner):
        """Only complete2024 detector trajectories; no replacements."""
        count = 6 if panel == "validation" else 12
        eligible = np.isfinite(data[panel][:, :12+count]).all(axis=1)
        ii = np.flatnonzero(eligible)
        residual = np.zeros((len(ii), count))
        for month in range(count):
            pp, _, _ = forecasts(panel, 12+month, winner)
            for row, i in enumerate(ii):
                error = data[panel][i,12+month]-pp[winner][i,0]
                residual[row,month] = error/scales[panel][i]/sigma[keys[panel][i][1]]
        return residual, [keys[panel][i] for i in ii]

    # Initialization/model loading is technical preparation, not inference.
    # Every performance operation after this exclusive claim is in the try.
    save(run/"receipts/BURNED.json", dict(at=now(), freeze_sha256=sha(run/"FREEZE_MANIFEST.json")))
    try:
        validation = []
        for n in range(12,18):
            validation.extend(forecast_rows("validation", n, (1,3,6),17))
        pd.DataFrame(validation).to_csv(output/"VALIDATION_PREDICTIONS.csv", index=False)
        val_metrics = metric_table(validation)
        winner, scores = rank_methods(val_metrics, METHODS)
        save(output/"FORECAST_SELECTION.json", dict(at=now(), winner=winner, validation_scores=scores,
            validation_sha256=sha(output/"VALIDATION_PREDICTIONS.csv"), holdout_evaluated=False))
        print("FORECAST_SELECTION "+winner+" "+json.dumps(scores), flush=True)
        # Reuse selected val forecasts without fitting them again.
        for n in range(12,18):
            pp, qq, mask = cache[("validation",n,None)]
            cache[("validation",n,winner)] = ({winner:pp[winner]}, {winner:qq[winner]} if winner in qq else {},mask)
        val_z, val_keys = detector_stream("validation", winner)
        thresholds, calibration = calibrate(val_z)
        save(output/"DETECTOR_CALIBRATION.json", dict(at=now(), thresholds=thresholds, search=calibration,
                                                    scope="retrospectiveJanJun development; not March exante"))
        cal_cases, cal_trace = synthetic_cases(val_z, val_keys, thresholds, 2, "calibration")
        cal_df = pd.DataFrame(cal_cases)
        detector_scores = cal_df.groupby("detector").agg(gain=("paired_gain","mean"), delay=("penalized_delay","mean"))
        choices = [(d,float(detector_scores.loc[d,"gain"]),float(detector_scores.loc[d,"delay"])) for d in DETECTORS]
        choices.append(("NoAlarm",0.,4.))
        detector_winner = min(choices,key=lambda x:(-x[1],x[2],list(DETECTORS+('NoAlarm',)).index(x[0])))[0]
        save(output/"DETECTOR_SELECTION.json", dict(at=now(), winner=detector_winner, scores=choices,
                                                    holdout_evaluated=False))
        cal_df.to_csv(output/"SYNTHETIC_CALIBRATION_CASES.csv",index=False)
        pd.DataFrame(cal_trace).to_csv(output/"SYNTHETIC_CALIBRATION_TRACE.csv",index=False)
        held = []
        for n in range(18,24):
            held.extend(forecast_rows("holdout",n,(1,3,6),23))
        pd.DataFrame(held).to_csv(output/"HOLDOUT_PREDICTIONS.csv",index=False)
        held_metrics = metric_table(held)
        diagnostic = forecast_rows("holdout",12,(12,),23)
        for row in diagnostic:
            row["split"] = "diagnostic_h12"
        pd.DataFrame(diagnostic).to_csv(output/"DIAGNOSTIC_H12_PREDICTIONS.csv",index=False)
        metrics = pd.concat([val_metrics,held_metrics,metric_table(diagnostic)],ignore_index=True)
        metrics.to_csv(output/"METRICS.csv",index=False)
        for n in range(18,24):
            pp,qq,mask = cache[("holdout",n,None)]
            cache[("holdout",n,winner)] = ({winner:pp[winner]}, {winner:qq[winner]} if winner in qq else {},mask)
        held_z, held_keys = detector_stream("holdout",winner)
        held_cases, held_trace = synthetic_cases(held_z,held_keys,thresholds,8,"holdout")
        pd.DataFrame(held_cases).to_csv(output/"SYNTHETIC_HOLDOUT_CASES.csv",index=False)
        pd.DataFrame(held_trace).to_csv(output/"SYNTHETIC_HOLDOUT_TRACE.csv",index=False)
        clean_rows = []
        for panel,zs,kk,start_month in (("calibration",val_z,val_keys,0),("holdout",held_z,held_keys,6)):
            for key,z in zip(kk,zs):
                for detector in DETECTORS+("NoAlarm",):
                    stat,alarm = trace(z,detector,thresholds[detector])
                    # Keep the full warm history for exact detector reconstruction.
                    for month in range(len(z)):
                        clean_rows.append(dict(split=panel,territory_id=key[0],category=key[1],detector=detector,
                            month_index=month+12,z=float(z[month]),statistic=float(stat[month]),alarm=int(alarm[month])))
        clean = pd.DataFrame(clean_rows)
        clean.to_csv(output/"DETECTORS_ORIGINAL.csv",index=False)
        rate_rows = clean[(clean.split == "calibration") | (clean.month_index >= 18)]
        rate_rows.groupby(["split","detector"]).alarm.agg(["mean","sum","count"]).reset_index().to_csv(
            output/"DETECTOR_ALARM_RATES.csv",index=False)
        combined_cases = pd.concat([cal_df,pd.DataFrame(held_cases)],ignore_index=True)
        combined_cases.groupby(["split","detector","kind"]).agg(
            paired_gain=("paired_gain","mean"),additional_hit=("additional_hit","mean"),
            penalized_delay=("penalized_delay","mean"),cases=("paired_gain","size")).reset_index().to_csv(
                output/"SYNTHETIC_SUMMARY.csv",index=False)
        save(output/"CONDITIONAL_BOOTSTRAP.json",conditional_bootstrap(held,winner))
        save(output/"FIT_METADATA.json",fit_metadata)
        save(output/"COVERAGE.json",dict(intended={p:len(keys[p]) for p in keys},
            detector_complete_series={"validation":len(val_keys),"holdout":len(held_keys)},
            missing_policy=protocol["missing_policy"]))
        # Full per-category metrics are derived from exact same saved forecast rows.
        category_metrics=[]
        for category in cats:
            part=metric_table([r for r in validation+held if r["category"]==category])
            part["category"]=category
            category_metrics.append(part)
        pd.concat(category_metrics).to_csv(output/"CATEGORY_METRICS.csv",index=False)
        result=dict(at=now(),by="Лея",status="EXECUTED",winner=winner,detector_winner=detector_winner,
                    validation_scores=scores,held_scores=held_metrics.groupby("method").macro_scaled_MAE.mean().to_dict(),
                    counters=counters,elapsed_seconds=time.monotonic()-start,
                    Cloud="NOT_RUN",CLAUDE_REVIEW="NOT_RUN",external_sends=0,
                    decision="Keep one validation-selected model and detector; do not tune on held outcomes.")
        save(output/"RESULT.json",result)
        save(run/"receipts/TERMINAL.json",dict(**result,output_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        save(run/"receipts/TERMINAL.json",dict(at=now(),status="FAILED",error=repr(exc),
            counters=counters,elapsed_seconds=time.monotonic()-start,
            decision="STOP scientific retry; inspect cause and already disclosed outputs."))
        raise


if __name__ == "__main__":
    main()
