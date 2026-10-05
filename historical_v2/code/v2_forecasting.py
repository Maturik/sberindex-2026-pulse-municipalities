"""Fixed candidate forecasters for the new economic competition experiment.

No proprietary core is imported. All learned pooled features use training
territories and labels available at the current monthly prefix.
"""
import numpy as np

METHODS = ["LastValue", "Prophet", "ProphetYearly", "ThetaLite",
           "SharedSeasonalHalf", "SharedSeasonal", "EqualBlend",
           "GlobalLightGBM", "Chronos2Plain", "Chronos2CalendarCross"]
HORIZONS = (1, 3, 6, 12)


def reference_scale(y):
    """Fixed Jan-Jun2023 scale; never fitted on validation/held targets."""
    return np.maximum(np.median(np.abs(y[:, :6]), axis=1), 1.)


def calendar_profiles(training, categories, category_names):
    """2023-only pooled calendar prior, detrended separately per series.

Only one annual cycle exists. This is a calendar hypothesis, not identified
repeating seasonality: a common 2023 shock can enter the profile.
"""
    design = np.column_stack([np.ones(12), np.arange(12)])
    log_y = np.log1p(training[:, :12])
    trend = design @ np.linalg.lstsq(design, log_y.T, rcond=None)[0]
    residual = log_y - trend.T
    result = np.array([np.median(residual[np.array(categories) == c], axis=0)
                       for c in category_names])
    return result - result.mean(axis=1, keepdims=True)


def theta_lite(y, steps=12):
    """Fixed-alpha nonseasonal Theta forecast (forecast R theta formula)."""
    alpha = .2
    level = float(y[0])
    for value in y[1:]:
        level = alpha * value + (1-alpha) * level
    drift = np.polyfit(np.arange(len(y)), y, 1)[0] / 2
    h = np.arange(1, steps+1)
    return level + drift * (h-1 + (1-(1-alpha)**len(y))/alpha)


def shared_seasonal(y, profile, gamma=1., steps=12):
    """Damped local log trend plus the frozen pooled monthly calendar."""
    n = len(y)
    adjusted = np.log1p(y) - gamma * profile[np.arange(n) % 12]
    slope = np.polyfit(np.arange(6), adjusted[-6:], 1)[0]
    damping = np.cumsum(.8 ** np.arange(1, steps+1))
    future_months = np.arange(n, n+steps) % 12
    return np.expm1(adjusted[-1] + slope*damping + gamma*profile[future_months])


def local_candidates(y, profile, dates, future_dates, wanted=None):
    """Two fixed Prophet configurations and simple/calendar competitors."""
    from prophet import Prophet
    import pandas as pd
    wanted = METHODS if wanted is None else wanted
    result = {"LastValue": np.repeat(y[-1], 12), "ThetaLite": theta_lite(y),
              "SharedSeasonalHalf": shared_seasonal(y, profile, .5),
              "SharedSeasonal": shared_seasonal(y, profile, 1.)}
    for name in ("Prophet", "ProphetYearly"):
        if name not in wanted and not (name == "Prophet" and "EqualBlend" in wanted):
            continue
        model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, n_changepoints=3,
                        changepoint_prior_scale=.05, uncertainty_samples=0)
        if name == "ProphetYearly":
            model.add_seasonality("yearly", period=365.25, fourier_order=2,
                                  prior_scale=.1)
        model.fit(pd.DataFrame({"ds": dates, "y": y}))
        result[name] = model.predict(pd.DataFrame({"ds": future_dates})).yhat.to_numpy()
    if "EqualBlend" in wanted:
        result["EqualBlend"] = (result["LastValue"] + result["Prophet"] +
                                 result["SharedSeasonal"]) / 3
    return result


def global_features(prefix, scale, cat, cat_names, horizon):
    """Past six values and known calendar; no future economic covariates."""
    a = prefix[-6:] / scale
    n = len(prefix)
    origin_angle = 2*np.pi*((n-1) % 12)/12
    target_angle = 2*np.pi*((n-1+horizon) % 12)/12
    return [a[-1], *np.diff(a), float(np.median(a)),
            float(np.polyfit(np.arange(6), a, 1)[0]),
            np.sin(origin_angle), np.cos(origin_angle),
            np.sin(target_angle), np.cos(target_angle),
            *[float(cat == c) for c in cat_names]]


def pooled_models(training, scales, categories, cat_names, n):
    """Fit direct horizon models using only label indices < n."""
    from lightgbm import LGBMRegressor
    models, metadata = {}, []
    for h in HORIZONS:
        xx, yy = [], []
        for y, scale, cat in zip(training, scales, categories):
            for origin in range(5, n-h):
                prefix = y[:origin+1]
                if not np.isfinite(prefix).all() or not np.isfinite(y[origin+h]):
                    continue
                xx.append(global_features(prefix, scale, cat, cat_names, h))
                yy.append((y[origin+h]-y[origin])/scale)
        if not xx:
            models[h] = None
            metadata.append({"horizon": h, "rows": 0, "fallback": "damped_h1_increment"})
            continue
        model = LGBMRegressor(n_estimators=100, max_depth=3, num_leaves=7,
                             learning_rate=.05, min_child_samples=30,
                             reg_lambda=5., random_state=20261004,
                             n_jobs=2, verbosity=-1, deterministic=True,
                             force_col_wise=True)
        model.fit(np.asarray(xx), np.asarray(yy))
        models[h] = model
        metadata.append({"horizon": h, "rows": len(xx), "max_label_index": n-1})
    return models, metadata


def pooled_predict(models, prefixes, scales, categories, cat_names):
    """Interpolate fixed direct forecasts at 1/3/6/12 for a full curve."""
    anchors = np.zeros((len(prefixes), 5))
    anchors[:, 0] = [y[-1] for y in prefixes]
    for column, h in enumerate(HORIZONS, 1):
        model = models[h]
        if model is None:
            one_increment = anchors[:, 1]-anchors[:, 0]
            anchors[:, column] = anchors[:, 0] + one_increment*sum(.8**i for i in range(h))
        else:
            xx = [global_features(y, s, c, cat_names, h)
                  for y, s, c in zip(prefixes, scales, categories)]
            anchors[:, column] = anchors[:, 0] + scales*model.predict(np.asarray(xx))
    return np.array([np.interp(np.arange(1, 13), [0, 1, 3, 6, 12], row)
                     for row in anchors])


def foundation_predict(pipeline, prefixes, scales, calendar=False):
    """Fixed groups passed by caller. Median and 10/90% quantiles."""
    inputs = []
    for y, scale in zip(prefixes, scales):
        item = {"target": np.asarray(y/scale, dtype=np.float32)}
        if calendar:
            angle = 2*np.pi*np.arange(len(y))/12
            future = 2*np.pi*np.arange(len(y), len(y)+12)/12
            item["past_covariates"] = {"month_sin": np.sin(angle).astype("float32"),
                                       "month_cos": np.cos(angle).astype("float32")}
            item["future_covariates"] = {"month_sin": np.sin(future).astype("float32"),
                                         "month_cos": np.cos(future).astype("float32")}
        inputs.append(item)
    quantiles, medians = pipeline.predict_quantiles(
        inputs, prediction_length=12, quantile_levels=[.1, .5, .9],
        batch_size=64 if calendar else 128, cross_learning=calendar)
    pred = np.array([q.detach().cpu().numpy()[0] for q in medians]) * scales[:, None]
    bounds = np.array([q.detach().cpu().numpy()[0] for q in quantiles]) * scales[:, None, None]
    return pred, bounds


def clip_predictions(values):
    """Predeclared nonnegative decision action for every candidate equally."""
    return np.maximum(values, 0.)
