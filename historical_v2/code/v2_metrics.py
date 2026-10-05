"""Equal-series scaled MAE and transparent secondary metrics."""
import numpy as np
import pandas as pd


def metric_table(rows):
    """Macro-average each series first; never weight larger regions by scale."""
    df = pd.DataFrame(rows)
    df["error"] = df.prediction-df.actual
    df["abs_error"] = df.error.abs()
    df["scaled_error"] = df.abs_error/df.reference_scale
    df["mase_error"] = np.where(df.mase_scale > 0, df.abs_error/df.mase_scale, np.nan)
    out = []
    for (split, method, horizon), group in df.groupby(["split", "method", "horizon"]):
        series = group.groupby(["territory_id", "category"])
        out.append(dict(split=split, method=method, horizon=int(horizon),
            MAE=float(group.abs_error.mean()), bias=float(group.error.mean()),
            macro_scaled_MAE=float(series.scaled_error.mean().mean()),
            MASE1=float(series.mase_error.mean().mean()),
            WAPE=float(group.abs_error.sum()/max(group.actual.abs().sum(), 1.)),
            forecast_pairs=len(group), series=series.ngroups,
            origins=group.prefix_n.nunique()))
    return pd.DataFrame(out)


def rank_methods(metrics, method_order):
    """One validation decision using equal weights for h1/h3/h6."""
    scores = metrics[metrics.horizon.isin([1, 3, 6])].groupby("method").macro_scaled_MAE.mean()
    assert set(scores.index) == set(method_order)
    ranked = sorted(method_order, key=lambda name: (scores[name], method_order.index(name)))
    return ranked[0], {name: float(scores[name]) for name in ranked}


def conditional_bootstrap(rows, winner, comparison="Prophet", repetitions=1000):
    """Territory-cluster bootstrap conditional on this observed macroperiod."""
    df = pd.DataFrame(rows)
    df = df[(df.method.isin([winner, comparison])) & df.horizon.isin([1, 3, 6])].copy()
    df["loss"] = np.abs(df.prediction-df.actual)/df.reference_scale
    per = df.groupby(["territory_id", "category", "horizon", "method"]).loss.mean().unstack("method")
    if winner == comparison:
        return dict(comparison=comparison, gain=0., conditional_CI=[0., 0.], repetitions=repetitions)
    per["gain"] = per[comparison]-per[winner]
    # Complete territory-horizon coverage for this secondary uncertainty estimate.
    cluster = per.gain.groupby(level=[0, 2]).mean().unstack(1).dropna()
    values = cluster[[1, 3, 6]].mean(axis=1).to_numpy()
    rng = np.random.default_rng(20261004)
    samples = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(repetitions)])
    return dict(comparison=comparison, gain=float(values.mean()),
                conditional_CI=np.quantile(samples, [.025, .975]).tolist(),
                territories=len(values), repetitions=repetitions,
                limitation="conditional on2024; correlated geography not fully resolved; not futuremacro confidence")
