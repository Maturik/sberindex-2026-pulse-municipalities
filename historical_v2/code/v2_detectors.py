"""Causal alarm issuance; calibrated alarm rate is not ground-truth FPR."""
import numpy as np

DETECTORS = ("Shewhart", "EWMA", "ResetCUSUM", "BOCPD", "PrefixPELT")
GRIDS = {
    "Shewhart": np.r_[0., np.geomspace(.1, 100., 45), 1e30],
    "EWMA": np.r_[0., np.geomspace(.02, 50., 45), 1e30],
    "ResetCUSUM": np.r_[0., np.geomspace(.1, 200., 45), 1e30],
    "BOCPD": np.r_[np.linspace(0., 1., 81), 1e30],
    "PrefixPELT": np.r_[0., np.geomspace(.02, 10000., 45), 1e30],
}


def bocpd_statistics(z):
    """Gaussian Normal-Inverse-Gamma BOCPD, hazard1/12.

Score sums posterior run lengths1 and2. Constant-hazard p(r=0) alone
would be uninformative. Priors mu0,kappa1,alpha2,beta1 are fixed.
"""
    from scipy.special import gammaln
    prob = np.array([1.])
    mu, kappa, alpha, beta = (np.array([v]) for v in (0., 1., 2., 1.))
    scores = []
    for value in z:
        dof = 2*alpha
        scale2 = beta*(kappa+1)/(alpha*kappa)
        log_density = (gammaln((dof+1)/2)-gammaln(dof/2)
                       -.5*np.log(dof*np.pi*scale2)
                       -(dof+1)/2*np.log1p((value-mu)**2/(dof*scale2)))
        likelihood = np.exp(log_density-log_density.max())
        weighted = prob*likelihood
        prob = np.r_[weighted.sum()/12, weighted*11/12]
        prob /= prob.sum()
        scores.append(float(prob[1:3].sum()))
        new_beta = beta+kappa*(value-mu)**2/(2*(kappa+1))
        new_mu = (kappa*mu+value)/(kappa+1)
        mu, kappa = np.r_[0., new_mu], np.r_[1., kappa+1]
        alpha, beta = np.r_[2., alpha+.5], np.r_[1., new_beta]
    return np.array(scores)


def trace(z, detector, threshold):
    """Actual-time alarms after a fixed six-zero artificial prior seed.

The seed is not economic history and is excluded from all alarm budgets.
It narrows BOCPD's prior; this explicit assumption avoids start-up alarms.
"""
    if detector == "NoAlarm":
        return np.zeros(len(z)), np.zeros(len(z), dtype=bool)
    seed = 6
    z = np.r_[np.zeros(seed), np.asarray(z, float)]
    stat = np.zeros(len(z))
    alarm = np.zeros(len(z), dtype=bool)
    if detector == "BOCPD":
        stat = bocpd_statistics(z)
        return stat[seed:], (stat > threshold)[seed:]
    ewma = plus = minus = 0.
    issued = set()
    for t, value in enumerate(z):
        if detector == "Shewhart":
            stat[t] = abs(value)
        elif detector == "EWMA":
            ewma = .3*value+.7*ewma
            stat[t] = abs(ewma)
        elif detector == "ResetCUSUM":
            plus = max(0., plus+value-.5)
            minus = max(0., minus-value-.5)
            stat[t] = max(plus, minus)
        elif detector == "PrefixPELT":
            if t+1 < 4 or not np.isfinite(threshold):
                continue
            import ruptures
            boundaries = ruptures.Pelt(model="l2", min_size=2, jump=1).fit(
                z[:t+1, None]).predict(pen=float(threshold))[:-1]
            recent = {b for b in boundaries if b >= seed and 0 < t+1-b <= 3}
            new = recent-issued
            alarm[t] = bool(new)
            stat[t] = len(new)
            issued.update(recent)
            continue
        else:
            raise ValueError(detector)
        alarm[t] = stat[t] > threshold
        if detector == "ResetCUSUM" and alarm[t]:
            plus = minus = 0.
    return stat[seed:], alarm[seed:]


def calibrate(clean, budget=.10):
    """One pooled series-month budget on retrospective Jan-Jun development."""
    thresholds, audit = {}, []
    for detector in DETECTORS:
        candidates = []
        for threshold in GRIDS[detector]:
            count = sum(int(trace(z, detector, threshold)[1].sum()) for z in clean)
            rate = count/(len(clean)*6)
            candidates.append((float(threshold), rate))
        eligible = [(v, rate) for v, rate in candidates if rate <= budget+1e-12]
        # Largest attainable rate below budget; stricter threshold wins rate ties.
        best = max(eligible, key=lambda p: (p[1], p[0]))
        thresholds[detector] = best[0]
        audit.append({"detector": detector, "threshold": best[0], "alarm_rate": best[1],
                      "grid": candidates, "series_months": len(clean)*6})
    thresholds["NoAlarm"] = 1e30
    return thresholds, audit


def injection(length, onset, kind, amplitude):
    """Fixed residual-domain alteration; forecaster is not refitted."""
    delta = np.zeros(length)
    if kind == "step":
        delta[onset:] = amplitude
    elif kind == "ramp":
        delta[onset:] = amplitude*np.minimum(np.arange(1, length-onset+1)/3, 1.)
    elif kind == "spike":
        delta[onset] = amplitude
    else:
        raise ValueError(kind)
    return delta


def synthetic_cases(clean, keys, thresholds, onset, split):
    """Matched injected/control cases, first additional alarm and censored delay."""
    rows, trajectories = [], []
    for key, z in zip(keys, clean):
        for detector in DETECTORS+("NoAlarm",):
            _, original_alarm = trace(z, detector, thresholds[detector])
            for kind in ("step", "ramp", "spike"):
                for amp in (-3., -2., -1., 1., 2., 3.):
                    changed = z+injection(len(z), onset, kind, amp)
                    stat, alarm = trace(changed, detector, thresholds[detector])
                    extra = np.flatnonzero((alarm & ~original_alarm)[onset:])
                    original_hit = bool(original_alarm[onset:].any())
                    injected_hit = bool(alarm[onset:].any())
                    rows.append(dict(split=split, territory_id=int(key[0]), category=key[1],
                                     detector=detector, kind=kind, amplitude=amp,
                                     original_hit=int(original_hit), injected_hit=int(injected_hit),
                                     paired_gain=int(injected_hit)-int(original_hit),
                                     additional_hit=int(len(extra)>0),
                                     penalized_delay=int(extra[0]) if len(extra) else len(z)-onset))
                    for t in range(onset, len(z)):
                        trajectories.append(dict(split=split, territory_id=int(key[0]), category=key[1],
                            detector=detector, kind=kind, amplitude=amp, month_index=t+12,
                            z_original=float(z[t]), z_injected=float(changed[t]), statistic=float(stat[t]),
                            original_alarm=int(original_alarm[t]), injected_alarm=int(alarm[t])))
    return rows, trajectories
