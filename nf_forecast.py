"""Naive Prognoseverfahren und Rolling-Origin-Auswertung.

Ein Ursprung t heißt: bekannt sind die Tage 0..t-1, prognostiziert werden die Tage t..t+h-1. Alle Verfahren sehen nur die Vergangenheit (auf Wunsch nur ein gleitendes Fenster der letzten Tage).

  mean         Mittel der Vergangenheit
  naive        letzter beobachteter Tag
  snaive       derselbe Wochentag der letzten Woche
  snaive_k     Mittel desselben Wochentags über die letzten k Wochen
  snaive_year  derselbe Wochentag vor 52 Wochen (braucht mindestens ein Jahr Vergangenheit; ignoriert ein kleineres Fenster)
  drift        letzter Tag plus die mittlere Tagesänderung der Vergangenheit, fortgeschrieben"""

import numpy as np

import nf_constants as C


def _history(y, t, window_weeks):
    if window_weeks and window_weeks > 0:
        return y[max(0, t - 7 * window_weeks):t]
    return y[:t]


def forecast(method, y, t, h, window_weeks=0, k=C.DEFAULT_K_WEEKS):
    """Prognose der Tage t..t+h-1 aus y[:t]."""
    hist = _history(y, t, window_weeks)
    m = len(hist)
    j = np.arange(h)
    if method == "mean":
        return np.full(h, hist.mean())
    if method == "naive":
        return np.full(h, hist[-1])
    if method == "snaive":
        return hist[m - 7 + (j % 7)]
    if method == "snaive_k":
        kk = max(1, min(k, m // 7))
        return np.mean([hist[m - 7 * (i + 1) + (j % 7)] for i in range(kk)], axis=0)
    if method == "snaive_year":
        full = y[:t]
        return full[len(full) - 364 + j] if len(full) >= 364 + h else full[len(full) - 7 + (j % 7)]
    if method == "drift":
        slope = (hist[-1] - hist[0]) / max(m - 1, 1)
        return hist[-1] + slope * (j + 1)
    raise ValueError(method)


def origins(n, h, first=C.FIRST_TEST, step=C.DEFAULT_STEP):
    return np.arange(first, n - h + 1, step)


def mase_scale(y, t0):
    """Mittlerer absoluter Fehler der saisonal naiven Prognose (Periode 7) innerhalb der Trainingsdaten y[:t0] - Nenner der MASE."""
    return float(np.abs(y[7:t0] - y[:t0 - 7]).mean())


def rolling_origin(y, h=C.DEFAULT_HORIZON, step=C.DEFAULT_STEP, window_weeks=0, k=C.DEFAULT_K_WEEKS, methods=C.METHODS, first=C.FIRST_TEST):
    """Prognosefehler (Prognose minus Ist) je Ursprung und Horizont für alle Verfahren: {Verfahren: (Ursprünge, h)} und die Ist-Werte (Ursprünge, h)."""
    org = origins(len(y), h, first, step)
    actual = np.stack([y[t:t + h] for t in org])
    errors = {}
    for mth in methods:
        F = np.stack([forecast(mth, y, t, h, window_weeks, k) for t in org])
        errors[mth] = F - actual
    return org, actual, errors


def summarize(errors, y, first=C.FIRST_TEST):
    """MAE, RMSE, ME (Verzerrung) und MASE je Verfahren über alle Ursprünge und Horizonte."""
    scale = mase_scale(y, first)
    out = {}
    for mth, E in errors.items():
        mae = float(np.abs(E).mean())
        out[mth] = {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}
    return out


def per_horizon(errors):
    return {m: np.abs(E).mean(axis=0) for m, E in errors.items()}


def per_origin(errors):
    """MAE je Ursprung (über den Horizont gemittelt)."""
    return {m: np.abs(E).mean(axis=1) for m, E in errors.items()}
