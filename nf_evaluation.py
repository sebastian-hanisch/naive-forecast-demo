"""Auswertung: naive Verfahren im Rolling-Origin-Vergleich auf einer Tagesaufträge-Reihe und vier Experimente (einzelner Testzeitraum gegen viele Ursprünge, Rauschen, Wochentag des Ursprungs, Trend und Fenster)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import nf_constants as C
import nf_forecast as F
import nf_scenario as S


@dataclass(frozen=True)
class Settings:
    trend: int = C.DEFAULT_TREND
    weekly: float = C.DEFAULT_WEEKLY
    yearly: float = C.DEFAULT_YEARLY
    noise: float = C.DEFAULT_NOISE
    shift: int = C.DEFAULT_SHIFT
    events: float = C.DEFAULT_EVENTS
    horizon: int = C.DEFAULT_HORIZON
    step: int = C.DEFAULT_STEP
    window: int = C.DEFAULT_WINDOW
    k: int = C.DEFAULT_K_WEEKS
    seed: int = 3


@dataclass
class Analysis:
    settings: Settings
    series: S.Series
    origins: np.ndarray
    actual: np.ndarray
    errors: dict
    summary: dict
    oracle: dict           # Kennzahlen der Orakel-Prognose (wahrer Erwartungswert)
    horizon_mae: dict
    origin_mae: dict
    winners: dict          # Anteil der Ursprünge, an denen das Verfahren den kleinsten Fehler hat

    @property
    def best(self):
        return min(self.summary, key=lambda m: self.summary[m]["mase"])


def make_series(s, seed=None):
    return S.generate(s.trend, s.weekly, s.yearly, s.noise, s.shift, s.events, seed=s.seed if seed is None else seed)


def oracle_summary(series, org, h):
    """Kennzahlen der Orakel-Prognose: der wahre Erwartungswert als Prognose, gemessen am beobachteten Wert (untere Grenze für jedes Verfahren im Mittel)."""
    E = np.stack([series.mu[t:t + h] - series.y[t:t + h] for t in org])
    scale = F.mase_scale(series.y, C.FIRST_TEST)
    mae = float(np.abs(E).mean())
    return {"mae": mae, "rmse": float(np.sqrt((E ** 2).mean())), "me": float(E.mean()), "mase": mae / scale}


@lru_cache(maxsize=48)
def analyse(s):
    series = make_series(s)
    org, actual, errors = F.rolling_origin(series.y, s.horizon, s.step, s.window, s.k)
    po = F.per_origin(errors)
    best_at = np.argmin(np.stack([po[m] for m in C.METHODS]), axis=0)
    winners = {m: float(np.mean(best_at == i)) for i, m in enumerate(C.METHODS)}
    return Analysis(s, series, org, actual, errors, F.summarize(errors, series.y), oracle_summary(series, org, s.horizon), F.per_horizon(errors), po, winners)


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _replace(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


# --- Experiment 1: einzelner Testzeitraum gegen viele Ursprünge ---------------------------------------------------------------------------


def split_experiment(seeds=None, base=None):
    """Je Reihe alle Ursprünge des letzten Jahres (Schritt 1): wer hat im einzelnen Ursprung den kleinsten Fehler, und wie oft weicht dieser Gewinner vom Gewinner über alle Ursprünge ab? Dazu die Streuung der MASE
    eines Verfahrens über die Ursprünge."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    wins = {m: [] for m in C.METHODS}
    flips, roll, spread_lo, spread_hi = [], [], [], []
    for sd in seeds:
        st = _replace(base, step=1, seed=sd)
        a = analyse(st)
        for m in C.METHODS:
            wins[m].append(a.winners[m])
        overall = a.best
        best_at = np.argmin(np.stack([a.origin_mae[m] for m in C.METHODS]), axis=0)
        flips.append(float(np.mean(np.array(C.METHODS)[best_at] != overall)))
        roll.append(overall)
        scale = F.mase_scale(a.series.y, C.FIRST_TEST)
        v = a.origin_mae[overall] / scale
        spread_lo.append(float(np.percentile(v, 10)))
        spread_hi.append(float(np.percentile(v, 90)))
    row = {"n_seeds": len(seeds), "flip": _mean_se(flips), "rolling_winners": roll, "spread_lo": float(np.mean(spread_lo)), "spread_hi": float(np.mean(spread_hi))}
    for m in C.METHODS:
        row["win_" + m] = float(np.mean(wins[m]))
    return row


# --- Experiment 2: Rauschen ---------------------------------------------------------------------------------------------------------------


def noise_experiment(levels=None, seeds=None, base=None):
    levels = C.NOISE_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for nz in levels:
        res = {m: [] for m in C.METHODS}
        floor = []
        for sd in seeds:
            a = analyse(_replace(base, noise=nz, seed=sd))
            for m in C.METHODS:
                res[m].append(a.summary[m]["mase"])
            floor.append(a.oracle["mase"])
        row = {"noise": nz, "n_seeds": len(seeds), "floor": float(np.mean(floor))}
        for m, v in res.items():
            row[m], row[m + "_se"] = _mean_se(v)
        d = np.array(res["snaive"]) - np.array(res["snaive_k"])
        row["k_gain"], row["k_gain_se"] = _mean_se(d)
        rows.append(row)
    return rows


# --- Experiment 3: Wochentag des Ursprungs -----------------------------------------------------------------------------------------------------


def weekday_experiment(seeds=None, base=None):
    """Ursprünge nur an einem Wochentag (Schritt 7): die Verfahren, die den letzten Tag fortschreiben (naiv, Drift), hängen davon ab, an welchem Wochentag der Ursprung liegt."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for d in range(7):
        res = {m: [] for m in C.METHODS}
        for sd in seeds:
            series = make_series(base, seed=sd)
            first = C.FIRST_TEST + ((d + 1 - C.FIRST_TEST) % 7)                 # der letzte bekannte Tag t-1 hat den Wochentag d
            org, act, err = F.rolling_origin(series.y, base.horizon, 7, base.window, base.k, first=first)
            summ = F.summarize(err, series.y)
            for m in C.METHODS:
                res[m].append(summ[m]["mase"])
        row = {"weekday": d, "n_seeds": len(seeds)}
        for m, v in res.items():
            row[m], row[m + "_se"] = _mean_se(v)
        rows.append(row)
    return rows


# --- Experiment 4: Trend und Fenster ------------------------------------------------------------------------------------------------------------


def trend_experiment(levels=None, seeds=None, base=None):
    levels = C.TREND_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for tr in levels:
        res = {m: [] for m in C.METHODS}
        for sd in seeds:
            a = analyse(_replace(base, trend=tr, seed=sd))
            for m in C.METHODS:
                res[m].append(a.summary[m]["mase"])
        row = {"trend": tr, "n_seeds": len(seeds)}
        for m, v in res.items():
            row[m], row[m + "_se"] = _mean_se(v)
        rows.append(row)
    return rows


def window_experiment(windows=None, seeds=None, base=None, shift=30):
    """Niveausprung (+30 %) im Testjahr: wächst das Fenster mit, dauert es lange, bis Mittelwert und Drift den neuen Stand sehen; ein gleitendes Fenster vergisst den alten Stand."""
    windows = C.WINDOW_OPTIONS if windows is None else windows
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = []
    for w in windows:
        res = {m: [] for m in C.METHODS}
        for sd in seeds:
            a = analyse(_replace(base, shift=shift, window=w, seed=sd))
            for m in C.METHODS:
                res[m].append(a.summary[m]["mase"])
        row = {"window": w, "n_seeds": len(seeds)}
        for m, v in res.items():
            row[m], row[m + "_se"] = _mean_se(v)
        rows.append(row)
    return rows
