"""Orakel-Test: Verfahren, Rolling-Origin-Kennzahlen und die Reihe gegen unabhängige Skalarschleifen (kein vektorisiertes numpy, andere Indexrechnung:
der Vortag gleichen Wochentags wird über das Zieldatum bestimmt, nicht über Modulo auf dem Fenster) sowie die Aussage zur Orakel-Untergrenze."""

import math
import random

import numpy as np

import nf_constants as C
import nf_forecast as F
import nf_scenario as S


def ref_forecast(method, y, t, h, win, k):
    y = [float(v) for v in y]
    hist = y[max(0, t - 7 * win):t] if win > 0 else y[:t]
    m = len(hist)
    out = []
    for j in range(h):
        target = t + j
        d = target - 7 * ((target - (t - 1) + 6) // 7)       # jüngster bekannter Tag mit dem Wochentag des Zieltags
        if method == "mean":
            out.append(sum(hist) / m)
        elif method == "naive":
            out.append(hist[-1])
        elif method == "snaive":
            out.append(y[d])
        elif method == "snaive_k":
            kk = max(1, min(k, m // 7))
            out.append(sum(y[d - 7 * i] for i in range(kk)) / kk)
        elif method == "snaive_year":
            out.append(y[target - 364] if t >= 364 + h else y[d])
        elif method == "drift":
            slope = (hist[-1] - hist[0]) / (m - 1) if m > 1 else 0.0
            out.append(hist[-1] + slope * (j + 1))
    return out


def test_every_method_matches_an_index_by_target_day_reference():
    rng = random.Random(1)
    for _ in range(120):
        n = rng.randint(30, 500)
        y = np.array([rng.randint(0, 200) for _ in range(n)], float)
        h, t = rng.randint(1, 28), rng.randint(28, n)
        win, k = rng.choice([0, 1, 2, 4, 8, 13, 26]), rng.randint(2, 12)
        for mth in C.METHODS:
            assert np.allclose(F.forecast(mth, y, t, h, win, k), ref_forecast(mth, y, t, h, win, k)), (mth, n, t, h, win, k)


def test_rolling_origin_and_summary_match_scalar_loops():
    rng = random.Random(2)
    for _ in range(12):
        n = rng.randint(60, 160)
        y = np.array([rng.randint(0, 200) for _ in range(n)], float)
        h, step, first = rng.randint(1, 14), rng.randint(1, 14), rng.randint(30, 50)
        win, k = rng.choice([0, 2, 4]), rng.randint(2, 6)
        org, _, errs = F.rolling_origin(y, h, step, win, k, first=first)
        ref_org = list(range(first, n - h + 1, step))
        assert list(org) == ref_org
        scale = sum(abs(y[i] - y[i - 7]) for i in range(7, first)) / (first - 7)
        summ = F.summarize(errs, y, first=first)
        for mth in C.METHODS:
            e = [ref_forecast(mth, y, t, h, win, k)[j] - y[t + j] for t in ref_org for j in range(h)]
            mae = sum(abs(v) for v in e) / len(e)
            assert abs(summ[mth]["mae"] - mae) < 1e-9 and abs(summ[mth]["mase"] - mae / scale) < 1e-9
            assert abs(summ[mth]["rmse"] - math.sqrt(sum(v * v for v in e) / len(e))) < 1e-9 and abs(summ[mth]["me"] - sum(e) / len(e)) < 1e-9


def test_expectation_of_the_series_matches_a_day_by_day_scalar_computation():
    rng = random.Random(3)
    pat = [p / (sum(C.WEEKLY_PATTERN) / 7) for p in C.WEEKLY_PATTERN]
    for _ in range(6):
        trend, weekly, yearly = rng.choice(range(-20, 41, 5)), rng.choice([0, .5, 1, 1.5]), rng.choice([0, .2, .5])
        shift, events, seed = rng.choice(range(-40, 41, 10)), rng.choice([0, .5, 1]), rng.randint(0, 9999)
        ser = S.generate(trend, weekly, yearly, 0.14, shift, events, seed=seed)
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        hol = {yr * 365 + d for yr in range(3) for d in C.HOLIDAY_DOY}
        promo = set(int(i) for i in np.nonzero(ser.promo)[0])
        for d in range(C.N_DAYS):
            f = 1 + trend / 100 * d / 365
            f *= 1 + weekly * (pat[d % 7] - 1)
            f *= 1 + yearly * math.sin(2 * math.pi * (d % 365) / 365 + phase)
            f *= (1 + shift / 100) if (ser.shift_day >= 0 and d >= ser.shift_day) else 1.0
            f *= 1 - (events * C.HOLIDAY_DROP if d in hol else 0) + (events * C.HOLIDAY_REBOUND if (d - 1) in hol else 0)
            f *= 1 + events * 0.5 * (d in promo)
            assert abs(max(C.LEVEL * f, 1.0) - ser.mu[d]) < 1e-9 * max(1.0, ser.mu[d])


def test_oracle_floor_holds_for_the_mean_but_the_median_is_slightly_better_for_mae():
    """Die Untergrenze gilt für Erwartungswert-Schätzer; die MAE-optimale Prognose ist der Median mu * exp(-sigma^2/2) (log-normal)."""
    for nz, min_gain in ((0.14, 0.0005), (0.4, 0.01)):
        gains = []
        for sd in range(12):
            ser = S.generate(noise=nz, seed=sd)
            org = F.origins(ser.n, 14)
            e_mean = np.mean([np.abs(ser.mu[t:t + 14] - ser.y[t:t + 14]).mean() for t in org])
            med = ser.mu * math.exp(-0.5 * nz ** 2)
            e_med = np.mean([np.abs(med[t:t + 14] - ser.y[t:t + 14]).mean() for t in org])
            gains.append(1 - e_med / e_mean)
        assert np.mean(gains) > min_gain                  # im Mittel über zwölf Reihen (einzelne Reihen können durch Zufall anders liegen)
