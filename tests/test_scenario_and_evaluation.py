"""Die Reihe (Komponenten einzeln geprüft) und die Auswertung (Analyse, vier Experimente)."""

import numpy as np
import pytest

import nf_constants as C
import nf_evaluation as E
import nf_scenario as S


def test_generate_is_reproducible_shaped_and_integer():
    a, b, c = S.generate(seed=4), S.generate(seed=4), S.generate(seed=5)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.n == C.N_DAYS and np.all(a.y >= 0) and np.all(a.y == np.round(a.y))
    assert np.array_equal(a.dow, np.arange(C.N_DAYS) % 7) and a.shift_day == -1


def test_components_can_be_switched_off_and_the_expectation_is_then_the_level():
    ser = S.generate(trend=0, weekly=0.0, yearly=0.0, shift=0, events=0.0, seed=1)
    assert ser.mu == pytest.approx(np.full(C.N_DAYS, C.LEVEL))


def test_weekly_pattern_averages_to_one_and_follows_the_weekdays():
    ser = S.generate(trend=0, yearly=0.0, shift=0, events=0.0, seed=1)
    means = [ser.mu[ser.dow == d].mean() / C.LEVEL for d in range(7)]
    pattern = np.array(C.WEEKLY_PATTERN) / np.mean(C.WEEKLY_PATTERN)
    assert means == pytest.approx(list(pattern)) and np.mean(means) == pytest.approx(1.0) and int(np.argmax(means)) == 4 and int(np.argmin(means)) == 6


def test_trend_is_linear_in_percent_per_year():
    ser = S.generate(trend=20, weekly=0.0, yearly=0.0, shift=0, events=0.0, seed=1)
    assert ser.mu[365] / ser.mu[0] == pytest.approx(1.2) and ser.mu[730] / ser.mu[0] == pytest.approx(1.4)


def test_holiday_and_rebound_and_promotions():
    ser = S.generate(trend=0, weekly=0.0, yearly=0.0, shift=0, events=1.0, seed=1)
    base = C.LEVEL
    hol = np.flatnonzero(ser.holiday)
    assert len(hol) == len(C.HOLIDAY_DOY) * 3 and set((hol % 365).tolist()) == set(C.HOLIDAY_DOY)
    clean = [d for d in hol if d + 1 < C.N_DAYS and ser.promo[d] == 0 and ser.promo[d + 1] == 0 and (d + 1) % 365 not in C.HOLIDAY_DOY]
    assert ser.mu[clean[0]] == pytest.approx(base * (1 - C.HOLIDAY_DROP)) and ser.mu[clean[0] + 1] == pytest.approx(base * (1 + C.HOLIDAY_REBOUND))
    promo_days = np.flatnonzero((ser.promo == 1) & (ser.holiday == 0) & (np.roll(ser.holiday, 1) == 0))
    assert ser.mu[promo_days[0]] == pytest.approx(base * 1.5) and ser.promo.sum() <= C.PROMO_LENGTH * C.PROMO_PER_YEAR * 3
    quiet = S.generate(trend=0, weekly=0.0, yearly=0.0, shift=0, events=0.0, seed=1)
    assert quiet.mu == pytest.approx(np.full(C.N_DAYS, base))


def test_level_shift_lies_in_the_test_year_and_scales_the_level():
    ser = S.generate(trend=0, weekly=0.0, yearly=0.0, events=0.0, shift=30, seed=3)
    assert C.FIRST_TEST + 30 <= ser.shift_day < C.N_DAYS - 60
    assert ser.mu[ser.shift_day - 1] == pytest.approx(C.LEVEL) and ser.mu[ser.shift_day] == pytest.approx(1.3 * C.LEVEL)


def test_noise_is_unbiased_and_grows_with_sigma():
    lo, hi = S.generate(noise=0.1, seed=1), S.generate(noise=0.4, seed=1)
    assert abs(np.mean(lo.y / lo.mu) - 1) < 0.02 and abs(np.mean(hi.y / hi.mu) - 1) < 0.05
    assert np.std(hi.y / hi.mu) > 2.5 * np.std(lo.y / lo.mu)


def test_analyse_is_cached_and_consistent():
    s = E.Settings(seed=2)
    a = E.analyse(s)
    assert a is E.analyse(s) and set(a.summary) == set(C.METHODS) and a.best in C.METHODS
    assert len(a.origins) == len(range(C.FIRST_TEST, C.N_DAYS - s.horizon + 1, s.step)) and sum(a.winners.values()) == pytest.approx(1.0)
    assert a.oracle["mase"] < min(v["mase"] for v in a.summary.values())


def test_window_and_step_change_the_result():
    base = E.analyse(E.Settings(seed=2))
    assert E.analyse(E.Settings(seed=2, window=4)).summary["mean"]["mase"] != base.summary["mean"]["mase"]
    assert len(E.analyse(E.Settings(seed=2, step=7)).origins) == pytest.approx(len(base.origins) / 7, abs=1)


def test_experiments_return_consistent_rows():
    r = E.split_experiment(seeds=(0, 1))
    assert sum(r["win_" + m] for m in C.METHODS) == pytest.approx(1.0) and 0 <= r["flip"][0] <= 1 and len(r["rolling_winners"]) == 2 and r["spread_lo"] < r["spread_hi"]
    rows = E.noise_experiment(levels=(0.04, 0.3), seeds=(0, 1))
    assert [x["noise"] for x in rows] == [0.04, 0.3] and rows[1]["floor"] > rows[0]["floor"] and all(x["k_gain"] == pytest.approx(x["snaive"] - x["snaive_k"]) for x in rows)
    wd = E.weekday_experiment(seeds=(0,))
    assert [x["weekday"] for x in wd] == list(range(7)) and max(x["mean"] for x in wd) - min(x["mean"] for x in wd) < 0.05 and max(x["naive"] for x in wd) > 1.3 * min(x["naive"] for x in wd)
    tr = E.trend_experiment(levels=(-20, 40), seeds=(0, 1))
    assert tr[1]["snaive_k"] > tr[0]["snaive_k"]
    wi = E.window_experiment(windows=(0, 4), seeds=(0, 1))
    assert wi[1]["mean"] < wi[0]["mean"] and wi[0]["snaive"] == pytest.approx(wi[1]["snaive"])
