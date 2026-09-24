"""Verfahren von Hand, Rolling-Origin gegen eine unabhängige Schleife, Kennzahlen und Orakel."""

import numpy as np
import pytest

import nf_constants as C
import nf_evaluation as E
import nf_forecast as F
import nf_scenario as S


def week3():
    """Drei gleiche Wochen 10, 20, ..., 70 (Ursprung t = 21, Horizont 7): alle Verfahren von Hand."""
    return np.tile(np.arange(10.0, 80.0, 10.0), 3)


def test_methods_by_hand_on_three_identical_weeks():
    y = week3()
    assert F.forecast("naive", y, 21, 7) == pytest.approx(np.full(7, 70.0))
    assert F.forecast("mean", y, 21, 7) == pytest.approx(np.full(7, 40.0))
    assert F.forecast("snaive", y, 21, 7) == pytest.approx([10, 20, 30, 40, 50, 60, 70])
    assert F.forecast("snaive_k", y, 21, 7, k=3) == pytest.approx([10, 20, 30, 40, 50, 60, 70])
    assert F.forecast("drift", y, 21, 3) == pytest.approx([73.0, 76.0, 79.0])                # (70 - 10) / 20 = 3 je Tag
    assert F.forecast("snaive", y, 21, 10)[7:] == pytest.approx([10, 20, 30])              # über eine Woche hinaus: wieder von vorn


def test_snaive_k_averages_the_same_weekday_and_clamps_k():
    y = np.concatenate([np.full(7, 10.0), np.full(7, 20.0), np.full(7, 60.0)])
    assert F.forecast("snaive_k", y, 21, 7, k=3) == pytest.approx(np.full(7, 30.0))
    assert F.forecast("snaive_k", y, 21, 7, k=2) == pytest.approx(np.full(7, 40.0))
    assert F.forecast("snaive_k", y, 21, 7, k=12) == pytest.approx(np.full(7, 30.0))                 # mehr Wochen als vorhanden: alle vorhandenen
    assert F.forecast("snaive_k", y, 21, 7, window_weeks=1, k=12) == pytest.approx(np.full(7, 60.0))     # Fenster von einer Woche


def test_window_limits_the_history_and_year_needs_a_year():
    y = np.concatenate([np.full(70, 100.0), np.full(28, 200.0)])
    assert F.forecast("mean", y, 98, 3) == pytest.approx(np.full(3, 100 * 70 / 98 + 200 * 28 / 98))
    assert F.forecast("mean", y, 98, 3, window_weeks=4) == pytest.approx(np.full(3, 200.0))
    assert F.forecast("drift", y, 98, 2, window_weeks=4) == pytest.approx([200.0, 200.0])
    long = np.arange(500.0)
    assert F.forecast("snaive_year", long, 450, 3) == pytest.approx(long[450 - 364:450 - 364 + 3])
    assert F.forecast("snaive_year", long[:100], 100, 3) == pytest.approx(long[93:96])           # zu kurz: fällt auf die letzte Woche zurück
    with pytest.raises(ValueError):
        F.forecast("unbekannt", long, 100, 3)


def test_mase_scale_by_hand():
    y = np.array([1.0, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17])
    assert F.mase_scale(y, 14) == pytest.approx(10.0)                                      # jeder Wert liegt 10 über dem der Vorwoche
    assert F.mase_scale(y, 10) == pytest.approx(10.0)


def test_rolling_origin_agrees_with_an_independent_loop():
    ser = S.generate(seed=2)
    y = ser.y
    h, step = 9, 5
    org, actual, errors = F.rolling_origin(y, h, step, window_weeks=8, k=3)
    assert list(org) == list(range(C.FIRST_TEST, len(y) - h + 1, step))
    hist = lambda t: y[max(0, t - 56):t]
    for i, t in enumerate(org):
        for j in range(h):
            assert actual[i, j] == y[t + j]
            assert errors["naive"][i, j] == pytest.approx(hist(t)[-1] - y[t + j])
            assert errors["snaive"][i, j] == pytest.approx(hist(t)[len(hist(t)) - 7 + j % 7] - y[t + j])
            assert errors["snaive_k"][i, j] == pytest.approx(np.mean([hist(t)[len(hist(t)) - 7 * (q + 1) + j % 7] for q in range(3)]) - y[t + j])
            assert errors["mean"][i, j] == pytest.approx(hist(t).mean() - y[t + j])


def test_summary_and_helpers_are_consistent():
    ser = S.generate(seed=1)
    org, actual, errors = F.rolling_origin(ser.y, 10, 3)
    summ = F.summarize(errors, ser.y)
    scale = F.mase_scale(ser.y, C.FIRST_TEST)
    for m in C.METHODS:
        E_ = errors[m]
        assert summ[m]["mae"] == pytest.approx(np.abs(E_).mean()) and summ[m]["mase"] == pytest.approx(summ[m]["mae"] / scale) and summ[m]["me"] == pytest.approx(E_.mean())
        assert summ[m]["rmse"] >= summ[m]["mae"] - 1e-12
        assert F.per_horizon(errors)[m].shape == (10,) and F.per_origin(errors)[m].shape == (len(org),)
        assert F.per_origin(errors)[m].mean() == pytest.approx(summ[m]["mae"]) and F.per_horizon(errors)[m].mean() == pytest.approx(summ[m]["mae"])


def test_oracle_is_a_lower_bound_for_every_method_and_tiny_without_noise():
    for seed in range(4):
        a = E.analyse(E.Settings(seed=seed))
        assert all(a.oracle["mase"] < a.summary[m]["mase"] for m in C.METHODS)
    ser = S.generate(noise=0.0, seed=1)
    org = F.origins(ser.n, 14)
    assert E.oracle_summary(ser, org, 14)["mae"] < 0.5                                     # nur die Rundung auf ganze Aufträge


def test_seasonal_naive_is_exact_on_a_pure_weekly_series():
    ser = S.generate(trend=0, yearly=0.0, noise=0.0, events=0.0, seed=1)
    org, actual, errors = F.rolling_origin(ser.y, 14, 7)
    assert np.abs(errors["snaive"]).max() <= 1.0 and np.abs(errors["snaive_k"]).max() <= 1.0 and np.abs(errors["naive"]).max() > 20
