"""Jede Zahl aus README und PRESET_HELP als Test. Die Reihen und Verfahren sind deterministisch (kein Training), die Bänder daher eng gehalten."""

import numpy as np
import pytest

import nf_constants as C
import nf_evaluation as E
import nf_presets as P


def _preset(name, **over):
    p = dict(P.PRESETS[name])
    p.update(over)
    return E.analyse(E.Settings(p["trend"], p["weekly"], p["yearly"], p["noise"], p["shift"], p["events"], p["horizon"], p["step"], p["window"], p["k"], p["seed"]))


def test_standard_preset():
    a = _preset("Standardfall")
    m = {k: v["mase"] for k, v in a.summary.items()}
    assert len(a.origins) == 352 and a.best == "snaive_k" and m["snaive_k"] == pytest.approx(0.88, abs=0.03) and m["snaive"] == pytest.approx(1.12, abs=0.03) and m["snaive_year"] == pytest.approx(1.14, abs=0.03)
    assert m["mean"] == pytest.approx(2.14, abs=0.05) and m["naive"] == pytest.approx(2.68, abs=0.05) and m["drift"] == pytest.approx(2.69, abs=0.05) and a.oracle["mase"] == pytest.approx(0.75, abs=0.03) and 1.15 < m["snaive_k"] / a.oracle["mase"] < 1.2


def test_strong_noise_preset_the_plain_mean_beats_last_week():
    m = {k: v["mase"] for k, v in _preset("Starkes Rauschen (0,4)").summary.items()}
    assert m["mean"] == pytest.approx(1.05, abs=0.03) and m["snaive"] == pytest.approx(1.14, abs=0.03) and m["mean"] < m["snaive"] and m["snaive_k"] == pytest.approx(0.86, abs=0.03)


def test_no_weekly_pattern_preset():
    a = _preset("Ohne Wochenmuster")
    m = {k: v["mase"] for k, v in a.summary.items()}
    assert m["naive"] == pytest.approx(m["snaive"], abs=0.02) and m["naive"] == pytest.approx(1.13, abs=0.03) and m["mean"] == pytest.approx(1.06, abs=0.03) and m["snaive_k"] == pytest.approx(0.89, abs=0.03)


def test_strong_trend_preset():
    m = {k: v["mase"] for k, v in _preset("Starker Trend (+40 %)").summary.items()}
    assert m["snaive_k"] == pytest.approx(1.11, abs=0.03) and m["snaive"] == pytest.approx(1.42, abs=0.03) and m["snaive_year"] == pytest.approx(1.88, abs=0.04) and m["naive"] == pytest.approx(3.39, abs=0.05)
    assert m["drift"] == pytest.approx(3.40, abs=0.05) and m["snaive_year"] > 1.5 * m["snaive_k"]


def test_level_shift_preset_and_the_growing_window_is_worse_for_mean_and_drift():
    a = _preset("Niveausprung +30 %, Fenster 8 Wochen")
    m = {k: v["mase"] for k, v in a.summary.items()}
    assert a.series.shift_day == 809 and m["mean"] == pytest.approx(2.49, abs=0.05) and m["snaive_k"] == pytest.approx(1.08, abs=0.03) and m["snaive"] == pytest.approx(1.36, abs=0.03) and m["snaive_year"] == pytest.approx(2.15, abs=0.05)
    grow = _preset("Niveausprung +30 %, Fenster 8 Wochen", window=0).summary
    assert grow["mean"]["mase"] == pytest.approx(3.02, abs=0.05) and grow["drift"]["mase"] == pytest.approx(3.25, abs=0.05) and a.summary["drift"]["mase"] == pytest.approx(3.46, abs=0.05) and grow["drift"]["mase"] < a.summary["drift"]["mase"]


def test_one_weekday_preset():
    a = _preset("Ursprünge nur an einem Wochentag")
    std = _preset("Standardfall")
    assert len(a.origins) == 51 and a.summary["naive"]["mase"] == pytest.approx(2.21, abs=0.05) and a.summary["naive"]["mase"] < std.summary["naive"]["mase"] - 0.3
    for m in ("snaive", "snaive_k", "mean"):
        assert a.summary[m]["mase"] == pytest.approx(std.summary[m]["mase"], abs=0.02)


@pytest.fixture(scope="module")
def standard_rows():
    rows = [E.analyse(E.Settings(seed=s)) for s in C.EXP_SEEDS]
    return {m: np.array([a.summary[m]["mase"] for a in rows]) for m in C.METHODS}, np.array([a.oracle["mase"] for a in rows])


def test_standard_over_twelve_series(standard_rows):
    m, floor = standard_rows
    mean = {k: float(v.mean()) for k, v in m.items()}
    assert mean["snaive_k"] == pytest.approx(0.95, abs=0.03) and mean["snaive"] == pytest.approx(1.17, abs=0.03) and mean["snaive_year"] == pytest.approx(1.18, abs=0.03) and mean["mean"] == pytest.approx(2.28, abs=0.05)
    assert mean["naive"] == pytest.approx(2.73, abs=0.05) and mean["drift"] == pytest.approx(2.74, abs=0.05) and float(floor.mean()) == pytest.approx(0.74, abs=0.03)
    assert mean["naive"] > 2 * mean["snaive"] and 0.15 < float((m["snaive"] - m["snaive_k"]).mean()) < 0.3 and all(np.all(m["snaive_k"] < m[k]) for k in C.METHODS if k != "snaive_k")


def test_split_experiment_names_the_wrong_winner_in_a_third_of_the_origins():
    r = E.split_experiment()
    assert all(w == "snaive_k" for w in r["rolling_winners"]) and 0.28 < r["flip"][0] < 0.38 and r["flip"][1] < 0.03
    assert r["win_snaive_k"] == pytest.approx(0.674, abs=0.03) and r["win_snaive"] == pytest.approx(0.16, abs=0.03) and r["win_snaive_year"] == pytest.approx(0.165, abs=0.03) and max(r["win_naive"], r["win_mean"], r["win_drift"]) < 0.001
    assert r["spread_lo"] == pytest.approx(0.64, abs=0.04) and r["spread_hi"] == pytest.approx(1.335, abs=0.05)


def test_noise_experiment():
    rows = {r["noise"]: r for r in E.noise_experiment()}
    assert [round(rows[n]["floor"], 2) for n in C.NOISE_LEVELS] == pytest.approx([0.48, 0.64, 0.74, 0.78, 0.81], abs=0.02)
    assert rows[0.04]["k_gain"] == pytest.approx(0.039, abs=0.02) and rows[0.04]["k_gain"] < 2 * rows[0.04]["k_gain_se"] + 0.02 and rows[0.4]["k_gain"] == pytest.approx(0.241, abs=0.03) and rows[0.14]["k_gain"] == pytest.approx(0.217, abs=0.03)
    assert rows[0.4]["mean"] == pytest.approx(1.116, abs=0.03) and rows[0.4]["snaive"] == pytest.approx(1.161, abs=0.03) and rows[0.4]["mean"] < rows[0.4]["snaive"] and rows[0.14]["mean"] > rows[0.14]["snaive"] + 0.8
    assert rows[0.04]["snaive_year"] == pytest.approx(1.58, abs=0.05) and rows[0.4]["snaive_year"] == pytest.approx(1.103, abs=0.04)


def test_weekday_experiment():
    rows = E.weekday_experiment()
    naive = [r["naive"] for r in rows]
    assert naive == pytest.approx([2.39, 2.31, 2.21, 2.28, 2.74, 3.12, 4.09], abs=0.05) and int(np.argmin(naive)) == 2 and int(np.argmax(naive)) == 6
    assert all(r["snaive_k"] == pytest.approx(0.95, abs=0.02) and r["snaive"] == pytest.approx(1.165, abs=0.02) for r in rows)


def test_trend_and_window_experiments():
    tr = {r["trend"]: r for r in E.trend_experiment()}
    assert [round(tr[t]["snaive_k"], 2) for t in C.TREND_LEVELS] == pytest.approx([0.53, 0.84, 0.95, 1.08, 1.19], abs=0.03)
    assert all(abs(tr[t]["drift"] - tr[t]["naive"]) < 0.02 for t in C.TREND_LEVELS) and tr[40]["snaive_year"] == pytest.approx(1.93, abs=0.05) and tr[-20]["snaive_year"] == pytest.approx(1.49, abs=0.05)
    wi = {r["window"]: r for r in E.window_experiment()}
    assert wi[0]["mean"] == pytest.approx(3.00, abs=0.06) and wi[4]["mean"] == pytest.approx(2.43, abs=0.05) and wi[26]["mean"] == pytest.approx(2.72, abs=0.06)
    assert wi[0]["drift"] == pytest.approx(3.20, abs=0.06) and wi[4]["drift"] == pytest.approx(3.67, abs=0.06) and wi[0]["snaive_k"] == pytest.approx(1.147, abs=0.03) and wi[0]["snaive_year"] == pytest.approx(1.96, abs=0.06)
