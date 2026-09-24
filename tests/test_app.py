"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Ursprungs-Regler, Wochen-Warnung, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import nf_constants as C
import nf_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_the_seasonal_success_and_all_charts():
    at = _run()
    _ok(at)
    assert len(at.metric) == 4 and len(at.get("plotly_chart")) == 5 and any("Das Wochenmuster trägt" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_no_weekly_pattern_preset_shows_the_level_verdict():
    at = _run()
    next(b for b in at.button if b.key == "preset_Ohne Wochenmuster").click().run()
    _ok(at)
    assert any("Ohne starkes Wochenmuster" in i.value for i in at.info)


def test_multiple_of_seven_step_shows_the_warning_and_other_steps_do_not():
    at = _run(step_slider=7)
    _ok(at)
    assert any("Vielfaches von 7" in w.value for w in at.warning)
    assert not any("Vielfaches von 7" in w.value for w in _run(step_slider=3).warning)


def test_origin_slider_survives_a_shorter_test_range():
    at = _run(horizon_slider=1, origin_slider=1090)
    _ok(at)
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Reihe generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["trend"] = "999"
    at.query_params["noise"] = "0.31"
    at.query_params["weekly"] = "abc"
    at.query_params["window"] = "5"
    at.query_params["horizon"] = "0"
    at.query_params["shift"] = "-35"
    at.run()
    _ok(at)
    assert at.session_state["trend_slider"] == C.TREND_MAX and at.session_state["noise_slider"] == 0.3 and at.session_state["weekly_slider"] == C.DEFAULT_WEEKLY
    assert at.session_state["window_select"] == C.DEFAULT_WINDOW and at.session_state["horizon_slider"] == C.HORIZON_MIN and at.session_state["shift_slider"] == -40


@pytest.mark.parametrize("kw", [dict(trend_slider=C.TREND_MIN, noise_slider=C.NOISE_MAX), dict(trend_slider=C.TREND_MAX, weekly_slider=0.0, yearly_slider=0.0), dict(horizon_slider=C.HORIZON_MAX, step_slider=C.STEP_MAX),
                                dict(shift_slider=-40, window_select=4, k_slider=12), dict(shift_slider=40, window_select=26, k_slider=2, events_slider=1.0), dict(noise_slider=C.NOISE_MIN, events_slider=0.0, horizon_slider=1)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_split_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "split_start").click().run()
    _ok(at)
    assert at.session_state["split_on"] and any(w.value.startswith("**Befund:** Über alle Ursprünge") for w in at.warning)


def test_noise_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    monkeypatch.setattr(C, "NOISE_LEVELS", (0.04, 0.4))
    at = _run()
    next(b for b in at.button if b.key == "noise_start").click().run()
    _ok(at)
    assert at.session_state["noise_on"] and any("Wochenmittel nur" in w.value for w in at.warning)


def test_weekday_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "weekday_start").click().run()
    _ok(at)
    assert at.session_state["weekday_on"] and any("Wochentag des letzten bekannten Tages" in w.value for w in at.warning)


def test_trend_and_window_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    monkeypatch.setattr(C, "TREND_LEVELS", (-20, 0, 40))
    monkeypatch.setattr(C, "WINDOW_OPTIONS", (0, 4, 8))
    at = _run()
    next(b for b in at.button if b.key == "trend_start").click().run()
    _ok(at)
    assert at.session_state["trend_on"] and any("Die Drift hilft hier nicht" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{signed(" not in el.value
