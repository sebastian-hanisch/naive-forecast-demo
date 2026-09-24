"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. gn_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import nf_constants as C


def _window(value):
    v = int(float(value))
    if v not in C.WINDOW_OPTIONS:
        raise ValueError(value)
    return v


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "weekly_slider": SettingSpec("weekly", float, C.DEFAULT_WEEKLY, C.WEEKLY_MIN, C.WEEKLY_MAX),
    "yearly_slider": SettingSpec("yearly", float, C.DEFAULT_YEARLY, C.YEARLY_MIN, C.YEARLY_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "shift_slider": SettingSpec("shift", int, C.DEFAULT_SHIFT, C.SHIFT_MIN, C.SHIFT_MAX),
    "events_slider": SettingSpec("events", float, C.DEFAULT_EVENTS, C.EVENTS_MIN, C.EVENTS_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "step_slider": SettingSpec("step", int, C.DEFAULT_STEP, C.STEP_MIN, C.STEP_MAX),
    "window_select": SettingSpec("window", _window, C.DEFAULT_WINDOW),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K_WEEKS, C.K_WEEKS_MIN, C.K_WEEKS_MAX),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"trend": "trend_slider", "weekly": "weekly_slider", "yearly": "yearly_slider", "noise": "noise_slider", "shift": "shift_slider", "events": "events_slider", "horizon": "horizon_slider",
               "step": "step_slider", "window": "window_select", "k": "k_slider", "seed": "seed_input"}
STEPS = {"trend_slider": C.TREND_STEP, "weekly_slider": C.WEEKLY_STEP, "yearly_slider": C.YEARLY_STEP, "noise_slider": C.NOISE_STEP, "shift_slider": C.SHIFT_STEP, "events_slider": C.EVENTS_STEP}


def _p(**kw):
    base = {"trend": C.DEFAULT_TREND, "weekly": C.DEFAULT_WEEKLY, "yearly": C.DEFAULT_YEARLY, "noise": C.DEFAULT_NOISE, "shift": C.DEFAULT_SHIFT, "events": C.DEFAULT_EVENTS, "horizon": C.DEFAULT_HORIZON,
            "step": C.DEFAULT_STEP, "window": C.DEFAULT_WINDOW, "k": C.DEFAULT_K_WEEKS, "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Starkes Rauschen (0,4)": _p(noise=0.4),
    "Ohne Wochenmuster": _p(weekly=0.0),
    "Starker Trend (+40 %)": _p(trend=40),
    "Niveausprung +30 %, Fenster 8 Wochen": _p(shift=30, window=8),
    "Ursprünge nur an einem Wochentag": _p(step=7),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "Seed 3, 14 Tage Horizont, 352 Ursprünge im Testjahr: MASE Wochenmittel (k = 4) 0,88, saisonal naiv 1,12, Vorjahr 1,14, Mittelwert 2,14, naiv 2,68, Drift 2,69; Orakel-Untergrenze 0,75. Im Mittel über 12 Reihen liegt das Wochenmittel bei 0,95, saisonal naiv bei 1,17.",
    "Starkes Rauschen (0,4)": "Seed 3, Rauschen 0,4: das Mittel über die ganze Vergangenheit (MASE 1,05) schlägt die saisonal naive letzte Woche (1,14); das Wochenmittel liegt bei 0,86, die Orakel-Untergrenze bei 0,81. Je stärker eine einzelne Woche rauscht, desto mehr lohnt das Mitteln.",
    "Ohne Wochenmuster": "Seed 3, Wochenmuster 0: naiv und saisonal naiv liegen gleichauf (je 1,13), das Mittel bei 1,06, das Wochenmittel bei 0,89. Ohne Wochentagsmuster gibt es nichts zu wiederholen; der letzte Tag ist kein schlechterer Schätzer als die letzte Woche.",
    "Starker Trend (+40 %)": "Seed 3, Trend +40 % je Jahr: alle Verfahren werden schlechter (Wochenmittel 1,11, saisonal naiv 1,42, Vorjahr 1,88, naiv 3,39; Orakel 0,94). Am stärksten trifft es das Vorjahr, das das Wachstum verpasst; die Drift hilft nicht (3,40).",
    "Niveausprung +30 %, Fenster 8 Wochen": "Seed 3, Niveausprung +30 % an Tag 809, Verfahren sehen nur die letzten 8 Wochen: Mittelwert 2,49, Wochenmittel 1,08, saisonal naiv 1,36, Vorjahr 2,15 (blind für den Sprung). Mit wachsendem Fenster läge der Mittelwert bei 3,02 statt 2,49; die Drift (3,46) wäre dagegen mit wachsendem Fenster besser (3,25).",
    "Ursprünge nur an einem Wochentag": "Seed 3, Abstand der Ursprünge 7 Tage (51 statt 352 Ursprünge, alle am selben Wochentag): naiv 2,21 statt 2,68, alle anderen Verfahren wie zuvor. Wer so auswertet, macht das naive Verfahren besser (oder je nach Wochentag schlechter), als es ist.",
}
