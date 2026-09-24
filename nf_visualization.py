"""Plotly-Abbildungen der Naive-Prognose-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nf_constants as C
import nf_forecast as F

COLORS = {"mean": "#7f7f7f", "naive": "#e45756", "snaive": "#4c78a8", "snaive_k": "#17becf", "snaive_year": "#b279a2", "drift": "#f58518"}
SHORT = {"mean": "Mittelwert", "naive": "Naiv", "snaive": "Saisonal naiv", "snaive_k": "Wochenmittel", "snaive_year": "Vorjahr", "drift": "Drift"}
ACTUAL = "#14233B"
ORACLE = "#54a24b"
WARN = "#f58518"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def method_label(m, k):
    return f"Saisonal: Mittel der letzten {k} Wochen" if m == "snaive_k" else C.METHOD_NAMES[m]


def build_series(a, origin):
    """Die ganze Reihe (drei Jahre) mit dem Testjahr und dem gewählten Ursprung."""
    s = a.series
    t = np.arange(s.n)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=s.y, mode="lines", name="Tagesaufträge", line=dict(color=ACTUAL, width=1)))
    fig.add_trace(go.Scatter(x=t, y=s.mu, mode="lines", name="Erwartungswert (ohne Rauschen)", line=dict(color=ORACLE, width=1.5, dash="dot")))
    fig.add_vrect(x0=C.FIRST_TEST, x1=s.n, fillcolor="rgba(245,133,24,0.08)", line_width=0, annotation_text="Testjahr (Ursprünge)", annotation_position="top left")
    fig.add_vline(x=origin, line=dict(color=WARN, dash="dash"))
    if s.shift_day >= 0:
        fig.add_vline(x=s.shift_day, line=dict(color="#e45756", dash="dot"), annotation_text="Niveausprung", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 300)


def build_origin(a, origin):
    """Die 42 Tage vor dem Ursprung und die nächsten h Tage: Ist, Erwartungswert und alle Prognosen."""
    s, st = a.series, a.settings
    h = st.horizon
    x_hist = np.arange(origin - 42, origin)
    x_fut = np.arange(origin, origin + h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=s.y[origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=s.y[origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=2), marker=dict(size=6, symbol="circle-open")))
    fig.add_trace(go.Scatter(x=x_fut, y=s.mu[origin:origin + h], mode="lines", name="Erwartungswert", line=dict(color=ORACLE, width=2, dash="dot")))
    for m in C.METHODS:
        f = F.forecast(m, s.y, origin, h, st.window, st.k)
        fig.add_trace(go.Scatter(x=x_fut, y=f, mode="lines", name=method_label(m, st.k), line=dict(color=COLORS[m], width=2 if m == "snaive_k" else 1.4, dash="dash")))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.35))


def build_bars(a):
    """MASE je Verfahren über alle Ursprünge und Horizonte, dazu die Orakel-Untergrenze."""
    ms = sorted(C.METHODS, key=lambda m: a.summary[m]["mase"])
    fig = go.Figure(go.Bar(x=[method_label(m, a.settings.k) for m in ms], y=[a.summary[m]["mase"] for m in ms], marker=dict(color=[COLORS[m] for m in ms]),
                           text=[f"{a.summary[m]['mase']:.2f}".replace(".", ",") for m in ms], textposition="outside", showlegend=False))
    fig.add_hline(y=a.oracle["mase"], line=dict(color=ORACLE, dash="dot"), annotation_text="Orakel (wahrer Erwartungswert)", annotation_position="top right")
    fig.add_hline(y=1.0, line=dict(color="#7f7f7f", dash="dash"), annotation_text="MASE 1 = saisonal naiv im Training", annotation_position="bottom right")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 360)


def build_horizon(a):
    scale = F.mase_scale(a.series.y, C.FIRST_TEST)
    xs = list(range(1, a.settings.horizon + 1))
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=a.horizon_mae[m] / scale, mode="lines+markers", name=method_label(m, a.settings.k), line=dict(color=COLORS[m], width=2 if m == "snaive_k" else 1.5), marker=dict(size=4)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if a.settings.horizon <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 340).update_layout(legend=dict(orientation="h", y=-0.3))


def build_spread(a):
    """Verteilung des Fehlers über die Ursprünge (MAE je Ursprung durch die MASE-Skala)."""
    scale = F.mase_scale(a.series.y, C.FIRST_TEST)
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Box(y=a.origin_mae[m] / scale, name=SHORT[m], marker=dict(color=COLORS[m]), boxmean=True, showlegend=False))
    fig.update_yaxes(title_text="MASE je Ursprung", rangemode="tozero")
    return _base(fig, 340)


def _line_chart(rows, xkey, xtitle, height=340, floor=False, categorical=None, k=C.DEFAULT_K_WEEKS, log=False):
    xs = [r[xkey] for r in rows] if categorical is None else categorical
    fig = go.Figure()
    for m in C.METHODS:
        fig.add_trace(go.Scatter(x=xs, y=[r[m] for r in rows], error_y=dict(type="data", array=[r[m + "_se"] for r in rows]), mode="lines+markers", name=method_label(m, k), line=dict(color=COLORS[m], width=2 if m == "snaive_k" else 1.5)))
    if floor:
        fig.add_trace(go.Scatter(x=xs, y=[r["floor"] for r in rows], mode="lines", name="Orakel (Untergrenze)", line=dict(color=ORACLE, dash="dot", width=2)))
    fig.update_xaxes(title_text=xtitle, type="category" if categorical is not None else None)
    if log:
        fig.update_yaxes(title_text="MASE (kleiner ist besser, logarithmisch)", type="log", tickvals=[0.5, 1, 2, 4], ticktext=["0,5", "1", "2", "4"])
    else:
        fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, height).update_layout(legend=dict(orientation="h", y=-0.3))


def build_noise(rows):
    return _line_chart(rows, "noise", "Streuung des Rauschens", floor=True, log=True)


def build_weekday(rows):
    return _line_chart(rows, "weekday", "Wochentag des letzten bekannten Tages", categorical=list(C.WEEKDAYS))


def build_trend(rows):
    return _line_chart(rows, "trend", "Trend in Prozent je Jahr")


def build_window(rows):
    return _line_chart(rows, "window", "Fenster in Wochen (0 = wachsend)", categorical=["wachsend" if r["window"] == 0 else str(r["window"]) for r in rows])


def build_split(row):
    """Wie oft ist ein Verfahren in einem einzelnen Ursprung das beste? (Mittel über die Reihen)"""
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in C.METHODS], y=[100 * row["win_" + m] for m in C.METHODS], marker=dict(color=[COLORS[m] for m in C.METHODS]), showlegend=False,
                           text=[f"{100 * row['win_' + m]:.0f} %" for m in C.METHODS], textposition="outside"))
    fig.update_yaxes(title_text="Anteil der Ursprünge, an denen das Verfahren gewinnt (%)", range=[0, 100])
    return _base(fig, 320)
