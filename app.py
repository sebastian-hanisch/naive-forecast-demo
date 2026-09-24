"""Naive Prognose und Rolling-Origin-Auswertung - die Messlatte jeder Prognose - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Erstes Stück (Wurzel) der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: Tagesaufträge eines Depots über drei Jahre, ein paar Verfahren, die nur die Vergangenheit fortschreiben, und die Frage, wie man Prognosen überhaupt
fair vergleicht. Die Messlatte, an der alle Nachfolger gemessen werden.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import nf_constants as C
import nf_forecast as F
from nf_evaluation import Settings, analyse, noise_experiment, split_experiment, trend_experiment, weekday_experiment, window_experiment
from nf_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from nf_visualization import SHORT, build_bars, build_horizon, build_noise, build_origin, build_series, build_split, build_spread, build_trend, build_weekday, build_window, method_label

st.set_page_config(page_title="Naive Prognose – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def signed(x, se=None, digits=2):
    s = f"{'+' if x >= 0 else '−'}{de(abs(x), digits)}"
    return s + (f" ± {de(se, digits)}" if se is not None else "")


@st.cache_data(show_spinner=False)
def _split(seeds):
    return split_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _noise(levels, seeds):
    return noise_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _weekday(seeds):
    return weekday_experiment(seeds=seeds)


@st.cache_data(show_spinner=False)
def _trend(levels, seeds):
    return trend_experiment(levels=levels, seeds=seeds)


@st.cache_data(show_spinner=False)
def _window(windows, seeds):
    return window_experiment(windows=windows, seeds=seeds)


st.title("📏 Naive Prognose – die Messlatte")
st.markdown(
    """
Bevor ein Prognoseverfahren etwas taugt, muss es eine **naive Prognose** schlagen: den letzten Wert fortschreiben, den Wert derselben Woche wiederholen, über ein paar Wochen mitteln. Die Demo zeigt an den **Tagesaufträgen eines Depots**
(drei Jahre, Wochenmuster, Trend, Feiertage, Aktionen), wie weit diese Verfahren kommen, und - genauso wichtig - **wie man Prognosen fair vergleicht**: mit vielen Ursprüngen statt einem Testzeitraum (Rolling-Origin) und mit einer Kennzahl,
die nicht vom Niveau der Reihe abhängt (MASE). Dazu eine **Untergrenze**, die kein Verfahren unterschreiten kann: das Rauschen der Reihe selbst.
"""
)
st.caption(
    "Erstes Stück (Wurzel) der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe und die erste Linie des Portfolios mit Prognosen; alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben. "
    "**Bezug zu OR:** jede Bestands-, Personal- und Tourenplanung beginnt mit einer Nachfrageprognose - und mit der Frage, wie gut sie ist."
)

with st.expander("So funktioniert der Vergleich", expanded=True):
    st.markdown(
        """
1. **Ursprung.** Ein Ursprung $t$ heißt: bekannt sind die Tage $0..t-1$, prognostiziert werden die nächsten $h$ Tage. Der Ursprung wandert durch das **letzte Jahr** (Rolling-Origin); über alle Ursprünge und Horizonte wird der Fehler gemittelt.
2. **Verfahren.** *Mittelwert* (Mittel der Vergangenheit), *naiv* (letzter Tag), *saisonal naiv* (derselbe Wochentag der letzten Woche), *Wochenmittel* (derselbe Wochentag, gemittelt über die letzten $k$ Wochen), *Vorjahr* (derselbe Wochentag vor 52 Wochen)
   und *Drift* (letzter Tag plus mittlere Tagesänderung). Auf Wunsch sehen sie nur ein gleitendes Fenster der letzten Wochen.
3. **MASE.** Der mittlere absolute Fehler geteilt durch den mittleren absoluten Fehler der saisonal naiven Prognose im Training (Hyndman/Koehler 2006). Eine MASE unter 1 heißt: besser als saisonal naiv im Training; sie ist von der Größe des Depots unabhängig.
4. **Orakel.** Weil die Reihe erzeugt ist, ist ihr wahrer Erwartungswert bekannt. Der Fehler der Prognose "Erwartungswert" ist die **Untergrenze** im Mittel: darunter liegt nur Glück.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.caption("Die Reihe")
    trend = st.slider("Trend (% je Jahr)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Lineares Wachstum (oder Schrumpfen) der Aufträge, in Prozent des Ausgangsniveaus je Jahr.")
    weekly = st.slider("Wochenmuster", *bounds("weekly_slider"), key="weekly_slider", step=C.WEEKLY_STEP, help="Stärke des Wochentagsmusters (1 = Standard: Freitag am stärksten, Sonntag am schwächsten; 0 = keines).")
    yearly = st.slider("Jahresmuster", *bounds("yearly_slider"), key="yearly_slider", step=C.YEARLY_STEP, help="Amplitude der jahreszeitlichen Schwankung (Anteil des Niveaus).")
    noise = st.slider("Rauschen", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung des multiplikativen Rauschens (log-normal, im Mittel unverzerrt).")
    shift = st.slider("Niveausprung (%)", *bounds("shift_slider"), key="shift_slider", step=C.SHIFT_STEP, help="Sprung des Niveaus an einem zufälligen Tag im letzten Jahr (0 = keiner), z. B. ein neuer Großkunde.")
    events = st.slider("Feiertage und Aktionen", *bounds("events_slider"), key="events_slider", step=C.EVENTS_STEP, help="Stärke der Effekte: am Feiertag ruht das Depot (bis -50 %), am Folgetag Nachholeffekt, dazu drei Aktionswochen je Jahr (bis +50 %). Die naiven Verfahren kennen sie nicht.")
    st.caption("Der Vergleich")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    step = st.slider("Abstand der Ursprünge (Tage)", *bounds("step_slider"), key="step_slider", help="Alle wie viele Tage ein neuer Ursprung beginnt. 1 = jeder Tag des letzten Jahres. Ein Vielfaches von 7 wertet Ursprünge nur an einem Wochentag aus.")
    window = st.selectbox("Fenster der Verfahren", C.WINDOW_OPTIONS, key="window_select", format_func=lambda w: "wachsend (alle bisherigen Tage)" if w == 0 else f"gleitend, {w} Wochen",
                          help="Wie viel Vergangenheit Mittelwert, Naiv, Drift und das Wochenmittel sehen. Das Vorjahr-Verfahren braucht immer das ganze Jahr.")
    k = st.slider("Wochen im Wochenmittel (k)", *bounds("k_slider"), key="k_slider", help="Über wie viele letzte Wochen das Verfahren 'Wochenmittel' den Wochentag mittelt.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Rauschen, Aktionstage, Phase des Jahresmusters und den Tag des Niveausprungs fest.")
    st.button("🎲 Neue Reihe generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"trend_slider": int(trend), "weekly_slider": round(float(weekly), 2), "yearly_slider": round(float(yearly), 2), "noise_slider": round(float(noise), 2), "shift_slider": int(shift), "events_slider": round(float(events), 2),
                   "horizon_slider": int(horizon), "step_slider": int(step), "window_select": int(window), "k_slider": int(k), "seed_input": int(seed)})

settings = Settings(int(trend), round(float(weekly), 2), round(float(yearly), 2), round(float(noise), 2), int(shift), round(float(events), 2), int(horizon), int(step), int(window), int(k), int(seed))
a = analyse(settings)
s = a.series
K = settings.k

# --- Die Reihe ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Reihe und die Prognosen an einem Ursprung")
lo_o, hi_o = int(a.origins[0]), int(a.origins[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, origin), width="stretch", key="series_chart")
st.plotly_chart(build_origin(a, origin), width="stretch", key="origin_chart")
weekday_name = C.WEEKDAYS[(origin - 1) % 7]
st.caption(
    f"Reihe mit {s.n} Tagen (Mittel {de(s.y.mean())} Aufträge je Tag); Ursprung an Tag {origin} (letzter bekannter Tag: ein {weekday_name}). Die gestrichelten Linien sind die Prognosen der sechs Verfahren für die nächsten {settings.horizon} Tage, "
    "die grüne gepunktete Linie der wahre Erwartungswert. Beachten Sie, wie der naive Wert (rot) den Wochentag des letzten bekannten Tages fortschreibt."
)

st.markdown("---")

# --- Auswertung ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Auswertung über alle Ursprünge")
n_org = len(a.origins)
if settings.step % 7 == 0:
    st.warning("⚠️ Der Abstand der Ursprünge ist ein Vielfaches von 7: alle Ursprünge liegen am selben Wochentag. Verfahren, die den letzten Tag fortschreiben (naiv, Drift), hängen dann vom Zufall dieses Wochentags ab (Experiment unten).")
best = a.best
sm = a.summary
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Bestes Verfahren: {SHORT[best]}", f"MASE {de(sm[best]['mase'], 2)}", help="Kleinste MASE über alle Ursprünge und Horizonte.")
c2.metric("Saisonal naiv (letzte Woche)", f"MASE {de(sm['snaive']['mase'], 2)}", help="Derselbe Wochentag der letzten Woche.")
c3.metric("Naiv (letzter Tag)", f"MASE {de(sm['naive']['mase'], 2)}", help="Der letzte beobachtete Tag, für alle Horizonte fortgeschrieben.")
c4.metric("Orakel-Untergrenze", f"MASE {de(a.oracle['mase'], 2)}", help="Fehler der Prognose 'wahrer Erwartungswert' gegen die beobachteten Werte: das Rauschen der Reihe. Kein Verfahren liegt im Mittel darunter.")
st.plotly_chart(build_bars(a), width="stretch", key="bars_chart")
rows = [{"Verfahren": method_label(m, K), "MASE": de(sm[m]["mase"], 2), "MAE (Aufträge)": de(sm[m]["mae"], 1), "RMSE": de(sm[m]["rmse"], 1), "Verzerrung (Prognose minus Ist)": de(sm[m]["me"], 1), "Bester Ursprung (Anteil)": pct(a.winners[m])} for m in sorted(C.METHODS, key=lambda m: sm[m]["mase"])]
st.dataframe(rows, hide_index=True)
gap = sm[best]["mase"] / a.oracle["mase"] - 1
if sm["naive"]["mase"] > 1.5 * sm["snaive"]["mase"]:
    st.success(f"✅ Das Wochenmuster trägt: saisonal naiv erreicht MASE {de(sm['snaive']['mase'], 2)} gegen {de(sm['naive']['mase'], 2)} beim letzten Tag. Das beste Verfahren ({SHORT[best]}, {de(sm[best]['mase'], 2)}) liegt noch {pct(gap)} über der "
               f"Orakel-Untergrenze ({de(a.oracle['mase'], 2)}) - so viel Spielraum bleibt für bessere Verfahren.")
elif sm["naive"]["mase"] < 1.1 * sm["snaive"]["mase"]:
    st.info(f"Ohne starkes Wochenmuster liegen naiv ({de(sm['naive']['mase'], 2)}) und saisonal naiv ({de(sm['snaive']['mase'], 2)}) gleichauf; das beste Verfahren ist {SHORT[best]} mit {de(sm[best]['mase'], 2)} (Orakel {de(a.oracle['mase'], 2)}).")
else:
    st.info(f"Naiv {de(sm['naive']['mase'], 2)}, saisonal naiv {de(sm['snaive']['mase'], 2)}, bestes Verfahren {SHORT[best]} {de(sm[best]['mase'], 2)} (Orakel {de(a.oracle['mase'], 2)}).")
st.caption(f"{n_org} Ursprünge im Testjahr (Abstand {settings.step} Tage), je {settings.horizon} Tage Horizont; MASE-Nenner: saisonal naiver Fehler in den ersten {C.FIRST_TEST} Tagen ({de(F.mase_scale(s.y, C.FIRST_TEST), 1)} Aufträge).")

st.markdown("##### Wie der Fehler mit dem Horizont wächst")
st.plotly_chart(build_horizon(a), width="stretch", key="horizon_chart")
st.markdown("##### Wie stark der Fehler von Ursprung zu Ursprung streut")
st.plotly_chart(build_spread(a), width="stretch", key="spread_chart")
spread = a.origin_mae[best] / F.mase_scale(s.y, C.FIRST_TEST)
st.caption(f"Beim besten Verfahren ({SHORT[best]}) liegt die MASE je Ursprung zwischen {de(float(spread.min()), 2)} und {de(float(spread.max()), 2)} (10. bis 90. Perzentil: {de(float(np.percentile(spread, 10)), 2)} bis {de(float(np.percentile(spread, 90)), 2)}). "
           "Ein einzelner Testzeitraum zeigt nur einen dieser Werte.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie verlässlich ist ein einzelner Testzeitraum?")
st.caption(f"Standardreihe, alle Ursprünge des Testjahres (Abstand 1 Tag); je Ursprung wird das Verfahren mit dem kleinsten Fehler bestimmt. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Einzelne Ursprünge auswerten", key="split_start"):
    st.session_state["split_on"] = True
if st.session_state.get("split_on"):
    r = _split(C.EXP_SEEDS)
    st.plotly_chart(build_split(r), width="stretch", key="split_chart")
    winner = max(C.METHODS, key=lambda m: r["win_" + m])
    st.warning(
        f"**Befund:** Über alle Ursprünge ist {SHORT[r['rolling_winners'][0]]} in allen {r['n_seeds']} Reihen das beste Verfahren. In einem einzelnen Ursprung gewinnt es aber nur in {pct(r['win_' + winner])} der Fälle; in {pct(r['flip'][0], 1)} ± {pct(r['flip'][1], 1)} "
        f"der Ursprünge steht ein anderes Verfahren vorn - wer nur einen Testzeitraum auswertet, kürt in etwa jedem dritten Fall den falschen Gewinner. Selbst beim besten Verfahren liegt die MASE je Ursprung zwischen {de(r['spread_lo'], 2)} und {de(r['spread_hi'], 2)} "
        "(10. bis 90. Perzentil, Mittel über die Reihen)."
    )

st.markdown("---")

st.subheader("🔬 Wann hilft Mitteln, wann schadet es?")
st.caption(f"Standardreihe, Rauschen {', '.join(de(x, 2) for x in C.NOISE_LEVELS)}; Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer wenige Sekunden.")
if st.button("Rauschen durchrechnen", key="noise_start"):
    st.session_state["noise_on"] = True
if st.session_state.get("noise_on"):
    rows_n = _noise(C.NOISE_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_noise(rows_n), width="stretch", key="noise_chart")
    lo, hi = rows_n[0], rows_n[-1]
    st.warning(
        f"**Befund:** Bei Rauschen {de(lo['noise'], 2)} liegt das Wochenmittel nur {de(lo['k_gain'], 2)} ± {de(lo['k_gain_se'], 2)} MASE-Punkte vor der letzten Woche, bei Rauschen {de(hi['noise'], 2)} sind es {de(hi['k_gain'], 2)} ± {de(hi['k_gain_se'], 2)}: je mehr die einzelne Woche "
        f"rauscht, desto mehr lohnt es, mehrere zu mitteln. Bei Rauschen {de(hi['noise'], 2)} schlägt sogar das Mittel über die ganze Vergangenheit ({de(hi['mean'], 2)}) die saisonal naive letzte Woche ({de(hi['snaive'], 2)}), und die Orakel-Untergrenze steigt von "
        f"{de(lo['floor'], 2)} auf {de(hi['floor'], 2)}."
    )

st.markdown("---")

st.subheader("🔬 Hängt das Ergebnis vom Wochentag des Ursprungs ab?")
st.caption(f"Ursprünge nur an einem Wochentag (Abstand 7); der Wochentag des letzten bekannten Tages läuft von Montag bis Sonntag; Standardreihe, Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Wochentage durchrechnen", key="weekday_start"):
    st.session_state["weekday_on"] = True
if st.session_state.get("weekday_on"):
    rows_w = _weekday(C.EXP_SEEDS)
    st.plotly_chart(build_weekday(rows_w), width="stretch", key="weekday_chart")
    nv = [r["naive"] for r in rows_w]
    imin, imax = int(np.argmin(nv)), int(np.argmax(nv))
    sv = [r["snaive_k"] for r in rows_w]
    st.warning(
        f"**Befund:** Die MASE des naiven Verfahrens reicht je nach Wochentag des letzten bekannten Tages von {de(min(nv), 2)} ({C.WEEKDAYS[imin]}) bis {de(max(nv), 2)} ({C.WEEKDAYS[imax]}); es schreibt den Wert dieses Tages für alle Horizonte fort. "
        f"Das Wochenmittel bleibt bei {de(min(sv), 2)} bis {de(max(sv), 2)}. Wer nur Ursprünge an einem Wochentag auswertet, macht das naive Verfahren besser oder schlechter, als es ist."
    )

st.markdown("---")

st.subheader("🔬 Trend und Fenster")
st.caption(f"Links: Trend {', '.join(str(x) for x in C.TREND_LEVELS)} % je Jahr. Rechts: Niveausprung +30 % im Testjahr, Fenster {', '.join('wachsend' if w == 0 else str(w) for w in C.WINDOW_OPTIONS)} Wochen. Standardreihe, Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer wenige Sekunden.")
if st.button("Trend und Fenster durchrechnen", key="trend_start"):
    st.session_state["trend_on"] = True
if st.session_state.get("trend_on"):
    rows_t = _trend(C.TREND_LEVELS, C.EXP_SEEDS)
    rows_f = _window(C.WINDOW_OPTIONS, C.EXP_SEEDS)
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(build_trend(rows_t), width="stretch", key="trend_chart")
    with d2:
        st.plotly_chart(build_window(rows_f), width="stretch", key="window_chart")
    drift_gap = max(abs(r["drift"] - r["naive"]) for r in rows_t)
    top = rows_t[-1]
    wr = {r["window"]: r for r in rows_f}
    st.warning(
        f"**Befund:** Die Drift hilft hier nicht: sie liegt höchstens {de(drift_gap, 2)} MASE-Punkte neben dem naiven Verfahren, weil der letzte Tag vom Wochentag dominiert wird. Bei Trend {top['trend']} % steigt der Fehler aller Verfahren (Wochenmittel {de(top['snaive_k'], 2)}), "
        f"am stärksten beim Vorjahr ({de(top['snaive_year'], 2)}), das das Wachstum verpasst. Nach einem Niveausprung von 30 % sinkt die MASE des Mittelwerts von {de(wr[0]['mean'], 2)} (wachsendes Fenster) auf {de(wr[4]['mean'], 2)} (Fenster 4 Wochen); "
        f"die Drift wird mit kleinem Fenster dagegen schlechter ({de(wr[0]['drift'], 2)} auf {de(wr[4]['drift'], 2)}), weil eine kurze Steigung vor allem den Wochentag misst."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zukunft ähnelt der letzten Woche** | Trend, Niveausprung, Feiertage und Aktionen kennen die naiven Verfahren nicht: der Fehler wächst mit dem Trend, nach einem Sprung dauert es Wochen. | Exponentielle Glättung, Dynamische Regression (nächste Stücke) |
| **Eine Reihe genügt** | Alle Verfahren sehen nur die eigene Reihe; ähnliche Depots teilen ihr Wissen nicht. | Globale Modelle (Boosting, Vortrainiertes Netz) |
| **Es gibt eine Punktprognose** | Das Rauschen der Reihe bleibt: die Untergrenze ist nicht null, und die Prognose sagt nichts über das Risiko. | Prognoseintervalle |
| **Der Bedarf ist nie null** | Bei vielen Nullen verzerren Mittelwerte die Prognose; hier liegt der Bedarf bei 100 Aufträgen je Tag. | Croston, SBA, TSB |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher. Die Zahlen gelten für diese Reihen und Größen. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, Kombination, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Reihe.** $y_t = \mathrm{round}\big(\mu_t \, e^{\sigma z_t - \sigma^2/2}\big)$ mit $z_t \sim \mathcal N(0,1)$ und $\mu_t = L\,(1 + g\,t/365)\,w_{t \bmod 7}\,(1 + a \sin(2\pi t/365 + \varphi))\,s_t\,f_t\,p_t$
(Niveau $L$, Trend $g$, Wochenmuster $w$, Jahresmuster, Niveausprung $s_t$, Feiertage $f_t$, Aktionen $p_t$).

**Verfahren** für den Ursprung $t$ und den Horizont $j = 1..h$ (bekannt: $y_0 .. y_{t-1}$, $m$ = Länge der sichtbaren Vergangenheit):
Mittelwert $\bar y$; naiv $y_{t-1}$; saisonal naiv $y_{t-7+((j-1) \bmod 7)}$; Wochenmittel $\frac1k\sum_{i=1}^{k} y_{t-7i+((j-1) \bmod 7)}$; Vorjahr $y_{t-364+j-1}$; Drift $y_{t-1} + j\,\frac{y_{t-1} - y_{t-m}}{m-1}$.

**Kennzahlen.** $\mathrm{MAE} = \frac1{|O|h}\sum_{t \in O}\sum_{j}\lvert \hat y_{t+j-1} - y_{t+j-1}\rvert$ über alle Ursprünge $O$, $\mathrm{RMSE}$ und Verzerrung $\mathrm{ME}$ analog;
$\mathrm{MASE} = \mathrm{MAE} / \frac1{T-7}\sum_{u=7}^{T-1}\lvert y_u - y_{u-7}\rvert$ mit $T$ = Länge der ersten Trainingsdaten (Hyndman/Koehler 2006). **Orakel:** dieselbe MAE mit $\hat y = \mu$ - die Untergrenze im Mittel.

Implementiert in `nf_forecast.py` (Verfahren, Rolling-Origin, Kennzahlen), `nf_scenario.py` (die Reihe), `nf_evaluation.py` (Analyse, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
