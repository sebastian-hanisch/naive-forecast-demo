# 📏 Naive Prognose – die Messlatte

**[→ Demo live ausprobieren](https://sebastianhanisch-naive-forecast-demo.streamlit.app/)**

Erstes Stück (Wurzel) der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und die erste Linie des Portfolios mit Prognosen
(geplant sind zehn weitere Stücke: Exponentielle Glättung, ARIMA, Dynamische Regression, Croston-Verfahren, Boosting, Prognoseintervalle, Hierarchische Abstimmung, Kombination, Prognose → Bestand und ein vortrainiertes Netz; noch nicht gebaut).

Bevor ein Prognoseverfahren etwas taugt, muss es eine **naive Prognose** schlagen: den letzten Wert fortschreiben, den Wert derselben Woche wiederholen, über ein paar Wochen mitteln. Die Demo zeigt an den **Tagesaufträgen eines Depots** (drei Jahre täglich, Wochenmuster, Trend,
Jahresmuster, Feiertage, Aktionen, auf Wunsch ein Niveausprung), wie weit sechs solche Verfahren kommen – und genauso wichtig, **wie man Prognosen fair vergleicht**: mit vielen Ursprüngen statt einem Testzeitraum (Rolling-Origin) und mit einer Kennzahl, die nicht vom Niveau der Reihe abhängt (MASE).
Weil die Reihe erzeugt ist, gibt es außerdem eine **Untergrenze**, die kein Verfahren im Mittel unterschreiten kann: das Rauschen der Reihe selbst (die Prognose "wahrer Erwartungswert"). Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben.

**Bezug zu OR:** jede Bestands-, Personal- und Tourenplanung beginnt mit einer Nachfrageprognose – und mit der Frage, wie gut sie ist.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Naive ist schwer zu schlagen". Die Messung differenziert das:

1. **Der letzte Tag ist bei einem Wochenmuster keine gute Messlatte.** Naiv (MASE 2,73) liegt mehr als doppelt so hoch wie saisonal naiv (1,17); erst die Wiederholung desselben Wochentags ist die Latte, die ein Verfahren reißen muss.
2. **Die schwerste einfache Latte ist nicht "saisonal naiv", sondern das Wochenmittel** (derselbe Wochentag, gemittelt über die letzten 4 Wochen; MASE 0,95): es gewinnt in allen zwölf Reihen und liegt 0,22 Punkte vor der letzten Woche allein. Ein Verfahren, das nur "die letzte Woche" wiederholt, ist keine faire Latte.
3. **Die Drift hilft nicht.** Sie liegt bei jedem Trend höchstens 0,02 MASE-Punkte neben dem naiven Verfahren, weil der letzte Tag vom Wochentag dominiert wird, nicht vom Trend.
4. **Ein einzelner Testzeitraum kürt in etwa jedem dritten Fall den falschen Gewinner** und hängt beim naiven Verfahren vom Wochentag des Ursprungs ab – das war der Anlass für die Rolling-Origin-Auswertung als eigentliches Thema dieses Stücks.

## Modell

- **Die Reihe** (`nf_scenario.py`): $y_t = \mathrm{round}(\mu_t\,e^{\sigma z_t - \sigma^2/2})$ mit $\mu_t = L\,(1+g\,t/365)\,w_{t \bmod 7}\,(1 + a\sin(2\pi t/365+\varphi))\,s_t\,f_t\,p_t$: Niveau $L = 100$ Aufträge, Trend, Wochenmuster (Mo bis So: 1,10 / 1,05 / 1,00 / 1,05 / 1,20 / 0,55 / 0,35, auf Mittel 1 normiert), Jahresmuster,
  Niveausprung, Feiertage (zehn Ruhetage je Jahr, am Folgetag ein Nachholeffekt), drei Aktionswochen je Jahr; multiplikatives, mittelwerttreues log-normales Rauschen. **1 095 Tage**; die Ursprünge liegen im **letzten Jahr** (ab Tag 730).
- **Verfahren** (`nf_forecast.py`): Mittelwert, naiv, saisonal naiv, Wochenmittel (k Wochen), Vorjahr (52 Wochen zurück), Drift; auf Wunsch nur ein gleitendes Fenster der letzten 4 bis 26 Wochen.
- **Kennzahlen:** MAE, RMSE, Verzerrung (Prognose minus Ist), **MASE** (MAE geteilt durch den saisonal naiven Fehler innerhalb der ersten 730 Trainingstage; Hyndman/Koehler 2006); Auswertung über alle Ursprünge (Standard: jeder Tag des Testjahres, Horizont 14 Tage).
- **Orakel:** dieselbe Kennzahl für die Prognose $\hat y = \mu$; im Mittel die untere Grenze.

## Methodik

- **Handrechnungen:** alle Verfahren auf drei gleichen Wochen 10, 20, …, 70 (naiv 70, Mittelwert 40, saisonal naiv 10…70, Drift 73/76/79), Wochenmittel mit Mittelwert 30 bzw. 40, Fenster und Vorjahres-Rückfall, MASE-Nenner auf einer Reihe mit konstantem Wochensprung (10).
- **Gegenprobe:** die vektorisierte Rolling-Origin-Auswertung gegen eine **unabhängige Schleife** (naiv, saisonal naiv, Wochenmittel, Mittelwert; Fenster von 8 Wochen, Abstand 5, Horizont 9); saisonal naiv ist auf einer reinen Wochenreihe ohne Rauschen exakt (Fehler höchstens 1, die Rundung).
- **Die Reihe:** jede Komponente einzeln (Niveau bei ausgeschalteten Komponenten, Wochenmuster gemittelt 1, Trend linear, Feiertag −50 % und Nachholeffekt +15 %, Aktionen +50 %, Niveausprung im Testjahr, Rauschen mittelwerttreu); die Orakel-Untergrenze liegt bei jeder Testreihe unter allen Verfahren.
- **Statistik:** zwölf feste Seeds, Fehlerbalken = Standardfehler; die Verfahren sind deterministisch (kein Training), die Zahlen daher exakt reproduzierbar.
- **Literatur** (nicht nachgebaut): Hyndman/Koehler 2006 ("Another look at measures of forecast accuracy", MASE); Hyndman/Athanasopoulos, *Forecasting: Principles and Practice* (3. Aufl.; einfache Verfahren, Zeitreihen-Kreuzvalidierung).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardreihen** (12 Seeds, Horizont 14, 352 Ursprünge je Reihe) | MASE: Wochenmittel **0,95**, saisonal naiv **1,17**, Vorjahr 1,18, Mittelwert 2,28, naiv **2,73**, Drift 2,74; Orakel-Untergrenze **0,74**. Das Wochenmittel gewinnt in allen zwölf Reihen und liegt im Mittel 0,22 vor der letzten Woche. | `test_standard_over_twelve_series` |
| Standardfall (Preset, Seed 3) | Wochenmittel 0,88, saisonal naiv 1,12, Vorjahr 1,14, Mittelwert 2,14, naiv 2,68, Drift 2,69, Orakel 0,75 (das beste Verfahren liegt 17 % darüber). | `test_standard_preset` |
| **Wie verlässlich ist ein einzelner Testzeitraum?** (alle Ursprünge des Testjahres, 12 Seeds) | Über alle Ursprünge gewinnt das Wochenmittel in allen 12 Reihen; in einem einzelnen Ursprung aber nur in **67,4 %** der Fälle (saisonal naiv 16,0 %, Vorjahr 16,5 %, naiv, Mittelwert und Drift praktisch nie). In **32,6 ± 2,1 %** der Ursprünge steht ein anderes Verfahren vorn. Die MASE des besten Verfahrens streut je Ursprung zwischen 0,64 und 1,34 (10. bis 90. Perzentil). | `test_split_experiment_names_the_wrong_winner_in_a_third_of_the_origins` |
| **Wann hilft Mitteln?** (Rauschen 0,04 / 0,08 / 0,14 / 0,24 / 0,40) | Vorsprung des Wochenmittels vor der letzten Woche: **+0,04 ± 0,02**, +0,17 ± 0,01, +0,22 ± 0,01, +0,24 ± 0,01, +0,24 ± 0,01 MASE-Punkte. Orakel-Untergrenze 0,48 / 0,64 / 0,74 / 0,78 / 0,81. Bei Rauschen 0,40 schlägt sogar der **Mittelwert über die ganze Vergangenheit (1,12) die saisonal naive letzte Woche (1,16)**; das Vorjahr liegt bei 0,04 mit 1,58, bei 0,40 mit 1,10. | `test_noise_experiment` |
| **Hängt das Ergebnis vom Wochentag des Ursprungs ab?** (Abstand 7, letzter bekannter Tag Mo bis So) | Naiv: 2,39 / 2,31 / **2,21** / 2,28 / 2,74 / 3,12 / **4,09**; Wochenmittel dagegen immer 0,95, saisonal naiv 1,17. Wer nur Ursprünge an einem Wochentag auswertet, macht das naive Verfahren besser oder schlechter, als es ist. Preset (Seed 3, 51 Ursprünge am selben Wochentag): naiv 2,21 statt 2,68. | `test_weekday_experiment`, `test_one_weekday_preset` |
| **Trend** (−20 / 0 / +10 / +25 / +40 % je Jahr) | Wochenmittel 0,53 / 0,84 / 0,95 / 1,08 / 1,19; Vorjahr 1,49 bei −20 % und **1,93** bei +40 % (verpasst das Wachstum); Drift höchstens 0,02 neben naiv. Preset "Starker Trend": Wochenmittel 1,11, saisonal naiv 1,42, Vorjahr 1,88, naiv 3,39, Drift 3,40, Orakel 0,94. | `test_trend_and_window_experiments`, `test_strong_trend_preset` |
| **Niveausprung +30 %** und Fenster (wachsend / 4 / 8 / 13 / 26 Wochen) | Mittelwert: **3,00** (wachsend), **2,43** (4 Wochen), 2,72 (26 Wochen); Drift dagegen schlechter mit kleinem Fenster: **3,20** (wachsend), **3,67** (4 Wochen), weil eine kurze Steigung vor allem den Wochentag misst; Wochenmittel 1,15, Vorjahr 1,96 (blind für den Sprung). Preset (Seed 3, Sprung an Tag 809, Fenster 8 Wochen): Mittelwert 2,49 (wachsend 3,02), Wochenmittel 1,08, saisonal naiv 1,36, Vorjahr 2,15, Drift 3,46 (wachsend 3,25). | `test_trend_and_window_experiments`, `test_level_shift_preset_and_the_growing_window_is_worse_for_mean_and_drift` |
| Starkes Rauschen (Preset, Seed 3, 0,4) | Mittelwert 1,05 schlägt saisonal naiv 1,14; Wochenmittel 0,86, Orakel 0,81. | `test_strong_noise_preset_the_plain_mean_beats_last_week` |
| Ohne Wochenmuster (Preset, Seed 3) | naiv und saisonal naiv gleichauf (je 1,13), Mittelwert 1,06, Wochenmittel 0,89. | `test_no_weekly_pattern_preset` |

Die Preset-Zeilen sind **Einzelreihen** (Seed 3); belastbar sind die Zeilen über zwölf Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Zukunft ähnelt der letzten Woche** | Trend, Niveausprung, Feiertage und Aktionen kennen die naiven Verfahren nicht: der Fehler wächst mit dem Trend, nach einem Sprung dauert es Wochen. | Exponentielle Glättung, Dynamische Regression (geplant) |
| **Eine Reihe genügt** | Alle Verfahren sehen nur die eigene Reihe; ähnliche Depots teilen ihr Wissen nicht. | Globale Modelle: Boosting, Vortrainiertes Netz (geplant) |
| **Es gibt eine Punktprognose** | Das Rauschen der Reihe bleibt: die Untergrenze ist nicht null, und die Prognose sagt nichts über das Risiko. | Prognoseintervalle (geplant) |
| **Der Bedarf ist nie null** | Bei vielen Nullen verzerren Mittelwerte die Prognose; hier liegt der Bedarf bei 100 Aufträgen je Tag. | Croston, SBA, TSB (geplant) |
| **Erzeugte Reihe, zwölf Seeds** | Das Vehikel kennt genau die Muster, die es erzeugt; echte Reihen sind unordentlicher (Ausreißer, Lücken, wechselnde Wochenmuster). Die Zahlen gelten für diese Reihen und Größen. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 15 Sekunden): Verfahren von Hand und gegen eine unabhängige Schleife, Kennzahlen und Orakel, die Reihe (jede Komponente einzeln), Auswertung und Experimentzeilen, Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Ursprungs-Regler bei kürzerem Testbereich,
Wochen-Warnung, Extremwerte, vier Experimente auf Abruf) und `test_claims.py` (jede Zahl aus diesem README; Reihen und Verfahren sind deterministisch, die Bänder daher eng).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `nf_constants.py` | Regler-Grenzen, Wochenmuster, Feiertage, Verfahrensnamen, Experiment-Seeds |
| `nf_presets.py` | Permalink/Presets-Mechanik |
| `nf_scenario.py` | Die Reihe (Komponenten, Rauschen, Erwartungswert) |
| `nf_forecast.py` | Verfahren, Rolling-Origin, MAE/RMSE/ME/MASE |
| `nf_evaluation.py` | Analyse, Orakel, vier Experimente |
| `nf_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- Alles, was die nächsten Stücke bringen: Glättung und Modelle, Regression auf Kalendermerkmale, Intervalle, mehrere Reihen, Nullen.
- Weitere Kennzahlen (sMAPE, RMSSE, Pinball-Verlust) und andere Formen der Kreuzvalidierung (gleitendes statt wachsendes Trainingsfenster für die Ursprünge); die Verfahren selbst kennen ein gleitendes Fenster.
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
