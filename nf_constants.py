"""Konstanten der Naive-Prognose-Demo: Vehikel "Tagesaufträge eines Depots" (Stück 1 der Zeitreihen-Prognose-Linie), Regler, Verfahren, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

N_DAYS = 1095                                       # drei Jahre täglich
FIRST_TEST = 730                                    # Ursprünge liegen im letzten Jahr (Anfangs-Trainingsfenster: zwei Jahre)
LEVEL = 100.0                                       # mittlere Tagesaufträge zu Beginn
WEEKDAYS = ("Mo", "Di", "Mi", "Do", "Fr", "Sa", "So")
WEEKLY_PATTERN = (1.10, 1.05, 1.00, 1.05, 1.20, 0.55, 0.35)
HOLIDAY_DOY = (0, 89, 92, 120, 134, 143, 275, 358, 359, 360)     # Tage im Jahr (0 = 1. Januar), an denen das Depot ruht
HOLIDAY_DROP = 0.5                                  # Rückgang am Feiertag (Faktor 1 - 0,5 * Stärke)
HOLIDAY_REBOUND = 0.15                              # Nachholeffekt am Folgetag
PROMO_LENGTH = 7
PROMO_PER_YEAR = 3

TREND_MIN, TREND_MAX, TREND_STEP, DEFAULT_TREND = -20, 40, 5, 10              # Prozent je Jahr
WEEKLY_MIN, WEEKLY_MAX, WEEKLY_STEP, DEFAULT_WEEKLY = 0.0, 1.5, 0.25, 1.0
YEARLY_MIN, YEARLY_MAX, YEARLY_STEP, DEFAULT_YEARLY = 0.0, 0.5, 0.05, 0.2
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.02, 0.5, 0.02, 0.14
SHIFT_MIN, SHIFT_MAX, SHIFT_STEP, DEFAULT_SHIFT = -40, 40, 10, 0             # Prozent, Niveausprung an einem zufälligen Tag im Testjahr
EVENTS_MIN, EVENTS_MAX, EVENTS_STEP, DEFAULT_EVENTS = 0.0, 1.0, 0.25, 0.5     # Stärke von Feiertagen und Aktionen
HORIZON_MIN, HORIZON_MAX, DEFAULT_HORIZON = 1, 28, 14
STEP_MIN, STEP_MAX, DEFAULT_STEP = 1, 14, 1
WINDOW_OPTIONS = (0, 4, 8, 13, 26)                  # gleitendes Fenster in Wochen; 0 = wachsend (alle bisherigen Tage)
DEFAULT_WINDOW = 0
K_WEEKS_MIN, K_WEEKS_MAX, DEFAULT_K_WEEKS = 2, 12, 4

METHODS = ("mean", "naive", "snaive", "snaive_k", "snaive_year", "drift")
METHOD_NAMES = {
    "mean": "Mittelwert",
    "naive": "Naiv (letzter Tag)",
    "snaive": "Saisonal naiv (letzte Woche)",
    "snaive_k": "Saisonal: Mittel der letzten k Wochen",
    "snaive_year": "Saisonal naiv (Vorjahr)",
    "drift": "Naiv mit Drift",
}

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(12))
NOISE_LEVELS = (0.04, 0.08, 0.14, 0.24, 0.4)
TREND_LEVELS = (-20, 0, 10, 25, 40)
SPLIT_DRAWS = 300
