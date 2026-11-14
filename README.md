# PokePonche Predict 🎯

**Predicting attendance at Pokémon GO community meetups using CatBoost, participation records, and an RSVP interest counter.**

![Actual vs. predicted attendance](assets/01_tendencia_real_vs_prediccion.png)

> **Key result:** **MAE of 6.75 people per meetup** across **81 test events**, using a retrospectively reconstructed count of interested participants. The median absolute error was **5.43 people**. These results do NOT constitute prospective validation using actual RSVP counter snapshots captured at the start of events.

## 1. Objective

Predict the number of users who will record a `CHECKED_IN` at a **Pokeponche de Frutas [Puebla]** meetup to support activity and resource planning. This is a **supervised regression** problem, with the numeric target `checkins` for each event. The dataset spans November 2023 through November 2026.

## 2. Data: What Was Actually Used for Training

| Item | Count | Role |
|---|---:|---|
| Meetups | **548** | **548 model observations**; each row represents one event |
| Individual records | **33,848** | User interaction history used to compute aggregated features |
| Unique users | **5,131** | Pseudonymized histories preserve repeat participation without exposing original IDs |
| Check-ins | **23,032** | Total target count across all meetups |
| Final test events | **81** | Held-out chronological evaluation |

**Important:** We did not train the model on 33,848 independent attendance examples. The final model **learns from 548 meetups**; the 33,848 individual records are used to derive activity, retention, and repeat-attendance features from prior dates, and to reconstruct the RSVP counter. Confusing these two levels would artificially inflate the reported training sample size.


## 3. Methodology

1. **Data integration:** Join the master dataset of 548 events with individual records using `event_id`; validate 23,032 check-ins without duplicate user/event combinations.
2. **Calendar and event features:** Day of the week, seasonality, event type, and the distinction between major global events and preparatory events.
3. **Historical features without look-ahead:** Rolling means/medians, recent events of the same type, active users over 30/90/180 days, retention, and historical conversion; these features use only events from previous dates.
4. **RSVP:** Reconstruct interested participants as `CHECKED_IN + ACCEPTED` for each meetup when evaluating Stage 5.
5. **Temporal split:** Divide training, validation, and test data chronologically, without splitting events from the same day across sets; reserve **81 meetups** for the final test.
6. **Optimization:** Run **20 Optuna trials (TPE, seed 42)** for CatBoost, using three temporal validation windows to select hyperparameters and the modeling strategy.
7. **Final model:** **Conversion-based CatBoost**, which estimates the proportion of interested participants who will check in. It was selected through validation, not by retrospectively choosing the best test result.

### Models and Techniques Evaluated Throughout the Project

| Experiment | Techniques | Key Findings |
|---|---|---|
| Stage 1 — Baselines | Historical medians, Random Forest, and Gradient Boosting | Establish a temporal benchmark and understand seasonality |
| Stage 2 — Alternatives | Gradient Boosting, CatBoost, and a PyTorch MLP | CatBoost performed best for advance predictions; distinguishing global events from ordinary meetups matters |
| Stage 3 — Hyperparameter search | Optuna, weighted CatBoost, `log1p` transformation, global-event adjustments | The best advance-prediction model reduced MAE to approximately 16.78 |
| Stage 4 — Community activity | Unique-user features, repeat participation, check-in history, and experimental RSVP features | Individual records enrich event-level features; the RSVP proxy is a stronger signal |
| Stage 5 — Final model | Smoothed historical conversion, direct CatBoost, logarithmic target, and **conversion-based CatBoost** | The winning RSVP-proxy model achieved an **MAE of 6.75** |
| Stage 6 — Calibration (not adopted) | A specialized correction based on the number of interested participants | It did not improve overall MAE (~6.76); Stage 5 was retained |

This repository includes **the final pipeline and its required feature-preparation modules**, not every discarded model. The table documents the earlier experiments without turning the repository into an archive of versions.

## 4. Model Interface: Inputs, Output, and Architecture

### 4.1 What Does the Model Need to Make a Prediction?

The predictor estimates the expected number of check-ins **for an entire event** (it does not classify individual users). It requires three groups of inputs:

| Input group | Examples | Source / Availability |
|---|---|---|
| **Event and date** | `fecha_puebla`, `tipo_campfire`, `nombre_campfire`, global or preparatory event phase, bonuses, and Pokémon-related features | Event calendar and description, available before prediction |
| **Community history** | Attendance over the last 5/10 meetups, recent events of the same type, active participants over 30/90/180 days, retention, and prior conversion | Computed from events and users **before the target date** |
| **Interested-participant counter** | Total number of people who selected “Going” in Campfire | In actual use: the count **observed at prediction time** |

The date and counter **alone are not sufficient** to reproduce the prediction: historical data are also needed to calculate the event's feature vector. User IDs and nicknames are not predictive model inputs; they are only used to count unique users and reconstruct historical activity.

**Output:** A non-negative numeric estimate of `checkins` for the meetup. It may be rounded to the nearest whole person for display; test metrics were calculated using continuous predictions.

### 4.2 Exact Architecture of the Selected Model

The file `models/modelo_catboost.cbm` contains a **`CatBoostRegressor`** based on decision-tree *gradient boosting*.

| Component | Final Model Configuration |
|---|---|
| Type | CatBoost regression with symmetric (*oblivious*) trees |
| Number of trees (`iterations`) | **400** |
| Tree depth (`depth`) | **3** (up to 8 leaves per tree) |
| Learning rate (`learning_rate`) | **0.0337569126** |
| L2 regularization (`l2_leaf_reg`) | **4.6493457** |
| Loss function (`loss_function`) | **MAE** |
| Random seed (`random_seed`) | **42** |
| Features accepted by the saved model | **46**: 41 numerical and 5 categorical |
| Hyperparameter search | **20 Optuna trials**, using TPE and temporal validation |

**Learned target:** The winning model (`conversion`) does not directly predict the number of attendees. During training, it uses:

```text
conversion_observada = checkins_reales / max(interesados_total_proxy, 1)
```

At inference time, it produces an estimated rate, clips it to `[0, 1]`, and converts it into a count of people:

```text
tasa_estimada = clip(modelo_catboost.predict(X_evento), 0, 1)
checkins_estimados = max(0, tasa_estimada * interesados_observados)
```

**Implementation detail:** In this variant, `interesados_total_proxy` is removed from the feature matrix `X` before calling `fit()` and `predict()`. This is why it **is not one of the 46 features stored in the `.cbm` file**, but it is still **an essential input to the complete predictor**, since the predicted conversion rate is multiplied by this count. The `scripts/verificar_modelo.py` script reproduces the same post-processing step.

### 4.3 The 46 Features Actually Used

Features are calculated in `train_etapa2.py`, `train_etapa4_mejorada.py`, `train_etapa5.py`, and `train_etapa5_rsvp.py` before the feature matrix is passed to CatBoost.

| Group | Features |
|---|---|
| Calendar (9) | `is_weekend`, `year_offset`, `month`, `day_of_week`, `week_of_year`, `is_holiday_season`, `event_sequence`, `days_since_first_event`, `days_since_previous_event` |
| Global events and context (3) | `is_global_major_event`, `is_global_pre_event`, `days_until_main_event` |
| Aggregated meetup history (8) | `prior_events_same_type`, `hist_last_5_mean`, `hist_last_10_mean`, `hist_last_10_median`, `hist_same_type_last_5_mean`, `hist_same_type_last_5_median`, `days_since_previous_same_type`, `prev_event_checkins` |
| Event and Pokémon attributes (9) | `has_paid_bonus_evidence`, `has_free_bonus_evidence`, `has_assumed_free_bonus`, `has_exclusive_move_section`, `has_shiny_boost_evidence`, `has_pokemon_debut_evidence`, `has_paid_research_evidence`, `pokemon_label_missing`, `primary_pokemon_count` |
| User activity (11) | `user_active_30d`, `user_active_90d`, `user_active_180d`, `user_checkins_30d`, `user_checkins_90d`, `user_retention_90d`, `users_with_3plus_prior_checkins`, `community_users_seen_prior`, `prior_conversion_10`, `prior_conversion_same_type`, `recent_accepted_mean_5` |
| Additional historical conversion (1) | `conversion_historica_anterior` (smoothed toward the recent overall rate when few same-type events are available) |
| Categorical features (5) | `tipo_campfire`, `event_phase`, `event_family`, `major_event_subtype`, `month_label` |

Total: **41 numerical + 5 categorical = 46**. Categorical features are handled natively by CatBoost; no external *one-hot encoding* was applied to the final model. Missing numerical values are encoded as `-99999`, and missing categorical values as `desconocido` in the final feature matrix. The 46-feature order must match the order expected by the `.cbm` file and must not be changed during inference.

### 4.4 Required Transformations and Design Decisions

1. **Event normalization:** Join by `event_id`, verify that each event has exactly one row, and validate check-ins and RSVP statuses without double-counting participants.
2. **Sequential feature engineering:** Compute means/medians from prior events, user activity over 30/90/180-day windows, and conversion rates. For multiple events on the same day, calculate historical statistics before incorporating outcomes from any event on that date.
3. **Conversion regression:** Learn a rate between zero and one and convert it to check-ins by multiplying by the count of interested participants; the final output is non-negative.
4. **Temporal split and model selection:** Train and validate on dates before the final 81-event test period; Optuna evaluates 20 combinations for direct CatBoost, followed by a comparison of historical conversion, direct prediction, `log1p` transformation, and conversion-based CatBoost.
5. **Evaluation and publication:** Report metrics on the held-out temporal test set, retain the original `.cbm` model, and replace participant IDs and nicknames before preparing the public release.

The code includes components originally used in earlier experiments, but the published model uses **only** the `anticipado` (advance-prediction) feature matrix **plus** `conversion_historica_anterior`, and **does not** use retrospective weather variables or competitive rankings.

## 5. Evaluation

**Test period: July 29, 2026 – November 10, 2026; 81 meetups.**

| Metric | Result |
|---|---:|
| MAE (mean absolute error) | **6.75 people** |
| Median absolute error (Q2) | **5.43 people** |
| First quartile (Q1) | 3.07 people |
| Third quartile (Q3) | 9.26 people |
| 90th percentile of absolute error | 13.80 people |
| RMSE | 8.98 people |
| WAPE | 13.65% |
| R² | 0.972 |
| Mean bias (predicted − actual) | −0.08 people |

Interpretation: In the test set, **half of the absolute errors were ≤5.43 people**, and **75% were ≤9.26 people**. An MAE of 6.75 represents the average magnitude of prediction errors.

### Attendance Over Time (Main Visualization)

![Actual vs. predicted attendance over time](assets/01_tendencia_real_vs_prediccion.png)

All 81 test meetups are shown in chronological order, without monthly aggregation or excluding large events.

### Scatter Plot: Actual vs. Predicted Attendance

![Actual vs. predicted attendance scatter plot](assets/02_dispersion_real_vs_prediccion.png)

The diagonal represents the ideal line of perfect predictions.

### Absolute Error Distribution

![Error box plot](assets/04_boxplot_error.png)

Quartiles summarize how errors are distributed in people, regardless of event attendance size.

## 6. Reproducing the Training Process

Python 3.11–3.13, Windows PowerShell (also compatible with other platforms).

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\verificar_datos.py
.\.venv\Scripts\python.exe scripts\train_etapa5_rsvp.py --root . --trials 20
```

The script loads the pseudonymized CSV files from `data/`, reconstructs date- and behavior-based features, trains the candidate models, applies temporal validation, and creates `resultados_etapa5_rsvp/`. Tested hyperparameters and published outputs are available in `results/`. **Retraining may produce slight differences due to library versions or parallel execution**; therefore, the original winning `.cbm` file and historical predictions are included.

To verify the saved model:

```powershell
.\.venv\Scripts\python.exe scripts\verificar_modelo.py
```

To regenerate the three charts from the published predictions:

```powershell
.\.venv\Scripts\python.exe scripts\generar_graficos.py
```

## 7. Repository Structure

```text
PokePonche_Predict_GitHub/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── dataset_maestro_548.csv
│   └── registros_usuarios_pseudonimizados.csv
├── scripts/
│   ├── train_etapa5_rsvp.py        # Final pipeline
│   ├── train_etapa5.py             # Feature construction
│   ├── train_etapa4_mejorada.py    # User-level history
│   ├── train_etapa3.py             # Temporal cross-validation
│   ├── train_etapa2.py             # Calendar and historical features
│   ├── fases_eventos.py            # Preparatory-event classification
│   ├── verificar_datos.py
│   ├── verificar_modelo.py
│   └── generar_graficos.py
├── models/
│   └── modelo_catboost.cbm        # Winning model from 20 trials
├── results/
│   ├── predicciones_test.csv
│   ├── metricas_test.csv
│   ├── validacion_temporal.csv
│   ├── seleccion_modelo.csv
│   ├── optuna_trials.csv
│   └── reporte.json
└── assets/
    ├── 01_tendencia_real_vs_prediccion.png
    ├── 02_dispersion_real_vs_prediccion.png
    └── 04_boxplot_error.png
```

## 8. Privacy and Publication

This repository includes **data minimization and pseudonymization**: `USUARIO1`, `USUARIO2`, etc. replace each person's internal ID and nickname consistently. Original meetup identifiers are replaced with `EVT0001`, etc. Profiles, event URLs, organizer IDs, and other original fields not needed to reproduce the pipeline were removed. The original-to-pseudonym mapping **is not distributed**.


## Technologies

`Python` · `pandas` · `NumPy` · `CatBoost` · `Optuna` · `scikit-learn` · `Matplotlib` · `PyTorch` (explored in Stage 2)

## Authorship and Usage

Independent project analyzing community meetup attendance. Pokémon, Campfire, and related trademarks belong to their respective owners.
