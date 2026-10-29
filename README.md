# 🚛 SmartFleet Route Optimizer

A Streamlit prototype that combines Chicago crime, taxi, traffic, weather, vehicle-sensor and event-sentiment data into a route risk score, with an interactive map, scenario controls and a what-if cost calculator. Supporting notebooks cover a Random Forest risk evaluation and an X (Twitter) sentiment + LLM-summary pipeline.

**Live demo:** http://35.89.230.31:8501/ (may be offline when the server isn't running)

<img width="1517" height="917" alt="SmartFleet dashboard" src="https://github.com/user-attachments/assets/749085b6-c912-4f66-98db-e2b35a274618" />

---

## What it does

**Streamlit app (`application.py`)**
- **Data fusion:** merges sampled Chicago taxi trips with crime, weather, traffic-congestion, ADAS sensor and event-sentiment data by time/zone
- **Risk scoring:** rule-based risk terms (crime rate, weather thresholds, obstacle distance, gyro labels, negative sentiment, traffic speed), weighted by user-controlled sliders
- **Scenario controls:** day/hour, temperature, rain, wind, humidity, visibility, surge multiplier, demand density, crime threshold, sensor weights
- **Sensor panel:** vision-sensor accuracy vs. humidity from a quadratic formula, plus gyro risk
- **Route view:** a simulated 5-stop route scored against the risk formula (next stops are sampled randomly, so this is an illustration, not an optimized route)
- **Chicago map:** Folium heatmap of crime locations with route and sensor-risk markers
- **Executive dashboard + what-if cost estimator:** KPIs and annual savings computed from your fleet inputs and **assumed** savings percentages. These are scenario estimates, not measured outcomes.

**Notebooks**
- `ML Model Evaluations/SmartFleet_Evaluation (1).ipynb`: trains and evaluates scikit-learn **Random Forest** classifiers for crime risk and composite risk (SMOTE, ROC/PR curves, confusion matrices)
- `LLM Model/LLM_X_WebScraper.ipynb`: pulls Chicago traffic posts from the X API (token entered at runtime), scores sentiment with `cardiffnlp/twitter-roberta-base-sentiment-latest` and VADER, and writes route alerts with `sshleifer/distilbart-cnn-12-6`; uses sentence-transformers and MLflow

<img width="1105" height="365" alt="Control panel" src="https://github.com/user-attachments/assets/8183b73b-fdff-4f41-a61c-23f3a4d59600" />

---

## Results (honest summary)

From `ML Model Evaluations/deployed_metrics.csv` and the evaluation notebook:

| Model | Data | Accuracy | F1 | ROC-AUC |
|-------|------|----------|----|---------|
| Crime risk (Random Forest) | 72 zone-hour rows (SMOTE applied), ~18 test rows | 0.94 | 0.93 | 0.95 |
| Composite risk (Random Forest) | 13,291 rows | 0.84 | 0.84 | 0.93 |

Caveats:
- The crime model's test set is very small, so its score is not a reliable estimate.
- In the logged run, gyro (Terra-D2) and sentiment risk fell back to **synthetic values** because their timestamps failed to parse, and random noise was added to features. The composite score therefore measures a partly synthetic target.
- The Streamlit app does **not** load these trained models; it uses the rule-based formulas above.
- Dollar savings shown in the app (and in `business_impact.txt` / the A/B testing folder) are outputs of assumption-based calculators, not observed results.

### Next steps
- Replace the random route sampling with a real optimizer (e.g. shortest path over a risk-weighted road graph)
- Serve the trained models in the app instead of fixed formulas
- Evaluate on a larger, fully real labelled dataset

---

## Data

| File | Source |
|------|--------|
| `Chicago Crime Sampled.csv` (1,000 rows) | Chicago Data Portal crimes data (sample) |
| `Chicago Taxi Sampled.xlsx` | Chicago Data Portal taxi trips (sample) |
| `Chicago_Traffic_Tracker.csv` | Chicago Traffic Tracker regional congestion data |
| `Chicago Weather.csv` | NOAA daily records, Chicago O'Hare station (USW00094846) |
| `ADAS_EV_Dataset.csv`, `News Sentiment Analysis.csv` | see notebooks |
| `Terra-D2-multi-labeled-interpolated.csv` | gyro sensor data; **not included in this repo** (needed by the app) |

---

## How to run locally

```bash
pip install -r requirements.txt
# add Data/Terra-D2-multi-labeled-interpolated.csv (not in repo), then:
streamlit run application.py
```

`Dockerfile.txt` holds the container setup used for deployment (rename to `Dockerfile`; it expects the app file as `app.py`).

---

## Tech stack
Streamlit, pandas, NumPy, Matplotlib, Folium / streamlit-folium, scikit-learn (Random Forest), imbalanced-learn (SMOTE), Hugging Face Transformers, sentence-transformers, NLTK VADER, Tweepy (X API), MLflow, Docker

---

## Project structure
```
application.py                     # Streamlit app
requirements.txt
Data/                              # sampled Chicago + sensor + sentiment data
ML Model Evaluations/              # Random Forest evaluation notebook, metrics, plots, A/B test simulation files
LLM Model/                         # X sentiment scraper + LLM alert notebook
Architecture/                      # deployment design notebook
Machine Learning Algorithms Details.docx
Dockerfile.txt, Dockerignore.txt
```

<img width="1081" height="725" alt="Route view" src="https://github.com/user-attachments/assets/917b6b74-0ee4-4b32-9553-25e4012ca2ff" />

<img width="973" height="589" alt="Executive dashboard" src="https://github.com/user-attachments/assets/4703930b-93bf-4bbc-9c3f-0e5b18979252" />
