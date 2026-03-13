# 🏎️ F1 Podium Predictor

A machine learning pipeline that predicts Formula 1 podium finishes using 14 years of race data (2010–2024). Built as a data science portfolio project featuring end-to-end feature engineering, model comparison, SHAP explainability, and an interactive Streamlit app.

---

## 🏆 Results

| Model | ROC-AUC | F1 Score | Per-Race Top-3 Accuracy |
|---|---|---|---|
| Logistic Regression | 0.926 | 0.605 | — |
| Random Forest | **0.930** | **0.656** | **63.7%** |
| XGBoost | 0.923 | 0.648 | — |

> The Random Forest model correctly identifies **~2 out of 3 podium finishers per race** on unseen 2022–2024 data — significantly outperforming a grid-position-only baseline.

**Spotlight prediction:** Max Verstappen @ Austrian Grand Prix 2023 — predicted **97.8% podium probability** ✅

---

## 📊 Key Insights from SHAP Analysis

- **Grid position** is the single most impactful feature (mean |SHAP| = 0.134)
- **Qualifying gap to pole** is almost equally important (0.116) — raw pace matters as much as starting spot
- **Recent driver form** (avg finish last 5 races) is the #3 predictor (0.086)
- Constructor reliability and circuit-specific history add meaningful signal on top of pace

---

## 🗂️ Project Structure

```
f1_predictor/
│
├── f1_eda_starter.py        # Phase 1: EDA & first feature engineering
├── f1_phase2_features.py    # Phase 2: Advanced feature engineering
├── f1_phase3_models.py      # Phase 3: Model training & comparison
├── f1_phase4_shap.py        # Phase 4: SHAP explainability
├── f1_app.py                # Phase 5: Interactive Streamlit app
│
├── charts/                  # All generated charts (12 total)
│   ├── chart1_grid_vs_win_rate.png
│   ├── chart2_constructor_dominance.png
│   ├── chart3_top_drivers.png
│   ├── chart4_nationality_points_rate.png
│   ├── chart5_feature_correlation.png
│   ├── chart6_model_comparison.png
│   ├── chart7_roc_curves.png
│   ├── chart8_feature_importance.png
│   ├── chart9_shap_global.png
│   ├── chart10_shap_beeswarm.png
│   ├── chart11_shap_waterfall.png
│   └── chart12_shap_dependence_grid.png
│
└── data/                    # Not included — download from Kaggle (see below)
```

---

## ⚙️ Features Used

| Feature | Description |
|---|---|
| `grid` | Starting grid position |
| `quali_gap_pct` | % gap to pole position in qualifying |
| `rolling_avg_finish` | Driver's average finishing position (last 5 races) |
| `career_races` | Total career races started (experience proxy) |
| `circuit_wins_before` | Driver's historical wins at this specific circuit |
| `circuit_podiums_before` | Driver's historical podiums at this circuit |
| `points_gap_to_leader` | Championship points gap to the leader |
| `constructor_rolling_points` | Constructor's avg points over last 3 races |
| `constructor_rolling_wins` | Constructor's wins over last 5 races |
| `season_progress` | Race round as a fraction of total season rounds |
| `home_race` | Whether the driver is racing in their home country |

---

## 🚀 Running the App

### 1. Clone the repo
```bash
git clone https://github.com/abhireddy20/F1-Podium-Predictor.git
cd F1-Podium-Predictor
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap streamlit
```

### 3. Download the dataset
Download the **Formula 1 World Championship (1950–2024)** dataset from Kaggle:
👉 https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020

Place the CSV files in a `data/` folder, or run:
```bash
python unzip_data.py   # if you downloaded archive.zip
```

### 4. Run the pipeline
```bash
python f1_eda_starter.py        # Phase 1
python f1_phase2_features.py    # Phase 2
python f1_phase3_models.py      # Phase 3
python f1_phase4_shap.py        # Phase 4
```

### 5. Launch the app
```bash
streamlit run f1_app.py
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)
![SHAP](https://img.shields.io/badge/SHAP-explainability-red)
![Streamlit](https://img.shields.io/badge/Streamlit-app-ff4b4b)

- **Data:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** matplotlib, seaborn
- **App:** Streamlit

---

## 📁 Dataset

**Source:** [Formula 1 World Championship (1950–2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) via Kaggle / Ergast API

The dataset is not included in this repo due to file size. Download it from Kaggle and place the CSVs in the `data/` directory.

---

## 👤 Author

**Abhiraam Thadur**  
Data Science Graduate  
[GitHub](https://github.com/abhireddy20)

---

*Built for portfolio purposes. Predictions are based on historical patterns and do not account for race incidents, strategy calls, or weather.*
