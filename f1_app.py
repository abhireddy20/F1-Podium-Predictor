import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .stApp { background-color: #0a0a0a; color: #f0f0f0; }

    h1 { color: #e8002d !important; font-size: 2.6rem !important; letter-spacing: 2px; }
    h2, h3 { color: #f5c842 !important; letter-spacing: 1px; }

    .metric-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-left: 4px solid #e8002d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .metric-card .driver { font-size: 1.1rem; font-weight: 700; color: #ffffff; }
    .metric-card .prob   { font-size: 2rem;   font-weight: 700; color: #e8002d; }
    .metric-card .detail { font-size: 0.85rem; color: #888; font-family: 'Share Tech Mono', monospace; }

    .podium-gold   { border-left-color: #FFD700 !important; }
    .podium-silver { border-left-color: #C0C0C0 !important; }
    .podium-bronze { border-left-color: #CD7F32 !important; }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #cccccc !important; font-family: 'Rajdhani', sans-serif;
    }
    .stButton > button {
        background-color: #e8002d; color: white; font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem; font-weight: 700; letter-spacing: 1px;
        border: none; border-radius: 4px; padding: 12px 40px;
        width: 100%; cursor: pointer; transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #ff1a3e; }
    .section-divider { border-top: 1px solid #333; margin: 24px 0; }
    .tag {
        display: inline-block; background: #1a1a1a; border: 1px solid #444;
        border-radius: 3px; padding: 2px 8px; font-size: 0.78rem;
        font-family: 'Share Tech Mono', monospace; color: #aaa; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────
FEATURES = [
    "grid", "quali_gap_pct", "rolling_avg_finish", "career_races",
    "circuit_wins_before", "circuit_podiums_before", "points_gap_to_leader",
    "constructor_rolling_points", "constructor_rolling_wins",
    "season_progress", "home_race",
]

FEATURE_LABELS = {
    "grid":                       "Grid Position",
    "quali_gap_pct":              "Quali Gap to Pole (%)",
    "rolling_avg_finish":         "Avg Finish (Last 5 Races)",
    "career_races":               "Career Races (Experience)",
    "circuit_wins_before":        "Wins at This Circuit",
    "circuit_podiums_before":     "Podiums at This Circuit",
    "points_gap_to_leader":       "Championship Points Gap",
    "constructor_rolling_points": "Constructor Form (pts, last 3)",
    "constructor_rolling_wins":   "Constructor Wins (last 5)",
    "season_progress":            "Season Progress (0–1)",
    "home_race":                  "Home Race",
}

CURRENT_DRIVERS = [
    "Max Verstappen", "Sergio Perez", "Lewis Hamilton", "George Russell",
    "Charles Leclerc", "Carlos Sainz", "Lando Norris", "Oscar Piastri",
    "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly",
    "Valtteri Bottas", "Zhou Guanyu", "Alexander Albon", "Logan Sargeant",
    "Yuki Tsunoda", "Daniel Ricciardo", "Kevin Magnussen", "Nico Hulkenberg",
]

CIRCUITS = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
    "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
    "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
    "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
    "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
    "Italian Grand Prix", "Singapore Grand Prix", "United States Grand Prix",
    "Mexico City Grand Prix", "São Paulo Grand Prix", "Las Vegas Grand Prix",
    "Qatar Grand Prix", "Abu Dhabi Grand Prix",
]


# ── Load & Train Model (cached) ───────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("f1_features_phase2.csv")
    df["podium"] = (df["positionOrder"] <= 3).astype(int)
    df_model = df[FEATURES + ["podium", "year"]].dropna()
    train = df_model[df_model["year"] <= 2021]
    X_train, y_train = train[FEATURES], train["podium"]
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf)
    return rf, explainer

@st.cache_data
def load_stats():
    df = pd.read_csv("f1_features_phase2.csv")
    return df


# ── Helper: build input row ───────────────────────────────────
def build_input(grid, quali_gap, avg_finish, career_races,
                circuit_wins, circuit_podiums, pts_gap,
                constructor_pts, constructor_wins,
                season_progress, home_race):
    return pd.DataFrame([{
        "grid":                       grid,
        "quali_gap_pct":              quali_gap,
        "rolling_avg_finish":         avg_finish,
        "career_races":               career_races,
        "circuit_wins_before":        circuit_wins,
        "circuit_podiums_before":     circuit_podiums,
        "points_gap_to_leader":       pts_gap,
        "constructor_rolling_points": constructor_pts,
        "constructor_rolling_wins":   constructor_wins,
        "season_progress":            season_progress,
        "home_race":                  int(home_race),
    }])


# ── Helper: SHAP bar chart ────────────────────────────────────
def shap_chart(shap_vals, driver_name, prob):
    sv = pd.Series(shap_vals, index=FEATURES)
    sv_sorted = sv.reindex(sv.abs().sort_values(ascending=True).index)
    labels = [FEATURE_LABELS[f] for f in sv_sorted.index]
    colors = ["#e8002d" if v > 0 else "#4a9eff" for v in sv_sorted.values]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0f0f0f")
    ax.set_facecolor("#1a1a1a")
    ax.barh(labels, sv_sorted.values, color=colors)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set_title(f"Why {driver_name}? ({prob:.0%} podium probability)",
                 color="white", fontsize=10, pad=10)
    ax.set_xlabel("SHAP value", color="#aaa")
    ax.tick_params(colors="#aaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    red  = mpatches.Patch(color="#e8002d", label="↑ Increases podium chance")
    blue = mpatches.Patch(color="#4a9eff", label="↓ Decreases podium chance")
    ax.legend(handles=[red, blue], framealpha=0.15, fontsize=7, labelcolor="white")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

st.markdown("# 🏎️ F1 PODIUM PREDICTOR")
st.markdown("##### *Machine learning model trained on 2010–2021 F1 data · Random Forest · 0.930 AUC*")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Load model
with st.spinner("Loading model..."):
    try:
        model, explainer = load_model()
        df_stats = load_stats()
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False

if not model_loaded:
    st.stop()

# ── Sidebar: Race Context ─────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏁 Race Setup")
    selected_circuit = st.selectbox("Circuit", CIRCUITS, index=6)
    season_round     = st.slider("Race Round", 1, 23, 7)
    total_rounds     = 23
    season_prog      = round(season_round / total_rounds, 2)
    st.caption(f"Season progress: {season_prog:.0%}")

    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown('<span class="tag">Random Forest</span><span class="tag">n=300 trees</span><span class="tag">AUC 0.930</span>', unsafe_allow_html=True)
    st.markdown('<span class="tag">Top-3 Acc: 63.7%</span><span class="tag">Train: 2010–2021</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔍 About")
    st.caption("Built as a data science portfolio project. Features include qualifying pace, driver form, circuit history, constructor reliability, and championship pressure.")

# ── Main: Driver Grid Builder ─────────────────────────────────
st.markdown("### 🏎️ Build Your Grid")
st.caption("Add up to 6 drivers, set their qualifying position and stats, then hit Predict.")

col_add, col_spacer = st.columns([2, 3])
with col_add:
    num_drivers = st.slider("Number of drivers to compare", 2, 6, 3)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

driver_inputs = []
cols = st.columns(min(num_drivers, 3))

for i in range(num_drivers):
    col = cols[i % 3]
    with col:
        st.markdown(f"**Driver {i+1}**")
        name      = st.selectbox("Driver", CURRENT_DRIVERS, index=i, key=f"name_{i}")
        grid_pos  = st.number_input("Grid position", 1, 20, i+1, key=f"grid_{i}")
        quali_gap = st.number_input("Quali gap to pole (%)", 0.0, 10.0, float(i)*0.4, step=0.1, key=f"qgap_{i}")

        with st.expander("Advanced stats"):
            avg_finish      = st.number_input("Avg finish (last 5)", 1.0, 20.0, float(max(1, i*3+2)), key=f"af_{i}")
            career_r        = st.number_input("Career races", 0, 350, 80 + i*10, key=f"cr_{i}")
            circ_wins       = st.number_input("Wins at circuit", 0, 10, 1 if i == 0 else 0, key=f"cw_{i}")
            circ_pods       = st.number_input("Podiums at circuit", 0, 15, 2 if i == 0 else 0, key=f"cp_{i}")
            pts_gap         = st.number_input("Pts gap to leader", 0, 500, i*20, key=f"pg_{i}")
            con_pts         = st.number_input("Constructor pts (last 3)", 0.0, 50.0, 15.0 - i*2, key=f"cpts_{i}")
            con_wins        = st.number_input("Constructor wins (last 5)", 0, 5, max(0, 3-i), key=f"cwins_{i}")
            home            = st.checkbox("Home race?", key=f"home_{i}")

        driver_inputs.append({
            "name": name, "grid": grid_pos, "quali_gap": quali_gap,
            "avg_finish": avg_finish, "career_races": career_r,
            "circuit_wins": circ_wins, "circuit_podiums": circ_pods,
            "pts_gap": pts_gap, "constructor_pts": con_pts,
            "constructor_wins": con_wins, "home_race": home,
        })

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────
predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_btn = st.button("⚑  PREDICT PODIUM")

if predict_btn:
    st.markdown("---")
    st.markdown(f"### 🏆 Podium Predictions — {selected_circuit}")

    # Build predictions
    predictions = []
    for d in driver_inputs:
        X = build_input(
            d["grid"], d["quali_gap"], d["avg_finish"], d["career_races"],
            d["circuit_wins"], d["circuit_podiums"], d["pts_gap"],
            d["constructor_pts"], d["constructor_wins"], season_prog, d["home_race"]
        )
        prob = model.predict_proba(X)[0][1]
        sv   = explainer.shap_values(X)
        if isinstance(sv, list):
            shap_vals = sv[1][0]
        elif hasattr(sv, "ndim") and sv.ndim == 3:
            shap_vals = sv[0, :, 1]
        else:
            shap_vals = sv[0]
        predictions.append({"name": d["name"], "prob": prob, "shap": shap_vals, "input": X})

    predictions.sort(key=lambda x: x["prob"], reverse=True)

    # ── Podium Cards ──────────────────────────────────────────
    podium_classes = ["podium-gold", "podium-silver", "podium-bronze"]
    podium_medals  = ["🥇", "🥈", "🥉"]

    res_cols = st.columns(len(predictions))
    for i, (pred, col) in enumerate(zip(predictions, res_cols)):
        css_class = podium_classes[i] if i < 3 else ""
        medal     = podium_medals[i]  if i < 3 else f"P{i+1}"
        with col:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div class="driver">{medal} {pred['name']}</div>
                <div class="prob">{pred['prob']:.1%}</div>
                <div class="detail">podium probability</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── SHAP Explanations ─────────────────────────────────────
    st.markdown("### 🔬 Why These Predictions?")
    st.caption("SHAP values show which features pushed each driver's podium probability up (red) or down (blue).")

    shap_cols = st.columns(min(len(predictions), 3))
    for i, pred in enumerate(predictions):
        with shap_cols[i % 3]:
            fig = shap_chart(pred["shap"], pred["name"], pred["prob"])
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # ── Probability Bar Chart ─────────────────────────────────
    st.markdown("### 📊 Head-to-Head Comparison")
    names = [p["name"].split()[-1] for p in predictions]
    probs = [p["prob"] for p in predictions]
    colors_bar = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#444"
                  for i in range(len(predictions))]

    fig2, ax2 = plt.subplots(figsize=(8, 3))
    fig2.patch.set_facecolor("#0f0f0f")
    ax2.set_facecolor("#1a1a1a")
    bars = ax2.bar(names, probs, color=colors_bar, width=0.5)
    for bar, prob in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{prob:.0%}", ha="center", color="white", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Podium Probability", color="#aaa")
    ax2.set_ylim(0, 1.15)
    ax2.tick_params(colors="#aaa")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.caption("⚠️ Predictions are based on historical patterns (2010–2021). Real races involve strategy, weather, and incidents not captured in this model.")