import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# Re-train Random Forest (or load if you saved it)
from sklearn.ensemble import RandomForestClassifier

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333333",   "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",      "ytick.color": "#888888",
    "text.color": "#ffffff",       "grid.color": "#2a2a2a",
    "grid.linestyle": "--",        "font.family": "monospace",
})
F1_RED = "#e8002d"; F1_GOLD = "#f5c842"; F1_BLUE = "#4a9eff"; F1_WHITE = "#ffffff"

FEATURES = [
    "grid", "quali_gap_pct", "rolling_avg_finish", "career_races",
    "circuit_wins_before", "circuit_podiums_before", "points_gap_to_leader",
    "constructor_rolling_points", "constructor_rolling_wins",
    "season_progress", "home_race",
]

FEATURE_LABELS = {
    "grid":                       "Grid Position",
    "quali_gap_pct":              "Quali Gap to Pole (%)",
    "rolling_avg_finish":         "Avg Finish (Last 5)",
    "career_races":               "Career Experience",
    "circuit_wins_before":        "Wins at This Circuit",
    "circuit_podiums_before":     "Podiums at This Circuit",
    "points_gap_to_leader":       "Points Gap to Leader",
    "constructor_rolling_points": "Constructor Form (pts)",
    "constructor_rolling_wins":   "Constructor Wins (last 5)",
    "season_progress":            "Season Progress",
    "home_race":                  "Home Race",
}


# ── 1. Load & Re-train ───────────────────────────────────────
print("Loading data and re-training Random Forest...")
df = pd.read_csv("f1_features_phase2.csv")
df["podium"] = (df["positionOrder"] <= 3).astype(int)
df_model = df[FEATURES + ["podium", "year", "raceId", "driver_name", "name_race"]].dropna()

train = df_model[df_model["year"] <= 2021]
test  = df_model[df_model["year"] >= 2022]

X_train, y_train = train[FEATURES], train["podium"]
X_test,  y_test  = test[FEATURES],  test["podium"]

rf = RandomForestClassifier(
    n_estimators=300, max_depth=8, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("✅ Model ready")


# ── 2. Compute SHAP Values ───────────────────────────────────
print("\nComputing SHAP values (this takes ~30 seconds)...")
explainer   = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Handle different shap output formats across versions:
# Older shap: list of 2 arrays [class_0, class_1]
# Newer shap: single 3D array of shape (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values
sv_df = pd.DataFrame(sv, columns=FEATURES)
print("✅ SHAP values computed")


# ── 3. Chart 1: Global Feature Importance (mean |SHAP|) ──────
print("\nPlotting Chart 9: Global SHAP importance...")

mean_shap = sv_df.abs().mean().sort_values()
labels    = [FEATURE_LABELS[f] for f in mean_shap.index]
colors    = [F1_RED if v == mean_shap.max() else "#555555" for v in mean_shap.values]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(labels, mean_shap.values, color=colors)
ax.set_title("Global Feature Importance (Mean |SHAP|)\nRandom Forest — Podium Prediction",
             color=F1_WHITE, fontsize=13, pad=15)
ax.set_xlabel("Mean |SHAP Value|  →  Impact on podium probability")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("chart9_shap_global.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart9_shap_global.png")


# ── 4. Chart 2: SHAP Beeswarm (summary plot) ─────────────────
print("Plotting Chart 10: SHAP beeswarm...")

fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    sv, X_test, feature_names=[FEATURE_LABELS[f] for f in FEATURES],
    show=False, plot_size=None, color_bar=True
)
plt.gcf().set_facecolor("#0f0f0f")
plt.gca().set_facecolor("#1a1a1a")
plt.title("SHAP Summary — Feature Impact on Podium Probability",
          color=F1_WHITE, fontsize=12, pad=12)
plt.tight_layout()
plt.savefig("chart10_shap_beeswarm.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f0f")
plt.close()
print("   → Saved: chart10_shap_beeswarm.png")


# ── 5. Chart 3: SHAP for a Single High-Profile Race ──────────
# Find a race where the model was very confident (use 2023 or 2024 races)
print("Plotting Chart 11: Single race waterfall...")

test_with_meta = test.reset_index(drop=True)
test_with_meta["pred_proba"] = rf.predict_proba(X_test)[:, 1]

# Pick the driver with highest predicted podium probability across 2023
top_pred_idx = test_with_meta[test_with_meta["year"] == 2023]["pred_proba"].idxmax()
driver_name  = test_with_meta.loc[top_pred_idx, "driver_name"]
race_name    = test_with_meta.loc[top_pred_idx, "name_race"]
actual       = "✅ Podium" if test_with_meta.loc[top_pred_idx, "podium"] == 1 else "❌ No Podium"
pred_prob    = test_with_meta.loc[top_pred_idx, "pred_proba"]

shap_row   = sv_df.iloc[top_pred_idx]
ev = explainer.expected_value
if isinstance(ev, list):
    base_value = ev[1]
elif hasattr(ev, "__len__"):
    base_value = ev[1] if len(ev) > 1 else ev[0]
else:
    base_value = ev

sorted_shap = shap_row.reindex(shap_row.abs().sort_values(ascending=True).index)
sorted_labels = [FEATURE_LABELS[f] for f in sorted_shap.index]
bar_colors = [F1_RED if v > 0 else F1_BLUE for v in sorted_shap.values]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(sorted_labels, sorted_shap.values, color=bar_colors)
ax.axvline(0, color="#888888", linewidth=1)
ax.set_title(
    f"SHAP Waterfall — {driver_name}\n{race_name} 2023  |  Predicted: {pred_prob:.0%}  |  Actual: {actual}",
    color=F1_WHITE, fontsize=11, pad=12
)
ax.set_xlabel("SHAP Value  →  Pushes toward podium (red) or away (blue)")
red_patch  = mpatches.Patch(color=F1_RED,  label="Increases podium probability")
blue_patch = mpatches.Patch(color=F1_BLUE, label="Decreases podium probability")
ax.legend(handles=[red_patch, blue_patch], framealpha=0.2, fontsize=9)
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("chart11_shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   → Saved: chart11_shap_waterfall.png  ({driver_name} @ {race_name})")


# ── 6. Chart 4: Grid Position SHAP Dependence ────────────────
print("Plotting Chart 12: SHAP dependence — grid position...")

grid_vals  = X_test["grid"].values
grid_shap  = sv_df["grid"].values

fig, ax = plt.subplots(figsize=(9, 5))
sc = ax.scatter(grid_vals, grid_shap,
                c=X_test["quali_gap_pct"].values,
                cmap="RdYlGn_r", alpha=0.5, s=15)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Quali Gap to Pole (%)", color=F1_WHITE)
cbar.ax.yaxis.set_tick_params(color=F1_WHITE)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=F1_WHITE)
ax.axhline(0, color="#555555", linewidth=1, linestyle="--")
ax.set_title("SHAP Dependence — Grid Position\n(color = qualifying gap to pole)",
             color=F1_WHITE, fontsize=12, pad=12)
ax.set_xlabel("Grid Position")
ax.set_ylabel("SHAP Value (impact on podium probability)")
ax.grid(True)
plt.tight_layout()
plt.savefig("chart12_shap_dependence_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart12_shap_dependence_grid.png")


# ── 7. Key Insights ──────────────────────────────────────────
print("\n── SHAP Insights ────────────────────────────────────")
top3_features = sv_df.abs().mean().nlargest(3)
print("Top 3 most impactful features:")
for feat, val in top3_features.items():
    print(f"   {FEATURE_LABELS[feat]:<30} mean |SHAP| = {val:.4f}")

print(f"\nSingle race spotlight:")
print(f"   Driver:   {driver_name}")
print(f"   Race:     {race_name} 2023")
print(f"   Predicted podium prob: {pred_prob:.1%}")
print(f"   Actual result: {actual}")

print("\n🏁 Phase 4 complete!")
print("   Next up → Phase 5: Streamlit app — make it interactive!")
