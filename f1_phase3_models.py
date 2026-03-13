import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, confusion_matrix)
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333333",   "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",      "ytick.color": "#888888",
    "text.color": "#ffffff",       "grid.color": "#2a2a2a",
    "grid.linestyle": "--",        "font.family": "monospace",
})
F1_RED = "#e8002d"; F1_GOLD = "#f5c842"; F1_BLUE = "#4a9eff"; F1_WHITE = "#ffffff"


# ── 1. Load & Prepare ────────────────────────────────────────
print("Loading feature dataset...")
df = pd.read_csv("f1_features_phase2.csv")

FEATURES = [
    "grid",
    "quali_gap_pct",
    "rolling_avg_finish",
    "career_races",
    "circuit_wins_before",
    "circuit_podiums_before",
    "points_gap_to_leader",
    "constructor_rolling_points",
    "constructor_rolling_wins",
    "season_progress",
    "home_race",
]

# Target: podium finish (top 3)
df["podium"] = (df["positionOrder"] <= 3).astype(int)

df_model = df[FEATURES + ["podium", "year", "raceId"]].dropna()
print(f"✅ Model dataset: {len(df_model):,} rows | Podium rate: {df_model['podium'].mean():.1%}")


# ── 2. Temporal Train/Test Split ─────────────────────────────
# Use 2010-2021 for training, 2022-2024 for testing
# This mimics real-world usage — never shuffle time-series data!
train = df_model[df_model["year"] <= 2021]
test  = df_model[df_model["year"] >= 2022]

X_train = train[FEATURES]
y_train = train["podium"]
X_test  = test[FEATURES]
y_test  = test["podium"]

print(f"   Train: {len(train):,} rows ({train['year'].min()}–{train['year'].max()})")
print(f"   Test:  {len(test):,} rows  ({test['year'].min()}–{test['year'].max()})")


# ── 3. Train Models ──────────────────────────────────────────
print("\nTraining models...")

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42, eval_metric="logloss", verbosity=0
    ),
}

results = {}
for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "y_proba": y_proba,
        "auc":     auc,
        "report":  report,
        "f1":      report["1"]["f1-score"],
        "precision": report["1"]["precision"],
        "recall":  report["1"]["recall"],
    }
    print(f"      AUC: {auc:.3f} | F1: {report['1']['f1-score']:.3f} | "
          f"Precision: {report['1']['precision']:.3f} | Recall: {report['1']['recall']:.3f}")


# ── 4. Per-Race Top-3 Accuracy ───────────────────────────────
# More meaningful metric: per race, did we correctly identify podium finishers?
print("\nCalculating per-race Top-3 accuracy...")

test_copy = test.copy()
best_model_name = max(results, key=lambda k: results[k]["auc"])
best_model = results[best_model_name]["model"]
test_copy["pred_proba"] = best_model.predict_proba(X_test)[:, 1]
test_copy["actual_podium"] = y_test.values

def top3_accuracy(group):
    predicted_top3 = set(group.nlargest(3, "pred_proba").index)
    actual_top3    = set(group[group["actual_podium"] == 1].index)
    if len(actual_top3) == 0:
        return np.nan
    return len(predicted_top3 & actual_top3) / 3

per_race_acc = test_copy.groupby("raceId").apply(top3_accuracy).dropna()
print(f"   {best_model_name} per-race Top-3 accuracy: {per_race_acc.mean():.1%}")
print(f"   (Correctly predicted {per_race_acc.mean()*3:.1f}/3 podium finishers per race on average)")


# ── 5. Chart: Model Comparison ───────────────────────────────
print("\nPlotting model comparison...")

metrics     = ["auc", "f1", "precision", "recall"]
labels      = ["ROC-AUC", "F1 Score", "Precision", "Recall"]
model_names = list(results.keys())
x           = np.arange(len(metrics))
width       = 0.25
colors      = [F1_BLUE, F1_GOLD, F1_RED]

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, color) in enumerate(zip(model_names, colors)):
    vals = [results[name][m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=F1_WHITE)

ax.set_title("Model Comparison — Podium Prediction (2022–2024 Test Set)", color=F1_WHITE, fontsize=13, pad=15)
ax.set_xticks(x + width)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.1)
ax.legend(framealpha=0.2)
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("chart6_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart6_model_comparison.png")


# ── 6. Chart: ROC Curves ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1, label="Random baseline")
ax.set_title("ROC Curves — Podium Prediction", color=F1_WHITE, fontsize=13, pad=15)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(framealpha=0.2, fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig("chart7_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart7_roc_curves.png")


# ── 7. Chart: XGBoost Feature Importance ─────────────────────
print("Plotting XGBoost feature importance...")

xgb_model   = results["XGBoost"]["model"]
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = [F1_RED if v == importances.max() else "#555555" for v in importances.values]
ax.barh(importances.index, importances.values, color=colors_bar)
ax.set_title("XGBoost Feature Importance — Podium Prediction", color=F1_WHITE, fontsize=13, pad=15)
ax.set_xlabel("Importance Score")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("chart8_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart8_feature_importance.png")


# ── 8. Save Predictions ──────────────────────────────────────
test_copy["xgb_proba"]   = results["XGBoost"]["y_proba"]
test_copy["xgb_pred"]    = results["XGBoost"]["y_pred"]
test_copy["rf_proba"]    = results["Random Forest"]["y_proba"]
test_copy["lr_proba"]    = results["Logistic Regression"]["y_proba"]
test_copy.to_csv("f1_predictions_phase3.csv", index=False)
print("\n✅ Saved predictions → f1_predictions_phase3.csv")


# ── 9. Print Summary ─────────────────────────────────────────
print("\n── Results Summary ──────────────────────────────────")
print(f"{'Model':<25} {'AUC':>6} {'F1':>6} {'Precision':>10} {'Recall':>8}")
print("─" * 58)
for name, res in results.items():
    marker = " ← best" if name == best_model_name else ""
    print(f"{name:<25} {res['auc']:>6.3f} {res['f1']:>6.3f} {res['precision']:>10.3f} {res['recall']:>8.3f}{marker}")

print(f"\n🏆 Best model: {best_model_name}")
print(f"   Per-race Top-3 accuracy: {per_race_acc.mean():.1%}")
print(f"\n🏁 Phase 3 complete!")
print("   Next up → Phase 4: SHAP explainability — why does the model predict what it predicts?")