import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#cccccc",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#ffffff",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})
F1_RED   = "#e8002d"
F1_GOLD  = "#f5c842"
F1_WHITE = "#ffffff"


# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
results      = pd.read_csv("data/results.csv")
races        = pd.read_csv("data/races.csv")
drivers      = pd.read_csv("data/drivers.csv")
constructors = pd.read_csv("data/constructors.csv")
qualifying   = pd.read_csv("data/qualifying.csv")

# Merge results with race metadata
df = results.merge(races[["raceId", "year", "name", "circuitId"]], on="raceId")
df = df.merge(drivers[["driverId", "forename", "surname", "nationality"]], on="driverId")
df = df.merge(constructors[["constructorId", "name"]], on="constructorId", suffixes=("_race", "_constructor"))
df["driver_name"] = df["forename"] + " " + df["surname"]

# Clean position column (some DNFs are stored as \N)
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
df["grid"]          = pd.to_numeric(df["grid"], errors="coerce")
df["points"]        = pd.to_numeric(df["points"], errors="coerce")

# Focus on modern era for cleaner signal
df_modern = df[df["year"] >= 2010].copy()

print(f"✅ Loaded {len(df_modern):,} race entries ({df_modern['year'].min()}–{df_modern['year'].max()})")
print(f"   Drivers: {df_modern['driver_name'].nunique()} | Constructors: {df_modern['name_constructor'].nunique()}")


# ── 2. Chart 1: Grid Position vs Win Rate ────────────────────
print("\nPlotting Chart 1: Grid position vs win rate...")

grid_wins = (
    df_modern[df_modern["positionOrder"] == 1]
    .groupby("grid").size()
    / df_modern.groupby("grid").size()
).reset_index()
grid_wins.columns = ["grid", "win_rate"]
grid_wins = grid_wins[grid_wins["grid"] <= 20]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(grid_wins["grid"], grid_wins["win_rate"] * 100,
              color=[F1_RED if g == 1 else "#444444" for g in grid_wins["grid"]])
ax.set_title("Win Rate by Grid Position (2010–2023)", color=F1_WHITE, fontsize=14, pad=15)
ax.set_xlabel("Starting Grid Position")
ax.set_ylabel("Win Rate (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_xticks(grid_wins["grid"])
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("chart1_grid_vs_win_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart1_grid_vs_win_rate.png")


# ── 3. Chart 2: Constructor Dominance Over Time ──────────────
print("Plotting Chart 2: Constructor dominance...")

top_constructors = (
    df_modern[df_modern["positionOrder"] == 1]
    .groupby("name_constructor").size()
    .nlargest(6).index.tolist()
)
wins_by_year = (
    df_modern[(df_modern["positionOrder"] == 1) & (df_modern["name_constructor"].isin(top_constructors))]
    .groupby(["year", "name_constructor"]).size()
    .reset_index(name="wins")
)

palette = [F1_RED, F1_GOLD, "#00d2ff", "#ff8c00", "#a8ff3e", "#c084fc"]
fig, ax = plt.subplots(figsize=(13, 5))
for i, team in enumerate(top_constructors):
    subset = wins_by_year[wins_by_year["name_constructor"] == team]
    ax.plot(subset["year"], subset["wins"], marker="o", linewidth=2,
            color=palette[i], label=team, markersize=5)

ax.set_title("Race Wins per Season — Top 6 Constructors (2010–2023)", color=F1_WHITE, fontsize=14, pad=15)
ax.set_xlabel("Season")
ax.set_ylabel("Wins")
ax.legend(loc="upper left", framealpha=0.2, fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.savefig("chart2_constructor_dominance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart2_constructor_dominance.png")


# ── 4. Chart 3: Top 10 Drivers by Wins ──────────────────────
print("Plotting Chart 3: Top drivers by wins...")

top_drivers = (
    df_modern[df_modern["positionOrder"] == 1]
    .groupby("driver_name").size()
    .nlargest(10)
    .sort_values()
)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_drivers.index, top_drivers.values,
               color=[F1_RED if v == top_drivers.max() else "#555555" for v in top_drivers.values])
ax.set_title("Top 10 Drivers by Race Wins (2010–2023)", color=F1_WHITE, fontsize=14, pad=15)
ax.set_xlabel("Total Wins")
for i, (name, val) in enumerate(top_drivers.items()):
    ax.text(val + 0.3, i, str(val), va="center", color=F1_WHITE, fontsize=9)
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("chart3_top_drivers.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart3_top_drivers.png")


# ── 5. Chart 4: Points Finish Rate by Nationality ────────────
print("Plotting Chart 4: Points finish rate by nationality...")

nationality_stats = (
    df_modern.groupby("nationality")
    .apply(lambda x: (x["positionOrder"] <= 10).sum() / len(x))
    .reset_index(name="points_finish_rate")
)
# Only nationalities with 50+ entries
counts = df_modern.groupby("nationality").size().reset_index(name="count")
nationality_stats = nationality_stats.merge(counts, on="nationality")
nationality_stats = nationality_stats[nationality_stats["count"] >= 50].nlargest(12, "points_finish_rate")

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(nationality_stats["nationality"], nationality_stats["points_finish_rate"] * 100,
       color=F1_RED, alpha=0.85)
ax.set_title("Points Finish Rate by Driver Nationality (2010–2023)", color=F1_WHITE, fontsize=14, pad=15)
ax.set_xlabel("Nationality")
ax.set_ylabel("Points Finish Rate (%)")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
plt.xticks(rotation=35, ha="right")
ax.grid(axis="y")
plt.tight_layout()
plt.savefig("chart4_nationality_points_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart4_nationality_points_rate.png")


# ── 6. Feature Engineering Preview ───────────────────────────
print("\nBuilding feature engineering preview...")

df_feat = df_modern.sort_values(["driver_name", "year", "raceId"]).copy()

# Feature: rolling avg finish position (last 5 races per driver)
df_feat["rolling_avg_finish"] = (
    df_feat.groupby("driver_name")["positionOrder"]
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Feature: DNF flag
df_feat["dnf"] = df_feat["positionOrder"].isna().astype(int)

# Feature: constructor DNF rate
constructor_dnf = (
    df_feat.groupby("name_constructor")["dnf"].mean().reset_index()
    .rename(columns={"dnf": "constructor_dnf_rate"})
)
df_feat = df_feat.merge(constructor_dnf, on="name_constructor")

# Preview
feature_cols = ["driver_name", "year", "name_race", "grid", "positionOrder",
                "rolling_avg_finish", "dnf", "constructor_dnf_rate"]
preview = df_feat[feature_cols].dropna(subset=["positionOrder"]).tail(10)

print("\n── Feature Engineering Preview (last 10 rows) ──")
print(preview.to_string(index=False))

# Save feature-engineered dataset
df_feat.to_csv("f1_features_phase1.csv", index=False)
print("\n✅ Saved feature dataset → f1_features_phase1.csv")


# ── 7. Summary Stats ─────────────────────────────────────────
print("\n── Quick Stats ──────────────────────────────────────")
pole_to_win = df_modern[df_modern["grid"] == 1]["positionOrder"].eq(1).mean()
print(f"Pole-to-win conversion rate:  {pole_to_win:.1%}")

avg_grid_of_winner = df_modern[df_modern["positionOrder"] == 1]["grid"].mean()
print(f"Average grid position of race winner:  {avg_grid_of_winner:.2f}")

top_team = df_modern[df_modern["positionOrder"] == 1]["name_constructor"].value_counts().idxmax()
print(f"Most successful constructor (2010–2023):  {top_team}")

print("\n🏁 Phase 1 complete! Check your folder for 4 chart PNGs.")
print("   Next up → Phase 2: Advanced feature engineering (weather, circuit history, quali gap)")