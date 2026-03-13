import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Styling ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a1a",
    "axes.edgecolor": "#333333",   "axes.labelcolor": "#cccccc",
    "xtick.color": "#888888",      "ytick.color": "#888888",
    "text.color": "#ffffff",       "grid.color": "#2a2a2a",
    "grid.linestyle": "--",        "font.family": "monospace",
})
F1_RED = "#e8002d"; F1_GOLD = "#f5c842"; F1_WHITE = "#ffffff"


# ── 1. Load Data ─────────────────────────────────────────────
print("Loading data...")
df        = pd.read_csv("f1_features_phase1.csv")
races     = pd.read_csv("data/races.csv")
qualifying = pd.read_csv("data/qualifying.csv")
results   = pd.read_csv("data/results.csv")
circuits  = pd.read_csv("data/circuits.csv")

df = df[df["year"] >= 2010].copy()
df["positionOrder"] = pd.to_numeric(df["positionOrder"], errors="coerce")
df["grid"]          = pd.to_numeric(df["grid"], errors="coerce")

print(f"✅ Base dataset: {len(df):,} rows")


# ── 2. Qualifying Gap to Pole ─────────────────────────────────
print("\nBuilding qualifying gap to pole...")

# Parse qualifying times to seconds
def time_to_seconds(t):
    try:
        if pd.isna(t) or t in ("\\N", "", "N"):
            return np.nan
        parts = str(t).split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return np.nan

qualifying["q_seconds"] = qualifying[["q1", "q2", "q3"]].apply(
    lambda row: np.nanmin([time_to_seconds(row["q1"]),
                          time_to_seconds(row["q2"]),
                          time_to_seconds(row["q3"])]), axis=1
)

# Pole time = fastest in each race
pole_times = (
    qualifying.groupby("raceId")["q_seconds"]
    .min().reset_index().rename(columns={"q_seconds": "pole_time"})
)
qualifying = qualifying.merge(pole_times, on="raceId")
qualifying["quali_gap_to_pole"] = qualifying["q_seconds"] - qualifying["pole_time"]
qualifying["quali_gap_pct"]     = qualifying["quali_gap_to_pole"] / qualifying["pole_time"] * 100

# Merge into main df via raceId + driverId
races_slim = races[["raceId", "year", "name"]].rename(columns={"name": "name_race"})
qualifying = qualifying.merge(races_slim, on="raceId")

df = df.merge(
    qualifying[["raceId", "driverId", "quali_gap_to_pole", "quali_gap_pct"]],
    on=["raceId", "driverId"], how="left"
)
print(f"   Qualifying gap coverage: {df['quali_gap_to_pole'].notna().mean():.1%} of rows")


# ── 3. Circuit-Specific Driver Win History ───────────────────
print("Building circuit win history...")

df = df.sort_values(["driver_name", "circuitId", "year", "raceId"])

def circuit_wins_before(group):
    won = (group["positionOrder"] == 1).astype(int)
    return won.shift(1).expanding().sum().fillna(0)

df["circuit_wins_before"] = (
    df.groupby(["driver_name", "circuitId"], group_keys=False)
    .apply(circuit_wins_before)
)

def circuit_podiums_before(group):
    podium = (group["positionOrder"] <= 3).astype(int)
    return podium.shift(1).expanding().sum().fillna(0)

df["circuit_podiums_before"] = (
    df.groupby(["driver_name", "circuitId"], group_keys=False)
    .apply(circuit_podiums_before)
)

print(f"   Max circuit wins by one driver at one track: {df['circuit_wins_before'].max():.0f}")


# ── 4. Constructor Rolling Form ──────────────────────────────
print("Building constructor rolling form...")

df = df.sort_values(["name_constructor", "year", "raceId"])

df["constructor_rolling_points"] = (
    df.groupby("name_constructor")["points"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

df["constructor_rolling_wins"] = (
    df.groupby("name_constructor")["positionOrder"]
    .transform(lambda x: (x.shift(1) == 1).rolling(5, min_periods=1).sum())
)


# ── 5. Season Progress ───────────────────────────────────────
print("Building season progress features...")

race_round = races[["raceId", "round", "year"]].copy()
df = df.merge(race_round, on=["raceId", "year"], how="left")

total_rounds = df.groupby("year")["round"].max().reset_index().rename(columns={"round": "total_rounds"})
df = df.merge(total_rounds, on="year")
df["season_progress"] = df["round"] / df["total_rounds"]  # 0.0 = race 1, 1.0 = final race

# Championship pressure: driver points gap to leader
df = df.sort_values(["year", "raceId", "driver_name"])
cumulative = (
    df.sort_values(["driver_name", "year", "raceId"])
    .groupby(["driver_name", "year"])["points"]
    .cumsum()
)
df["cumulative_points"] = cumulative.values

race_leader_points = (
    df.groupby(["year", "raceId"])["cumulative_points"]
    .max().reset_index().rename(columns={"cumulative_points": "leader_points"})
)
df = df.merge(race_leader_points, on=["year", "raceId"])
df["points_gap_to_leader"] = df["leader_points"] - df["cumulative_points"]


# ── 6. Driver Experience ─────────────────────────────────────
print("Building driver experience feature...")

df = df.sort_values(["driver_name", "year", "raceId"])
df["career_races"] = df.groupby("driver_name").cumcount()  # races started before this one


# ── 7. Home Race Flag ────────────────────────────────────────
print("Building home race flag...")

circuits_slim = circuits[["circuitId", "country"]].copy()
df = df.merge(circuits_slim, on="circuitId", how="left")

nationality_country_map = {
    "British": ["UK", "Great Britain"],
    "German": ["Germany"],
    "Dutch": ["Netherlands"],
    "Spanish": ["Spain"],
    "Finnish": ["Finland"],
    "French": ["France"],
    "Australian": ["Australia"],
    "Canadian": ["Canada"],
    "Italian": ["Italy"],
    "Brazilian": ["Brazil"],
    "Mexican": ["Mexico"],
    "Japanese": ["Japan"],
    "Monegasque": ["Monaco"],
    "Belgian": ["Belgium"],
    "Austrian": ["Austria"],
    "American": ["USA", "United States"],
}

def is_home_race(row):
    countries = nationality_country_map.get(row["nationality"], [])
    return int(any(c.lower() in str(row["country"]).lower() for c in countries))

df["home_race"] = df.apply(is_home_race, axis=1)
print(f"   Home race entries: {df['home_race'].sum():,}")


# ── 8. Final Feature Set ─────────────────────────────────────
feature_cols = [
    "raceId", "driverId", "constructorId",
    "driver_name", "name_constructor", "name_race",
    "year", "round", "season_progress", "circuitId", "country",

    # Target
    "positionOrder",

    # Core race features
    "grid", "quali_gap_to_pole", "quali_gap_pct",

    # Driver form
    "rolling_avg_finish", "career_races",
    "circuit_wins_before", "circuit_podiums_before",
    "cumulative_points", "points_gap_to_leader",

    # Constructor features
    "constructor_dnf_rate", "constructor_rolling_points", "constructor_rolling_wins",

    # Context
    "home_race", "dnf",
]

df_final = df[feature_cols].dropna(subset=["positionOrder", "grid"])
df_final.to_csv("f1_features_phase2.csv", index=False)
print(f"\n✅ Saved → f1_features_phase2.csv  ({len(df_final):,} rows, {len(feature_cols)} columns)")


# ── 9. Feature Correlation Chart ────────────────────────────
print("\nPlotting feature correlation heatmap...")

numeric_features = [
    "grid", "quali_gap_pct", "rolling_avg_finish", "career_races",
    "circuit_wins_before", "circuit_podiums_before", "points_gap_to_leader",
    "constructor_dnf_rate", "constructor_rolling_points",
    "season_progress", "home_race", "positionOrder"
]

corr = df_final[numeric_features].corr()[["positionOrder"]].drop("positionOrder")
corr = corr.sort_values("positionOrder")

fig, ax = plt.subplots(figsize=(7, 8))
colors = [F1_RED if v > 0 else "#4a9eff" for v in corr["positionOrder"]]
ax.barh(corr.index, corr["positionOrder"], color=colors)
ax.axvline(0, color="#555555", linewidth=1)
ax.set_title("Feature Correlation with Finishing Position", color=F1_WHITE, fontsize=13, pad=15)
ax.set_xlabel("Pearson Correlation")
ax.annotate("← predicts better finish     worse finish →", xy=(0.5, -0.08),
            xycoords="axes fraction", ha="center", fontsize=8, color="#888888")
ax.grid(axis="x")
plt.tight_layout()
plt.savefig("chart5_feature_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("   → Saved: chart5_feature_correlation.png")


# ── 10. Summary ──────────────────────────────────────────────
print("\n── Feature Summary ──────────────────────────────────")
print(df_final[numeric_features].describe().round(2).to_string())
print("\n🏁 Phase 2 complete!")
print("   Next up → Phase 3: Train Logistic Regression, Random Forest & XGBoost")