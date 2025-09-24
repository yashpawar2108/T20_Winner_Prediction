import pandas as pd
import numpy as np

def rolling_adjusted_sum(series, N):
    rolled_sum = series.shift().rolling(N, min_periods=1).sum()
    rolled_count = series.shift().rolling(N, min_periods=1).count()
    return np.where(
        rolled_count < N,
        (rolled_sum / np.where(rolled_count.eq(0), 1, rolled_count)) * N,
        rolled_sum
    )

def process_batting_data(N=5):
    bat = pd.read_csv('batsman_level_scorecard.csv')
    bat["match_dt"] = pd.to_datetime(bat["match_dt"], errors="coerce")

    match_stats = (
        bat.groupby(["batsman_id", "match id", "match_dt"], as_index=False)
        .agg(
            runs=("runs", "sum"),
            balls=("balls_faced", "sum"),
            fours=("Fours", "sum"),
            sixes=("Sixes", "sum"),
            dismissals=("wicket kind", lambda x: np.sum(x.notna())),
            dismissal_types=("wicket kind", lambda x: x.dropna().iloc[0] if len(x.dropna()) else np.nan)
        )
        .sort_values(["batsman_id", "match_dt"])
    )

    g = match_stats.groupby("batsman_id", group_keys=False)

    match_stats["lastN_runs"] = g["runs"].transform(lambda s: rolling_adjusted_sum(s, N))

    match_stats["lastN_runs_sum"] = g["runs"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())
    match_stats["lastN_dismissals_sum"] = g["dismissals"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())

    match_stats["lastN_avg"] = np.where(
        match_stats["lastN_dismissals_sum"] > 0,
        match_stats["lastN_runs_sum"] / match_stats["lastN_dismissals_sum"],
        0.0
    )

    match_stats["lastN_sr"] = (
        match_stats["lastN_runs_sum"] /
        g["balls"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()).replace({0: np.nan}) * 100
    ).fillna(0.0)

    match_stats["lastN_boundary_pct"] = (
        (g["fours"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()) +
         g["sixes"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())) /
        g["balls"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()).replace({0: np.nan}) * 100
    ).fillna(0.0)

    dismissal_types_unique = bat["wicket kind"].dropna().unique()

    for dtype in dismissal_types_unique:
        match_stats[f"lastN_{dtype}"] = (
            g["dismissal_types"]
            .transform(lambda s: (s.shift().eq(dtype)).rolling(N, min_periods=1).sum())
            .fillna(0.0)
        )

    match_stats["career_runs"] = g["runs"].cumsum() - match_stats["runs"]
    career_dismissals_before = g["dismissals"].cumsum() - match_stats["dismissals"]
    match_stats["career_avg"] = np.where(
        career_dismissals_before > 0,
        (g["runs"].cumsum() - match_stats["runs"]) / career_dismissals_before,
        0.0
    )

    career_balls_before = (g["balls"].cumsum() - match_stats["balls"]).replace({0: np.nan})
    match_stats["career_sr"] = (
        (g["runs"].cumsum() - match_stats["runs"]) / career_balls_before * 100
    ).fillna(0.0)

    dismissal_cols = [f"lastN_{dtype}" for dtype in dismissal_types_unique]
    batsman_features = match_stats[[
        "batsman_id", "match_dt", "lastN_runs", "lastN_avg", "lastN_sr",
        "lastN_boundary_pct", "career_runs", "career_avg", "career_sr"
    ] + dismissal_cols].copy()

    batsman_features = batsman_features.fillna(0.0)
    batsman_features.to_csv('batsman_features2.csv', index=False)
    print("Batsman features created and saved to batsman_features2.csv")

if __name__ == '__main__':
    process_batting_data()
