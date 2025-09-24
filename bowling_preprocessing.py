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

def process_bowling_data(N=5):
    bowl = pd.read_csv('bowler_level_scorecard.csv')
    bowl["match_dt"] = pd.to_datetime(bowl["match_dt"], errors="coerce")

    match_stats_bowl = (
        bowl.groupby(["bowler_id", "match id", "match_dt"], as_index=False)
        .agg(
            runs_conceded=("runs", "sum"),
            balls_bowled=("balls_bowled", "sum"),
            wickets=("wicket_count", "sum"),
            maidens=("maiden", "sum"),
            dots=("dots", "sum"),
            fours_conceded=("Fours", "sum"),
            sixes_conceded=("Sixes", "sum"),
            wides=("wides", "sum"),
            noballs=("noballs", "sum"),
            economy=("economy", "mean")
        )
        .sort_values(["bowler_id", "match_dt"])
    )

    g_bowl = match_stats_bowl.groupby("bowler_id", group_keys=False)

    match_stats_bowl["lastN_runs_conceded"] = g_bowl["runs_conceded"].transform(lambda s: rolling_adjusted_sum(s, N))
    match_stats_bowl["lastN_balls_bowled"] = g_bowl["balls_bowled"].transform(lambda s: rolling_adjusted_sum(s, N))
    match_stats_bowl["lastN_wickets"] = g_bowl["wickets"].transform(lambda s: rolling_adjusted_sum(s, N))

    match_stats_bowl["lastN_runs_sum"] = g_bowl["runs_conceded"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())
    match_stats_bowl["lastN_balls_sum"] = g_bowl["balls_bowled"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())
    match_stats_bowl["lastN_wickets_sum"] = g_bowl["wickets"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())

    match_stats_bowl["lastN_bowling_avg"] = np.where(
        match_stats_bowl["lastN_wickets_sum"] > 0,
        match_stats_bowl["lastN_runs_sum"] / match_stats_bowl["lastN_wickets_sum"],
        np.nan
    )

    match_stats_bowl["lastN_economy"] = np.where(
        match_stats_bowl["lastN_balls_sum"] > 0,
        (match_stats_bowl["lastN_runs_sum"] / match_stats_bowl["lastN_balls_sum"]) * 6,  # 6 balls per over
        np.nan
    )

    match_stats_bowl["lastN_strike_rate"] = np.where(
        match_stats_bowl["lastN_wickets_sum"] > 0,
        match_stats_bowl["lastN_balls_sum"] / match_stats_bowl["lastN_wickets_sum"],
        np.nan
    )

    match_stats_bowl["lastN_dot_pct"] = (
        g_bowl["dots"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()) /
        match_stats_bowl["lastN_balls_sum"] * 100
    )

    match_stats_bowl["lastN_boundary_conceded_pct"] = (
        (g_bowl["fours_conceded"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()) +
         g_bowl["sixes_conceded"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())) /
        match_stats_bowl["lastN_balls_sum"] * 100
    )

    match_stats_bowl["lastN_extras_pct"] = (
        (g_bowl["wides"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum()) +
         g_bowl["noballs"].transform(lambda s: s.shift().rolling(N, min_periods=1).sum())) /
        match_stats_bowl["lastN_balls_sum"] * 100
    )

    match_stats_bowl["career_runs_conceded"] = g_bowl["runs_conceded"].cumsum() - match_stats_bowl["runs_conceded"]
    match_stats_bowl["career_balls_bowled"] = g_bowl["balls_bowled"].cumsum() - match_stats_bowl["balls_bowled"]
    match_stats_bowl["career_wickets"] = g_bowl["wickets"].cumsum() - match_stats_bowl["wickets"]

    match_stats_bowl["career_bowling_avg"] = np.where(
        match_stats_bowl["career_wickets"] > 0,
        match_stats_bowl["career_runs_conceded"] / match_stats_bowl["career_wickets"],
        np.nan
    )

    match_stats_bowl["career_economy"] = np.where(
        match_stats_bowl["career_balls_bowled"] > 0,
        (match_stats_bowl["career_runs_conceded"] / match_stats_bowl["career_balls_bowled"]) * 6,
        np.nan
    )

    match_stats_bowl["career_strike_rate"] = np.where(
        match_stats_bowl["career_wickets"] > 0,
        match_stats_bowl["career_balls_bowled"] / match_stats_bowl["career_wickets"],
        np.nan
    )

    bowler_features = match_stats_bowl[[
        "bowler_id", "match_dt", "lastN_runs_conceded", "lastN_bowling_avg", "lastN_economy",
        "lastN_strike_rate", "lastN_dot_pct", "lastN_boundary_conceded_pct", "lastN_extras_pct",
        "career_runs_conceded", "career_bowling_avg", "career_economy", "career_strike_rate"
    ]]

    bowler_features = bowler_features.fillna(0)
    bowler_features.to_csv('bowler_features.csv', index=False)
    print("Bowler features created and saved to bowler_features.csv")

if __name__ == '__main__':
    process_bowling_data()
