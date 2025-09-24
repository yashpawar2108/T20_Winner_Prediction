import pandas as pd
import numpy as np

def attach_player_features(match_df, player_col, prefix, player_groups_dict, batsman_features):
    out_index = match_df.index
    cols = [c for c in batsman_features.columns if c not in ["batsman_id", "match_dt"]]
    result = pd.DataFrame(np.nan, index=out_index, columns=[f"{prefix}_{c}" for c in cols])

    for idx, row in match_df.iterrows():
        pid = row[player_col]
        if pd.isna(pid) or pid == "nan":
            continue

        pid_str = str(pid)
        player_df = player_groups_dict.get(pid_str)
        if player_df is None or player_df.empty:
            continue

        cutoff = pd.to_datetime(row["match_dt"])  # ensure Timestamp
        player_dates = pd.to_datetime(player_df["match_dt"])  # ensure DatetimeIndex

        pos = player_dates.searchsorted(cutoff, side="right") - 1
        if pos < 0:
            continue

        latest_feats = player_df.iloc[pos].drop(labels=["batsman_id", "match_dt"])
        result.loc[idx, [f"{prefix}_{c}" for c in cols]] = latest_feats.values

    return result

def role_for_position(pos, role_splits):
    for role, (start, end) in role_splits.items():
        if start <= pos <= end:
            return role
    return "lower_order"

def calc_player_batting_score(row, prefix, role, ROLE_WEIGHTS, N):
    lastN_avg = row.get(f"{prefix}_lastN_avg", 0.0) or 0.0
    career_avg = row.get(f"{prefix}_career_avg", 0.0) or 0.0
    career_sr = row.get(f"{prefix}_career_sr", 0.0) or 0.0
    lastN_runs = row.get(f"{prefix}_lastN_runs", 0.0) or 0.0
    lastN_sr = row.get(f"{prefix}_lastN_sr", 0.0) or 0.0
    lastN_bpct = row.get(f"{prefix}_lastN_boundary_pct", 0.0) or 0.0

    feats = {
        "lastN_avg": lastN_avg,
        "career_avg": career_avg,
        "career_sr": career_sr,
        "lastN_runs_per_match": lastN_runs / float(N) if N else 0.0,
        "lastN_sr": lastN_sr,
        "lastN_boundary_pct": lastN_bpct,
    }

    weights = ROLE_WEIGHTS.get(role, {})
    score = 0.0
    for k, w in weights.items():
        score += w * feats.get(k, 0.0)
    return score

def calc_team_role_scores_by_order(df, team_num, role_splits, ROLE_WEIGHTS, N):
    for p in range(1, 13):
        prefix = f"T{team_num}_P{p}"
        role = role_for_position(p, role_splits)
        df[f"{prefix}_bat_score"] = df.apply(lambda r: calc_player_batting_score(r, prefix, role, ROLE_WEIGHTS, N), axis=1)

    role_totals = {f"team{team_num}_{role}_score": pd.Series(0.0, index=df.index) for role in role_splits.keys()}

    for role, (start, end) in role_splits.items():
        cols = [f"T{team_num}_P{p}_bat_score" for p in range(start, end + 1)]
        role_totals[f"team{team_num}_{role}_score"] = df[cols].fillna(0.0).sum(axis=1)

    return pd.DataFrame(role_totals)

def attach_bowler_features(match_df, player_col, prefix, bowler_df):
    merged_list = []
    
    for idx, row in match_df.iterrows():
        pid = row[player_col]
        if pd.isna(pid) or pid == "nan":
            merged_list.append(pd.Series(np.nan, index=[f"{prefix}_{col}" for col in bowler_df.columns if col not in ["bowler_id", "match_dt"]]))
            continue
        
        pid_str = str(pid)
        bowler_feats = bowler_df[bowler_df["bowler_id"] == pid_str]
        
        if bowler_feats.empty:
            merged_list.append(pd.Series(np.nan, index=[f"{prefix}_{col}" for col in bowler_df.columns if col not in ["bowler_id", "match_dt"]]))
            continue
        
        bowler_feats_before = bowler_feats[bowler_feats["match_dt"] <= row["match_dt"]]
        if bowler_feats_before.empty:
            merged_list.append(pd.Series(np.nan, index=[f"{prefix}_{col}" for col in bowler_df.columns if col not in ["bowler_id", "match_dt"]]))
            continue
        
        latest_feats = bowler_feats_before.iloc[-1]
        merged_list.append(latest_feats.drop(["bowler_id", "match_dt"]).rename(lambda c: f"{prefix}_{c}"))
    
    return pd.DataFrame(merged_list, index=match_df.index)

def calc_player_bowling_score(row, prefix):
    try:
        lastN_economy = row[f"{prefix}_lastN_economy"]
        career_economy = row[f"{prefix}_career_economy"]
        lastN_bowling_avg = row[f"{prefix}_lastN_bowling_avg"]
        lastN_dot_pct = row[f"{prefix}_lastN_dot_pct"]
        
        if pd.isna(lastN_economy) or pd.isna(career_economy) or pd.isna(lastN_bowling_avg):
            return np.nan
        
        economy_score = max(0, 10 - lastN_economy)
        career_economy_score = max(0, 10 - career_economy)
        bowling_avg_score = max(0, 50 - lastN_bowling_avg)
        dot_ball_score = lastN_dot_pct if not pd.isna(lastN_dot_pct) else 0
        
        score = (
            (economy_score * 0.4) +
            (career_economy_score * 0.3) +
            (bowling_avg_score * 0.2) +
            (dot_ball_score * 0.1)
        )
        return score
    except KeyError:
        return np.nan

def calc_team_bowling_scores(df, team_num, bowling_roles):
    for p in range(1, 13):
        prefix = f"T{team_num}_P{p}_bowl"
        df[f"{prefix}_score"] = df.apply(lambda r: calc_player_bowling_score(r, prefix), axis=1)
    
    def row_role_scores(row):
        scores = [(p, row[f"T{team_num}_P{p}_bowl_score"]) for p in range(1, 13)]
        scores = [s for s in scores if not pd.isna(s[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        role_totals = {role: 0 for role in bowling_roles.keys()}
        
        for role, (start, end) in bowling_roles.items():
            for _, sc in scores[start:end]:
                role_totals[role] += sc
        
        return pd.Series({f"team{team_num}_{role}_bowling_score": val for role, val in role_totals.items()})
    
    return df.apply(row_role_scores, axis=1)

def create_features(N=5):
    match = pd.read_csv('top5_train1.csv')
    batsman_features = pd.read_csv('batsman_features2.csv')
    bowler_features = pd.read_csv('bowler_features.csv')

    match["team1_roster_ids"] = match["team1_roster_ids"].astype(str)
    match["team2_roster_ids"] = match["team2_roster_ids"].astype(str)
    match["match_dt"] = pd.to_datetime(match["match_dt"], errors="coerce")

    batsman_features["batsman_id"] = batsman_features["batsman_id"].astype(str)
    batsman_features["match_dt"] = pd.to_datetime(batsman_features["match_dt"], errors="coerce")

    for team_prefix in ["team1", "team2"]:
        match[[f"{team_prefix}_P{i}" for i in range(1, 13)]] = (
            match[team_prefix + "_roster_ids"]
            .str.split(":", expand=True)
            .iloc[:, :12]
            .fillna(np.nan)
        )

    batsman_features = batsman_features.sort_values(["batsman_id", "match_dt"]).reset_index(drop=True)
    match = match.sort_values("match_dt").reset_index(drop=True)

    player_groups = {pid: df.reset_index(drop=True) for pid, df in batsman_features.groupby("batsman_id")}

    for team_num in [1, 2]:
        for p in range(1, 13):
            player_col = f"team{team_num}_P{p}"
            prefix = f"T{team_num}_P{p}"
            feats_df = attach_player_features(match, player_col, prefix, player_groups, batsman_features)
            match = pd.concat([match, feats_df], axis=1)

    role_splits = {
        "opener": (1, 2),
        "top_order": (3, 4),
        "middle_order": (5, 7),
        "lower_order": (8, 12)
    }

    ROLE_WEIGHTS = {
        "opener": {
            "lastN_sr": 0.35,
            "lastN_boundary_pct": 0.25,
            "lastN_avg": 0.20,
            "career_sr": 0.15,
            "career_avg": 0.05,
        },
        "top_order": {
            "lastN_avg": 0.30,
            "lastN_sr": 0.25,
            "career_avg": 0.25,
            "career_sr": 0.15,
            "lastN_runs_per_match": 0.05,
        },
        "middle_order": {
            "lastN_avg": 0.35,
            "career_avg": 0.30,
            "career_sr": 0.20,
            "lastN_runs_per_match": 0.10,
            "lastN_boundary_pct": 0.05,
        },
        "lower_order": {
            "career_sr": 0.40,
            "lastN_sr": 0.25,
            "lastN_boundary_pct": 0.20,
            "lastN_runs_per_match": 0.10,
            "career_avg": 0.05,
        },
    }

    team1_roles = calc_team_role_scores_by_order(match, 1, role_splits, ROLE_WEIGHTS, N)
    team2_roles = calc_team_role_scores_by_order(match, 2, role_splits, ROLE_WEIGHTS, N)

    match = pd.concat([match, team1_roles, team2_roles], axis=1)

    bowler_features["bowler_id"] = bowler_features["bowler_id"].astype(str)
    bowler_features["match_dt"] = pd.to_datetime(bowler_features["match_dt"], errors="coerce")
    bowler_features = bowler_features.sort_values(["bowler_id", "match_dt"])

    for team_num in [1, 2]:
        for p in range(1, 13):
            player_col = f"team{team_num}_P{p}"
            prefix = f"T{team_num}_P{p}_bowl"
            feats_df = attach_bowler_features(match, player_col, prefix, bowler_features)
            match = pd.concat([match, feats_df], axis=1)

    bowling_roles = {
        "pace_attack": (0, 3),
        "spin_support": (3, 6),
        "all_rounder": (6, 9),
        "part_time": (9, 12)
    }

    team1_bowling_roles = calc_team_bowling_scores(match, 1, bowling_roles)
    team2_bowling_roles = calc_team_bowling_scores(match, 2, bowling_roles)

    match = pd.concat([match, team1_bowling_roles, team2_bowling_roles], axis=1)

    match["winner12"] = np.where(
        match["team1_id"] == match["winner_id"], 1,
        np.where(match["team2_id"] == match["winner_id"], 2, 0)
    )

    match.to_csv("match_features.csv", index=False)
    print("Match features created and saved to match_features.csv")

if __name__ == '__main__':
    create_features()
