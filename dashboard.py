
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    match_df = pd.read_csv('match_features.csv')
    batsman_df = pd.read_csv('batsman_features2.csv')
    bowler_df = pd.read_csv('bowler_features.csv')
    return match_df, batsman_df, bowler_df

match_df, batsman_df, bowler_df = load_data()

st.sidebar.title("Cricket Match Analysis")
page = st.sidebar.radio("Choose a page", ["Match Analysis", "Player Analysis", "Team Analysis", "Team vs Team"])

if page == "Match Analysis":
    st.title("Match Analysis")

    st.header("Batting Role Analysis")
    batting_roles = ["opener", "top_order", "middle_order", "lower_order"]
    selected_batting_role = st.selectbox("Select Batting Role", batting_roles)

    # Scatter plot
    scatter_chart = alt.Chart(match_df).mark_circle(size=60).encode(
        x=f'team1_{selected_batting_role}_score',
        y=f'team2_{selected_batting_role}_score',
        color=alt.Color('winner12:N', title="Winner"),
        tooltip=['match_id:N', 'team1_id:N', 'team2_id:N', 'winner12:N']
    ).interactive()
    st.altair_chart(scatter_chart, use_container_width=True)

    # Violin plot
    melted_df = match_df.melt(
        id_vars=["winner12"],
        value_vars=[f"team1_{selected_batting_role}_score", f"team2_{selected_batting_role}_score"],
        var_name="team_role",
        value_name="score"
    )
    violin_chart = alt.Chart(melted_df).transform_density(
        'score',
        as_=['score', 'density'],
        extent=[melted_df['score'].min(), melted_df['score'].max()],
        groupby=['team_role']
    ).mark_area(orient='horizontal').encode(
        x='density:Q',
        y='score:Q',
        color='team_role:N'
    ).properties(width=300)
    st.altair_chart(violin_chart)


    st.header("Bowling Role Analysis")
    bowling_roles = ["pace_attack", "spin_support", "all_rounder", "part_time"]
    selected_bowling_role = st.selectbox("Select Bowling Role", bowling_roles, key='bowling_role_selector')

    # Scatter plot for bowling
    bowling_scatter_chart = alt.Chart(match_df).mark_circle(size=60).encode(
        x=alt.X(f'team1_{selected_bowling_role}_bowling_score', title='Team 1 Bowling Score'),
        y=alt.Y(f'team2_{selected_bowling_role}_bowling_score', title='Team 2 Bowling Score'),
        color=alt.Color('winner12:N', title="Winner"),
        tooltip=['match_id:N', 'team1_id:N', 'team2_id:N', 'winner12:N']
    ).interactive()
    st.altair_chart(bowling_scatter_chart, use_container_width=True)


elif page == "Player Analysis":
    st.title("Player Analysis")
    
    all_players = sorted(pd.concat([batsman_df['batsman_id'], bowler_df['bowler_id']]).unique())
    selected_player = st.selectbox("Select Player", all_players)

    st.header(f"Batting Stats for {selected_player}")
    player_batting_stats = batsman_df[batsman_df['batsman_id'] == selected_player]
    if not player_batting_stats.empty:
        st.dataframe(player_batting_stats)
        
        st.subheader("Batting Performance Over Time")
        batting_perf_chart = alt.Chart(player_batting_stats).mark_line(point=True).encode(
            x='match_dt:T',
            y='lastN_avg:Q',
            tooltip=['match_dt', 'lastN_avg', 'lastN_sr']
        ).interactive()
        st.altair_chart(batting_perf_chart, use_container_width=True)

    else:
        st.write("No batting data for this player.")

    st.header(f"Bowling Stats for {selected_player}")
    player_bowling_stats = bowler_df[bowler_df['bowler_id'] == selected_player]
    if not player_bowling_stats.empty:
        st.dataframe(player_bowling_stats)

        st.subheader("Bowling Performance Over Time")
        bowling_perf_chart = alt.Chart(player_bowling_stats).mark_line(point=True).encode(
            x='match_dt:T',
            y='lastN_economy:Q',
            tooltip=['match_dt', 'lastN_economy', 'lastN_bowling_avg']
        ).interactive()
        st.altair_chart(bowling_perf_chart, use_container_width=True)
    else:
        st.write("No bowling data for this player.")


elif page == "Team Analysis":
    st.title("Team Analysis")
    
    all_teams = sorted(pd.concat([match_df['team1_id'], match_df['team2_id']]).unique())
    selected_team = st.selectbox("Select Team", all_teams)

    st.header(f"Performance for {selected_team}")
    
    team_matches = match_df[(match_df['team1_id'] == selected_team) | (match_df['team2_id'] == selected_team)]
    wins = len(team_matches[team_matches['winner_id'] == selected_team])
    losses = len(team_matches) - wins
    
    st.write(f"Matches Played: {len(team_matches)}")
    st.write(f"Wins: {wins}")
    st.write(f"Losses: {losses}")

    st.subheader("Average Role Scores")
    avg_scores = {}
    for role in ["opener", "top_order", "middle_order", "lower_order"]:
        team1_scores = team_matches[team_matches['team1_id'] == selected_team][f'team1_{role}_score']
        team2_scores = team_matches[team_matches['team2_id'] == selected_team][f'team2_{role}_score']
        avg_scores[role] = pd.concat([team1_scores, team2_scores]).mean()
    
    st.bar_chart(pd.Series(avg_scores))


elif page == "Team vs Team":
    st.title("Team vs Team Analysis")

    all_teams = sorted(pd.concat([match_df['team1_id'], match_df['team2_id']]).unique())
    
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1", all_teams, index=0)
    with col2:
        team2 = st.selectbox("Select Team 2", all_teams, index=1)

    if team1 and team2 and team1 != team2:
        h2h_matches = match_df[((match_df['team1_id'] == team1) & (match_df['team2_id'] == team2)) | ((match_df['team1_id'] == team2) & (match_df['team2_id'] == team1))]
        
        if not h2h_matches.empty:
            st.header(f"Head-to-Head: {team1} vs {team2}")
            
            team1_wins = len(h2h_matches[h2h_matches['winner_id'] == team1])
            team2_wins = len(h2h_matches[h2h_matches['winner_id'] == team2])
            
            st.write(f"{team1} Wins: {team1_wins}")
            st.write(f"{team2} Wins: {team2_wins}")

            st.subheader("Recent Matches")
            st.dataframe(h2h_matches[['match_dt', 'team1_id', 'team2_id', 'winner_id']].sort_values('match_dt', ascending=False))
        else:
            st.write("No matches found between these two teams.")

