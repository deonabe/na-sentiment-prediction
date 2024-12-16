import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file for Reddit API credentials (optional, if used)
load_dotenv("../../.env")

# --- NBA Boxscore Data Processing ---
# Load NBA Boxscore data (team-level game statistics)
boxscore_df = pd.read_csv("../../data/processed/nba_boxscores_2023_2024.csv")

# Convert 'date' to datetime format
boxscore_df["date"] = pd.to_datetime(boxscore_df["date"])

# Encode home/away (1 for home, 0 for away)
boxscore_df["home_game"] = (boxscore_df["home"] == boxscore_df["team"]).astype(int)

# Calculate shooting percentages
boxscore_df["FG%"] = boxscore_df["FGM"] / boxscore_df["FGA"]
boxscore_df["3P%"] = boxscore_df["3PM"] / boxscore_df["3PA"]
boxscore_df["FT%"] = boxscore_df["FTM"] / boxscore_df["FTA"]

# Normalize points scored (based on team averages)
boxscore_df["normalized_pts"] = boxscore_df.groupby("team")["PTS"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Feature: Win/loss encoded as 1 for win and 0 for loss (for predictive modeling)
boxscore_df["win_encoded"] = boxscore_df["win"].apply(lambda x: 1 if x == 1 else 0)

# Save the processed boxscore data to a CSV file
boxscore_df.to_csv(
    "../../data/processed/processed_nba_boxscores_2023_2024.csv", index=False
)

print("NBA Boxscore data processed.")

# --- NBA Player Boxscore Data Processing ---
# Load NBA Player Boxscore data (player-level statistics)
player_boxscore_df = pd.read_csv(
    "../../data/processed/nba_boxscores_player_2023_2024.csv"
)

# Convert 'date' to datetime format
player_boxscore_df["date"] = pd.to_datetime(player_boxscore_df["date"])

# Calculate individual shooting percentages for players
player_boxscore_df["FG%"] = player_boxscore_df["FGM"] / player_boxscore_df["FGA"]
player_boxscore_df["3P%"] = player_boxscore_df["3PM"] / player_boxscore_df["3PA"]
player_boxscore_df["FT%"] = player_boxscore_df["FTM"] / player_boxscore_df["FTA"]

# Feature engineering: Create Player Efficiency Rating (PER) - simplified version
player_boxscore_df["player_efficiency"] = (
    player_boxscore_df["PTS"]
    + player_boxscore_df["REB"]
    + player_boxscore_df["AST"]
    + player_boxscore_df["STL"]
    + player_boxscore_df["BLK"]
    - player_boxscore_df["TOV"]
)

# Normalize player points
player_boxscore_df["normalized_pts"] = player_boxscore_df.groupby("player")[
    "PTS"
].transform(lambda x: (x - x.mean()) / x.std())

# Save the processed player boxscore data to a CSV file
player_boxscore_df.to_csv(
    "../../data/processed/processed_nba_boxscores_player_2023_2024.csv", index=False
)

print("NBA Player Boxscore data processed.")

# --- Reddit Sentiment Data Processing ---
# Load Reddit Sentiment data (analysis of posts mentioning NBA teams)
reddit_sentiment_df = pd.read_csv("../../data/sentiment/nba_teams_sentiment_2023.csv")

# Convert 'created_at' to datetime format
reddit_sentiment_df["created_at"] = pd.to_datetime(
    reddit_sentiment_df["created_at"], unit="s"
)

# Filter for posts in the 2023-2024 season (assuming 'created_at' is in timestamp format)
start_timestamp = pd.to_datetime("2023-10-01")
end_timestamp = pd.to_datetime("2024-04-01")

reddit_sentiment_df = reddit_sentiment_df[
    (reddit_sentiment_df["created_at"] >= start_timestamp)
    & (reddit_sentiment_df["created_at"] <= end_timestamp)
]

# Save the processed sentiment data to a CSV file
reddit_sentiment_df.to_csv(
    "../../data/sentiment/processed_nba_teams_sentiment_2023.csv", index=False
)

print("Reddit Sentiment data processed.")

# --- Fix the Team Name Mismatch for Merging ---
# Mapping of full team names to abbreviations
team_name_mapping = {
    "Lakers": "LAL",
    "Bucks": "MIL",
    "Warriors": "GSW",
    "Nets": "BKN",
    "Heat": "MIA",
    "76ers": "PHI",
    "Celtics": "BOS",
    "Nuggets": "DEN",
    "Suns": "PHX",
    "Mavericks": "DAL",
    "Clippers": "LAC",
    "Raptors": "TOR",
    "Knicks": "NYK",
    "Pelicans": "NOP",
    "Pacers": "IND",
    "Kings": "SAC",
    "Bulls": "CHI",
    "Magic": "ORL",
    "Spurs": "SAS",
    "Hornets": "CHA",
    "Wizards": "WAS",
    "Hawks": "ATL",
    "Grizzlies": "MEM",
    "Rockets": "HOU",
    "Pistons": "DET",
    "Cavaliers": "CLE",
    "Timberwolves": "MIN",
    "Trail Blazers": "POR",
    "Jazz": "UTA",
    "Thunder": "OKC",
}

# Apply the mapping to convert full team names to abbreviations in the Reddit sentiment dataset
reddit_sentiment_df["team"] = (
    reddit_sentiment_df["team"]
    .map(team_name_mapping)
    .fillna(reddit_sentiment_df["team"])
)

# --- Merging Data for Analysis ---
# Merge Team-level data (Boxscore and Reddit Sentiment)
merged_df = pd.merge(boxscore_df, reddit_sentiment_df, on="team", how="inner")

# Merge Player-level data (Player Boxscore and Reddit Sentiment)
merged_player_df = pd.merge(
    player_boxscore_df, reddit_sentiment_df, on="team", how="inner"
)

# Save merged data for future analysis
merged_df.to_csv("../../data/processed/merged_data.csv", index=False)
merged_player_df.to_csv("../../data/processed/merged_player_data.csv", index=False)

print("Merged data for analysis saved.")

# --- Optional: Additional Feature Engineering or Aggregations ---

# You can now proceed to perform additional feature engineering or aggregations
# For example, you might want to calculate rolling averages of sentiment or performance metrics

# Rolling average sentiment for each team
merged_df["rolling_avg_sentiment"] = (
    merged_df.groupby("team")["compound"]
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Save final dataset with rolling average sentiment
merged_df.to_csv("../../data/processed/final_team_data_with_sentiment.csv", index=False)

print("Rolling average sentiment added and final data saved.")
