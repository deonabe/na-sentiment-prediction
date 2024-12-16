import os
import time

import pandas as pd
import praw
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load environment variables from .env file
load_dotenv("../../.env")

# Set up Reddit authentication using environment variables
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API with PRAW
reddit = praw.Reddit(
    client_id=client_id, client_secret=client_secret, user_agent=user_agent
)

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# List of NBA teams
teams = [
    "Lakers",
    "Bucks",
    "Warriors",
    "Nets",
    "Heat",
    "76ers",
    "Celtics",
    "Nuggets",
    "Suns",
    "Mavericks",
    "Clippers",
    "Raptors",
    "Knicks",
    "Pelicans",
    "Pacers",
    "Kings",
    "Bulls",
    "Magic",
    "Spurs",
    "Hornets",
    "Wizards",
    "Hawks",
    "Grizzlies",
    "Rockets",
    "Pistons",
    "Cavaliers",
    "Timberwolves",
    "Trail Blazers",
    "Jazz",
    "Thunder",
]

# Initialize list to hold Reddit post data
all_team_data = []

# Define the start and end timestamps for the 2023-2024 season
start_timestamp = 1695907200  # October 1, 2023 (start of the 2023-2024 season)
end_timestamp = 1714435200  # April 1, 2024 (or adjust as needed for end of season)

# List of subreddits to search (NBA teams and NBA betting related subreddits)
subreddits_to_search = [
    "nba",  # General NBA
    "nbadiscussion",
    "nbabetting",
    "sportsbook",  # Betting-focused subreddits
]

# Search Reddit for posts mentioning each team and analyze sentiment
for team in teams:
    print(f"Collecting posts for {team}...")

    # Iterate over all relevant subreddits
    for subreddit_name in subreddits_to_search:
        subreddit = reddit.subreddit(subreddit_name)  # Access the subreddit

        # Search for posts mentioning the team in the selected subreddit
        posts = subreddit.search(
            team, sort="new", time_filter="year", limit=100  # Adjust limit as needed
        )

        for submission in posts:
            # Get submission's creation time (in UTC) and check if it falls in the 2023-2024 season
            created_at = submission.created_utc
            if (
                start_timestamp <= created_at <= end_timestamp
            ):  # Filter based on season time range
                sentiment_score = analyzer.polarity_scores(
                    submission.title + " " + submission.selftext
                )

                # Append the sentiment data to the list
                all_team_data.append(
                    {
                        "team": team,
                        "subreddit": subreddit_name,
                        "post_title": submission.title,
                        "post_text": submission.selftext,
                        "positive": sentiment_score["pos"],
                        "neutral": sentiment_score["neu"],
                        "negative": sentiment_score["neg"],
                        "compound": sentiment_score["compound"],
                        "created_at": submission.created_utc,
                        "url": submission.url,
                    }
                )

    # Sleep to avoid hitting Reddit API rate limits
    time.sleep(2)

# Step 2: Convert to DataFrame and save to CSV
df_teams_sentiment = pd.DataFrame(all_team_data)
df_teams_sentiment.to_csv(
    "../../data/sentiment/nba_teams_sentiment_2023.csv", index=False
)

print("Sentiment data for NBA teams collected and saved.")
