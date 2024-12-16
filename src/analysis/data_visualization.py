import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the final merged dataset (from the data directory)
merged_df = pd.read_csv("../../data/processed/final_team_data_with_sentiment.csv")

# --- Visualization 1: Sentiment Distribution ---
# Sentiment Distribution Summary
sentiment_stats = merged_df["compound"].describe()
print("Sentiment Distribution Summary (Compound Sentiment Score):")
print(sentiment_stats)

# Plot distribution of sentiment scores (compound) across teams
plt.figure(figsize=(12, 6))
sns.boxplot(x="team", y="compound", data=merged_df)
plt.xticks(rotation=90)
plt.title("Sentiment Distribution Across Teams", fontsize=14)
plt.xlabel("Team", fontsize=12)
plt.ylabel("Compound Sentiment Score", fontsize=12)
plt.tight_layout()
plt.show()

# --- Visualization 2: Correlation Heatmap ---
# Calculate correlations between performance metrics and sentiment
performance_cols = [
    "PTS",
    "FG%",
    "3P%",
    "FT%",
    "AST",
    "REB",
    "TOV",
    "STL",
    "BLK",
    "compound",
]
corr = merged_df[performance_cols].corr()

# Print the correlation matrix
print("Correlation Matrix (Performance Metrics and Sentiment):")
print(corr)

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Performance and Sentiment", fontsize=14)
plt.tight_layout()
plt.show()

# --- Visualization 4: Performance vs Sentiment (Aggregated by Team) ---
# Aggregate performance and sentiment data by team (average points and sentiment)
agg_df = (
    merged_df.groupby("team")
    .agg(avg_pts=("PTS", "mean"), avg_sentiment=("compound", "mean"))
    .reset_index()
)

# Scatter plot of aggregated performance (points) vs sentiment for each team
plt.figure(figsize=(10, 6))
sns.scatterplot(x="avg_pts", y="avg_sentiment", data=agg_df, hue="team", palette="Set2")

# Print aggregated data summary
print("Average Points and Sentiment by Team:")
print(agg_df.describe())


# Adding labels for each team in the scatter plot
for i in range(agg_df.shape[0]):
    plt.text(
        x=agg_df["avg_pts"].iloc[i]
        + 0.1,  # Adjusting the position slightly to the right
        y=agg_df["avg_sentiment"].iloc[i],
        s=agg_df["team"].iloc[i],
        fontweight="bold",
        fontsize=9,
        color="black",
    )

plt.title("Average Points vs Average Sentiment by Team", fontsize=14)
plt.xlabel("Average Points Scored", fontsize=12)
plt.ylabel("Average Compound Sentiment Score", fontsize=12)
plt.tight_layout()
plt.show()

# --- Optional: Save the visualizations to files ---
# Uncomment the lines below to save the plots as images
# plt.savefig("sentiment_distribution.png")
# plt.savefig("correlation_heatmap.png")
# plt.savefig("sentiment_over_time.png")
# plt.savefig("performance_vs_sentiment.png")

# # Ensure 'date' is converted to datetime
# merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")

# # Check if 'date' conversion is successful
# print("Data types after conversion:")
# print(merged_df.dtypes)

# # Print the first few rows of 'date' to see the correct conversion
# print("\nFirst few rows of the data with date:")
# print(merged_df[["date"]].head())

# # --- Convert 'date' to month period and print it to inspect ---
# merged_df["month"] = merged_df["date"].dt.to_period("M")  # Convert to monthly period

# # Check unique values of 'month' column to see how the month is being extracted
# print("\nUnique months extracted:")
# print(merged_df["month"].unique())

# # --- Visualization 3: Sentiment Over Time (Monthly) ---
# sentiment_time_series = (
#     merged_df.groupby(["month", "team"])["compound"].mean().reset_index()
# )

# # Check the first few rows of the sentiment time series to confirm the month format
# print("\nFirst few rows of sentiment time series:")
# print(sentiment_time_series.head())

# # Filter for a few teams to make the plot less cluttered
# teams_of_interest = ["BOS", "DEN", "MIA"]  # Example of fewer teams for clarity
# filtered_sentiment_time_series = sentiment_time_series[
#     sentiment_time_series["team"].isin(teams_of_interest)
# ]

# # Plot sentiment over time for the selected teams
# plt.figure(figsize=(12, 6))  # Increase figure size for better readability
# sns.lineplot(
#     x="month", y="compound", hue="team", data=filtered_sentiment_time_series, marker="o"
# )

# # Title and axis labels
# plt.title("Sentiment Over Time (Monthly Aggregation) for Selected Teams", fontsize=14)
# plt.xlabel("Month", fontsize=12)
# plt.ylabel("Average Compound Sentiment Score", fontsize=12)
# plt.xticks(rotation=45)  # Rotate month labels for readability

# # Adjust layout
# plt.tight_layout()
# plt.show()
