import pandas as pd

# Load the datasets
df1 = pd.read_csv(
    "../../data/raw/nba_boxscores_2023_2024.csv"
)  # Replace with your actual file path for dataset 1
df2 = pd.read_csv(
    "../../data/raw/nba_boxscores_player_2023_2024.csv"
)  # Replace with your actual file path for dataset 2

# Filter for the 2023-2024 season (assuming 'season' column exists and has a value for 2023-2024)
df1_filtered = df1[(df1["season"] == 2024)]  # Filter for 2023-2024 season in dataset 1
df2_filtered = df2[(df2["season"] == 2024)]  # Filter for 2023-2024 season in dataset 2

# Filter out playoff games (assuming 'type' column indicates game type)
df1_filtered = df1_filtered[df1_filtered["type"] != "playoff"]
df2_filtered = df2_filtered[df2_filtered["type"] != "playoff"]

# You can now check the first few rows of the filtered datasets to confirm
print("Filtered Dataset 1:")
print(df1_filtered.head())

print("\nFiltered Dataset 2:")
print(df2_filtered.head())

# Optional: Save the filtered data to new CSV files
df1_filtered.to_csv("../../data/processed/nba_boxscores_2023_2024.csv", index=False)
df2_filtered.to_csv(
    "../../data/processed/nba_boxscores_player_2023_2024.csv", index=False
)
