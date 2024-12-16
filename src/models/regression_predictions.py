import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
merged_df = pd.read_csv("../../data/processed/final_team_data_with_sentiment.csv")

# --- Data Preprocessing ---
# Convert 'date' to datetime (if necessary)
merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")

# Handle missing values in 'compound' (sentiment)
merged_df["compound"] = pd.to_numeric(merged_df["compound"], errors="coerce")

# Drop rows with NaN values in 'compound' (sentiment)
merged_df_clean = merged_df.dropna(subset=["compound"])

# --- Calculate Point Spread ---
# Filter home and away teams separately
home_games = merged_df_clean[merged_df_clean["home"] == merged_df_clean["team"]]
away_games = merged_df_clean[merged_df_clean["away"] == merged_df_clean["team"]]

# Rename the columns for clarity
home_games = home_games.rename(columns={"PTS": "home_pts"})
away_games = away_games.rename(columns={"PTS": "away_pts", "team": "away_team"})

# Merge the home and away data on the gameid to get the points for both teams
merged_point_spread = pd.merge(
    home_games[["gameid", "home_pts", "team", "date"]],
    away_games[["gameid", "away_pts", "away_team"]],
    on="gameid",
    how="inner",
)

# Calculate point spread (home_pts - away_pts)
merged_point_spread["point_spread"] = (
    merged_point_spread["home_pts"] - merged_point_spread["away_pts"]
)

# --- Feature Selection ---
# Select performance-related features (excluding 'gameid', 'date', 'team', 'home', 'away', etc.)
features = [
    "PTS",
    "FG%",
    "3P%",
    "FT%",
    "AST",
    "REB",
    "TOV",
    "STL",
    "BLK",
    "PF",
    "+/-",
    "win",
    "normalized_pts",
    "rolling_avg_sentiment",  # Including sentiment as a feature
]

# Merge point spread with the cleaned dataset for feature engineering
merged_df_clean = pd.merge(
    merged_df_clean, merged_point_spread[["gameid", "point_spread"]], on="gameid"
)

# Create the feature matrix X and target vector y (predicting point spread)
X = merged_df_clean[features]
y = merged_df_clean["point_spread"]

# --- Check for Missing Values ---
missing_values = X.isnull().sum()
print("\nMissing values in feature columns:")
print(missing_values)

# --- Handle Missing Values ---
# If there are missing values in any feature, we'll impute them using the median strategy
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# --- Train-Test Split ---
# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Print shape of training and test sets
print(f"\nShape of training features: {X_train.shape}")
print(f"Shape of test features: {X_test.shape}")
print(f"Shape of training target: {y_train.shape}")
print(f"Shape of test target: {y_test.shape}")

# --- Feature Scaling ---
# Scale the features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check first few rows of scaled features
print("\nFirst few rows of scaled training features:")
print(X_train_scaled[:5, :])

# --- Train Linear Regression Model ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Make Predictions ---
y_pred = model.predict(X_test_scaled)

# --- Model Evaluation ---
# Calculate MSE (Mean Squared Error)
mse = mean_squared_error(y_test, y_pred)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# Calculate R² (Coefficient of Determination)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# --- Visualize Actual vs Predicted Point Spread ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue", alpha=0.6)
plt.plot(
    [min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--"
)  # Ideal line
plt.title("Actual vs Predicted Point Spread")
plt.xlabel("Actual Point Spread")
plt.ylabel("Predicted Point Spread")
plt.tight_layout()
plt.show()

# --- Visualize Feature Importance (Coefficients) ---
# This will help to see which performance metrics have the most impact on the point spread
coefficients = pd.DataFrame(model.coef_, features, columns=["Coefficient"])
coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=coefficients.index, y=coefficients["Coefficient"], palette="coolwarm")
plt.title("Feature Importance: Performance Metrics Impact on Point Spread")
plt.xlabel("Performance Metric")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
