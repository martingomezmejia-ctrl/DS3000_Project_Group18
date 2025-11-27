"""
Chess Win Probability Estimation Script
--------------------------------------
This script loads a chess game dataset, extracts key variables such as ELO ratings
and piece color, and applies basic statistical and machine learning models to
estimate the probability that White or Black will win each game.

NOTE: Replace 'games.csv' with the path to your dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ===============================
# 1. Load the dataset
# ===============================

df = pd.read_csv(r"C:\Users\hongy\Hongyue Tian\Western\Year 3\DS3000 - Machine Learning\Project\chess_games.csv")
  # <-- update this to your dataset file

# Example expected columns: 'white_elo', 'black_elo', 'winner', 'white_rating_diff', ...
# If your dataset differs, modify the code below accordingly.

# ===============================
# 2. Extract key variables
# ===============================

df = df.dropna(subset=["Event", "White", "Black", "Result", "UTCDate", "UTCTime", "WhiteElo", "BlackElo", "WhiteRatingDiff", "BlackRatingDiff", "ECO", "Opening", "TimeControl", "Termination", "AN"])

# Encode winner: White = 1, Black = 0
label_encoder = LabelEncoder()
df["winner_encoded"] = label_encoder.fit_transform(df["Result"])

# Create features
# White advantage can be represented by ELO difference

df["elo_diff"] = df["WhiteElo"] - df["BlackElo"]

X = df[["WhiteElo", "BlackElo", "elo_diff"]]
y = df["winner_encoded"]

# ===============================
# 3. Split into train/test sets
# ===============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===============================
# 4. Logistic Regression Model
# ===============================

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ===============================
# 5. Evaluate performance
# ===============================

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ===============================
# 6. Predict probability of White winning
# ===============================

sample = {
    "WhiteElo": 1600,
    "BlackElo": 1500,
    "elo_diff": 100
}

sample_df = pd.DataFrame([sample])
prob_white_wins = model.predict_proba(sample_df)[0][1]

print(f"\nPredicted Probability White Wins: {prob_white_wins:.4f}")
