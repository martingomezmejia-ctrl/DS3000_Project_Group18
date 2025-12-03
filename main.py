# DS3000 Project - Group 18
# Predicting Chess Outcomes from Elo and Opening Selection

import pandas as pd
import numpy as np
import time  # for timing model training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


# =====================================================
# 1. LOAD DATA
# =====================================================

# Path to the Lichess / chess game dataset (local CSV file)
csv_path = r"C:\Users\marti\OneDrive - The University of Western Ontario\DS3000_Project_Group18\chess_games.csv"

print("Loading CSV from:", csv_path)
df = pd.read_csv(csv_path)

print("\nRaw shape:", df.shape)         # (rows, columns) before cleaning
print("Columns:", df.columns.tolist())  # original column names


# =====================================================
# 2. CLEANING & FEATURE ENGINEERING
# =====================================================

# Standardize column names to something shorter / easier to work with
df = df.rename(columns={
    'WhiteElo': 'white_elo',
    'BlackElo': 'black_elo',
    'Result': 'result',
    'ECO': 'eco',
    'Opening': 'opening_name',
    'TimeControl': 'time_control'
})

# Keep only the columns we actually use in this experiment
df = df[['white_elo', 'black_elo', 'result', 'eco', 'opening_name', 'time_control']]

# Drop rows where Elo or result is missing
df = df.dropna(subset=['white_elo', 'black_elo', 'result'])

# Convert Elo ratings to numeric (coerce errors to NaN) and drop NaNs
df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')
df = df.dropna(subset=['white_elo', 'black_elo'])

# Keep only decisive games: white win (1-0) or black win (0-1)
df = df[df['result'].isin(['1-0', '0-1'])].copy()

# Target variable: 1 if white wins, 0 if white loses
df['white_win'] = (df['result'] == '1-0').astype(int)

# ---- Elo-based features ----
# Elo difference (positive means white is higher rated)
df['elo_diff'] = df['white_elo'] - df['black_elo']
# Average Elo of the two players (proxy for overall game strength)
df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2
# Indicator: 1 if white is higher rated, 0 otherwise
df['white_higher'] = (df['elo_diff'] > 0).astype(int)
# Absolute Elo difference (magnitude of rating mismatch)
df['elo_diff_abs'] = df['elo_diff'].abs()
# Squared Elo difference to allow for non-linear effect in Logistic Regression
df['elo_diff_sq'] = df['elo_diff'] ** 2

# ---- Opening + time control features ----
df['eco'] = df['eco'].fillna('UNKNOWN').astype(str)
# ECO code truncated to first 3 characters (e.g., "C65", "B20")
df['eco_code3'] = df['eco'].str[:3]

# Time control as string, with unknowns filled
df['time_control'] = df['time_control'].fillna('UNKNOWN').astype(str)


def parse_time_control(tc: str) -> str:
    """
    Convert raw time control string (e.g. '600+0') into coarse bucket:
    'bullet', 'blitz', 'rapid', or 'classical'.
    If format is not recognized, return 'unknown'.
    """
    if tc in ['UNKNOWN', '?', None] or tc == '':
        return 'unknown'
    if '+' not in tc:
        return 'unknown'
    try:
        base, inc = tc.split('+')
        base = int(base)
    except ValueError:
        return 'unknown'

    # Thresholds in seconds for each time category
    if base <= 180:
        return 'bullet'
    elif base <= 600:
        return 'blitz'
    elif base <= 1800:
        return 'rapid'
    else:
        return 'classical'


# Add a categorical feature for time control bucket
df['tc_bucket'] = df['time_control'].apply(parse_time_control)

# Collapse rare ECO codes into 'OTHER' to avoid very sparse one-hot columns
eco_counts = df['eco_code3'].value_counts()
rare_ecos = eco_counts[eco_counts < 100].index
df['eco_code3_clean'] = df['eco_code3'].where(~df['eco_code3'].isin(rare_ecos), 'OTHER')

print("\nAfter cleaning:", df.shape)
print(df.head())


# =====================================================
# 3. DOWNSAMPLE FOR ML
# =====================================================

# Limit dataset size for computational reasons (original file is huge)
N = 5_000_000
if len(df) > N:
    # Random sample of N rows, with fixed seed for reproducibility
    df_small = df.sample(n=N, random_state=42).copy()
    print(f"\nDownsampled to {N}")
else:
    df_small = df.copy()

print("ML dataframe shape:", df_small.shape)


# =====================================================
# 4. BUILD ML DATASET
# =====================================================

# Numerical features used in the model
num_features = ['elo_diff', 'avg_elo', 'elo_diff_abs', 'elo_diff_sq']

# Categorical features used in the model
# Note: 'white_higher' is a binary feature but treated as a categorical here
cat_features = ['eco_code3_clean', 'tc_bucket', 'white_higher']

# Keep only selected features + target column
df_model = df_small[num_features + cat_features + ['white_win']].dropna()

# X: feature matrix, y: target (1 = white wins, 0 = black wins)
X = df_model[num_features + cat_features]
y = df_model['white_win'].values

print("\nClass balance:", np.bincount(y))  # show how many white wins vs losses

# Train/test split (80/20), stratified to preserve win/loss ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =====================================================
# 5. LOGISTIC REGRESSION MODEL
# =====================================================

# Preprocessing pipeline:
# - One-hot encode categorical features (ECO, time control, white_higher)
# - Standardize numerical features (mean 0, std 1) for better optimization
preprocess_lr = ColumnTransformer(
    transformers=[
        # OneHotEncoder(handle_unknown='ignore'):
        #   - turns categories into binary columns
        #   - ignores unseen categories at test time instead of crashing
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),

        # StandardScaler:
        #   - centers each numeric feature (mean=0)
        #   - scales to unit variance (std=1)
        #   - helps Logistic Regression converge and makes coefficients comparable
        ('num', StandardScaler(), num_features),
    ]
)

# Core Logistic Regression classifier wrapped inside a full pipeline
log_reg_clf = Pipeline(steps=[
    ('preprocess', preprocess_lr),

    # LogisticRegression parameters explanation:
    # - max_iter=2000:
    #       Increase maximum optimization iterations so that the solver
    #       has enough steps to converge on a large, high-dimensional dataset.
    # - n_jobs=-1:
    #       Use all available CPU cores to parallelize computation
    #       (e.g., during gradient computations and matrix operations),
    #       which speeds up training on a big dataset.
    #
    # NOTE: other key parameters are left as scikit-learn defaults:
    # - penalty='l2': standard L2 regularization to prevent overfitting.
    # - C=1.0: regularization strength (inverse of lambda). We keep default
    #          to balance bias and variance without heavy tuning.
    # - solver='lbfgs': robust quasi-Newton optimizer, works well with
    #                   L2 penalty and supports multinomial logistic regression
    #                   (here we only have a binary problem).
    ('model', LogisticRegression(max_iter=2000, n_jobs=-1))
])

print("\nTraining Logistic Regression...")
start_lr = time.time()
log_reg_clf.fit(X_train, y_train)  # fit preprocessing + LR model end-to-end
end_lr = time.time()

training_time = end_lr - start_lr

# Predictions: class labels and predicted probabilities for "white_win = 1"
y_pred_lr = log_reg_clf.predict(X_test)
y_prob_lr = log_reg_clf.predict_proba(X_test)[:, 1]

# Evaluate performance: accuracy and ROC-AUC
acc = accuracy_score(y_test, y_pred_lr)
auc = roc_auc_score(y_test, y_prob_lr)

print("\n=== Logistic Regression Results ===")
print(f"Training time: {training_time:.2f} sec")
print(f"Accuracy:      {acc:.4f}")
print(f"ROC AUC:       {auc:.4f}")
print(classification_report(y_test, y_pred_lr, digits=4))


# =====================================================
# 6. OPENING PERFORMANCE BY ELO BRACKET
# =====================================================

def elo_bracket(r: float) -> str:
    """
    Map average Elo rating to a coarse rating bracket.
    These buckets are used to compare opening performance at different levels.
    """
    if r <= 999:
        return 'Under 999'
    elif r <= 1399:
        return '1000–1399'
    elif r <= 1799:
        return '1400–1799'
    elif r <= 2199:
        return '1800–2199'
    else:
        return '2200+'


# Apply Elo bracket mapping based on avg_elo
df['elo_bracket'] = df['avg_elo'].apply(elo_bracket)

# Aggregate opening stats: number of games + white win rate per (bracket, ECO)
MIN_GAMES = 100  # minimum number of games to consider opening stats reliable
stats = (
    df.groupby(['elo_bracket', 'eco_code3'])
    .agg(
        games=('white_win', 'size'),
        white_win_rate=('white_win', 'mean')
    )
    .reset_index()
)

# Filter out (bracket, ECO) pairs with too few games
stats = stats[stats['games'] >= MIN_GAMES]

print("\n=== Top Openings Per Elo Bracket (by White Win Rate) ===")
for bracket in stats['elo_bracket'].unique():
    print("\nELO BRACKET:", bracket)
    print(
        stats[stats['elo_bracket'] == bracket]
        .sort_values('white_win_rate', ascending=False)
        .head(5)
    )

print("\n=== Most Played Openings Per Elo Bracket (by Games) ===")
for bracket in stats['elo_bracket'].unique():
    print("\nELO BRACKET:", bracket)
    print(
        stats[stats['elo_bracket'] == bracket]
        .sort_values('games', ascending=False)
        .head(5)[['elo_bracket', 'eco_code3', 'games', 'white_win_rate']]
    )


# =====================================================
# 7. ROC CURVE
# =====================================================

# Compute ROC curve points from predicted probabilities
fpr, tpr, _ = roc_curve(y_test, y_prob_lr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc:.3f})")

# Diagonal line represents a random classifier baseline
plt.plot([0, 1], [0, 1], 'k--')

plt.title("ROC Curve — Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
