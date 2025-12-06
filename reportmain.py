# DS3000 Project - Group 18
# Predicting Chess Outcomes from Elo and Opening Selection

import pandas as pd
import numpy as np
import time  # for timing model training

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
N = 500_000
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
# 5. PREPROCESSORS FOR DIFFERENT MODELS
# =====================================================

# Logistic Regression:
#   - One-hot encode cats
#   - Standardize numerics
preprocess_lr = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features),
    ]
)

# Random Forest:
#   - One-hot encode cats
#   - Pass numerics through (no scaling needed)
preprocess_rf = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features),
    ]
)


# =====================================================
# 6. MODEL COMPARISON: MULTIPLE LOGISTIC REGRESSIONS + RANDOM FORESTS
# =====================================================

results = []          # to store model-level metrics
roc_curves = {}       # model_name -> (fpr, tpr, auc)
trained_models = {}   # keep fitted pipelines if you want to inspect later

# ---- Logistic Regression configurations (3 models) ----
log_reg_configs = [
    {
        "name": "LR_C0.1_L2",
        "params": {"C": 0.1, "penalty": "l2", "class_weight": None}
    },
    {
        "name": "LR_C1.0_L2",
        "params": {"C": 1.0, "penalty": "l2", "class_weight": None}
    },
    {
        "name": "LR_C10_L2",
        "params": {"C": 10.0, "penalty": "l2", "class_weight": None}
    },
]

# ---- Random Forest configurations  ----
rf_configs = [
    {
        "name": "RF_50depth_5",
        "params": {
            "n_estimators": 50,
            "max_depth": 5,
            "min_samples_leaf": 10
        }
    },
    {
        "name": "RF_100depth_10",
        "params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 10
        }
    },
    {
        "name": "RF_150depth_15",
        "params": {
            "n_estimators": 150,
            "max_depth": 15,
            "min_samples_leaf": 20
        }
    },
]

print("\n==============================")
print("Training Logistic Regression models...")
print("==============================")

for cfg in log_reg_configs:
    name = cfg["name"]
    p = cfg["params"]

    lr_model = LogisticRegression(
        C=p["C"],
        penalty=p["penalty"],
        class_weight=p["class_weight"],
        max_iter=2000,
        n_jobs=-1,
        solver="lbfgs",
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess_lr),
        ("model", lr_model)
    ])

    print(f"\n>>> Training {name} "
          f"(C={p['C']}, penalty='{p['penalty']}', class_weight={p['class_weight']})")
    start = time.time()
    pipe.fit(X_train, y_train)
    end = time.time()

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr, auc)

    # Store metrics
    results.append({
        "model": name,
        "family": "LogisticRegression",
        "C": p["C"],
        "penalty": p["penalty"],
        "class_weight": str(p["class_weight"]),
        "n_estimators": np.nan,
        "max_depth": np.nan,
        "min_samples_leaf": np.nan,
        "accuracy": acc,
        "roc_auc": auc,
        "train_time_sec": end - start,
    })

    trained_models[name] = pipe


print("\n==============================")
print("Training Random Forest models...")
print("==============================")

for cfg in rf_configs:
    name = cfg["name"]
    p = cfg["params"]

    rf_model = RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        min_samples_leaf=p["min_samples_leaf"],
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocess_rf),
        ("model", rf_model)
    ])

    print(f"\n>>> Training {name} "
          f"(n_estimators={p['n_estimators']}, max_depth={p['max_depth']}, "
          f"min_samples_leaf={p['min_samples_leaf']})")
    start = time.time()
    pipe.fit(X_train, y_train)
    end = time.time()

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr, auc)

    # Store metrics
    results.append({
        "model": name,
        "family": "RandomForest",
        "C": np.nan,
        "penalty": np.nan,
        "class_weight": np.nan,
        "n_estimators": float(p["n_estimators"]),
        "max_depth": float(p["max_depth"]),
        "min_samples_leaf": float(p["min_samples_leaf"]),
        "accuracy": acc,
        "roc_auc": auc,
        "train_time_sec": end - start,
    })

    trained_models[name] = pipe


# =====================================================
# 7. RESULTS TABLE + PLOTS
# =====================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("roc_auc", ascending=False).reset_index(drop=True)

print("\n=== Model comparison table (sorted by ROC AUC) ===")
print(results_df.to_string(index=False))

# ---------- Zoomed-in bar charts (monochrome, Word-friendly) ----------

# Accuracy bar chart (zoomed)
plt.figure(figsize=(6, 3.5), dpi=150)
plt.bar(results_df["model"], results_df["accuracy"],
        edgecolor='black', linewidth=0.8)
plt.title("Model Accuracy Comparison", fontsize=10)
plt.xlabel("Model", fontsize=9)
plt.ylabel("Accuracy", fontsize=9)
plt.xticks(rotation=45, ha="right", fontsize=8)
ymin_acc = results_df["accuracy"].min() - 0.002
ymax_acc = results_df["accuracy"].max() + 0.002
plt.ylim(ymin_acc, ymax_acc)
plt.grid(axis="y", alpha=0.4, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ROC AUC bar chart (zoomed)
plt.figure(figsize=(6, 3.5), dpi=150)
plt.bar(results_df["model"], results_df["roc_auc"],
        edgecolor='black', linewidth=0.8)
plt.title("Model ROC AUC Comparison", fontsize=10)
plt.xlabel("Model", fontsize=9)
plt.ylabel("ROC AUC", fontsize=9)
plt.xticks(rotation=45, ha="right", fontsize=8)
ymin_auc = results_df["roc_auc"].min() - 0.002
ymax_auc = results_df["roc_auc"].max() + 0.002
plt.ylim(ymin_auc, ymax_auc)
plt.grid(axis="y", alpha=0.4, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# ---------- Horizontal dot plot for ROC AUC ----------

plt.figure(figsize=(6, 3.5), dpi=150)
plt.scatter(results_df["roc_auc"], results_df["model"],
            s=25, marker='o')
plt.title("ROC AUC by Model", fontsize=10)
plt.xlabel("ROC AUC", fontsize=9)
plt.ylabel("Model", fontsize=9)
plt.grid(axis="x", alpha=0.4, linestyle='--', linewidth=0.5)

plt.xlim(ymin_auc, ymax_auc)
plt.tight_layout()
plt.show()

# --- ROC curves for all models ---
plt.figure(figsize=(6, 4), dpi=150)
for name, (fpr, tpr, auc) in roc_curves.items():
    plt.plot(fpr, tpr, linewidth=1.0, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random baseline")
plt.title("ROC Curves", fontsize=10)
plt.xlabel("False Positive Rate", fontsize=9)
plt.ylabel("True Positive Rate", fontsize=9)
plt.legend(loc="lower right", fontsize=7)
plt.grid(alpha=0.4, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Print classification report for the single best model by AUC
best_model_name = results_df.iloc[0]["model"]
print(f"\n=== Best model by ROC AUC: {best_model_name} ===")
best_model = trained_models[best_model_name]
best_pred = best_model.predict(X_test)
print(classification_report(y_test, best_pred, digits=4))


# =====================================================
# 8. OPENING PERFORMANCE BY ELO BRACKET (SAME AS BEFORE)
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
