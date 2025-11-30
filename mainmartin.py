# DS3000 Project - Group 18
# Predicting Chess Outcomes from Elo and Opening Selection

import pandas as pd
import numpy as np

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

csv_path = r"C:\Users\marti\OneDrive - The University of Western Ontario\DS3000_Project_Group18\chess_games.csv"

print("Loading CSV from:", csv_path)
df = pd.read_csv(csv_path)

print("\nRaw shape:", df.shape)
print("Columns:", df.columns.tolist())
# Expected columns:
# ['Event','White','Black','Result','UTCDate','UTCTime',
#  'WhiteElo','BlackElo','WhiteRatingDiff','BlackRatingDiff',
#  'ECO','Opening','TimeControl','Termination','AN']


# =====================================================
# 2. CLEANING & FEATURE ENGINEERING
# =====================================================

# Standardize names
df = df.rename(columns={
    'WhiteElo': 'white_elo',
    'BlackElo': 'black_elo',
    'Result': 'result',
    'ECO': 'eco',
    'Opening': 'opening_name',
    'TimeControl': 'time_control'
})

# Keep only relevant columns
df = df[['white_elo', 'black_elo', 'result', 'eco', 'opening_name', 'time_control']]

# Drop rows with missing crucial info
df = df.dropna(subset=['white_elo', 'black_elo', 'result'])

# Ratings to numeric
df['white_elo'] = pd.to_numeric(df['white_elo'], errors='coerce')
df['black_elo'] = pd.to_numeric(df['black_elo'], errors='coerce')
df = df.dropna(subset=['white_elo', 'black_elo'])

# Keep only decisive games: 1-0 and 0-1
df = df[df['result'].isin(['1-0', '0-1'])].copy()

# Target: 1 if White wins, 0 if Black wins
df['white_win'] = (df['result'] == '1-0').astype(int)

# Elo-based features
df['elo_diff'] = df['white_elo'] - df['black_elo']
df['avg_elo'] = (df['white_elo'] + df['black_elo']) / 2

# Extra Elo features for more signal
df['white_higher'] = (df['elo_diff'] > 0).astype(int)
df['elo_diff_abs'] = df['elo_diff'].abs()
df['elo_diff_sq'] = df['elo_diff'] ** 2

# Opening / time control features
df['eco'] = df['eco'].fillna('UNKNOWN').astype(str)
df['eco_code3'] = df['eco'].str[:3]  # e.g., 'C20', 'B01'
df['time_control'] = df['time_control'].fillna('UNKNOWN').astype(str)


def parse_time_control(tc: str) -> str:
    """
    Parse Lichess-style time control strings like '600+5' into buckets.
    Returns one of: 'bullet', 'blitz', 'rapid', 'classical', 'unknown'.
    """
    if tc in ['UNKNOWN', '?', None] or tc == '':
        return 'unknown'
    if '+' not in tc:
        return 'unknown'
    try:
        base, inc = tc.split('+')
        base = int(base)
        inc = int(inc)
    except ValueError:
        return 'unknown'

    # Very rough but standard-ish buckets (base time in seconds)
    if base <= 180:
        return 'bullet'
    elif base <= 600:
        return 'blitz'
    elif base <= 1800:
        return 'rapid'
    else:
        return 'classical'


df['tc_bucket'] = df['time_control'].apply(parse_time_control)

# Collapse very rare ECO codes into "OTHER" for the ML model
eco_counts = df['eco_code3'].value_counts()
rare_ecos = eco_counts[eco_counts < 200].index  # threshold can be tuned

df['eco_code3_clean'] = df['eco_code3'].where(~df['eco_code3'].isin(rare_ecos), 'OTHER')

print("\nAfter cleaning:", df.shape)
print(df[['white_elo', 'black_elo', 'result', 'white_win',
          'elo_diff', 'avg_elo', 'eco_code3', 'time_control', 'tc_bucket']].head())


# =====================================================
# 3. SIMPLE EDA (PRINTS ONLY)
# =====================================================

print("\n=== Basic EDA ===")
print("Result distribution (1=white win, 0=black win):")
print(df['white_win'].value_counts(normalize=True))

print("\nTop 10 ECO codes:")
print(df['eco_code3'].value_counts().head(10))

print("\nTime control distribution (top 10 raw strings):")
print(df['time_control'].value_counts().head(10))

print("\nTime control buckets:")
print(df['tc_bucket'].value_counts())


# =====================================================
# 4. DOWNSAMPLE FOR ML (AVOID 60GB ARRAY)
# =====================================================

N = 200_000  # you can try 300_000 if your PC is decent
if len(df) > N:
    df_small = df.sample(n=N, random_state=42).copy()
    print(f"\nDownsampled to {N} rows for ML.")
else:
    df_small = df.copy()
    print("\nNo downsampling needed; dataset is small enough.")

print("ML dataframe shape:", df_small.shape)


# =====================================================
# 5. BUILD ML DATASET (X, y)
# =====================================================

# Numeric and categorical feature sets for the models
num_features = ['elo_diff', 'avg_elo', 'elo_diff_abs', 'elo_diff_sq']
cat_features = ['eco_code3_clean', 'tc_bucket', 'white_higher']

df_model = df_small[num_features + cat_features + ['white_win']].dropna()

X = df_model[num_features + cat_features]
y = df_model['white_win'].values

print("\nClass balance (0=Black win, 1=White win):")
print(np.bincount(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape (raw):", X_train.shape, "Test shape (raw):", X_test.shape)

# Preprocessing for Logistic Regression (scales numeric features)
preprocess_lr = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features),
    ]
)

# Preprocessing for Random Forest (no need to scale numeric features)
preprocess_rf = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features),
    ]
)


# =====================================================
# 6. BASELINE MODEL: LOGISTIC REGRESSION (PIPELINE)
# =====================================================

log_reg_clf = Pipeline(steps=[
    ('preprocess', preprocess_lr),
    ('model', LogisticRegression(max_iter=2000, n_jobs=-1))
])

print("\nTraining Logistic Regression (pipeline)...")
log_reg_clf.fit(X_train, y_train)

y_pred_lr = log_reg_clf.predict(X_test)
y_prob_lr = log_reg_clf.predict_proba(X_test)[:, 1]

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)

print("\n=== Logistic Regression Performance ===")
print(f"Accuracy: {acc_lr:.4f}")
print(f"ROC AUC:  {auc_lr:.4f}")
print(classification_report(y_test, y_pred_lr, digits=4))


# =====================================================
# 7. STRONGER MODEL: RANDOM FOREST (PIPELINE)
# =====================================================

rf_clf = Pipeline(steps=[
    ('preprocess', preprocess_rf),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    ))
])

print("\nTraining Random Forest (pipeline)...")
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
y_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

print("\n=== Random Forest Performance ===")
print(f"Accuracy: {acc_rf:.4f}")
print(f"ROC AUC:  {auc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, digits=4))


# =====================================================
# 8. OPENING PERFORMANCE BY ELO BRACKET (FULL DATA)
# =====================================================

def elo_bracket(r):
    if r <= 599:
        return 'Under 600'
    elif r <= 999:
        return '600–999'
    elif r <= 1399:
        return '1000–1399'
    elif r <= 1799:
        return '1400–1799'
    elif r <= 2199:
        return '1800–2199'
    else:
        return '2200+'


df['elo_bracket'] = df['avg_elo'].apply(elo_bracket)

MIN_GAMES = 150  # avoid tiny sample artifacts

group_cols = ['elo_bracket', 'eco_code3']
stats = (
    df
    .groupby(group_cols)
    .agg(
        games=('white_win', 'size'),
        white_win_rate=('white_win', 'mean')
    )
    .reset_index()
)

stats_filtered = stats[stats['games'] >= MIN_GAMES]

print("\n=== Top openings by Elo bracket (White win rate) ===")
for bracket in stats_filtered['elo_bracket'].unique():
    print("\nELO BRACKET:", bracket)
    sub = (
        stats_filtered[stats_filtered['elo_bracket'] == bracket]
        .sort_values('white_win_rate', ascending=False)
        .head(5)
    )
    print(sub)


# =====================================================
# 9. ROC CURVES FOR BOTH MODELS
# =====================================================

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")

# Random guessing line
plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing")

plt.title("ROC Curve Comparison: Logistic Regression vs Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
