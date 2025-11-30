# DS3000 Project - Group 18
# Predicting Chess Outcomes from Elo and Opening Selection


import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve




# =====================================================
# 1. LOAD DATA
# =====================================================


csv_path = r"C:\Users\konra\DS3000_Project_Group18\chess_games (1).csv"


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
df['avg_elo']  = (df['white_elo'] + df['black_elo']) / 2


# Opening / time control features
df['eco'] = df['eco'].fillna('UNKNOWN').astype(str)
df['eco_code3'] = df['eco'].str[:3]  # e.g., 'C20', 'B01'
df['time_control'] = df['time_control'].fillna('UNKNOWN').astype(str)


print("\nAfter cleaning:", df.shape)
print(df[['white_elo','black_elo','result','white_win','elo_diff','avg_elo','eco_code3','time_control']].head())




# =====================================================
# 3. SIMPLE EDA (PRINTS ONLY)
# =====================================================


print("\n=== Basic EDA ===")
print("Result distribution (1=white win, 0=black win):")
print(df['white_win'].value_counts(normalize=True))


print("\nTop 10 ECO codes:")
print(df['eco_code3'].value_counts().head(10))


print("\nTime control distribution (top 10):")
print(df['time_control'].value_counts().head(10))




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


num_features = ['elo_diff', 'avg_elo']
cat_features = ['eco_code3', 'time_control']


df_model = df_small[num_features + cat_features + ['white_win']].dropna()


X = df_model[num_features + cat_features]
y = df_model['white_win'].values


# One-hot encode categorical features (SPARSE to save memory)
ohe = OneHotEncoder(handle_unknown='ignore')  # sparse_output=True by default


preprocess = ColumnTransformer(
    transformers=[
        ('cat', ohe, cat_features),
        ('num', 'passthrough', num_features)
    ]
)


X_encoded = preprocess.fit_transform(X)


print("\nEncoded feature matrix shape:", X_encoded.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Train shape:", X_train.shape, "Test shape:", X_test.shape)




# =====================================================
# 6. BASELINE MODEL: LOGISTIC REGRESSION
# =====================================================


log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)


print("\nTraining Logistic Regression...")
log_reg.fit(X_train, y_train)


y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)[:, 1]


acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_prob_lr)


print("\n=== Logistic Regression Performance ===")
print(f"Accuracy: {acc_lr:.4f}")
print(f"ROC AUC:  {auc_lr:.4f}")
print(classification_report(y_test, y_pred_lr, digits=4))




# =====================================================
# 7. STRONGER MODEL: RANDOM FOREST
# =====================================================


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)


print("\nTraining Random Forest...")
rf.fit(X_train, y_train)


y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]


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
    if r < 800:
        return 'Under 800'
    elif r < 1400:
        return 'Under 1400'
    elif r < 1800:
        return '1400–1799'
    elif r < 2200:
        return '1800–2199'
    else:
        return '2200+'


df['elo_bracket'] = df['avg_elo'].apply(elo_bracket)


MIN_GAMES = 200  # avoid tiny sample artifacts


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
        .head(10)
    )
    print(sub)




# =====================================================
# 9. ROC CURVES FOR BOTH MODELS
# =====================================================


# Compute ROC curve points
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



