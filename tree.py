import pandas as pd
import numpy as np
from xgboost import XGBRegressor

# ================================
# LOAD DATA
# ================================
df = pd.read_csv(r"C:\Users\Tanul\OneDrive\Desktop\mlpr project\final_dataset.csv")
df.columns = df.columns.str.strip()

targets = ['stress', 'pam', 'phq4_score']

# Remove leakage columns for phq4
leakage_cols = ['phq4-1','phq4-2','phq4-3','phq4-4',
                'phq4_resp_mean','phq4_resp_median']

# Store results
feature_scores = {}

# ================================
# RUN MODEL FOR EACH TARGET
# ================================
for target in targets:
    print(f"\nTraining for: {target}")

    drop_cols = ['uid','day', target]

    # Remove leakage if target is phq4
    if target == 'phq4_score':
        drop_cols += leakage_cols

    X = df.drop(columns=drop_cols, errors='ignore')
    X = X.fillna(X.median(numeric_only=True))
    y = df[target]

    model = XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X, y)

    importance = model.feature_importances_

    for feat, imp in zip(X.columns, importance):
        if feat not in feature_scores:
            feature_scores[feat] = {'score':0, 'count':0}

        feature_scores[feat]['score'] += imp
        feature_scores[feat]['count'] += 1 if imp > 0 else 0


# ================================
# CREATE FINAL RANKING
# ================================
final_df = pd.DataFrame([
    (feat, vals['score'], vals['count'])
    for feat, vals in feature_scores.items()
], columns=['Feature','Total_Importance','Frequency'])

# Normalize importance
final_df['Normalized_Importance'] = final_df['Total_Importance'] / len(targets)

# Sort by BOTH importance + frequency
final_df = final_df.sort_values(
    by=['Frequency','Normalized_Importance'],
    ascending=False
)

# ================================
# TOP 20 FINAL FEATURES
# ================================
top_20 = final_df.head(20)

print("\n=== FINAL TOP 20 FEATURES ===")
for i, row in enumerate(top_20.itertuples(), 1):
    print(f"{i}. {row.Feature} (Freq: {row.Frequency}, Score: {row.Normalized_Importance:.4f})")

# Save
top_20.to_csv("top_20_multi_target_xgboost.csv", index=False)