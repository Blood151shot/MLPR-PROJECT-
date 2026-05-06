from split_data import split_data
from model import ShallowNN
from train import train_model
from predict import predict
from threshold_tuning import tune_thresholds
from evaluation_metrics import evaluate_metrics

import pandas as pd

TARGETS = ['stress', 'phq4_score', 'pam']

def run_for_target(target):
    print(f"\n========== TARGET: {target.upper()} ==========")

    # Set ground truth thresholds based on actual distributions
    if target == 'stress':
        true_t1, true_t2 = 0.25, 0.26
    elif target == 'phq4_score':
        true_t1, true_t2 = 0.16, 0.17
    elif target == 'pam':
        true_t1, true_t2 = 0.39, 0.41
    else:
        true_t1, true_t2 = 0.3, 0.6

    df = pd.read_csv(r"C:\Users\Tanul\OneDrive\Desktop\mlpr project\final_data_set_1000.csv")

    # Remove leakage for PHQ4
    drop_cols = ['uid', 'day', target]

    if target == 'phq4_score':
        drop_cols += [
            'phq4-1','phq4-2','phq4-3','phq4-4',
            'phq4_resp_mean','phq4_resp_median'
        ]

    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target].values

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    feature_names = list(X.shape)

    # Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Model
    model = ShallowNN(input_dim=X.shape[1])
    model = train_model(model, X_train, y_train, X_val, y_val, true_t1, true_t2)

    # Validation
    val_preds = predict(model, X_val)
    pred_t1, pred_t2 = tune_thresholds(y_val, val_preds, true_t1, true_t2)

    # Test
    test_preds = predict(model, X_test)
    acc, f1 = evaluate_metrics(y_test, test_preds, true_t1, true_t2, pred_t1, pred_t2)

    print("Accuracy:", acc)
    print("F1 Score:", f1)

    return acc, f1


def main():
    results = {}

    for target in TARGETS:
        acc, f1 = run_for_target(target)
        results[target] = {'Accuracy': acc, 'F1': f1}

    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k} -> Accuracy: {v['Accuracy']:.3f}, F1: {v['F1']:.3f}")


if __name__ == "__main__":
    main()