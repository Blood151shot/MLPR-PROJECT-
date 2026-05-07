from split_data import split_data
from model import ShallowNN
from train import train_model
from predict import predict
from evaluation_metrics import evaluate_metrics
from predict import classify
import itertools

import pandas as pd

TARGETS = ['stress', 'phq4_score', 'pam']

def grid_search(X_train, y_train, X_val, y_val, true_t1, true_t2, input_dim):
    print("  Running Hyperparameter Grid Search...")
    h1_options = [32, 64]
    h2_options = [16, 32]
    dropout_options = [0.2, 0.4]
    lr_options = [0.001, 0.005]
    l1_options = [0.0, 1e-4, 1e-3]

    best_model = None
    best_f1 = -1
    best_params = {}
    best_t1 = 0.3
    best_t2 = 0.6

    for h1, h2, dropout, lr, l1_lambda in itertools.product(h1_options, h2_options, dropout_options, lr_options, l1_options):
        # Create model with specific params
        model = ShallowNN(input_dim=input_dim, h1=h1, h2=h2, dropout=dropout)
        
        # Train model
        model = train_model(model, X_train, y_train, X_val, y_val, true_t1, true_t2, lr=lr, l1_lambda=l1_lambda)
        
        # Validate
        val_preds_labels, _ = predict(model, X_val)
        y_val_labels = [classify(v, true_t1, true_t2) for v in y_val]
        
        acc, val_f1, prec, rec, cm = evaluate_metrics(y_val_labels, val_preds_labels)

        # Keep best
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model = model
            best_params = {'h1': h1, 'h2': h2, 'dropout': dropout, 'lr': lr, 'l1': l1_lambda}

    print(f"  -> Best Validation F1: {best_f1:.3f} with Params: {best_params}")
    return best_model, best_params

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

    df = pd.read_csv(r"C:\Users\Tanul\OneDrive\Desktop\mlpr project\final_data_set_10000.csv")

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

    # Grid Search Model
    model, best_params = grid_search(
        X_train, y_train, X_val, y_val, true_t1, true_t2, input_dim=X.shape[1]
    )

    # Test
    test_preds_labels, test_probs = predict(model, X_test)
    y_test_labels = [classify(v, true_t1, true_t2) for v in y_test]
    
    acc, f1, prec, rec, cm = evaluate_metrics(y_test_labels, test_preds_labels)

    print("Accuracy:", acc)
    print(f"F1 Score: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print("Confusion Matrix (Low, Medium, High):")
    print(cm)
    
    print("\nSample Probabilities (First 3 users in Test Set):")
    for i in range(min(3, len(test_probs))):
        print(f"User {i+1} [Low, Med, High]: [{test_probs[i][0]:.3f}, {test_probs[i][1]:.3f}, {test_probs[i][2]:.3f}] -> Predicted: {test_preds_labels[i]}")

    return acc, f1, prec, rec, cm


def main():
    results = {}

    for target in TARGETS:
        acc, f1, prec, rec, cm = run_for_target(target)
        results[target] = {'Accuracy': acc, 'F1': f1, 'Precision': prec, 'Recall': rec}

    print("\n===== FINAL COMPARISON =====")
    for k, v in results.items():
        print(f"{k} -> Acc: {v['Accuracy']:.3f}, F1: {v['F1']:.3f}, Prec: {v['Precision']:.3f}, Rec: {v['Recall']:.3f}")


if __name__ == "__main__":
    main()