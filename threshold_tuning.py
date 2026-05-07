import numpy as np
from sklearn.metrics import f1_score
from predict import classify

def tune_thresholds(y_true, y_pred, true_t1, true_t2):
    best_f1 = 0
    best_t1, best_t2 = 0.3, 0.6

    y_true_labels = [classify(v, true_t1, true_t2) for v in y_true]

    for t1 in np.linspace(0.2, 0.4, 10):
        for t2 in np.linspace(0.5, 0.8, 10):
            if t2 <= t1:
                continue

            y_pred_labels = [
                "Low" if p < t1 else "Medium" if p < t2 else "High"
                for p in y_pred
            ]

            f1 = f1_score(y_true_labels, y_pred_labels, average="macro")

            if f1 > best_f1:
                best_f1 = f1
                best_t1, best_t2 = t1, t2

    return best_t1, best_t2, best_f1