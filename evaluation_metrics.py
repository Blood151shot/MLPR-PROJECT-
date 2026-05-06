from sklearn.metrics import accuracy_score, f1_score
from predict import classify

def evaluate_metrics(y_true, y_pred, true_t1, true_t2, pred_t1, pred_t2):
    y_true_labels = [classify(v, true_t1, true_t2) for v in y_true]
    y_pred_labels = [classify(v, pred_t1, pred_t2) for v in y_pred]

    acc = accuracy_score(y_true_labels, y_pred_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average="macro")

    return acc, f1