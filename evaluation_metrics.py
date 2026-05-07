from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from predict import label_to_string

def evaluate_metrics(y_true_labels, y_pred_labels):
    y_true_str = [label_to_string(v) for v in y_true_labels]
    y_pred_str = [label_to_string(v) for v in y_pred_labels]

    acc = accuracy_score(y_true_str, y_pred_str)
    f1 = f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    prec = precision_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    rec = recall_score(y_true_str, y_pred_str, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_str, y_pred_str, labels=["Low", "Medium", "High"])

    return acc, f1, prec, rec, cm