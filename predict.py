import torch

def predict(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        return predictions.numpy(), probabilities.numpy()

def classify(score, t1=0.3, t2=0.6):
    if score < t1:
        return 0
    elif score < t2:
        return 1
    else:
        return 2

def label_to_string(label_idx):
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    return mapping.get(label_idx, "Unknown")