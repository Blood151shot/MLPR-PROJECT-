import torch

def predict(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return model(X).squeeze().numpy()

def classify(score, t1=0.3, t2=0.6):
    if score < t1:
        return "Low"
    elif score < t2:
        return "Medium"
    else:
        return "High"