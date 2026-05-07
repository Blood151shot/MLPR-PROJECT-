import torch
import torch.optim as optim
import torch.nn as nn
from imbalance import compute_class_weights
from predict import classify

def train_model(model, X_train, y_train, X_val, y_val, true_t1, true_t2, lr=0.001, l1_lambda=0.0, epochs=100):

    y_train_labels = [classify(v, true_t1, true_t2) for v in y_train]
    y_val_labels = [classify(v, true_t1, true_t2) for v in y_val]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_labels, dtype=torch.long)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_labels, dtype=torch.long)

    class_weights = torch.tensor(compute_class_weights(y_train_labels), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train)
        loss = criterion(preds, y_train_tensor)

        if l1_lambda > 0:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val_tensor).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model