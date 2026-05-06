import torch
import torch.optim as optim
from imbalance import compute_sample_weights

def train_model(model, X_train, y_train, X_val, y_val, true_t1, true_t2, epochs=100):

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    weights = torch.tensor(compute_sample_weights(y_train.numpy(), true_t1, true_t2), dtype=torch.float32).view(-1, 1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        preds = model(X_train)
        loss = ((preds - y_train) ** 2 * weights).mean()

        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = ((val_preds - y_val) ** 2).mean().item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    return model