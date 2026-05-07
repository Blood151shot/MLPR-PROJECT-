import torch
import torch.nn as nn

class ShallowNN(nn.Module):
    def __init__(self, input_dim, h1=64, h2=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 3)
        )

    def forward(self, x):
        return self.net(x)