import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def run_autoencoder(X, true_labels=None, threshold=0.02, epochs=30):
    model = Autoencoder(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, X_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        reconstructed = model(X_tensor)
        loss = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

    anomaly_scores = (loss > threshold).int().numpy()  # 1 = anomaly, 0 = normal

    metrics = None
    if true_labels is not None:
        precision = precision_score(true_labels, anomaly_scores)
        recall = recall_score(true_labels, anomaly_scores)
        f1 = f1_score(true_labels, anomaly_scores)
        accuracy = accuracy_score(true_labels, anomaly_scores)
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy
        }

    return X.copy(), anomaly_scores, metrics
