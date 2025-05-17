from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def run_isolation_forest(X, true_labels=None, contamination=0.05):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(X)
    preds = model.predict(X)
    # Convert {-1, 1} to {1, 0} where 1 means anomaly
    anomaly_scores = np.where(preds == -1, 1, 0)

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
