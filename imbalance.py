# imbalance.py

import numpy as np
from collections import Counter
from predict import classify

def compute_sample_weights(y, t1, t2):
    # Convert continuous scores to labels
    labels = [classify(v, t1, t2) for v in y]

    counts = Counter(labels)
    total = len(labels)

    # Inverse frequency weighting
    weights = {
        cls: total / (len(counts) * count)
        for cls, count in counts.items()
    }

    # Assign weight per sample
    return np.array([weights[l] for l in labels])