import numpy as np
from collections import Counter

def compute_class_weights(y_true_labels):
    counts = Counter(y_true_labels)
    total = len(y_true_labels)
    
    weights = []
    for cls in [0, 1, 2]:
        if cls in counts:
            weights.append(total / (len(counts) * counts[cls]))
        else:
            weights.append(1.0)
            
    return np.array(weights)