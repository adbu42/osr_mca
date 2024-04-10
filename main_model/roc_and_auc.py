import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# Load the error distributions from CSV files
def load_error_distribution(file_path):
    df = pd.read_csv(file_path)
    return df.values

# Load error distributions from CSV files
error_dist1 = load_error_distribution("tests/match_errors.csv")
error_dist2 = load_error_distribution("tests/non_match_errors.csv")

# Combine the error distributions
combined_errors = np.concatenate([error_dist1, error_dist2])

# Create labels (0 for error_dist1, 1 for error_dist2)
labels = np.concatenate([np.zeros(len(error_dist1)), np.ones(len(error_dist2))])

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(labels, combined_errors)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

# Calculate AUC score
auc_score = roc_auc_score(labels, combined_errors)

print(f"AUC score for the combined error distributions: {auc_score:.4f}")
