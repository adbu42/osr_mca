import numpy as np
import matplotlib.pyplot as plt

match_errors = np.genfromtxt("tests/match_errors.csv", delimiter=",")
non_match_errors = np.genfromtxt("tests/non_match_errors.csv", delimiter=",")
match_errors_sorted = np.sort(match_errors)

P = len(non_match_errors)
N = len(match_errors)
roc_curve = np.zeros((len(match_errors), 2))

for i, threshold in enumerate(match_errors_sorted):
    TP = (non_match_errors >= threshold).sum()
    FP = (match_errors > threshold).sum()
    FN = (non_match_errors < threshold).sum()
    TN = (match_errors < threshold).sum()
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    roc_curve[i, 0] = TPR
    roc_curve[i, 1] = FPR


plt.plot(roc_curve[:, 1], roc_curve[:, 0])
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()

auc = 0
for i in range(1, len(roc_curve)):
    auc += ((roc_curve[i-1, 1] - roc_curve[i, 1]) * roc_curve[i, 0]
            + 0.5 * ((roc_curve[i-1, 1] - roc_curve[i, 1]) * (roc_curve[i-1, 0] - roc_curve[i, 0])))
print(auc)
