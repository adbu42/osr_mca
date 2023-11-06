import numpy as np
import matplotlib.pyplot as plt

match_errors = np.genfromtxt("match_errors.csv", delimiter=",")
non_match_errors = np.genfromtxt("non_match_errors.csv", delimiter=",")
print(np.mean(match_errors))
print(np.mean(non_match_errors))
counts_match, bins_match = np.histogram(match_errors, bins=40)
plt.stairs(counts_match, bins_match)
counts_non_match, bins_non_match = np.histogram(non_match_errors, bins=40)
plt.stairs(counts_non_match, bins_non_match)
plt.show()
