import numpy as np
import matplotlib.pyplot as plt
import torch

match_errors = torch.tensor(np.genfromtxt("match_errors.csv", delimiter=","))
non_match_errors = torch.tensor(np.genfromtxt("non_match_errors.csv", delimiter=","))
print(torch.mean(match_errors))
print(torch.mean(non_match_errors))
counts_match, bins_match = torch.histogram(match_errors)
plt.stairs(counts_match, bins_match)
counts_non_match, bins_non_match = torch.histogram(non_match_errors)
plt.stairs(counts_non_match, bins_non_match)
plt.show()
