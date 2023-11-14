from EVT.libmr.libmrTorch import weibull
import matplotlib.pyplot as plt
import torch
import numpy as np


dist_match = torch.tensor(np.genfromtxt("match_errors.csv", delimiter=",")).unsqueeze(0)
dist_non_match = torch.tensor(np.genfromtxt("non_match_errors.csv", delimiter=",")).unsqueeze(0)
testing_data = torch.arange(torch.min(dist_non_match).item(), torch.max(dist_match).item(), 0.01)
tailsize = int(len(dist_match[0]) * 0.4)
mr_high = weibull(translateAmount=0.001)
mr_high.FitHigh(dist_match, tailsize)
FitHigh_pdf = mr_high.pdf(testing_data)
mr_low = weibull(translateAmount=0.001)
mr_low.FitLow(dist_non_match, tailsize)
FitLow_pdf = mr_low.pdf(testing_data)
counts_match, bins_match = torch.histogram(dist_match)
counts_non_match, bins_non_match = torch.histogram(dist_non_match)
plt.stairs(counts_match, bins_match)
plt.stairs(counts_non_match, bins_non_match)
plt.plot(testing_data, FitHigh_pdf[0])
plt.plot(testing_data, FitLow_pdf[0])
plt.show()

# line search
prior_probability = 0.5
lowest_value = 10
cutoff_point = 0
for candidate in testing_data:
    if torch.isnan(mr_low.pdf(candidate)).item() is False and torch.isnan(mr_high.pdf(candidate)).item() is False:
        error_probability = ((1-prior_probability) * mr_high.pdf(candidate).item() +
                             prior_probability * mr_low.pdf(candidate).item())
        if error_probability < lowest_value:
            lowest_value = error_probability
            cutoff_point = candidate
print(cutoff_point)
true_positive = len(dist_match[0][dist_match[0] <= cutoff_point])/len(dist_match[0])
false_positive = len(dist_match[0][dist_match[0] > cutoff_point])/len(dist_match[0])
true_negative = len(dist_non_match[0][dist_non_match[0] > cutoff_point])/len(dist_non_match[0])
false_negative = len(dist_non_match[0][dist_non_match[0] <= cutoff_point])/len(dist_non_match[0])
f1_score = (2*true_positive) / (2*true_positive + false_positive + false_negative)
print(f'tp: {true_positive}, fp: {false_positive}, tn: {true_negative}, fn: {false_negative}')
print(f'f1 score: {f1_score}')
