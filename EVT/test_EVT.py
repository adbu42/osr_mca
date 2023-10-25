import torch
import matplotlib.pyplot as plt
import numpy as np
from fit_general_parento import mean_excess_function, gpdfit, gpd_function

first_distribution = torch.normal(3, 4, size=(1, 500))
second_distribution = torch.normal(8, 4, size=(1, 500))
#u, alpha = mean_excess_function((first_distribution[0]))
#print(f'u: {u}')
#print(f'alpha: {alpha}')


first_counts, first_bins = np.histogram(first_distribution[0], bins=40)
second_counts, second_bins = np.histogram(second_distribution[0], bins=40)
plt.stairs(first_counts, first_bins)
plt.stairs(second_counts, second_bins)
plt.show()

training_data, _ = first_distribution[0].sort(descending=False)
u = np.array(training_data[int(len(training_data)*0.95)])
print(u)
data_over_threshold = [x for x in training_data if np.array(x) > u]
gpdfit = gpdfit(sample=training_data, threshold=u)
plt.plot(data_over_threshold, gpd_function(data_over_threshold, gpdfit[0], gpdfit[1], u))
plt.show()
print(gpdfit)
