import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ExtremeLy as ely
from evt.dataset import Dataset
from evt.methods.peaks_over_threshold import PeaksOverThreshold
from evt.estimators.gpdmle import GPDMLE
import pandas as pd


def mean_excess_function(training_data: torch.Tensor) -> (int, int):
    size = training_data.size()[0]
    mef = torch.zeros(size-1)
    training_data, _ = training_data.sort(descending=False)
    for i in range(size-1):
        m = []
        for j in range(size):
            if training_data[j] > training_data[i]:
                #m.append(training_data[j])
                m.append(training_data[j]-training_data[i])
        #mef[i] = np.mean(m)/len(m)
        mef[i] = np.sum(m) / len(m)
    training_data = training_data[:-1]
    mef, _ = mef.sort(descending=True)
    training_data, _ = training_data.sort(descending=True)
    plt.plot(training_data, mef)
    plt.show()
    errors = torch.zeros(int(size/5))
    for k in range(int(size/5)):
        k_biggest_training_samples = training_data[:k+1]
        k_biggest_mef = mef[:k+1]
        fitted_line = LinearRegression().fit(k_biggest_training_samples.reshape(-1, 1), k_biggest_mef)
        mse_value = mean_squared_error(fitted_line.predict(k_biggest_training_samples.reshape(-1, 1)),
                                       k_biggest_training_samples)
        errors[k] = torch.from_numpy(np.array([mse_value]))
    print(errors[:10])
    u = training_data[torch.argmin(errors)]
    values_over_threshold = []
    for j in range(size-1):
        if training_data[j] > u:
            values_over_threshold.append(training_data[j])
    alpha = (size-len(values_over_threshold)-1)/size
    return u, alpha


def gpd_u_estimation_and_fitting(training_data: torch.Tensor) -> (int, int):
    training_data, _ = training_data.sort(descending=False)
    u = training_data[int(len(training_data)*0.95)]
    fitted_gpd = gpdfit(sample=training_data, threshold=u)
    return fitted_gpd


def gpdfit(sample, threshold):
    sample = np.sort(sample)
    series = pd.Series(sample)
    series.index.name = "index"
    dataset = Dataset(series)

    # Using PeaksOverThreshold function from evt library.
    pot = PeaksOverThreshold(dataset, threshold)
    mle = GPDMLE(pot)
    shape_estimate, scale_estimate = mle.estimate()
    shape = getattr(shape_estimate, 'estimate')
    scale = getattr(scale_estimate, 'estimate')
    return shape, scale


def gpd_function(x, shape, scale, threshold):
    return (1 + (shape * (x - threshold))/scale)**(-1/shape)
