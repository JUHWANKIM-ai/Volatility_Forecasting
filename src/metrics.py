from sklearn.metrics import *

import numpy as np


def calculate_nd(gt, output):
    errors = np.abs(np.array(gt) - np.array(output))
    mean = np.mean(errors)
    std_deviation = np.std(errors)

    nd_list = (errors - mean) / std_deviation
    nd = np.mean(nd_list)
    return nd


def calculate_mae(gt, output):
    mae = mean_absolute_error(gt, output)
    return mae


def calculate_rmse(gt, output):
    rmse = np.sqrt(mean_squared_error(gt, output))
    return rmse


def MAE(pred, true):
    return np.mean(np.abs((pred - true)))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def calculate_R2(gt, output):
    if output.shape == ():
        output = output.reshape(1, 1)
    output = output.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    r2 = np.mean(r2_score(gt, output))
    return r2


def calculate_adjusted_r2_score(gt, output, n, p):
    r2 = r2_score(gt, output)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adjusted_r2


def calculate_r2_score(gt, output):
    r2 = r2_score(gt, output)
    return r2
