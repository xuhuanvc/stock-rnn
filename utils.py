import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

def logret(X):
    log_ret = np.zeros_like(X)
    log_ret[0] = 0
    for i in range(1, X.shape[0]):
        log_ret[i] = math.log(X[i] / X[i-1])
    return log_ret

def data_process(X, train_size, num_steps):
    X_result = X[:num_steps, :, :]
    Y_result = X[num_steps, :, :]
    for s in range(1, train_size - num_steps):
        X_result = torch.cat((X_result, X[s : s + num_steps, :, :]), dim = 1)
        Y_result = torch.cat((Y_result, X[s + num_steps, :, :]), dim = 0)
    return X_result, Y_result

def plot(axislengths, prices, colors, xLabels, yLabels, Title, Legends):
    plt.figure()
    for i in range(0, len(axislengths)):
        length = axislengths[i]
        plt.plot(range(0, length), prices[i][:length], color = colors[i], label = Legends[i])
    legend = plt.legend(loc='upper left')
    legend.get_frame()

    plt.xlabel(xLabels)
    plt.ylabel(yLabels)
    plt.title(Title)
    plt.show()