import numpy as np
import torch
from torch import nn

from data_loading import Loader
import utils

class EvalSet:

    def __init__(self, filename, window_size = 3, LogReturn = True):

        self.prices = Loader(filename, window_size, LogReturn = LogReturn)

    def __call__(self, modelname, split_rate = .9, seq_length = 30, 
                                                batch_size = 8, num_layers = 2):

        train_size = int(self.prices.train_size * split_rate)
        X = self.prices.X[ train_size : train_size + 300, :]
        X = torch.unsqueeze(torch.from_numpy(X).float(), 1)
        X_test, Y_test = utils.data_process(X, X.shape[0], seq_length)
        model = torch.load(modelname + '.model')
        model.eval()
        loss_fn = nn.MSELoss()
        with torch.no_grad():
            loss_sum = 0
            Y_pred = model(X_test[:, :batch_size, :])
            Y_pred = torch.squeeze(Y_pred[num_layers - 1, :, :])
            for i in range(batch_size, X_test.shape[1], batch_size):
                y = model(X_test[:, i : i + batch_size, :])
                y = torch.squeeze(y[num_layers - 1, :, :])
                Y_pred = torch.cat((Y_pred, y))

                loss = loss_fn(Y_test[i : i + batch_size, :], y)
                loss_sum += loss.item()

        print(loss_sum)
        Y_pred.resize_(Y_pred.shape[0] * Y_pred.shape[1])
        Y_test.resize_(Y_test.shape[0] * Y_test.shape[1])

        utils.plot([Y_pred.shape[0], Y_test.shape[0]], [Y_pred.numpy(), Y_test.numpy()], 
        ['blue', 'red'], 'Time (Days)', 'Price', 
        'Sample ' + modelname + ' Price Result', ['Prediction', 'Ground Truth'])