import numpy as np
import torch
from torch import nn
from torch import optim

from nets import SimpleLSTM, SimpleGRU
from data_loading import Loader
import utils

class TrainSet:

    def __init__(self, filename, window_size = 3, LogReturn = True):

        self.prices      = Loader(filename, window_size, LogReturn = LogReturn)
        self.window_size = window_size

    def __call__(self, model_name, hidden_size = 128, seq_length = 30, 
            split_rate = .9, batch_size = 8, num_epochs = 500, num_layers = 2):

        train_size = int(self.prices.train_size * split_rate)
        X = torch.unsqueeze(torch.from_numpy(self.prices.X[:train_size, :]).float(), 1)
        X_train, Y_train = utils.data_process(X, train_size, seq_length)

        if model_name == 'LSTM':
            model = SimpleLSTM(self.window_size, hidden_size, num_layers = num_layers)
        else:
            model = SimpleGRU(self.window_size, hidden_size, num_layers = num_layers)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        loss_plt = []

        for epoch in range(num_epochs):
            loss_sum = 0
            for i in range(0, X_train.shape[1] - batch_size, batch_size):
                Y_pred = model(X_train[:, i : i + batch_size, :])
                Y_pred = torch.squeeze(Y_pred[num_layers - 1, :, :])
                loss = loss_fn(Y_train[i : i + batch_size, :], Y_pred)
                loss_sum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print('epoch [%d] finished, Loss Sum: %f' % (epoch, loss_sum))
            loss_plt.append(loss_sum)
        
        torch.save(model, model_name + '.model')
        utils.plot([len(loss_plt)], [np.array(loss_plt)], 'black', 
                                    'Epoch', 'Loss Sum', 'MSE Loss Function')