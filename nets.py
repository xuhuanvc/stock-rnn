import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(SimpleLSTM, self).__init__()
        self.rnn    = nn.LSTM(input_size, hidden_size, dropout = .5, num_layers = num_layers)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, X):
        (_, X) = self.rnn(X)
        X = torch.squeeze(self.output(X[0]))
        return X

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 2):
        super(SimpleGRU, self).__init__()
        self.rnn    = nn.GRU(input_size, hidden_size, dropout = .5, num_layers = num_layers)
        self.output = nn.Linear(hidden_size, input_size)

    def forward(self, X):
        (_, X) = self.rnn(X)
        X = torch.squeeze(self.output(X))
        return X

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, T):
        super(Encoder, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.T           = T

        self.rnn         = nn.LSTM(input_size, hidden_size)
        self.attention   = nn.Sequential(nn.Linear(hidden_size * 2 + T, T), 
                                         nn.Tanh(),
                                         nn.Linear(T, 1))

    def forward(self, X):
        batch_size = X.size(0)
        code = torch.zeros(batch_size, self.T, self.hidden_size)
        h    = torch.zeros(1, batch_size, self.hidden_size)
        s    = torch.zeros(1, batch_size, self.hidden_size)

        for t in range(self.T - 1):
            from_last = torch.cat((h.repeat(self.input_size, 1, 1).transpose(0, 1), 
                                   s.repeat(self.input_size, 1, 1).transpose(0, 1),
                                   X.transpose(1, 2)), 
                                   dim = 2)

            Z = self.attention(from_last)
            E = F.softmax(Z.view(batch_size, self.input_size), dim = 1)
            alphas = torch.mul(E, X[:, t, :])

            (_, states) = self.rnn(alphas.unsqueeze(0), (h, s))
            h, s = states[0], states[1]
            code[:, t, :] = h

        return code

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, T):

        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.T = T

        self.attention = nn.Sequential(nn.Linear(hidden_size * 2 + input_size, input_size),
                                       nn.Tanh(),
                                       nn.Linear(input_size, 1))

        self.rnn       = nn.LSTM(1, hidden_size)
        self.fc        = nn.Linear(input_size + 1, 1)
        self.output    = nn.Linear(input_size + hidden_size, 1)

    def forward(self, X, Y):
        batch_size = X.size(0)
        d       = torch.zeros(1, batch_size, self.hidden_size)
        s       = torch.zeros(1, batch_size, self.hidden_size)
        context = torch.zeros(batch_size, self.hidden_size)

        for t in range(self.T - 1):
            from_last  = torch.cat((d.repeat(self.T - 1, 1, 1).transpose(0, 1), 
                                    s.repeat(self.T - 1, 1, 1).transpose(0, 1),
                                    X), 
                                    dim = 2)

            Z = self.attention(from_last)
            betas = F.softmax(Z.view(batch_size, -1), dim = 1)

            context = torch.bmm(betas.unsqueeze(1), X).squeeze(1)

            y_update  = self.fc(torch.cat((Y[:, t].unsqueeze(1), context), dim = 1))
            _, states = self.rnn(y_update.unsqueeze(0), (d, s))
            d, s = states[0], states[1]

        y = self.output(torch.cat((d.squeeze(0), context), dim = 1))

        return y