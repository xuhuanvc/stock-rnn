import numpy as np
import utils

class Loader:

    def __init__(self, filename, window_size, LogReturn = True):

        adjusted_close = np.genfromtxt(filename, delimiter = ',', skip_header = 1, usecols = (5))

        if (LogReturn):
            log_return = utils.logret(adjusted_close) 
        else:
            log_return = adjusted_close

        self.train_size = log_return.shape[0] // window_size

        log_return = log_return[:self.train_size * window_size]
        self.X = log_return.reshape(self.train_size, window_size)