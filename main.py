import sys
from train import TrainSet
from evaluate import EvalSet

ticker = sys.argv[2] + '.csv'

if sys.argv[1] == 'train':
    Trainer = TrainSet(ticker, LogReturn = True)
    Trainer(sys.argv[3])
else:
    Evaluator = EvalSet(ticker, LogReturn = True)
    Evaluator(sys.argv[3])
