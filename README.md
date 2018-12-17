# stock-rnn

1. Run `python3 data_cleaning.py {ticker}` to fetch the historical prices from Yahoo! Finance.
2. Run `python3 main.py train {ticker} {MODEL}` to train a save a RNN model.
3. Run `python3 main.py test {ticker} {MODEL}` to test a saved RNN model.
 
For examples,
- To download from Apple prices
```bash
python3 data_cleaning.py AAPL
AAPL Download Complete
```
- To train Apple's price by an LSTM model and save the model
```
python3 main.py train AAPL LSTM
```
- To test the performance of the model
```
python3 main.py test AAPL LSTM
```
