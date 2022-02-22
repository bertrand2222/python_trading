import pandas as pd
import IPython
import IPython.display
# from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime as dt
import os
import tensorflow as tf# This code has been tested with TensorFlow 1.6
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf
from models import *

R_WIN_SIZE = 40

symbol ="CA.PA"

tk = yf.Ticker(symbol)

#earning per share
eps = tk.info['trailingEps']

#df = yf.download(symbol, period="20y", interval = "1d").bfill()
#df.to_pickle("../python_trading_data/df.pkl")
#stop
df = pd.read_pickle("../python_trading_data/df.pkl")

val_cols = ['Open' , 'High', 'Low', 'Close']
input_cols = val_cols + ["Volume"]

#plot_features = df[plot_cols]
##plot_features.index = date_time
#_ = plot_features.plot(subplots=True)

#plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)

df = df[input_cols]
train_df = df[0:int(n*0.7)].copy()
val_df = df[int(n*0.7):int(n*0.9)].copy()
test_df = df[int(n*0.9):].copy()

#Normalisation
for dfi in [train_df, val_df, test_df] :
  # calculate Simple Moving Average with 20 days window
  sma = dfi['Close'].rolling(window=R_WIN_SIZE).mean()
  vrmean = dfi['Volume'].rolling(window = R_WIN_SIZE).mean()
  # calculate the standar deviation
  rstd = dfi['Close'].rolling(window=R_WIN_SIZE).std()
  vrstd = dfi['Volume'].rolling(window = R_WIN_SIZE).std()

  dfi[val_cols] = dfi[val_cols].sub(sma, axis = 0)
  dfi[val_cols]  = dfi[val_cols].divide(rstd, axis = 0)

  dfi['Volume'] = dfi['Volume'].sub(vrmean, axis = 0)
  dfi['Volume'] = dfi['Volume'].divide(vrstd, axis = 0)

  dfi.dropna(inplace= True)


val_performance = {}
performance = {}


##### 1 step model
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(200, return_sequences=True, dropout=DROPOUT,),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

#history = compile_and_fit(lstm_model, window)
#val_performance['LSTM'] = lstm_model.evaluate(window.val)
#performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)
multi_val_performance = {}
multi_performance = {}

OUT_STEPS = 10

multi_window = WindowGenerator(input_width=50,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               train_df=train_df,
                               val_df=val_df,
                               test_df=test_df,
                               label_columns = ["Close"])

##### multi step model
#multi_lstm_model = MultiLSTM(window=multi_window, units=200, out_steps=OUT_STEPS)
#history = compile_and_fit(multi_lstm_model, multi_window)

#multi_lstm_model.save_weights('./multi_lstm_checkpoints/my_checkpoints')
#compile(multi_lstm_model)
#multi_lstm_model.load_weights('./multi_lstm_checkpoints/my_checkpoints')


#multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
#multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
#multi_window.plot(multi_lstm_model)

##### feedback_model
feedback_model = FeedBack(window=multi_window, units=200, out_steps=OUT_STEPS)
#history = compile_and_fit(feedback_model, multi_window)
#feedback_model.save_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')
compile(feedback_model)
feedback_model.load_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')

IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)


plt.show()

