import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import tensorflow as tf
import numpy as np
import yfinance as yf
from models import *

symbol ="CA.PA"



if not os.path.isdir(DATA_PATH):
    os.mkdirs(DATA_PATH)

#tk = yf.Ticker(symbol)
#earning per share
#eps = tk.info['trailingEps']

#df = yf.download(symbol, period="20y", interval = "1d").bfill()
#df.to_pickle(os.path.join(DATA_PATH,"df.pkl"))

df = pd.read_pickle(os.path.join(DATA_PATH,"df.pkl"))

#plot_features = df[plot_cols]
##plot_features.index = date_time
#_ = plot_features.plot(subplots=True)

#plt.show()

val_performance = {}
performance = {}

##### 1 step model
#lstm_model = tf.keras.models.Sequential([
    #Shape [batch, time, features] => [batch, time, lstm_units]
    #tf.keras.layers.LSTM(200, return_sequences=True, dropout=DROPOUT,),
    #Shape => [batch, time, features]
    #tf.keras.layers.Dense(units=1)
#])

#history = compile_and_fit(lstm_model)
#val_performance['LSTM'] = lstm_model.evaluate(window.val)
#performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)
multi_val_performance = {}
multi_performance = {}

OUT_STEPS = 10

multi_window = WindowGenerator(input_width=50,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               df=df,
                               label_columns = ["Close"])

##### multi step model
multi_lstm_model = MultiLSTM(window=multi_window, units=200, out_steps=OUT_STEPS, )
#history = compile_and_fit(multi_lstm_model, name = "LSTM")

compile_and_load(multi_lstm_model, "LSTM")

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)
plt.show()

##### feedback_model
#feedback_model = FeedBack(window=multi_window, units=200, out_steps=OUT_STEPS)
#history = compile_and_fit(feedback_model)
#feedback_model.save_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')
#compile(feedback_model)
#feedback_model.load_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')

#IPython.display.clear_output()

#multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
#multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
#multi_window.plot(feedback_model)


plt.show()

