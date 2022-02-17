import pandas as pd
# from pandas_datareader import data
import matplotlib.pyplot as plt
import datetime as dt
import os
import tensorflow as tf# This code has been tested with TensorFlow 1.6
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import yfinance as yf

WIN_SIZE = 40
symbol ="CA.PA"

tk = yf.Ticker(symbol)

#earning per share
eps = tk.info['trailingEps']

#df = yf.download(symbol, period="20y", interval = "1d").bfill()
#df.to_pickle("df.pkl")
df = pd.read_pickle("df.pkl")

val_cols = ['Open' , 'High', 'Low', 'Close']
input_cols = val_cols + ["Volume"]

#plot_features = df[plot_cols]
##plot_features.index = date_time
#_ = plot_features.plot(subplots=True)

#plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)

train_df = df[input_cols][0:int(n*0.7)].copy()
val_df = df[input_cols][int(n*0.7):int(n*0.9)].copy()
test_df = df[input_cols][int(n*0.9):].copy()

num_features = df.shape[1]

#Normalisation
for dfi in [train_df, val_df, test_df] :
  # calculate Simple Moving Average with 20 days window
  sma = dfi['Close'].rolling(window=WIN_SIZE).mean()
  vrmean = dfi['Volume'].rolling(window = WIN_SIZE).mean()
  # calculate the standar deviation
  rstd = dfi['Close'].rolling(window=WIN_SIZE).std()
  vrstd = dfi['Volume'].rolling(window = WIN_SIZE).std()

  dfi[val_cols] = dfi[val_cols].sub(sma, axis = 0)
  dfi[val_cols]  = dfi[val_cols].divide(rstd, axis = 0)

  dfi['Volume'] = dfi['Volume'].sub(vrmean, axis = 0)
  dfi['Volume'] = dfi['Volume'].divide(vrstd, axis = 0)


class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels

  def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result


w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['Close'])
print(w2)


# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')






