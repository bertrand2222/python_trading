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


WIN_SIZE = 40
BATCH_SIZE = 200
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

  dfi.dropna(inplace= True)


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
        batch_size=BATCH_SIZE,)

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

  def plot(self, model=None, plot_col='Close', max_subplots=5):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      plt.ylabel(f'{plot_col} [normed]')
      plt.plot(self.input_indices, inputs[n, :, plot_col_index],
              label='Inputs', marker='.', zorder=-10)

      if self.label_columns:
        label_col_index = self.label_columns_indices.get(plot_col, None)
      else:
        label_col_index = plot_col_index

      if label_col_index is None:
        continue

      plt.scatter(self.label_indices, labels[n, :, label_col_index],
                  edgecolors='k', label='Labels', c='#2ca02c', s=64)
      if model is not None:
        predictions = model(inputs)
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [d]')

val_performance = {}
performance = {}

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

class FeedBack(tf.keras.Model):
  def __init__(self, units, out_steps):
    super().__init__()
    self.out_steps = out_steps
    self.units = units
    self.lstm_cell = tf.keras.layers.LSTMCell(units)
    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
    self.dense = tf.keras.layers.Dense(num_features)

  def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state

  def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the LSTM state.
    prediction, state = self.warmup(inputs)

    # Insert the first prediction.
    predictions.append(prediction)

    # Run the rest of the prediction steps.
    for n in range(1, self.out_steps):
      # Use the last prediction as input.
      x = prediction
      # Execute one lstm step.
      x, state = self.lstm_cell(x, states=state,
                                training=training)
      # Convert the lstm output to a prediction.
      prediction = self.dense(x)
      # Add the prediction to the output.
      predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions


MAX_EPOCHS = 20

def compile_model(model):
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

def compile_and_fit(model, window, patience=2, name=None):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  compile_model(model)

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  if not name is None:
    model.save_weights(name+'_checkpoints/my_checkpoints')

  return history

window = WindowGenerator(input_width=30, label_width=30, shift=30, label_columns=['Close'])

DROPOUT = 0.2

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
                               shift=OUT_STEPS)


multi_val_performance = {}
multi_performance = {}

##### multi step model
#multi_lstm_model = tf.keras.Sequential([
    ## Shape [batch, time, features] => [batch, lstm_units].
    ## Adding more `lstm_units` just overfits more quickly.
    #tf.keras.layers.LSTM(200, return_sequences=False, dropout = DROPOUT),
    ## Shape => [batch, out_steps*features].
    #tf.keras.layers.Dense(OUT_STEPS*num_features,
                          #kernel_initializer=tf.initializers.zeros()),
    ## Shape => [batch, out_steps, features].
    #tf.keras.layers.Reshape([OUT_STEPS, num_features])
#])


#history = compile_and_fit(multi_lstm_model, multi_window)

#multi_lstm_model.save_weights('./multi_lstm_checkpoints/my_checkpoints')
#compile_model(multi_lstm_model)
#multi_lstm_model.load_weights('./multi_lstm_checkpoints/my_checkpoints')


#multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
#multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
#multi_window.plot(multi_lstm_model)

##### feedback_model
feedback_model = FeedBack(units=200, out_steps=OUT_STEPS)
#history = compile_and_fit(feedback_model, multi_window)
#feedback_model.save_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')
compile_model(feedback_model)
feedback_model.load_weights('../python_trading_data/feedback_checkpoints/my_checkpoints')

IPython.display.clear_output()

multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(feedback_model)


plt.show()

