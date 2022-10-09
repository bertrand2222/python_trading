from time import perf_counter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from constant import *
import pandas as pd
from typing import Union
from sklearn.preprocessing import MinMaxScaler
#learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    #0.001, 1, 0.5, staircase=True, name=None
#)

def z_score(s : Union[pd.Series, pd.DataFrame] , periods : int = 50, shift = 1) :
      rolling = s.rolling(window= periods)
      sma = rolling.mean().shift(shift)
      std = rolling.std().shift(shift)

      r = s.sub(sma, axis = 0).divide( std, axis = 0)
      return r



class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               df : pd.DataFrame, input_cols, cols_to_rolling_normalize = [],
               label_columns=None, r_win_size = 20):
    # Store the raw data.
    df = df[input_cols]
    self.df = df
    self.cols_to_rolling_normalize = cols_to_rolling_normalize
    self.r_win_size = r_win_size
    self.df_scaled = self.get_normalized_df()

    n = len(self.df_scaled.index)
    self.train_df = self.df_scaled[0:int(n*0.7)].copy()
    self.val_df = self.df_scaled[int(n*0.7):int(n*0.9)].copy()
    self.test_df = self.df_scaled[int(n*0.9):].copy()
    self.num_features = df.shape[1]
    print("num features : ", self.num_features)


    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(df.columns)}

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

  

  def get_normalized_df(self) -> pd.DataFrame:

    df_scaled = self.df.copy()
    df_scaled[self.cols_to_normalize] = z_score(self.df[self.cols_to_rolling_normalize], self.r_win_size)
    return(df_scaled.dropna())
  #   #Normalisation
  #   for dfi in [self.train_df, self.val_df, self.test_df] :

  #       # calculate Simple Moving Average with 20 days window


  #       if self.normalizing_method == 'scale_to_close':
  #         sma = dfi["Close"].rolling(window= self.r_win_size).mean()
  #         std = dfi["Close"].rolling(window= self.r_win_size).std()

  #         dfi[PRICE_COLS] = dfi[PRICE_COLS].sub(sma, axis = 0)
  #         dfi[PRICE_COLS]  =dfi[PRICE_COLS].divide( 2 * std, axis = 0)
  #         if 'Volume' in dfi.columns :
  #           vmean = dfi['Volume'].mean()
  #           vstd = dfi['Volume'].std()
  #           dfi['Volume'] = dfi['Volume'].sub(vmean, axis = 0)
  #           dfi['Volume'] = dfi['Volume'].divide(vstd, axis = 0)

  #       elif self.normalizing_method == 'scale_to_all' :

  #         dfi[self.cols_to_normalize] = z_score(dfi[self.cols_to_normalize], self.r_win_size) 

 
  #       elif self.normalizing_method == 'diff_high_low' : 
  #         mean_ac = (dfi['Close'] + dfi['Open']) / 2
  #         sma = dfi["Close"].rolling(window= self.r_win_size).mean()
  #         std = dfi["Close"].rolling(window= self.r_win_size).std()

  #         close_init = dfi['Close']
  #         dfi['Close'] = dfi['Close'].sub(sma, axis = 0)
  #         dfi['Close']  =dfi['Close'].divide( 2 * std, axis = 0)

  #         dfi["High"] = dfi["High"].sub(mean_ac, axis = 0).divide(mean_ac, axis = 0)     
  #         dfi['High'] = dfi['High'].sub(dfi['High'].mean(), axis = 0).divide(dfi['High'].std())
  #         dfi["Low"] = dfi["Low"].sub(mean_ac, axis = 0).divide(mean_ac, axis = 0)     
  #         dfi['Low'] = dfi['Low'].sub(dfi['Low'].mean(), axis = 0).divide(dfi['Low'].std())
  #         dfi["Open"] = dfi["Open"].sub(close_init, axis = 0).divide(close_init, axis = 0)     
  #         dfi['Open'] = dfi['Open'].sub(dfi['Open'].mean(), axis = 0).divide(dfi['Open'].std())
        
 
  #       dfi.dropna(inplace= True)

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

  # @property
  # def real(self):
  #   return self.make_dataset(self.df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.test))
      # And cache it for next time
      self._example = result
    return result

  @property
  def last_inputs(self):
    return tf.expand_dims(tf.constant(self.test_df.iloc[-self.input_width -self.label_width :-self.label_width]),0)

  def iloc_past_inputs(self,i ):
    return tf.expand_dims(tf.constant(self.df_scaled.iloc[-self.input_width -self.label_width -i :-self.label_width - i]),0)
  

  def convert_real_outputs(self, outputs_z_score, real_inputs : pd.Series):

      rolling = real_inputs.rolling(self.r_win_size)
      sma = rolling.mean()
      std = rolling.std()
      real = std[-1] * outputs_z_score + sma[-1]

      return real

  def test_convert(self,):

    close_z = z_score(self.df['Close'], self.r_win_size)

    real = self.convert_real_outputs(close_z[-1], self.df['Close'][-self.r_win_size -1:-1] )
    print("real : " + str(self.df['Close'][-1]))
    print("computed real : " + str(real))

  def predict_real_iloc(self, model, i ):
    last_scaled_inputs = self.iloc_past_inputs(i)
    prediction = model(last_scaled_inputs)
    last_real_inputs = self.df[self.label_columns[0]].iloc[ - self.r_win_size - self.label_width - i: - self.label_width - i]
    real_prediction = self.convert_real_outputs(prediction[:,:,0].numpy()[0], last_real_inputs,)
    return real_prediction

  def test_real_perfo(self, model):
    print('eval perfo')
    res = np.zeros(len(self.test_df))
    err = np.zeros(len(self.test_df))
    for i in range(len(self.test_df.index) ) :
      prediction  = self.predict_real_iloc(model, i)[0]
      real_value = self.df['Close'][ - self.label_width - i]
      real_value_prev = self.df['Close'][ - 2 * self.label_width - i ]
      err[i] = (prediction - real_value)/ real_value
      res[i] = (real_value - real_value_prev) *  (prediction - real_value_prev)
    print('sum : ', res.sum())
    print('err moy :', err.mean())


  def plot(self, model=None, plot_col='Close', max_subplots=5):
    inputs, labels = self.example

    if model is not None:
      predictions = model(inputs)

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
        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                    marker='X', edgecolors='k', label='Predictions',
                    c='#ff7f0e', s=64)

      if n == 0:
        plt.legend()

    plt.xlabel('Time [d]')



class FeedBack(tf.keras.Model):
  def __init__(self, window, nb_units, out_steps, dropout ):
    super().__init__()
    self.window = window
    self.out_steps = out_steps
    self.nb_units = nb_units
    #self.lstm_cell = tf.keras.layers.LSTMCell(units, dropout = DROPOUT)
    self.lstm_cells = [tf.keras.layers.LSTMCell(nb, dropout = dropout ) for nb in nb_units]


    # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
    self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cells, return_state=True,)
    self.dense = tf.keras.layers.Dense(window.num_features)

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
      x, *state = self.lstm_rnn(tf.expand_dims(x,1), initial_state=state, training=training)

      #x, state = self.lstm_cell(x, states=state, training=training)

      # Convert the lstm output to a prediction.
      prediction = self.dense(x)
      # Add the prediction to the output.
      predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions

class MultiLSTM(tf.keras.Sequential):
  def __init__(self, window : WindowGenerator, nb_units, out_steps, dropout ):
    super().__init__()
    self.window = window
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.

    for nb in nb_units[:-1]:
      self.add(tf.keras.layers.LSTM(nb, return_sequences=True, dropout = dropout))
    self.add(tf.keras.layers.LSTM(nb_units[-1], return_sequences=False, dropout = dropout))
    # Shape => [batch, out_steps*features].
    self.add(tf.keras.layers.Dense(out_steps*window.num_features,
                          kernel_initializer=tf.initializers.zeros() ))
    # Shape => [batch, out_steps, features].
    self.add(tf.keras.layers.Reshape([out_steps, window.num_features]))

def compile(model):
  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

def compile_and_fit(model, patience=2, name=None):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  compile(model)

  history = model.fit(model.window.train, epochs=MAX_EPOCHS,
                      validation_data=model.window.val,
                      callbacks=[early_stopping])
  if not name is None:
    # model.save_weights(os.path.join(DATA_PATH,name+'/checkpoints'))
    model.save(os.path.join(DATA_PATH,name+'/model'))
  return history

# def compile_and_load(model, name):
#     compile(model)
    # model.load_weights(os.path.join(DATA_PATH,name+'/checkpoints'))

def load_model(name):
    model = tf.keras.models.load_model(os.path.join(DATA_PATH,name+'/model'))
    return model


