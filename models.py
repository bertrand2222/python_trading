import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
BATCH_SIZE = 300
DROPOUT = 0.1
MAX_EPOCHS = 30
DATA_PATH = "../python_trading_data"
VAL_COLS = ['Open' , 'High', 'Low', 'Close']
INPUT_COLS = VAL_COLS + ["Volume"]
R_WIN_SIZE = 50

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               df,
               label_columns=None):
    # Store the raw data.
    df = df[INPUT_COLS]
    self.df = df
    n = len(df)
    self.train_df = df[0:int(n*0.7)].copy()
    self.val_df = df[int(n*0.7)-R_WIN_SIZE:int(n*0.9)].copy()
    self.test_df = df[int(n*0.9)-R_WIN_SIZE:].copy()
    self.num_features = df.shape[1]
    self.normalize_data()

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

  def normalize_data(self):

    #Normalisation
    for dfi in [self.train_df, self.val_df, self.test_df] :
        # calculate Simple Moving Average with 20 days window
        sma = dfi['Close'].rolling(window=R_WIN_SIZE).mean()
        vrmean = dfi['Volume'].rolling(window = R_WIN_SIZE).mean()
        # calculate the standar deviation
        rstd = dfi['Close'].rolling(window=R_WIN_SIZE).std()
        vrstd = dfi['Volume'].rolling(window = R_WIN_SIZE).std()

        dfi[VAL_COLS] = dfi[VAL_COLS].sub(sma, axis = 0)
        dfi[VAL_COLS]  =dfi[VAL_COLS].divide(rstd, axis = 0)

        dfi['Volume'] = dfi['Volume'].sub(vrmean, axis = 0)
        dfi['Volume'] = dfi['Volume'].divide(vrstd, axis = 0)

        dfi.dropna(inplace= True)

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

class FeedBack(tf.keras.Model):
  def __init__(self, window, nb_units, out_steps, ):
    super().__init__()
    self.window = window
    self.out_steps = out_steps
    self.nb_units = nb_units
    #self.lstm_cell = tf.keras.layers.LSTMCell(units, dropout = DROPOUT)
    self.lstm_cells = [tf.keras.layers.LSTMCell(nb, dropout = DROPOUT) for nb in nb_units]


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
  def __init__(self, window, nb_units, out_steps, ):
    super().__init__()
    self.window = window
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.

    for nb in nb_units[:-1]:
      self.add(tf.keras.layers.LSTM(nb, return_sequences=True, dropout = DROPOUT))
    self.add(tf.keras.layers.LSTM(nb_units[-1], return_sequences=False, dropout = DROPOUT))
    # Shape => [batch, out_steps*features].
    self.add(tf.keras.layers.Dense(out_steps*window.num_features,
                          kernel_initializer=tf.initializers.zeros()))
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
    model.save_weights(os.path.join(DATA_PATH,name+'/checkpoints'))
  return history

def compile_and_load(model, name):
    compile(model)
    model.load_weights(os.path.join(DATA_PATH,name+'/checkpoints'))
