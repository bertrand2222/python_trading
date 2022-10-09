import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
import numpy as np
import yfinance as yf
from constant import *


symbol ="CA.PA"
ref = '^FCHI'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

df = yf.download(symbol, period="20y", interval = "1d").bfill()
tk = yf.Ticker(symbol)

df_ref = yf.download(ref, period="20y", interval = "1d").bfill()
df_fund = pd.read_csv(FED_FUND_LINK)
df_fund.set_index('DATE', inplace= True)
df_fund.index = pd.to_datetime(df_fund.index)



df = pd.merge(df, df_fund, left_index = True, right_index = True, how='outer').fillna(method='ffill').dropna()

# adjust dividend
dividends = tk.get_dividends()
df = df.merge(dividends, how = "left", left_index = True, right_index = True,).fillna(0)
sum_div = df['Dividends'].cumsum()
df[PRICE_COLS] = df[PRICE_COLS].add(sum_div, axis = 0)

#earning per share
#eps = tk.info['trailingEps']
INP_STEPS = 15
OUT_STEPS = 1
from models import *

df['Close_200'] = z_score(df['Close'], 200) 
df['Close_100'] = z_score(df['Close'], 100)
scaler = scaler=MinMaxScaler(feature_range=(-1,1))
df['FEDFUNDS'] = scaler.fit_transform(df[['FEDFUNDS']])
# df['FEDFUNDS'] = (df['FEDFUNDS'] - df['FEDFUNDS'].mean()) / df['FEDFUNDS'].std()
df['ref'] = df_ref['Close']

df['FEDFUNDS'].plot()
plt.show()
stop
INPUT_COLS = PRICE_COLS  + ['Close_100','Close_200',  'Volume', 'ref', 'FEDFUNDS' ]
TO_ROLLING_NORMALIZE = PRICE_COLS  + ['Volume', 'ref']


val_performance = {}
performance = {}

multi_val_performance = {}
multi_performance = {}

multi_window = WindowGenerator(input_width=INP_STEPS,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS,
                               df=df,
                               input_cols= INPUT_COLS,
                               cols_to_rolling_normalize = TO_ROLLING_NORMALIZE,
                               label_columns = ["Close"],
                               r_win_size= 25)

##### multi step model
multi_lstm_model = MultiLSTM(window=multi_window, nb_units=LSTM_UNITS, out_steps=OUT_STEPS, dropout = DROPOUT)
history = compile_and_fit(multi_lstm_model, name = "LSTM")

# multi_lstm_model = load_model('LSTM')


multi_window.test_real_perfo(multi_lstm_model)
# id_pred = 50 
# prediction  = multi_window.predict_real_iloc(multi_lstm_model, id_pred)
# close_last_real_inputs = df['Close'][-multi_window.input_width - multi_window.label_width -id_pred: - multi_window.label_width-id_pred]
# close_last_real_output = df['Close'][ - multi_window.label_width -id_pred: -id_pred].values
# print('close_last_inputs ', close_last_real_inputs)
# print('close_last_outputs ', close_last_real_output)
# print("prediction ", prediction)


multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)



##### feedback_model
#feedback_model = FeedBack(window=multi_window, nb_units=LSTM_UNITS, out_steps=OUT_STEPS, dropout = DROPOUT)
#history = compile_and_fit(feedback_model, name = "recursif_LSTM")

##compile_and_load(feedback_model, "recursif_LSTM")


# multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
# multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
# multi_window.plot(feedback_model)


#x = np.arange(len(multi_performance))
#width = 0.3

#metric_name = 'mean_absolute_error'
#metric_index = multi_lstm_model.metrics_names.index('mean_absolute_error')
#val_mae = [v[metric_index] for v in multi_val_performance.values()]
#test_mae = [v[metric_index] for v in multi_performance.values()]

#plt.bar(x - 0.17, val_mae, width, label='Validation')
#plt.bar(x + 0.17, test_mae, width, label='Test')
#plt.xticks(ticks=x, labels=multi_performance.keys(),
           #rotation=45)
#plt.ylabel(f'MAE (average over all times and outputs)')
#_ = plt.legend()

plt.show()
