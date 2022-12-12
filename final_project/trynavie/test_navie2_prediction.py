
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from m4_data import naive2_predictions
from ESRNN import ESRNN

df = pd.read_csv('run-of-river_production_load.csv',
                 index_col='Date_Time', parse_dates=True)
# df=pd.read_csv('run-of-river_production_load.csv',parse_dates=True)
df['Date_Time'] = df.index
df = df.asfreq('H')
X = pd.DataFrame()
X['unique_id'] = pd.to_datetime(df['Date_Time']).dt.date
X['ds'] = df['Date_Time']
X['x'] = df['pressure_hpa_134']
X['ds'] = X['ds'].astype('<M8[ns]')
y = pd.DataFrame()
y['unique_id'] = pd.to_datetime(df['Date_Time']).dt.date
y['ds'] = df['Date_Time']
y['y'] = df['Value']
y['ds'] = X['ds'].astype('<M8[ns]')
new_idx = pd.DataFrame([i for i in range(0, len(X))])
X.index = new_idx.index
y.index = new_idx.index
predi = np.random.uniform(-200, 200, (len(y)))
e_dataframe = pd.DataFrame(predi)
e_dataframe[0]
y['y_hat_naive2'] = y['y']+e_dataframe[0]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=0)
model = ESRNN(max_epochs=25, freq_of_test=5, batch_size=4, learning_rate=1e-4,
              per_series_lr_multip=0.8, lr_scheduler_step_size=10,
              lr_decay=0.1, gradient_clipping_threshold=50,
              rnn_weight_decay=0.0, level_variability_penalty=100,
              testing_percentile=50, training_percentile=50,
              ensemble=False, max_periods=25, seasonality=[],
              input_size=4, output_size=6,
              cell_type='LSTM', state_hsize=40,
              dilations=[[1], [6]], add_nl_layer=False,
              random_seed=1, device='cpu')
model.fit(X_train, y_train, X_test, y_test)
