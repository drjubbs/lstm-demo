---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Modeling

Use linear and LSTM models

```python
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import plotly.graph_objects as go
import lstmutil

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

# TensorFlow setup
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
```

# Configuration

```python
# Window sizes
IN_WINDOW = 24
IN_FEATURES = 4
OUT_WINDOW = 3
OUT_FEATURES = 1

# Months to use for validation
VALIDATION = 5*24 
```

# Load Data and Split Train/Test

Deserialize JSON from the preprocessing routines.

```python
def load_data():
    with open("./scaled/fred.json","r") as output_file:
        data_json=json.loads(output_file.read())
    scaler=lstmutil.Scaler()
    times=data_json['times']
    columns=data_json['columns']
    dates=np.array([datetime.utcfromtimestamp(t) for t in times])
    x_rolling=np.array(data_json['x_rolling'])
    y_rolling=np.array(data_json['y_rolling'])
    x_cols=data_json['x_cols']
    y_cols=data_json['y_cols']
    
    
    # Reserve last N years of data as a test set to estimate modeling error.
    x_train=x_rolling[:-VALIDATION, :].copy()
    y_train=y_rolling[:-VALIDATION, :].copy()
    t_train=dates[:-VALIDATION]

    x_test=x_rolling[-VALIDATION:, :].copy()
    y_test=y_rolling[-VALIDATION:, :].copy()
    t_test=dates[-VALIDATION:]
    
    print("Train: {0}  to  {1}".format(t_train[0], t_train[-1]))
    print("Test:  {0}  to  {1}".format(t_test[0], t_test[-1]))
    
    # Replot data to make sure it deserialized properly...
    idx1=x_cols.index("IPG2211A2N_interp_minus0")
    idx2=y_cols.index("IPG2211A2N_interp_plus1")
    
    # Plot
    traces=[]
    traces.append(go.Scatter(
                    x=t_test,
                    y=x_test[:,idx1],
                    mode='lines',                    
                    name='IPG2211A2N (t=0)',
                    ))
    traces.append(go.Scatter(
                    x=t_test,
                    y=y_test[:,idx2],
                    mode='lines',                    
                    name='IPG2211A2N (t+1)',
                    ))

    fig=go.Figure(data=traces, layout=dict(title="Target values offset by 1 month"))
    fig.show()
    
    return x_train, y_train, t_train, x_test, y_test, t_test

x_train, y_train, t_train, x_test, y_test, t_test = load_data() 
```

# Lasso

Search for optimal setting of LASSO parameter `alpha`. We will only include the target column (electricity demand) so that this represents a basic auto-regressive model.

```python
def lasso_search():
    df_col=3
    summary=[]
    for alpha in np.logspace(-4, -1, 50):
        lasso=Lasso(alpha=alpha, max_iter=30000, tol=1.0e-7)
        tss=TimeSeriesSplit(n_splits=5)
        err=[]
        for tindex, vindex in tss.split(x_train):
            lasso.fit(x_train[tindex, :], y_train[tindex, :])
            y_bar=lasso.predict(x_train[vindex, :])
            err.append(mean_absolute_error(y_train[vindex, :], y_bar))
        summary.append((alpha,np.mean(err)))

    df_lasso=pd.DataFrame(data=summary, columns=["alpha", "MAE"])


    trace=(go.Scatter(
            x=df_lasso['alpha'],
            y=df_lasso['MAE'],
            mode='markers',))
    fig=go.Figure(data=trace, layout=dict(title= "Alpha vs. Mean Absolute Error"))
    fig.update_xaxes(type="log")
    fig.show()

    # Return best alpha
    alpha=df_lasso[df_lasso['MAE']==df_lasso['MAE'].min()]['alpha'].values[0]
    return alpha
```

```python
alpha = lasso_search()
```

```python
def lasso_final(alpha):

    print("Refitting with alpha = {}".format(alpha))
    lasso=Lasso(alpha=alpha, max_iter=30000, tol=1.0e-7)
    lasso.fit(x_train, y_train[:, :])
    
    # Time plots
    y_bar=lasso.predict(x_train)
    for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
        s_model=(go.Scatter(
                    x=t_train,
                    y=y_bar[:, i],
                    name='Model',
                    mode='lines',))

        s_actual=(go.Scatter(
                    x=t_train,
                    y=y_train[:, i],
                    name='Actual',
                    mode='lines',))


        fig=go.Figure(data=[s_model, s_actual], layout=dict(title=title))
        fig.show()
        
    # Parity Plots
    for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
        parity=go.Scatter(
                    x=y_train[:, i],
                    y=y_bar[:, i],
                    mode='markers',)

        fig=go.Figure(data=parity, layout=\
                    dict(title="LASSO {0} {1:7.4f}".format(title, mean_absolute_error(y_train, y_bar)),
                         xaxis=dict(range=(0, 1)),
                         yaxis=dict(range=(0, 1)),                     
                        width=600,
                        height=600,
                     ))
        fig.show()
```

```python
lasso_final(alpha)
```

# LSTM

```python
def get_tf_model(lstm_size, hidden1_size=None):
    """Create and return a TensorFlow model"""
    this_model = Sequential()
    this_model.add(LSTM(lstm_size))
    if  hidden1_size is not None:
        this_model.add(Dense(hidden1_size))
    this_model.add(Dense(3))
    this_model.compile(optimizer='Adam', 
                       loss='MeanSquaredError',
                       metrics='MeanSquaredError')
    
    return this_model
```

```python
def prep_data(flat_x, flat_y):
    """Reshape data and convert to native TensorFlow objects to improve efficiency.
    Tensorflow expects 3D matrix: i,j,k = samples, window/sequence data.
    """
    num_pts=len(flat_x)
    x_rs=tf.cast(x_train.reshape(num_pts, IN_WINDOW, IN_FEATURES), dtype=tf.float32)
    y_rs=tf.cast(y_train.reshape(num_pts, OUT_WINDOW, OUT_FEATURES), dtype=tf.float32)
    
    return x_rs, y_rs
```

```python
x_train_rs, y_train_rs = prep_data(x_train, y_train)
tss=TimeSeriesSplit(n_splits=5)
summary = []
for hidden1_size in [4, 8, 16, 32]:
    counter = 1
    for tindex, vindex in tss.split(x_train):
        
        lstm = get_tf_model(lstm_size=18, hidden1_size=hidden1_size)
        
        begin = tindex[0]
        end = tindex[-1]
        history = lstm.fit(x_train_rs[begin:end, :], 
                 y_train_rs[begin:end, :],
                 epochs=4000,
                 verbose=0)                    
        print("Model: {}".format(counter))
        counter += 1
    
    summary.append((hidden1_size, history.history['loss'][-1]))
```

```python
print(pd.DataFrame(summary))
```

```python
# Refit whole model
lstm = get_tf_model(18, 16)
history = lstm.fit(x_train_rs, y_train_rs, epochs=4000, verbose=0)
print(history.history['loss'][-1])
```

```python
# Time plots
y_bar=lstm.predict(x_train_rs)
for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
    s_model=(go.Scatter(
                x=t_train,
                y=y_bar[:, i],
                name='Model',
                mode='lines',))

    s_actual=(go.Scatter(
                x=t_train,
                y=y_train[:, i],
                name='Actual',
                mode='lines',))


    fig=go.Figure(data=[s_model, s_actual], layout=dict(title=title))
    fig.show()
    
# Parity Plots
for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
    parity=go.Scatter(
                x=y_train[:, i],
                y=y_bar[:, i],
                mode='markers',)

    fig=go.Figure(data=parity, layout=\
                dict(title="LASSO {0} {1:7.4f}".format(title, mean_absolute_error(y_train, y_bar)),
                     xaxis=dict(range=(0, 1)),
                     yaxis=dict(range=(0, 1)),                     
                    width=600,
                    height=600,
                 ))
    fig.show()
```

```python

```
