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
```

# Load Data and Split Train/Test

Deserialize JSON from the preprocessing routines.

```python
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
```

Reserve last 5 years of data as a test set to estimate modeling error.

```python
x_train=x_rolling[:-24*5, :].copy()
y_train=y_rolling[:-24*5, :].copy()
t_train=dates[:-24*5]

x_test=x_rolling[-24*5:, :].copy()
y_test=y_rolling[-24*5:, :].copy()
t_test=dates[-24*5:]
```

```python
print("Train: {0}  to  {1}".format(t_train[0], t_train[-1]))
print("Test:  {0}  to  {1}".format(t_test[0], t_test[-1]))
```

Replot data to make sure it deserialized properly..

```python
idx1=x_cols.index("IPG2211A2N_interp_minus0")
idx2=y_cols.index("IPG2211A2N_interp_plus1")
```

```python
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
```

# Lasso

Search for optimal setting of LASSO parameter `alpha`. We will only include the target column (electricity demand) so that this represents a basic auto-regressive model.

```python
df_col=3
summary=[]
for alpha in np.logspace(-4, -1, 50):
    lasso=Lasso(alpha=alpha, max_iter=30000, tol=1.0e-7)
    tss=TimeSeriesSplit(n_splits=10)
    err=[]
    for tindex, vindex in tss.split(x_train):
        lasso.fit(x_train[tindex, :], y_train[tindex, :])
        y_bar=lasso.predict(x_train[vindex, :])
        err.append(mean_absolute_error(y_train[vindex, :], y_bar))
    summary.append((alpha,np.mean(err)))
```

```python
df_lasso=pd.DataFrame(data=summary, columns=["alpha", "MAE"])
```

```python
# First columm should be lag 0, 10th lag ten, 20th lag 20, etc... plot an example# Plot
trace=(go.Scatter(
            x=df_lasso['alpha'],
            y=df_lasso['MAE'],
            mode='markers',))
fig=go.Figure(data=trace, layout=dict(title= "Alpha vs. Mean Absolute Error"))
fig.update_xaxes(type="log")
fig.show()
```

```python
alpha=df_lasso[df_lasso['MAE']==df_lasso['MAE'].min()]['alpha'].values[0]
print("Refitting with alpha = {}".format(alpha))
```

```python
lasso=Lasso(alpha=alpha, max_iter=30000, tol=1.0e-7)
lasso.fit(x_train, y_train[:, :])
```

### Time Plots

```python
y_bar=lasso.predict(x_train)
for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
    s_model=(go.Scatter(
                x=t_train,
                y=y_bar[:, i],
                mode='lines',))

    s_actual=(go.Scatter(
                x=t_train,
                y=y_train[:, i],
                mode='lines',))

    
    fig=go.Figure(data=[s_model, s_actual], layout=dict(title=title))
    fig.show()
```

### Parity Plots

```python
y_bar=lasso.predict(x_train)
for title, i in zip(["M+1", "M+2", "M+3"],[0,1,2]):
    parity=go.Scatter(
                x=y_train[:, i],
                y=y_bar[:, i],
                mode='markers',)

    fig=go.Figure(data=parity, layout=\
                dict(title="LASSO - "+title, 
                     xaxis=dict(range=(0, 1)),
                     yaxis=dict(range=(0, 1)),                     
                    width=600,
                    height=600,
                 ))
    fig.show()
```

# LSTM

```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
```

```python
tf_model = Sequential()
tf_model.add(LSTM(4))
tf_model.compile()
```
