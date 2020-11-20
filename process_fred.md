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

# Workup of Data From FRED

```python
import os
import json
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import plotly.graph_objects as go
from lstmutil import TimeSeries
from lstmutil import Scaler
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

%load_ext autoreload
%autoreload 2
```

# Globals

Size of input and output windows

```python
IN_WINDOW = 24
OUT_WINDOW = 3
```

## Data Interpolation

Data sets will in general mismatch on both frequency and days available (e.g. financial data only available on trading days). Use step interpolation to fill in missed data and upsample as appropriate to a daily frequency. All times given in UTC.

```python
timeseries=TimeSeries(begin=datetime(1990,1,1), end=datetime.utcnow())
```

## Main Preprocessing

```python
raw={}
clean={}
interp={}

target_timestamps = timeseries.get_target_timestamps()
series_list = ["DGS1", "DGS5", "DGS10", "IPG2211A2N"]
df_merge = []
for series in series_list:
    
    # Clean & interpolate
    raw[series]=pd.read_csv("./input/"+series+".csv")
    clean[series]=timeseries.clean_ts(raw[series], "DATE", series)
    interp[series]=timeseries.interp_ts(clean[series], series, target_timestamps)
    
    # Merge back with the raw data, used in sanity checks later...
    interp[series]=interp[series].merge(clean[series], on=['date', 'timestamp'], how='left', suffixes=('_interp', '_raw'))
    
    if len(df_merge)==0:
        df_merge=interp[series].copy()
    else:
        df_merge=df_merge.merge(interp[series], on=['timestamp', 'date'], how='inner')    
```

```python
df_merge
```

```python
for series in series_list:
    p_interp1=go.Scatter(
                    x=df_merge['date'], 
                    y=df_merge[series+'_interp'],                        
                    mode='lines',
                    line=dict(color='#A9A9A9'),
                    name=series+'_interp',
                    )
    p_original1=go.Scatter(
                    x=df_merge['date'], 
                    y=df_merge[series+'_raw'], 
                    mode='markers', 
                    marker=dict(size=6, color='#006500'),
                    name=series+'_raw',
                    )

    fig=go.Figure(data=[p_interp1, p_original1])
    fig.show()
```

```python
# Keep only interpolated columns now
interp_cols = [t+"_interp" for t in series_list]
keep = ["date", "timestamp"] + interp_cols
df_interp=df_merge[keep]
x_unscaled = df_interp[interp_cols].values
```

```python
# Rescale so that all signals are [0,1]
mms=Scaler()
x_scaled=mms.fit_transform(x_unscaled)
df_scaled = df_interp.copy()
df_scaled[interp_cols]=x_scaled
```

```python
 # Down sample to monthly data
df_scaled=df_scaled[[t.day==1 for t in df_scaled['date']]]
```

```python
# Plot
traces=[]
for series in series_list:
    traces.append(go.Scatter(
                    x=df_scaled['date'],
                    y=df_scaled[series+'_interp'],
                    mode='lines',                    
                    name=series+'_interp',
                    ))

fig=go.Figure(data=traces, layout=dict(title="Scaled & interpolated x varaibles"))
fig.show()    
```

## Train/Test/Validation Splitting

This is a complex topic, see notes in `README.md`. For now we'll keep things simple and predict 
\[M+1, M+2, M+3\] based on the previous 2 years of data in a rolling window. We'll retain the last 5 years for final error esimation.

```python
x_cols=[t+"_interp" for t in series_list]
y_cols=["IPG2211A2N_interp"]

times, x_cols, y_cols, x_rolling, y_rolling = TimeSeries.rolling_horizon(
    df_scaled,
    time_col="timestamp",
    x_cols=x_cols,
    y_cols=y_cols,
    in_window=IN_WINDOW,
    out_window=OUT_WINDOW)
dates=[datetime.utcfromtimestamp(t) for t in times]
```

```python
# Plot lags... the data should be shifting to the left as lag is decresed 
traces=[]
idxs = [x_cols.index(t) for t in ["DGS10_interp_minus23", "DGS10_interp_minus13", "DGS10_interp_minus0"]]
for idx, lag in zip(idxs, [-23, -13, 0]):
    traces.append(go.Scatter(
                    x=dates,
                    y=x_rolling[:, idx],
                    mode='lines',
                    name="Lag {}".format(lag),
                    ))

fig=go.Figure(data=traces, layout=dict(title= "1 Year T-Bill @ various lags"))
fig.show()
```

## Output to JSON

```python
json_dict = {}
json_dict['minmax_scaler']=mms.to_json()
json_dict['df_clean']=df_scaled.to_json()
json_dict['columns']=interp_cols
json_dict['times']=times.tolist()
json_dict['x_rolling']=x_rolling.tolist()
json_dict['x_cols']=x_cols
json_dict['y_rolling']=y_rolling.tolist()
json_dict['y_cols']=y_cols
if not os.path.exists('scaled'):
    os.makedirs('scaled')
with open("./scaled/fred.json","w") as output_file:
    output_file.write(json.dumps(json_dict))
```
