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

## Todo
- Write unit testing for functions, particularly how the step-interploation
  is working in extrapolation.

```python
import os
import json
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
from scipy.interpolate import interp1d
import plotly.graph_objects as go
```

## Data Interpolation

Data sets will in general mismatch on both frequency and days available (e.g. financial data only available on trading days). Use step interpolation to fill in missed data and upsample as appropriate to a daily frequency. Change global varaible `BEGIN` to work with a smaller dataset.

```python
BEGIN=int((datetime(1990,1,1)-datetime(1970,1,1)).total_seconds())

def get_target_timestamps():
    """Return evenly space daily POSIX timestamps spanning Jan 1, 1970 to 
    current UTC. This is the "grid" over which we will interpolate.
    """
    times=[]
    curr = BEGIN
    while curr<datetime.utcnow().timestamp():
        times.append(curr)
        curr = curr + 24 * 60 * 60
    return times

def clean_ts(df_in, date_col, ts_col):
    """Clean timeseries:
    Convert dates to datetime objects, coerce the timeseries column to numeric
    format, remove NaNs.
    
    df_in - Pandas DataFrame of original data
    date_col - String, column which contains date
    ts_col - String, column which contains values
    """
    df_dates=[parser.parse(t) for t in df_in[date_col]]    
    df_in[date_col]=df_dates    
    df_in['timestamp'] = [(t - datetime(1970, 1, 1)) / timedelta(seconds=1) for t in df_dates]
    df_in[ts_col]=pd.to_numeric(df_in[ts_col], errors='coerce')
    df_in=df_in[df_in['timestamp']>=BEGIN]
    df_in=df_in[['timestamp', ts_col]].dropna().copy()
    
    return df_in

def interpolate_series(df_in, ts_col, interp_ts):
    """Interpolate time series to the requested points (interp_ts)"""
    
    interp=interp1d(df_in['timestamp'], df_in[ts_col], kind='previous', fill_value='extrapolate')
    df_interp = pd.DataFrame({
        'date' : [datetime.utcfromtimestamp(t) for t in interp_ts],
        'timestamp' : interp_ts,
        ts_col : interp(interp_ts)
    })
    
    return df_interp
```

## Main Preprocessing

```python
raw={}
clean={}
interp={}

target_timestamps = get_target_timestamps()
series_list = ["DGS1", "DGS5", "DGS10", "IPG2211A2N"]
df_merge = []
for series in series_list:
    
    # Clean & interpolate
    raw[series]=pd.read_csv("./input/"+series+".csv")
    clean[series]=clean_ts(raw[series], "DATE", series)
    interp[series]=interpolate_series(clean[series], series, target_timestamps)
    
    # Merge back with the raw data, used in sanity checks later...
    interp[series]=interp[series].merge(clean[series], on='timestamp', how='left', suffixes=('_interp', '_raw'))
    
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
# Sanity check... we should not have lost data in the merge:
for series in series_list:
    num_before=sum([not pd.isnull(t) for t in clean[series][series]])
    num_after=sum([not pd.isnull(t) for t in df_merge[series+'_raw']])    
    if not num_before==num_after:
        raise ValueError("Original data being lost, check alignment of timestamps")
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
mms=MinMaxScaler()
x_scaled=mms.fit_transform(x_unscaled)
df_scaled = df_interp.copy()
df_scaled[interp_cols]=x_scaled
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

fig=go.Figure(data=traces)
fig.show()    
```

## Output to JSON

```python
def serialize_minmax_scaler(scaler):
    """Serializes a scikit learn min/max scaler to JSON"""
    
    scaler_json=scaler.__dict__.copy()
    scaler_json['scale_']=scaler_json['scale_'].tolist()
    scaler_json['min_']=scaler_json['min_'].tolist()
    scaler_json['data_min_']=scaler_json['data_min_'].tolist()
    scaler_json['data_max_']=scaler_json['data_max_'].tolist()
    scaler_json['data_range_']=scaler_json['data_range_'].tolist()
    
    return json.dumps(scaler_json)
```

```python
json_dict = {}
json_dict['minmax_scaler']=serialize_minmax_scaler(mms)
json_dict['df_scaled']=df_scaled.to_json()

if not os.path.exists('scaled'):
    os.makedirs('scaled')
with open("./scaled/fred.json","w") as output_file:
    output_file.write(json.dumps(json_dict))
```

```python

```
