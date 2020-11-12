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
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import plotly.graph_objects as go
from lstmutil import TimeSeries
from lstmutil import Scaler
```

## Data Interpolation

Data sets will in general mismatch on both frequency and days available (e.g. financial data only available on trading days). Use step interpolation to fill in missed data and upsample as appropriate to a daily frequency. Change global varaible `BEGIN` to work with a smaller dataset.

```python
timeseries=TimeSeries(begin=datetime(1990,1,1))
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
    interp[series]=timeseries.interpolate_series(clean[series], series, target_timestamps)
    
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
json_dict = {}
json_dict['minmax_scaler']=mms.to_json()
json_dict['df_scaled']=df_scaled.to_json()

if not os.path.exists('scaled'):
    os.makedirs('scaled')
with open("./scaled/fred.json","w") as output_file:
    output_file.write(json.dumps(json_dict))
```
