# -*- coding: utf-8 -*-
"""
Implements functions to clean up time series code as well as serialization/
JSON deserialization code for the scikit-learn MinMaxScaler.
"""
import json
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

class TimeSeries:
    """Implements code to clean and interoploate time series to upsample
    to daily values and impute missing values. Interpolation uses previous
    values (step-interpolate) and extrapolation uses the last values seen.
    """

    def __init__(self, begin):
        """Constructor, begin is a datetime object specifying the oldest
        date to be included.
        """
        self.begin_ts = int((begin-datetime(1970,1,1)).total_seconds())


    def get_target_timestamps(self):
        """Return evenly space daily POSIX timestamps spanning Jan 1, 1970 to
        current UTC. This is the "grid" over which we will interpolate.
        """
        times=[]
        curr = self.begin_ts
        while curr<datetime.utcnow().timestamp():
            times.append(curr)
            curr = curr + 24 * 60 * 60
        return times


    def clean_ts(self, df_in, date_col, ts_col):
        """Clean timeserie -- convert dates to datetime objects, coerce
        the timeseries column to numeric format, remove NaNs.

        df_in:      Pandas DataFrame of original data
        date_col:   String, column which contains date
        ts_col:     String, column which contains values
        """
        df_dates=[parser.parse(t) for t in df_in[date_col]]
        df_in[date_col]=df_dates
        df_in['timestamp'] = [(t - datetime(1970, 1, 1)) / \
                                timedelta(seconds=1) for t in df_dates]
        df_in[ts_col]=pd.to_numeric(df_in[ts_col], errors='coerce')
        df_in=df_in[df_in['timestamp']>=self.begin_ts]
        df_in=df_in[['timestamp', ts_col]].dropna().copy()

        return df_in


    def interpolate_series(self, df_in, ts_col, interp_ts):
        """Interpolate time series to the requested points (interp_ts)"""

        interp=interp1d(df_in['timestamp'],
                        df_in[ts_col],
                        kind='previous',
                        fill_value='extrapolate')
        df_interp = pd.DataFrame({
            'date' : [datetime.utcfromtimestamp(t) for t in interp_ts],
            'timestamp' : interp_ts,
            ts_col : interp(interp_ts)
        })

        return df_interp

class Scaler(MinMaxScaler):
    """Extend class to support JSON serialization."""
    def to_json(self):
        """Serializes a scikit learn min/max scaler to JSON"""
        scaler_json=self.__dict__.copy()
        scaler_json['scale_']=scaler_json['scale_'].tolist()
        scaler_json['min_']=scaler_json['min_'].tolist()
        scaler_json['data_min_']=scaler_json['data_min_'].tolist()
        scaler_json['data_max_']=scaler_json['data_max_'].tolist()
        scaler_json['data_range_']=scaler_json['data_range_'].tolist()

        return json.dumps(scaler_json)

"""
# Sanity check... we should not have lost data in the merge:
for series in series_list:
    num_before=sum([not pd.isnull(t) for t in clean[series][series]])
    num_after=sum([not pd.isnull(t) for t in df_merge[series+'_raw']])    
    if not num_before==num_after:
        raise ValueError("Original data being lost, check alignment of timestamps")
"""
def main():
    """Temporary unit testing for functions..."""
    pass

if __name__ == "__main__":
    main()
