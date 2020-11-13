# -*- coding: utf-8 -*-
"""
Implements functions to clean up time series code as well as serialization/
JSON deserialization code for the scikit-learn MinMaxScaler.
"""
import json
from datetime import datetime, timedelta
from dateutil import parser
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

class TimeSeries:
    """Implements code to clean and interoploate time series to upsample
    to daily values and impute missing values. Interpolation uses previous
    values (step-interpolate) and extrapolation uses the last values seen.
    """

    def __init__(self, begin, end):
        """Constructor, begin is a datetime object specifying the oldest
        date to be included, end is when to stop. Round to match midnight UTC
        if hour/seconds are not zero.
        """
        begin = datetime(begin.year, begin.month, begin.day)
        self.begin_ts = int((begin-datetime(1970,1,1)).total_seconds())
        end = datetime(end.year, end.month, end.day)
        self.end_ts = int((end-datetime(1970,1,1)).total_seconds())


    def get_target_timestamps(self):
        """Return evenly space daily POSIX timestamps spanning Jan 1, 1970 to
        current UTC. This is the "grid" over which we will interpolate.
        """
        times=[]
        curr = self.begin_ts
        while curr<=self.end_ts:
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
        df_in=df_in[df_in['timestamp']<=self.end_ts]
        df_in=df_in[['timestamp', ts_col]].dropna().\
                                          sort_values(by='timestamp').\
                                          copy()
        df_in['date']=[datetime.utcfromtimestamp(t)\
                                for t in df_in['timestamp']]
        return df_in


    @classmethod
    def interp_ts(cls, df_in, ts_col, interp_ts):
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

    def from_json(self, scaler_json):
        """Deserializes a scikit learn min/max scaler"""

        json_dict=json.loads(scaler_json)

        # Basic fields
        for key in ['feature_range', 'copy',
                    'n_features_in_', 'n_samples_seen_']:
            self.__setattr__(key, json_dict[key])

        # Some fields need to be numpy arraysget
        for key in ['scale_', 'min_', 'data_min_', 'data_max_', 'data_range_']:
            self.__setattr__(key, np.array(json_dict[key]))
