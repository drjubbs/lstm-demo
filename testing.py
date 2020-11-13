# -*- coding: utf-8 -*-
"""Unit testing for LSTM demo."""

import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import lstmutil
from lstmutil import Scaler


class TestTimeSeries(unittest.TestCase):
    """Unittest for time series data"""

    def test_get_target_timestamps(self):
        """Test generator for evenly spaced times, begin to end"""

        now = datetime.utcnow()
        begin = now - timedelta(days = 3)
        ts1 = lstmutil.TimeSeries(begin, now)
        tts = ts1.get_target_timestamps()

        # We should get four timestamps - today, day-1, day-2, day-3
        self.assertEqual(len(tts), 4)

        # Times should be rounded to midnight
        times = [datetime.utcfromtimestamp(t) for t in tts]
        self.assertTrue(all([t.hour==0 for t in times]))
        self.assertTrue(all([t.minute==0 for t in times]))
        self.assertTrue(all([t.second==0 for t in times]))


class TestScaler(unittest.TestCase):
    """Test serialization and de-serialization of scikit-learn min/max scaler"""

    def test_scaler(self):
        """Test serialization and de-serialization of scikit-learn min/max
        scaler. Use a random numpy array.
        """

        # Generate random data and normalize
        mms = Scaler()
        array1 = np.random.rand(50, 3)
        array1[:,0] = 2 * array1[:,0]
        array1[:,1] = 4 * array1[:,1]
        array1[:,2] = 8 * array1[:,2]

        array1_scaled = mms.fit_transform(array1)
        txt = mms.to_json()

        # Deserialize scaler 1 to scaler 2 and unscale
        mms2 = Scaler()
        mms2.from_json(txt)
        array2 = mms2.inverse_transform(array1_scaled)
        self.assertTrue(np.all(np.isclose(array1, array2)))

    def test_clean_ts(self):
        """Test time series cleaning function"""

        df_raw = pd.DataFrame({
            'dates' : ['2020.11.03',
                       '11/6/2020',
                       '2020-11-9 1:30PM',
                       '11/10/2020 12:00AM',
                       '11/13/2020 2:00PM',
                       '11/21/2020',
                       ],
            'junk' : ["A", "B", "C", "D", "E", "F"],
        })

        df_raw['values']=[160.25, 150.5, 'foo', 140, 145, 130]

        ts1=lstmutil.TimeSeries(begin=datetime(2020, 11, 5),
                                end=datetime(2020, 11, 23)
        )

        # Two outer timestamps should be reject, and the non-numeric
        # value should be dropped.
        df_clean1 = ts1.clean_ts(df_raw, 'dates', 'values')
        self.assertEqual(len(df_clean1), 4)


        # Check interpolate within and beyond region
        df_interp1 = ts1.interp_ts(df_clean1,
                                 'values',
                                 ts1.get_target_timestamps())

        self.assertEqual(df_interp1['values'].values[0], 150.5)
        self.assertEqual(df_interp1['values'].values[-1], 130.0)
        mask=df_interp1['date']=='2020-11-11'
        self.assertEqual(df_interp1[mask]['values'].values[0], 140.0)

        # Make sure we didn't lose good data
        df_merge1=df_interp1.merge(df_clean1,
                                   on='date',
                                   suffixes=['_i', '_c'],
                                   how='left')

        num_before=sum([not pd.isnull(t) for t in df_clean1['values']])
        num_after=sum([not pd.isnull(t) for t in df_merge1['values_c']])
        self.assertTrue(num_before, num_after)


if __name__ == "__main__":
    unittest.main()
