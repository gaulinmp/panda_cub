#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import datetime as dt
import pandas as pd

SAS_ZERO = dt.datetime(1960,1,1)
TD_DAY = pd.Timedelta(days=1)
TD_YEAR = pd.Timedelta(days=1) * 365

def sas_date_to_datetime(df_col):
    """
    Convert dates from SAS to datetime objects based on:
    SAS date (as Timedelta in days) + 1960-01-01
    """
    return pd.to_timedelta(df_col, unit='d') + SAS_ZERO


def qtr_offset(qtr_string, delta=-1):
    """
    Takes in quarter string (2005Q1) and outputs quarter
    string offset by ``delta`` quarters.
    """
    old_y, old_q = map(int, qtr_string.split('Q'))
    old_q -= 1
    new_q = (old_q + delta) % 4 + 1
    if new_q == 0:
        new_q = 4
    new_y = old_y + (old_q + delta)//4
    return '{:.0f}Q{:d}'.format(new_y, new_q)
