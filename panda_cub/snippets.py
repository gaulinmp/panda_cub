#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

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