#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import datetime as dt
from io import BytesIO
from zipfile import ZipFile
import requests
import pandas as pd

def download_ffind_zip(ind_num):
    """
    Download SIC code file from Ken French's website and return the
    text file of the requested FF industry number `ind_num`.
    """
    zip_url = ('http://mba.tuck.dartmouth.edu/pages/faculty/'
               'ken.french/ftp/Siccodes{}.zip'.format(ind_num))

    data = requests.get(zip_url)
    zipfile = ZipFile(BytesIO(data.content))
    return zipfile.open('Siccodes{}.txt'.format(ind_num)).read().decode()

def get_ffind_df(ind_num):
    """
    Download Fama French industry classification and return DataFrame in the form

    ``sic(int),ff#(int),ff#_name(string),detail(string)``

    For merging into a dataframe by SIC code. Example:

    ``df = df.merge(get_ffind_df(12), on='sic', how='left')``
    """
    if ind_num not in [5, 10, 12, 17, 30, 38, 48, 49]:
        raise ValueError('Industry number must be one of {} not {}.'
                         .format([5, 10, 12, 17, 30, 38, 48, 49], ind_num))

    re_nameline = re.compile((r'^\s*(?P<ff{0}>\d\d?)\s+'
                              r'(?P<ff{0}_name>[a-z]+)\s+'
                              r'(?P<detail>.+)\s*$')
                             .format(ind_num), re.I|re.M)
    re_rangeline = re.compile(r'^\s*(?P<sicfrom>\d{3,4})-(?P<sicto>\d{3,4})'
                              r'(?P<notes>\s+.+)?\s*$', re.I|re.M)
    data = download_ffind_zip(ind_num)
    # init SIC to 'other'
    try:
        current_ind = [_.groupdict() for _ in re_nameline.finditer(data)
                       if _.group('ff{0}_name'.format(ind_num)).lower() == 'other'][0]
    except IndexError:
        current_ind = {'ff{0}'.format(ind_num):ind_num,
                       'ff{0}_name'.format(ind_num):'Other',
                       'detail':''}
    vals = {i:current_ind for i in range(10000)}
    for line in data.split('\n'):
        match = re_nameline.search(line.strip())
        if match:
            current_ind = match.groupdict()
            continue
        match = re_rangeline.search(line.strip())
        if not match:
            continue
        match = match.groupdict()
        sicfrom,sicto = int(match['sicfrom']), int(match['sicto'])
        for i in range(sicfrom, sicto+1):
            vals[i] = current_ind
    df = pd.DataFrame.from_dict(vals, orient='index')
    df.index.name = 'sic'
    df['ff{0}'.format(ind_num)] = df['ff{0}'.format(ind_num)].astype(int)
    return df.reset_index()
