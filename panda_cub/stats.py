#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import scipy.stats as stats

def t_test(df, x_rank, y_rank, value):
    """Create table of means with t-statistics on the margins."""
    _cols = [x_rank, y_rank, value]
    _df = df.ix[df[_cols].notnull().all(axis=1), _cols]

    _xs = sorted(_df[x_rank].unique())
    _ys = sorted(_df[y_rank].unique())

    _ret = (_df.groupby([x_rank, y_rank])
              .mean()
              .reset_index()
              .pivot(index=y_rank, columns=x_rank, values=value)
              )
    # Check that all intersections of x and y are non-empty
    if _ret.isnull().any().any():
        _ = _ret.ix[_ret.isnull().any(axis=1),_ret.isnull().any(axis=0)]
        raise ValueError("One of the intersections was NaN:\n\n{}"
                         .format(_.fillna('NaN')[_.isnull()].fillna('')))
    # Add difference column
    _ret.ix[_ys, 'diff'] = _ret.ix[_ys, max(_xs)] - _ret.ix[_ys, min(_xs)]
    # Add difference row
    _ret.ix['diff', _xs] = _ret.ix[max(_ys), _xs] - _ret.ix[min(_ys), _xs]

    for _x in _xs:
        sel = (_df[x_rank] == _x) & (_df[value].notnull())
        test = stats.ttest_ind(_df.ix[(_df[y_rank] == max(_ys)) & sel, value],
                               _df.ix[(_df[y_rank] == min(_ys)) & sel, value])
        _ret.ix['t-stat', _x] = test.statistic
        _ret.ix['p-value', _x] = test.pvalue

    for _y in _ys:
        sel = (_df[y_rank] == _y) & (_df[value].notnull())
        test = stats.ttest_ind(_df.ix[(_df[x_rank] == max(_xs)) & sel, value],
                               _df.ix[(_df[x_rank] == min(_xs)) & sel, value])
        _ret.ix[_y, 't-stat'] = test.statistic
        _ret.ix[_y, 'p-value'] = test.pvalue

    return _ret

# Now monkey patch pandas.
pd.DataFrame.t_test = t_test
