#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import scipy.stats as stats

def _listify(obj):
    if obj is None:
        return None
    if not isinstance(obj, (tuple, list, set)):
        return [obj]
    return list(obj)

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

def winsor(df, columns, p=0.01, inplace=False, prefix=None, suffix=None, verbose=False):
    """
    Winsorize columns by setting values outside of the p and 1-p percentiles
    equal to the p and 1-p percentiles respectively.
    Set inplace=True to make new column called prefix+'column'+suffix.
    Set inplace=False to return array of winsorized Series conforming to length of `columns`
    """
    new_cols = []
    for column in _listify(columns):
        new_col_name = '{}{}{}'.format(prefix or '', column, suffix or '')
        p = max(0, min(.5, p))
        low=df[column].quantile(p)
        hi=df[column].quantile(1-p)
        if pd.np.isnan(low) or pd.np.isnan(hi):
            if verbose:
                print("One of the quantiles is NAN! low: {}, high: {}"
                      .format(low,hi))
            continue
        if verbose:
            print("{}: Num < {:0.2f}: {} ({:0.3f}), num > {:0.2f}: {} ({:0.3f})"
                  .format(column, low, sum(df[column]<low),
                          sum(df[column]<low)/len(df[column]), hi,
                          sum(df[column]>hi ),
                          sum(df[column]>hi )/len(df[column])))
        if inplace:
            df[new_col_name] = df[column].copy()
            df.ix[df[new_col_name]>hi, new_col_name] = hi
            df.ix[df[new_col_name]<low, new_col_name] = low
        else:
            _tmpcol = df[column].copy()
            _tmpcol.ix[_tmpcol<low] = low
            _tmpcol.ix[_tmpcol>hi] = hi
            new_cols.append(_tmpcol)
    if inplace:
        return df
    return new_cols

def normalize(df, columns, p=0, inplace=False, prefix=None, suffix=None, verbose=False):
    """
    Normalize columns to have mean=0 and standard deviation = 1.
    Mean and StdDev. are calculated excluding the <p and >1-p percentiles.
    Set inplace=True to make new column called prefix+'column'+suffix.
    Set inplace=False to return array of winsorized Series conforming to length of `columns`
    """
    new_cols = []
    for column in _listify(columns):
        if p > 0 & p < .5:
            low=df[column].quantile(p)
            hi=df[column].quantile(1-p)
            sel = (df[column]>=low)&(df[column]<=hi)
        else:
            sel = df[column].notnull()
        _mu = df.ix[sel, column].mean()
        _rho = df.ix[sel,column].std()
        if not _rho > 0:
            raise ValueError('0 standard deviation found when normalizing '
                             '{} (mean={})'.format(column, _mu))
        new_col_name = '{}{}{}'.format(prefix or '', column, suffix or '')
        if verbose:
            print('{} = ({} - {:.2f}) / {:.2f}'
                  .format(new_col_name, column, _mu, _rho))
        if inplace:
            df[new_col_name] = (df[column] - _mu) / _rho
        else:
            new_cols.append((df[column] - _mu) / _rho)
    if inplace:
        return df
    return new_cols

def coalesce(df, *cols, no_scalar=False):
    """Fills in missing values with subsequently defined columns.
    Element-wise equivalent of: (col[0] or col[1] or ... or col[-1])
    The last provided value in *cols is assumed to be a scalar,
    and .fillna(col[-1]) is called unless `no_scalar` is set to True.
    """
    if len(cols) < 1:
        raise ValueError('must specify list of columns, got: {!r}'
                         .format(cols))
    if len(cols) == 1:
        return df[cols[0]].copy()

    _cols = list(cols) if no_scalar else list(cols[:-1])
    _return_column = df[_cols.pop(0)].copy()
    for col in _cols:
        if col in df:
            _return_column = _return_column.fillna(df[col])
    if not no_scalar:
        _return_column = _return_column.fillna(cols[-1])
    return _return_column

def get_duplicated(df, columns):
    """
    Return dataframe of all rows which match duplicated criteria.
    Differs from df[df.duplicated(cols)] in that the latter returns only the second
    occurance of the duplicated rows, this returns both.
    """
    _cols = _listify(columns) if columns else df.columns
    dups = df.ix[df.duplicated(_cols), _cols].sort_values(_cols)
    return df.merge(dups, on=_cols, how='right')

# Now monkey patch pandas.
print("Run monkey_patch_pandas() to monkey patch pandas.")
def monkey_patch_pandas():
    pd.DataFrame.t_test = t_test
    pd.DataFrame.normalize = normalize
    pd.DataFrame.winsor = winsor
    pd.DataFrame.coalesce = coalesce
    pd.DataFrame.get_duplicated = get_duplicated
    print("Added t_test, normalize, winsor, coalesce, and get_duplicated.")
