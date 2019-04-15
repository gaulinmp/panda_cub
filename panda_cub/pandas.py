#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import pandas as pd
import scipy.stats as stats
try:
    import statsmodels.api as sm
except ImportError:
    sm = None

__logger = logging.getLogger(__name__)

def _listify(obj):
    if obj is None:
        return None
    if not isinstance(obj, (tuple, list, set)):
        return [obj]
    return list(obj)


def two_way_t_test(df, x_rank, y_rank, value):
    """
    Create table of means by two rank variables, with differences
    [max(x/y_rank)-min(x/y_rank)] and T-/P-values.
    If `statsmodel` package is available, use that to calculate
    diff-in-diff estimator.

    Example: df.two_way_t_test('age_decile', 'size_decile', 'ROA')

    Args:
        x_rank: Variable for separating groups in X-dimension
        y_rank: Variable for separating groups in Y-dimension
        value: Variable to t-test across groups.
    Returns:
        Data frame of the results, with:
        Columns: X-rank unique values, and diff (+ t-stat/p-value) across min/max x-rank groups.
        Rows: Y-rank unique values, and diff (+ t-stat/p-value) across min/max y-rank groups.
    Raises:
        ValueError:
    """
    _cols = [x_rank, y_rank, value]
    _df = df.loc[df[_cols].notnull().all(axis=1), _cols]

    _xs = sorted(_df[x_rank].unique())
    _ys = sorted(_df[y_rank].unique())

    _ret = (_df.groupby([x_rank, y_rank])
            .mean()
            .reset_index()
            .pivot(index=y_rank, columns=x_rank, values=value)
            )
    # Check that all intersections of x and y are non-empty
    if _ret.isnull().any().any():
        _ = _ret.loc[_ret.isnull().any(axis=1), _ret.isnull().any(axis=0)]
        raise ValueError("One of the intersections was NaN:\n\n{}"
                         .format(_.fillna('NaN')[_.isnull()].fillna('')))
    # Add difference column
    _ret.loc[_ys, 'diff'] = _ret.loc[_ys, max(_xs)] - _ret.loc[_ys, min(_xs)]
    # Add difference row
    _ret.loc['diff', _xs] = _ret.loc[max(_ys), _xs] - _ret.loc[min(_ys), _xs]

    for _x in _xs:  # Iterate across X-values
        sel = (_df[x_rank] == _x) & (_df[value].notnull())
        test = stats.ttest_ind(_df.loc[(_df[y_rank] == max(_ys)) & sel, value],
                               _df.loc[(_df[y_rank] == min(_ys)) & sel, value])
        _ret.loc['t-stat', _x] = test.statistic
        _ret.loc['p-value', _x] = test.pvalue

    for _y in _ys:  # Iterate across Y-values
        sel = (_df[y_rank] == _y) & (_df[value].notnull())
        test = stats.ttest_ind(_df.loc[(_df[x_rank] == max(_xs)) & sel, value],
                               _df.loc[(_df[x_rank] == min(_xs)) & sel, value])
        _ret.loc[_y, 't-stat'] = test.statistic
        _ret.loc[_y, 'p-value'] = test.pvalue

    # diff in diff estimator
    if sm is not None:
        _df[x_rank+'_max'] = (_df[x_rank] == max(_xs))+0
        _df[x_rank+'_min'] = (_df[x_rank] == min(_xs))+0
        _df[y_rank+'_max'] = (_df[y_rank] == max(_ys))+0
        _df[y_rank+'_min'] = (_df[y_rank] == min(_ys))+0

        # Diff-in-diff estimation is the interaction term in
        # value = intercept + Y_max + X_max + Y_max*X_max
        dind_name = '_rank_interaction__'
        _df[dind_name] = _df[x_rank+'_max'] * _df[y_rank+'_max']

        dd_axes = _df.columns[-5:]

        sel = ( (_df[dd_axes[0:2]].sum(axis=1) > 0)
              & (_df[dd_axes[2:4]].sum(axis=1) > 0)
              &  _df[value].notnull() )

        # [0::2] grabs maxes and interaction
        y, x = _df.loc[sel, value], _df.loc[sel, dd_axes[0::2]]
        try:
            # cov_type='HC1' uses the robust sandwich estimator
            fit = sm.OLS(y, sm.add_constant(x)).fit(cov_type='HC1')
            _ret.loc['diff', 'diff'] = fit.params[dind_name]
            _ret.loc['t-stat', 't-stat'] = fit.tvalues[dind_name]
            _ret.loc['p-value', 'p-value'] = fit.pvalues[dind_name]
        except:
            # Must not have had statsmodels
            pass

    return _ret.fillna('')


def winsor(df, columns, p=0.01, inplace=False, prefix=None, suffix=None):
    """
    Winsorize columns by setting values outside of the p and 1-p percentiles
    equal to the p and 1-p percentiles respectively.
    Set inplace=True to make new column called prefix+'column'+suffix.
    Set inplace=False to return array of winsorized Series conforming to length of `columns`
    """
    new_cols = []
    for column in _listify(columns):
        new_col_name = '{}{}{}'.format(prefix or '', column, suffix or '')
        if column not in df:
            if inplace:
                __logger.warning("Column %s not found in df.", column)
                continue
            else:
                __logger.warning("Column %s not found in df.", column)
                raise KeyError("Column {} not found in data frame".format(column))
        p = max(0, min(.5, p))
        low=df[column].quantile(p)
        hi=df[column].quantile(1-p)
        if pd.np.isnan(low) or pd.np.isnan(hi):
            __logger.warning("One of the quantiles is NAN! low: {}, high: {}"
                             .format(low, hi))
            continue

        __logger.info("{}: Num < {:0.2f}: {} ({:0.3f}), num > {:0.2f}: {} ({:0.3f})"
                      .format(column, low, sum(df[column]<low),
                              sum(df[column]<low)/len(df[column]), hi,
                              sum(df[column]>hi ),
                              sum(df[column]>hi )/len(df[column])))

        if inplace:
            df[new_col_name] = df[column].copy()
            df.loc[df[new_col_name]>hi, new_col_name] = hi
            df.loc[df[new_col_name]<low, new_col_name] = low
        else:
            _tmpcol = df[column].copy()
            _tmpcol.loc[_tmpcol<low] = low
            _tmpcol.loc[_tmpcol>hi] = hi
            new_cols.append(_tmpcol)
    if inplace:
        return df
    return new_cols

def normalize(df, columns, p=0, inplace=False, prefix=None, suffix=None):
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
        _mu = df.loc[sel, column].mean()
        _rho = df.loc[sel,column].std()
        if not _rho > 0:
            raise ValueError('0 standard deviation found when normalizing '
                             '{} (mean={})'.format(column, _mu))
        new_col_name = '{}{}{}'.format(prefix or '', column, suffix or '')

        __logger.info('{} = ({} - {:.2f}) / {:.2f}'
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
    Returns `df[df.duplicated(cols, keep=False)]` and a sort on `cols`.
    """
    _cols = _listify(columns) if columns else df.columns
    return df[df.duplicated(_cols, keep=False)].sort_values(_cols)

def value_counts_full(series, normalize=False, sort=True, cumulative=True, **kwargs):
    """
    Series.value_counts() gives a series with the counts OR frequencies (normalize=True),
    but doesn't show both. Also doesn't show the cumulative frequency.
    This method provides that in a pretty little table (DataFrame).

    Monkey-patch onto pandas with pd.Series.value_counts_full = value_counts_full to
    be able to call it like: ``df.column_to_count.value_counts_full()`` just like you
    would the normal ``Series.value_counts()``.

    """
    _v = series.value_counts(normalize=False, **kwargs)
    _p = series.value_counts(normalize=True, **kwargs)*100
    _ret = pd.merge(_v, _p, left_index=True,
                    right_index=True, suffixes=('', ' %'))

    # Some cosmetics
    _ret.columns = ('Count', 'Percent')
    _ret.index.name = series.name

    # sort=False doesn't seem to work as expected with dropna=False,
    # so just force the index sort.
    if not sort:
        _ret.sort_index(inplace=True)

    if cumulative:
        _ret['Cumulative'] = _ret['Percent'].cumsum()

    return _ret

# Now monkey patch pandas.
__logger.info("Run monkey_patch_pandas() to monkey patch pandas.")
def monkey_patch_pandas():
    pd.DataFrame.two_way_t_test = two_way_t_test
    pd.DataFrame.normalize = normalize
    pd.DataFrame.winsor = winsor
    pd.DataFrame.coalesce = coalesce
    pd.DataFrame.get_duplicated = get_duplicated
    pd.Series.value_counts_full = value_counts_full
    __logger.info("Added to DataFrame: two_way_t_test, normalize, winsor, coalesce, and get_duplicated.")
    __logger.info("Added to Series: value_counts_full.")
