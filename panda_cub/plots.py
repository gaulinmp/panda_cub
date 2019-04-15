#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import logging

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


__logger = logging.getLogger(__name__)

def ciplot(x=None, y=None, hue=None, data=None, conf_level=.95, area_alpha=.5,
           legend=True, colors=None, markers=None, ax=None, hue_order=None, **kwargs):
    """
    Line plot of mean with confidence intervals. Like seaborn's tseries plot,
    but doesn't assume unit level observations to pivot on.
    Also doesn't bootstrap.
    ``colors`` and ``markers`` can be lists or dicts of ``{hue:color}``.
    """
    if (x is None or y is None) and data is None:
        raise AttributeError("Please input an x and y variable to plot.")
    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()
    BACKUP_HUE = '_got_no_hue_but_one_'
    # Handle different types of input data. Just DataFrame for now.
    if isinstance(data, pd.DataFrame):
        xlabel = x
        ylabel = y
        keep_cols = [x, y]
        if hue is not None:
            keep_cols.append(hue)

        data = data.loc[data[keep_cols].notnull().all(axis=1), keep_cols]

        # Condition is optional
        if hue is None:
            hue = BACKUP_HUE
            data[BACKUP_HUE] = 1
            legend = False
            # legend_name = None

        legend = True and legend
        # legend_name = hue

        _hue_order = sorted(data[hue].unique())
        n_hue = len(_hue_order)
        if hue_order is None:
            hue_order = _hue_order
        else:
            assert(len(hue_order) <= n_hue)

    else:
        raise NotImplementedError("Use a DataFrame please.")

    # Set up the color palette
    if colors is None:
        current_palette = sns.utils.get_color_cycle()
        if len(current_palette) < n_hue:
            colors = sns.color_palette("husl", n_hue)
        else:
            colors = sns.color_palette(n_colors=n_hue)
        colors = {c: colors[i] for i, c in enumerate(data[hue].unique())}
    elif isinstance(colors, dict):
        colors = {c: colors[c] for c in data[hue].unique()}
    elif isinstance(colors, list):
        colors = itertools.cycle(colors)
        colors = {c: next(colors) for c in data[hue].unique()}
    else:
        try:
            colors = sns.color_palette(colors, n_hue)
        except ValueError:
            colors = mpl.colors.colorConverter.to_rgb(colors)
            colors = [colors] * n_hue
        colors = {c: colors[i] for i, c in enumerate(data[hue].unique())}

    # Set up markers to rotate through
    if markers is None:
        markers = {c: 'o' for c in data[hue].unique()}
    elif isinstance(markers, dict):
        markers = {c: markers[c] for c in data[hue].unique()}
    else:
        markers = itertools.cycle(markers)
        markers = {c: next(markers) for c in data[hue].unique()}

    # Do a groupby with condition and plot each trace and area
#     for _h, (_hue, _huedf) in enumerate(data.groupby(hue, sort=False)):
    for _h, _thishue in enumerate(hue_order):
        _huedf = data[data[hue] == _thishue]

        label = _thishue if legend else "_nolegend_"

        _byx = _huedf.groupby(x)
        _byxmean = _byx[y].mean()
        _byxstd = _byx[y].std()
        _byxn = _byx[y].count()

        _cis = stats.norm.interval(
            conf_level, _byxmean, _byxstd / np.sqrt(_byxn))

        _x = _byxmean.index.astype(np.float)

        ax.fill_between(_x, _cis[0], _cis[1],
                        color=colors[_thishue], alpha=area_alpha)
        ax.plot(_x, _byxmean, color=colors[_thishue], marker=markers[_thishue],
                label=label, **kwargs)

    # Add the plot labels
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=0)
    return ax


# Now monkey patch pandas.
__logger.info("Run monkey_patch_seaborn() to monkey patch seaborn.")
def monkey_patch_seaborn():
    sns.ciplot = ciplot
    __logger.info("Added to seaborn (sns): ciplot ")
