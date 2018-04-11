#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_corrplot(
        x, y, dataframe, scatter_kws=None, line_kws=None, annot_kws=None, marginal_kws=None, add_hline=True, add_vline=True
):
    # - Fillin defaults
    if scatter_kws is None:
        scatter_kws = dict(edgecolor='w', lw=.3, s=12)

    if line_kws is None:
        line_kws = dict(lw=1.)

    if annot_kws is None:
        annot_kws = dict(stat='R')

    if marginal_kws is None:
        marginal_kws = dict(kde=False)

    # - Joint and Marginal plot
    g = sns.jointplot(
        x, y, data=dataframe, kind='reg', space=0, color=cdrug.PAL_SET2[8], annot_kws=annot_kws,
        marginal_kws=marginal_kws, joint_kws=dict(scatter_kws=scatter_kws, line_kws=line_kws)
    )

    # - Extras
    if add_hline:
        g.ax_joint.axhline(0, ls='-', lw=0.1, c=cdrug.PAL_SET2[7])

    if add_vline:
        g.ax_joint.axvline(0, ls='-', lw=0.1, c=cdrug.PAL_SET2[7])

    # - Labels
    g.set_axis_labels('{} (log10 FC)'.format(x), '{} (ln IC50)'.format(y))

    return g
