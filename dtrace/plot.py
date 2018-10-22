#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from scipy.stats import pearsonr


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Plot(object):
    # - DEFAULT AESTHETICS
    SNS_RC = {
        'axes.linewidth': .3,
        'xtick.major.width': .3,
        'ytick.major.width': .3,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in'
    }

    PAL_SET2 = sns.color_palette('Set2', n_colors=8).as_hex()
    PAL_DTRACE = [PAL_SET2[1], '#E1E1E1', '#656565']

    def __init__(self):
        sns.set(style='ticks', context='paper', rc=self.SNS_RC, font_scale=.75)

    def plot_corrplot(
            self, x, y, dataframe, scatter_kws=None, line_kws=None, annot_kws=None, marginal_kws=None, add_hline=True,
            add_vline=True, lowess=False
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor='w', lw=.3, s=12)

        if line_kws is None:
            line_kws = dict(lw=1., color=self.PAL_DTRACE[0])

        if annot_kws is None:
            annot_kws = dict(stat='R')

        if marginal_kws is None:
            marginal_kws = dict(kde=False, hist_kws=dict(linewidth=0))

        # Joint and Marginal plot
        g = sns.jointplot(
            x, y, data=dataframe, kind='reg', space=0, color=self.PAL_DTRACE[2], annot_kws=annot_kws,
            marginal_kws=marginal_kws, joint_kws=dict(lowess=lowess, scatter_kws=scatter_kws, line_kws=line_kws)
        )

        #
        g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', frameon=False, loc=4)

        # Extras
        if add_hline:
            g.ax_joint.axhline(0, ls='-', lw=0.1, c=self.PAL_DTRACE[1], zorder=0)

        if add_vline:
            g.ax_joint.axvline(0, ls='-', lw=0.1, c=self.PAL_DTRACE[1], zorder=0)

        # Labels
        g.set_axis_labels('{} (log2 FC)'.format(x), '{} (ln IC50)'.format(y))

        return g

    @staticmethod
    def _marginal_boxplot(a, xs=None, ys=None, zs=None, vertical=False, **kws):
        if vertical:
            ax = sns.boxplot(x=zs, y=ys, orient='v', **kws)
        else:
            ax = sns.boxplot(x=xs, y=zs, orient='h', **kws)

        ax.set_ylabel('')
        ax.set_xlabel('')

    def plot_corrplot_discrete(
            self, x, y, z, plot_df, scatter_kws=None, line_kws=None, legend_title='', discrete_pal=None, hue_order=None
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor='w', lw=.3, s=12)

        if line_kws is None:
            line_kws = dict(lw=1., color=self.PAL_DTRACE[0])

        pal = {0: self.PAL_DTRACE[2], 1: self.PAL_DTRACE[0]}

        #
        g = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

        g.plot_marginals(
            self._marginal_boxplot, palette=pal if discrete_pal is None else discrete_pal, data=plot_df, linewidth=.3,
            fliersize=1, notch=False, saturation=1.0,
            xs=x, ys=y, zs=z
        )

        sns.regplot(
            x=x, y=y, data=plot_df, color=pal[0], truncate=True, fit_reg=True, scatter_kws=scatter_kws,
            line_kws=line_kws, ax=g.ax_joint
        )
        sns.regplot(
            x=x, y=y, data=plot_df[plot_df[z] == 1], color=pal[1], truncate=True, fit_reg=False,
            scatter_kws=scatter_kws, ax=g.ax_joint
        )

        g.annotate(pearsonr, template='R={val:.2g}, p={p:.1e}', loc=4, frameon=False)

        g.ax_joint.axhline(0, ls='-', lw=0.3, c=pal[0], alpha=.2)
        g.ax_joint.axvline(0, ls='-', lw=0.3, c=pal[0], alpha=.2)

        g.set_axis_labels('{} (log2 FC)'.format(x), '{} (ln IC50)'.format(y))

        if discrete_pal is None:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label='Yes' if t else 'No') for t, c in pal.items()]
        elif hue_order is None:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=t) for t, c in discrete_pal.items()]
        else:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=discrete_pal[t], label=t) for t in hue_order]

        g.ax_marg_y.legend(handles=handles, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5),
                           frameon=False)

        plt.suptitle(z, y=1.05, fontsize=8)

        return g
