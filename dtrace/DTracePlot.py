#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from crispy import CrispyPlot
from scipy.stats import pearsonr


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class DTracePlot(CrispyPlot):
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

    PAL_DTRACE = [PAL_SET2[1], '#E1E1E1', '#656565', '#2b8cbe']

    BOXPROPS = dict(linewidth=1.)
    WHISKERPROPS = dict(linewidth=1.)
    MEDIANPROPS = dict(linestyle='-', linewidth=1., color=PAL_DTRACE[0])
    FLIERPROPS = dict(
        marker='o', markerfacecolor='black', markersize=2., linestyle='none', markeredgecolor='none', alpha=.6
    )

    MARKERS = dict(Sanger='o', Broad='X')

    def __init__(self):
        sns.set(style='ticks', context='paper', rc=self.SNS_RC, font_scale=.75)

    @classmethod
    def plot_corrplot(
            cls, x, y, style, dataframe, add_hline=True, add_vline=True, annot_text=None, lowess=False
    ):
        grid = sns.JointGrid(x, y, data=dataframe, space=0)

        # Joint
        for t, df in dataframe.groupby(style):
            grid.ax_joint.scatter(
                x=df[x], y=df[y], edgecolor='w', lw=.1, s=5, color=cls.PAL_DTRACE[2], marker=cls.MARKERS[t], label=t,
                alpha=.8
            )

        grid.plot_joint(
            sns.regplot, data=dataframe, line_kws=dict(lw=1., color=cls.PAL_DTRACE[0]), marker='', lowess=lowess,
            truncate=True
        )

        # Annotation
        if annot_text == '':
            cor, pval = pearsonr(dataframe[x], dataframe[y])
            annot_text = f'R={cor:.2g}, p={pval:.1e}'

        grid.ax_joint.text(.95, .05, annot_text, fontsize=5, transform=grid.ax_joint.transAxes, ha='right')

        # Marginals
        grid.plot_marginals(sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=cls.PAL_DTRACE[2])

        # Extra
        if add_hline:
            grid.ax_joint.axhline(0, ls='-', lw=0.1, c=cls.PAL_DTRACE[1], zorder=0)

        if add_vline:
            grid.ax_joint.axvline(0, ls='-', lw=0.1, c=cls.PAL_DTRACE[1], zorder=0)

        grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        return grid

    @staticmethod
    def _marginal_boxplot(a, xs=None, ys=None, zs=None, vertical=False, **kws):
        if vertical:
            ax = sns.boxplot(x=zs, y=ys, orient='v', **kws)
        else:
            ax = sns.boxplot(x=xs, y=zs, orient='h', **kws)

        ax.set_ylabel('')
        ax.set_xlabel('')

    @classmethod
    def plot_corrplot_discrete(
            cls, x, y, z, style, plot_df, scatter_kws=None, line_kws=None, legend_title='', discrete_pal=None,
            hue_order=None, annot_text=None
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor='w', lw=.3, s=12)

        if line_kws is None:
            line_kws = dict(lw=1., color=cls.PAL_DTRACE[0])

        pal = {0: cls.PAL_DTRACE[2], 1: cls.PAL_DTRACE[0]}

        #
        grid = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

        grid.plot_marginals(
            cls._marginal_boxplot, palette=pal if discrete_pal is None else discrete_pal, data=plot_df, linewidth=.3,
            fliersize=1, notch=False, saturation=1.0, xs=x, ys=y, zs=z, showcaps=False, boxprops=cls.BOXPROPS,
            whiskerprops=cls.WHISKERPROPS, flierprops=cls.FLIERPROPS, medianprops=dict(linestyle='-', linewidth=1.)
        )

        sns.regplot(
            x=x, y=y, data=plot_df, color=pal[0], truncate=True, fit_reg=True, scatter=False, line_kws=line_kws,
            ax=grid.ax_joint
        )

        for feature in [0, 1]:
            for t, df in plot_df[plot_df[z] == feature].groupby(style):
                sns.regplot(
                    x=x, y=y, data=df, color=pal[feature], fit_reg=False, scatter_kws=scatter_kws, label=t if feature == 0 else None,
                    marker=cls.MARKERS[t], ax=grid.ax_joint
                )

        grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        # Annotation
        if annot_text is None:
            cor, pval = pearsonr(plot_df[x], plot_df[y])
            annot_text = f'R={cor:.2g}, p={pval:.1e}'

        grid.ax_joint.text(.95, .05, annot_text, fontsize=5, transform=grid.ax_joint.transAxes, ha='right')

        grid.ax_joint.axhline(0, ls='-', lw=0.3, c=pal[0], alpha=.2)
        grid.ax_joint.axvline(0, ls='-', lw=0.3, c=pal[0], alpha=.2)

        grid.set_axis_labels('{} (log2 FC)'.format(x), '{} (ln IC50)'.format(y))

        if discrete_pal is None:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label='Yes' if t else 'No') for t, c in pal.items()]

        elif hue_order is None:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=c, label=t) for t, c in discrete_pal.items()]

        else:
            handles = [mpatches.Circle([.0, .0], .25, facecolor=discrete_pal[t], label=t) for t in hue_order]

        grid.ax_marg_y.legend(
            handles=handles, title=legend_title, loc='center left', bbox_to_anchor=(1, 0.5), frameon=False
        )

        return grid

    def plot_multiple(self, x, y, dataframe, order=None, ax=None, notch=False, n_offset=1.15, n_fontsize=3.5):
        if ax is None:
            ax = plt.gca()

        if order is None:
            order = list(dataframe.groupby(y)[x].mean().sort_values(ascending=False).index)

        dataframe = dataframe.dropna(subset=[x, y])

        pal = pd.Series(self.get_palette_continuous(len(order), self.PAL_DTRACE[2]), index=order)

        sns.boxplot(
            x=x, y=y, data=dataframe, orient='h', palette=pal, saturation=1., showcaps=False,
            order=order, notch=notch, flierprops=self.FLIERPROPS, ax=ax
        )

        #
        text_x = max(dataframe[x]) * n_offset

        for i, c in enumerate(order):
            n = np.sum(dataframe[y] == c)
            ax.text(text_x, i, f'N={n}', ha='left', va='center', fontsize=n_fontsize)

        x_lim = ax.get_xlim()
        ax.set_xlim(x_lim[0], text_x)

        return ax
