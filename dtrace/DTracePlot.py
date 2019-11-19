#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from crispy import CrispyPlot
from scipy.stats import pearsonr
from matplotlib.colors import LinearSegmentedColormap


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
        "axes.linewidth": 0.3,
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }

    PAL_SET2 = sns.color_palette("Set2", n_colors=8).as_hex()
    PAL_DTRACE = [PAL_SET2[1], "#E1E1E1", "#656565", "#2b8cbe", "#de2d26"]
    PAL_YES_NO = dict(Yes=PAL_DTRACE[0], No=PAL_DTRACE[2])
    PAL_1_0 = {1: PAL_DTRACE[0], 0: PAL_DTRACE[2]}

    CMAP_DTRACE = cmap = LinearSegmentedColormap.from_list(
        name="DTraceCMAP", colors=[PAL_DTRACE[0], "0.9", PAL_DTRACE[1]], N=33
    )

    BOXPROPS = dict(linewidth=.3)
    WHISKERPROPS = dict(linewidth=.3)
    MEDIANPROPS = dict(linestyle="-", linewidth=.3, color="black")
    FLIERPROPS = dict(
        marker="o",
        markerfacecolor="black",
        markersize=2.0,
        linestyle="none",
        markeredgecolor="none",
        alpha=0.6,
    )

    MARKERS = dict(Sanger="o", Broad="X")

    @classmethod
    def plot_corrplot(
        cls,
        x,
        y,
        style,
        dataframe,
        diag_line=False,
        annot_text=None,
        lowess=False,
        fit_reg=True,
    ):
        grid = sns.JointGrid(x, y, data=dataframe, space=0)

        # Joint
        for t, df in dataframe.groupby(style):
            grid.ax_joint.scatter(
                x=df[x],
                y=df[y],
                edgecolor="w",
                lw=0.05,
                s=10,
                color=cls.PAL_DTRACE[2],
                marker=cls.MARKERS[t],
                label=t,
                alpha=0.8,
            )

        if fit_reg:
            grid.plot_joint(
                sns.regplot,
                data=dataframe,
                line_kws=dict(lw=1.0, color=cls.PAL_DTRACE[0]),
                marker="",
                lowess=lowess,
                truncate=True,
            )

        # Annotation
        if annot_text == "":
            cor, pval = pearsonr(dataframe[x], dataframe[y])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"

        grid.ax_joint.text(
            0.95,
            0.05,
            annot_text,
            fontsize=4,
            transform=grid.ax_joint.transAxes,
            ha="right",
        )

        # Marginals
        grid.plot_marginals(
            sns.distplot, kde=False, hist_kws=dict(linewidth=0), color=cls.PAL_DTRACE[2]
        )

        # Extra
        grid.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        if diag_line:
            (x0, x1), (y0, y1) = grid.ax_joint.get_xlim(), grid.ax_joint.get_ylim()
            lims = [max(x0, y0), min(x1, y1)]
            grid.ax_joint.plot(
                lims, lims, ls="--", lw=0.3, zorder=0, c=cls.PAL_DTRACE[1]
            )

        grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        return grid

    @staticmethod
    def _marginal_boxplot(a, xs=None, ys=None, zs=None, vertical=False, **kws):
        if vertical:
            ax = sns.boxplot(x=zs, y=ys, orient="v", **kws)
        else:
            ax = sns.boxplot(x=xs, y=zs, orient="h", **kws)

        ax.set_ylabel("")
        ax.set_xlabel("")

    @classmethod
    def plot_corrplot_discrete(
        cls,
        x,
        y,
        z,
        style,
        plot_df,
        scatter_kws=None,
        line_kws=None,
        legend_title="",
        discrete_pal=None,
        hue_order=None,
        annot_text=None,
    ):
        # Defaults
        if scatter_kws is None:
            scatter_kws = dict(edgecolor="w", lw=0.3, s=12)

        if line_kws is None:
            line_kws = dict(lw=1.0, color=cls.PAL_DTRACE[0])

        #
        grid = sns.JointGrid(x, y, plot_df, space=0, ratio=8)

        grid.plot_marginals(
            cls._marginal_boxplot,
            palette=cls.PAL_1_0 if discrete_pal is None else discrete_pal,
            data=plot_df,
            linewidth=0.3,
            fliersize=1,
            notch=False,
            saturation=1.0,
            xs=x,
            ys=y,
            zs=z,
            showcaps=False,
            boxprops=cls.BOXPROPS,
            whiskerprops=cls.WHISKERPROPS,
            flierprops=cls.FLIERPROPS,
            medianprops=cls.MEDIANPROPS,
        )

        sns.regplot(
            x=x,
            y=y,
            data=plot_df,
            color=cls.PAL_1_0[0],
            truncate=True,
            fit_reg=True,
            scatter=False,
            line_kws=line_kws,
            ax=grid.ax_joint,
        )

        for feature in [0, 1]:
            for t, df in plot_df[plot_df[z] == feature].groupby(style):
                sns.regplot(
                    x=x,
                    y=y,
                    data=df,
                    color=cls.PAL_1_0[feature],
                    fit_reg=False,
                    scatter_kws=scatter_kws,
                    label=t if feature == 0 else None,
                    marker=cls.MARKERS[t],
                    ax=grid.ax_joint,
                )

        grid.ax_joint.legend(prop=dict(size=4), frameon=False, loc=2)

        # Annotation
        if annot_text is None:
            cor, pval = pearsonr(plot_df[x], plot_df[y])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"

        grid.ax_joint.text(
            0.95,
            0.05,
            annot_text,
            fontsize=4,
            transform=grid.ax_joint.transAxes,
            ha="right",
        )

        grid.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        grid.set_axis_labels("{} (log2 FC)".format(x), "{} (ln IC50)".format(y))

        if discrete_pal is None:
            handles = [
                mpatches.Circle(
                    [0.0, 0.0], 0.25, facecolor=c, label="Yes" if t else "No"
                )
                for t, c in cls.PAL_1_0.items()
            ]

        elif hue_order is None:
            handles = [
                mpatches.Circle([0.0, 0.0], 0.25, facecolor=c, label=t)
                for t, c in discrete_pal.items()
            ]

        else:
            handles = [
                mpatches.Circle([0.0, 0.0], 0.25, facecolor=discrete_pal[t], label=t)
                for t in hue_order
            ]

        grid.ax_marg_y.legend(
            handles=handles,
            title=legend_title,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
        )

        return grid

    @classmethod
    def plot_multiple(
        cls,
        x,
        y,
        dataframe,
        order=None,
        ax=None,
        notch=False,
        n_offset=1.15,
        n_fontsize=3.5,
    ):
        if ax is None:
            ax = plt.gca()

        if order is None:
            order = list(
                dataframe.groupby(y)[x].mean().sort_values(ascending=False).index
            )

        dataframe = dataframe.dropna(subset=[x, y])

        pal = pd.Series(
            cls.get_palette_continuous(len(order), cls.PAL_DTRACE[2]), index=order
        )

        sns.boxplot(
            x=x,
            y=y,
            data=dataframe,
            orient="h",
            palette=pal,
            saturation=1.0,
            showcaps=False,
            order=order,
            notch=notch,
            flierprops=cls.FLIERPROPS,
            ax=ax,
        )

        #
        text_x = max(dataframe[x]) * n_offset

        for i, c in enumerate(order):
            n = np.sum(dataframe[y] == c)
            ax.text(text_x, i, f"N={n}", ha="left", va="center", fontsize=n_fontsize)

        return ax

    @classmethod
    def plot_boxplot_discrete(cls, x, y, df, pal=None, notch=False, ax=None):

        if ax is None:
            ax = plt.gca()

        sns.boxplot(
            x=x,
            y=y,
            palette=cls.PAL_1_0 if pal is None else pal,
            data=df,
            linewidth=0.1,
            fliersize=1,
            notch=notch,
            saturation=1.0,
            showcaps=False,
            boxprops=cls.BOXPROPS,
            whiskerprops=cls.WHISKERPROPS,
            flierprops=cls.FLIERPROPS,
            medianprops=cls.MEDIANPROPS,
            ax=ax,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        return ax
