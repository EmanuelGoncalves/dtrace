#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from DTracePlot import DTracePlot


class Preliminary(DTracePlot):
    HIST_KDE_KWS = dict(cumulative=True, cut=0)

    @classmethod
    def _pairplot_fix_labels(cls, g, pca, by):
        for i, ax in enumerate(g.axes):
            vexp = pca[by]["vex"]["PC{}".format(i + 1)]
            ax[0].set_ylabel("PC{} ({:.1f}%)".format(i + 1, vexp * 100))

        for i, ax in enumerate(g.axes[2]):
            vexp = pca[by]["vex"]["PC{}".format(i + 1)]
            ax.set_xlabel("PC{} ({:.1f}%)".format(i + 1, vexp * 100))

    @classmethod
    def pairplot_pca_by_rows(cls, pca, hue="VERSION"):
        df = pca["row"]["pcs"].reset_index()

        pal = None if hue is None else dict(v17=cls.PAL_DTRACE[2], RS=cls.PAL_DTRACE[0])
        color = cls.PAL_DTRACE[2] if hue is None else None

        g = sns.PairGrid(
            df,
            vars=["PC1", "PC2", "PC3"],
            despine=False,
            size=1.5,
            hue=hue,
            palette=pal,
        )

        g = g.map_diag(plt.hist, color=color, linewidth=0, alpha=0.5)

        g = g.map_offdiag(
            plt.scatter, s=8, edgecolor="white", lw=0.1, alpha=0.8, color=color
        )

        if hue is not None:
            g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by="row")

    @classmethod
    def pairplot_pca_by_columns(cls, pca, hue=None, hue_vars=None):
        df = pca["column"]["pcs"]

        if hue_vars is not None:
            df = pd.concat([df, hue_vars], axis=1, sort=False).dropna()

        pal = (
            None
            if hue is None
            else dict(Broad=cls.PAL_DTRACE[2], Sanger=cls.PAL_DTRACE[0])
        )
        color = cls.PAL_DTRACE[2] if hue is None else None

        g = sns.PairGrid(
            df,
            vars=["PC1", "PC2", "PC3"],
            despine=False,
            size=1.5,
            hue=hue,
            palette=pal,
        )

        g = g.map_diag(plt.hist, color=color, linewidth=0, alpha=0.5)

        g = g.map_offdiag(
            plt.scatter, s=8, edgecolor="white", lw=0.1, alpha=0.8, color=color
        )

        if hue is not None:
            g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by="column")

    @classmethod
    def pairplot_pca_samples_cancertype(cls, pca, cancertypes, min_cell_lines=20):
        # Build data-frame
        df = pd.concat(
            [pca["column"]["pcs"], cancertypes.rename("Cancer Type")],
            axis=1,
            sort=False,
        ).dropna()

        # Order
        order = df["Cancer Type"].value_counts()
        df = df.replace(
            {"Cancer Type": {i: "Other" for i in order[order < min_cell_lines].index}}
        )

        order = ["Other"] + list(order[order >= min_cell_lines].index)
        pal = [cls.PAL_DTRACE[1]] + sns.color_palette(
            "tab20", n_colors=len(order) - 1
        ).as_hex()
        pal = dict(zip(*(order, pal)))

        # Plot
        g = sns.PairGrid(
            df,
            vars=["PC1", "PC2", "PC3"],
            despine=False,
            size=1.5,
            hue="Cancer Type",
            palette=pal,
            hue_order=order,
        )

        g = g.map_diag(sns.distplot, hist=False)
        g = g.map_offdiag(plt.scatter, s=8, edgecolor="white", lw=0.1, alpha=0.8)
        g = g.add_legend()

        cls._pairplot_fix_labels(g, pca, by="column")

    @classmethod
    def corrplot_pcs_growth(cls, pca, growth, pc):
        df = pd.concat([pca["column"]["pcs"], growth], axis=1, sort=False).dropna()

        annot_kws = dict(stat="R")
        marginal_kws = dict(kde=False, hist_kws={"linewidth": 0})

        line_kws = dict(lw=1.0, color=cls.PAL_DTRACE[0], alpha=1.0)
        scatter_kws = dict(edgecolor="w", lw=0.3, s=10, alpha=0.6)
        joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

        g = sns.jointplot(
            pc,
            "growth",
            data=df,
            kind="reg",
            space=0,
            color=cls.PAL_DTRACE[2],
            marginal_kws=marginal_kws,
            annot_kws=annot_kws,
            joint_kws=joint_kws,
        )

        g.annotate(pearsonr, template="R={val:.2g}, p={p:.1e}", frameon=False)

        g.ax_joint.axvline(0, ls="-", lw=0.1, c=cls.PAL_DTRACE[1], zorder=0)

        vexp = pca["column"]["vex"][pc]
        g.set_axis_labels(
            "{} ({:.1f}%)".format(pc, vexp * 100), "Growth rate\n(median day 1 / day 4)"
        )

    @classmethod
    def histogram_strong_response(cls, df):
        sns.distplot(
            df["n_resp"],
            color=cls.PAL_DTRACE[2],
            hist=False,
            kde_kws=cls.HIST_KDE_KWS,
            label=None,
        )

        plt.legend().remove()


class DrugPreliminary(Preliminary):
    DRUG_PAL = dict(v17=Preliminary.PAL_DTRACE[2], RS=Preliminary.PAL_DTRACE[0])

    @classmethod
    def histogram_drug(cls, drug_count):
        df = drug_count.rename("count").reset_index()

        for s in cls.DRUG_PAL:
            sns.distplot(
                df[df["VERSION"] == s]["count"],
                color=cls.DRUG_PAL[s],
                hist=False,
                label=s,
                kde_kws=cls.HIST_KDE_KWS,
            )

        plt.xlabel("Number of cell lines screened")
        plt.ylabel(f"Fraction of {df.shape[0]} drugs")

        plt.title("Cumulative distribution of drug measurements")

        plt.legend(loc=4, frameon=False, prop={"size": 6})

    @classmethod
    def histogram_sample(cls, samples_count):
        df = samples_count.rename("count").reset_index()

        sns.distplot(
            df["count"],
            color=cls.PAL_DTRACE[2],
            hist=False,
            kde_kws=cls.HIST_KDE_KWS,
            label=None,
        )

        plt.xlabel("Number of drugs screened")
        plt.ylabel(f"Fraction of {df.shape[0]} cell lines")

        plt.title("Cumulative distribution of drug measurements")

        plt.legend().remove()

    @classmethod
    def growth_correlation_histogram(cls, g_corr):
        for i, s in enumerate(cls.DRUG_PAL):
            hist_kws = dict(alpha=0.4, zorder=i + 1, linewidth=0)
            kde_kws = dict(cut=0, lw=1, zorder=i + 1, alpha=0.8)

            sns.distplot(
                g_corr[g_corr["VERSION"] == s]["pearson"],
                color=cls.DRUG_PAL[s],
                kde_kws=kde_kws,
                hist_kws=hist_kws,
                bins=15,
                label=s,
            )

        plt.axvline(0, c=cls.PAL_DTRACE[1], lw=0.1, ls="-", zorder=0)

        plt.xlabel("Drug correlation with growth rate\n(Pearson's R)")
        plt.ylabel("Density")

        plt.legend(prop={"size": 6}, frameon=False)

    @classmethod
    def growth_correlation_top_drugs(cls, g_corr, n_features=20):
        sns.barplot(
            "pearson",
            "DRUG_NAME",
            data=g_corr.head(n_features),
            color=cls.PAL_DTRACE[2],
            linewidth=0,
        )

        plt.axvline(0, c=cls.PAL_DTRACE[1], lw=0.1, ls="-", zorder=0)

        plt.xlabel("Drug correlation with growth rate\n(Pearson's R)")
        plt.ylabel("")

    @classmethod
    def growth_corrs_pcs_barplot(cls, df):
        sns.barplot(df["pearson"], df["index"], color=cls.PAL_DTRACE[2])

        plt.grid(axis="x", lw=0.1, color=cls.PAL_DTRACE[1], zorder=0)

        plt.xlabel("Pearson correlation coefficient")
        plt.ylabel("")
        plt.title("Drug-response PCA\ncorrelation with growth rate")


class CrisprPreliminary(Preliminary):
    @classmethod
    def corrplot_pcs_essentiality(cls, pca, num_resp_crispr, pc):
        df = pd.concat(
            [
                pca["row"]["pcs"][pc],
                num_resp_crispr.set_index("GeneSymbol"),
            ],
            axis=1,
        )

        annot_kws = dict(stat="R", loc=2)
        marginal_kws = dict(kde=False, hist_kws={"linewidth": 0})

        scatter_kws = dict(edgecolor="w", lw=0.3, s=6, alpha=0.3)
        line_kws = dict(lw=1.0, color=cls.PAL_DTRACE[0], alpha=1.0)
        joint_kws = dict(lowess=True, scatter_kws=scatter_kws, line_kws=line_kws)

        g = sns.jointplot(
            "n_resp",
            pc,
            data=df,
            kind="reg",
            space=0,
            color=cls.PAL_DTRACE[2],
            marginal_kws=marginal_kws,
            annot_kws=annot_kws,
            joint_kws=joint_kws,
        )

        g.annotate(pearsonr, template="R={val:.2g}, p={p:.1e}", frameon=False)

        g.ax_joint.axhline(0, ls="-", lw=0.1, c=cls.PAL_DTRACE[1], zorder=0)
        g.ax_joint.axvline(0, ls="-", lw=0.1, c=cls.PAL_DTRACE[1], zorder=0)

        vexp = pca["row"]["vex"][pc]
        g.set_axis_labels(
            "Gene significantly essential count", "{} ({:.1f}%)".format(pc, vexp * 100)
        )

    @classmethod
    def growth_corrs_pcs_barplot(cls, df):
        sns.barplot(df["pearson"], df["index"], color=cls.PAL_DTRACE[2])

        plt.grid(axis="x", lw=0.1, color=cls.PAL_DTRACE[1], zorder=0)

        plt.xlabel("Pearson correlation coefficient")
        plt.ylabel("")
        plt.title("CRISPR-Cas9 PCA\ncorrelation with growth rate")
