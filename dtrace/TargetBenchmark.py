#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import textwrap
import upsetplot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy.QCPlot import QCplot
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu
from dtrace.DTracePlot import DTracePlot
from dtrace.DataImporter import KinobeadCATDS
from sklearn.preprocessing import MinMaxScaler


class TargetBenchmark(DTracePlot):
    """
    Main class containing the analysis and plotting of the drug association analysis.

    """

    DRUG_TARGETS_COLORS = {
        "#8ebadb": {"RAF1", "BRAF"},
        "#5f9ecc": {"MAPK1", "MAPK3"},
        "#3182bd": {"MAP2K1", "MAP2K2"},
        "#f2a17a": {"PIK3CA", "PIK3CB"},
        "#ec7b43": {"AKT1", "AKT2", "AKT3"},
        "#e6550d": {"MTOR"},
        "#6fc088": {"EGFR"},
        "#31a354": {"IGF1R"},
        "#b2acd3": {"CHEK1", "CHEK2"},
        "#938bc2": {"ATR"},
        "#756bb1": {"WEE1", "TERT"},
        "#eeaad3": {"BTK"},
        "#e78ac3": {"SYK"},
        "#66c2a5": {"PARP1"},
        "#fedf57": {"BCL2", "BCL2L1"},
        "#fefb57": {"MCL1"},
        "#636363": {"GLS"},
        "#dd9a00": {"AURKA", "AURKB"},
        "#bc80bd": {"BRD2", "BRD4", "BRD3"},
        "#983539": {"JAK1", "JAK2", "JAK3"},
        "#ffffff": {"No target"},
        "#e1e1e1": {"Other target"},
        "#bbbbbb": {"Multiple targets"},
    }

    PPI_ORDER = ["T", "1", "2", "3", "4", "5+", "-"]

    PPI_PAL = {
        "T": "#fc8d62",
        "1": "#c3c3c3",
        "2": "#ababab",
        "3": "#949494",
        "4": "#7c7c7c",
        "5+": "#656565",
        "-": "#2b8cbe",
        "X": "#E1E1E1",
    }

    def __init__(self, assoc, fdr=0.1):
        self.fdr = fdr
        self.assoc = assoc

        self.dinfo = ["DRUG_ID", "DRUG_NAME", "VERSION"]

        # Drug targets
        self.d_targets = self.assoc.drespo_obj.get_drugtargets(by="Name")
        self.d_targets_id = self.assoc.drespo_obj.get_drugtargets(by="id")

        # Define sets of drugs
        self.d_sets_name = self.define_drug_sets()

        # Define sets of drugs PPI distance
        self.d_signif_ppi = self.define_drug_sets_ppi()
        self.d_signif_ppi_count = self.d_signif_ppi["target"].value_counts()[
            self.PPI_ORDER
        ]

        # Import kinobead measurements
        self.catds = KinobeadCATDS(assoc=self.assoc).get_data().dropna()

        super().__init__()

    def surrogate_pathway_ratio(self):
        df = self.assoc.lmm_drug_crispr.sort_values("fdr")

        df = pd.concat(
            [
                df.query("target != 'T'")
                    .groupby("DRUG_NAME")
                    .first()[["fdr", "GeneSymbol", "target"]]
                    .add_prefix("proxy_"),

                df.query("target == 'T'")
                    .groupby("DRUG_NAME")
                    .first()[["fdr", "DRUG_TARGETS", "GeneSymbol"]]
                    .add_prefix("target_"),
            ],
            axis=1,
            sort=False,
        ).dropna()

        df["ratio_fdr"] = np.log(df.eval("target_fdr/proxy_fdr"))
        df = df.sort_values("ratio_fdr")

        df["proxy_signif"] = (df["proxy_fdr"] < self.fdr).astype(int)
        df["target_signif"] = (df["target_fdr"] < self.fdr).astype(int)

        return df

    def define_drug_sets(self):
        df_genes = set(self.assoc.lmm_drug_crispr["GeneSymbol"])

        d_sets_name = dict(all=set(self.assoc.lmm_drug_crispr["DRUG_NAME"]))

        d_sets_name["significant"] = set(
            self.assoc.by(self.assoc.lmm_drug_crispr, fdr=self.fdr)["DRUG_NAME"]
        )

        d_sets_name["not_significant"] = {
            d for d in d_sets_name["all"] if d not in d_sets_name["significant"]
        }

        d_sets_name["annotated"] = {
            d for d in d_sets_name["all"] if d in self.d_targets
        }

        d_sets_name["tested"] = {
            d
            for d in d_sets_name["annotated"]
            if len(self.d_targets[d].intersection(df_genes)) > 0
        }

        d_sets_name["tested_significant"] = {
            d for d in d_sets_name["tested"] if d in d_sets_name["significant"]
        }

        d_sets_name["tested_corrected"] = {
            d
            for d in d_sets_name["tested_significant"]
            if d
            in set(
                self.assoc.by(self.assoc.lmm_drug_crispr, fdr=self.fdr, target="T")[
                    "DRUG_NAME"
                ]
            )
        }

        return d_sets_name

    def define_drug_sets_ppi(self):
        df = self.assoc.by(
            self.assoc.lmm_drug_crispr,
            fdr=self.fdr,
            drug_name=self.d_sets_name["tested_significant"],
        )

        d_signif_ppi = []
        for d in self.d_sets_name["tested_significant"]:
            df_ppi = df[df["DRUG_NAME"] == d].sort_values("fdr")
            df_ppi["target"] = pd.Categorical(df_ppi["target"], self.PPI_ORDER)

            d_signif_ppi.append(df_ppi.sort_values("target").iloc[0])

        d_signif_ppi = pd.DataFrame(d_signif_ppi)
        d_signif_ppi["target"] = pd.Categorical(d_signif_ppi["target"], self.PPI_ORDER)
        d_signif_ppi = d_signif_ppi.set_index("DRUG_NAME").sort_values("target")

        return d_signif_ppi

    def get_drug_target_color(self, drug_id):
        if drug_id not in self.d_targets_id:
            return "#ffffff"

        drug_targets = [
            c
            for c in self.DRUG_TARGETS_COLORS
            if len(self.d_targets_id[drug_id].intersection(self.DRUG_TARGETS_COLORS[c]))
            > 0
        ]

        if len(drug_targets) == 0:
            return "#e1e1e1"

        elif len(drug_targets) == 1:
            return drug_targets[0]

        else:
            return "#bbbbbb"

    def boxplot_kinobead(self):
        plt.figure(figsize=(0.75, 2.), dpi=300)

        order = ["No", "Yes"]
        pal = {"No": self.PAL_DTRACE[1], "Yes": self.PAL_DTRACE[0]}

        #
        catds_signif = {s: self.catds.query(f"signif == '{s}'")["catds"] for s in order}

        #
        t, p = mannwhitneyu(catds_signif["Yes"], catds_signif["No"])
        logging.getLogger("DTrace").info(
            f"Mann-Whitney U statistic={t:.2f}, p-value={p:.2e}"
        )

        # Plot
        ax = sns.boxplot(
            self.catds["signif"],
            self.catds["catds"],
            palette=pal,
            linewidth=0.3,
            fliersize=1.5,
            order=order,
            flierprops=self.FLIERPROPS,
            showcaps=False,
            orient="v",
        )

        # for i, s in enumerate(order):
        #     ax.scatter(gmean(catds_signif[s]), i, marker="+", lw=0.3, color="k", s=3)

        ax.set_yscale("log")

        ax.set_title(f"Drug-gene association")
        ax.set_xlabel("Significant")
        ax.set_ylabel("Kinobeads affinity (pKd [nM])")

    def beta_histogram(self):
        kde_kws = dict(cut=0, lw=1, zorder=1, alpha=0.8)
        hist_kws = dict(alpha=0.4, zorder=1, linewidth=0)

        plot_df = {
            c: self.assoc.lmm_drug_crispr.query(f"target {c} 'T'")["beta"]
            for c in ["!=", "=="]
        }

        for c in plot_df:
            sns.distplot(
                plot_df[c],
                hist_kws=hist_kws,
                bins=30,
                kde_kws=kde_kws,
                label="Target" if c == "==" else "All",
                color=self.PAL_DTRACE[0] if c == "==" else self.PAL_DTRACE[2],
            )

        t, p = mannwhitneyu(plot_df["!="], plot_df["=="])
        logging.getLogger("DTrace").info(
            f"Mann-Whitney U statistic={t:.2f}, p-value={p:.2e}"
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        plt.xlabel("Association beta")
        plt.ylabel("Density")

        plt.legend(prop={"size": 6}, loc=2, frameon=False)

    def pval_histogram(self):
        hist_kws = dict(alpha=0.5, zorder=1, linewidth=0.3, density=True)

        plot_df = {
            c: self.assoc.lmm_drug_crispr.query(f"target {c} 'T'")["pval"]
            for c in ["!=", "=="]
        }

        for c in plot_df:
            sns.distplot(
                plot_df[c],
                hist_kws=hist_kws,
                bins=30,
                kde=False,
                label="Target" if c == "==" else "All",
                color=self.PAL_DTRACE[0] if c == "==" else self.PAL_DTRACE[2],
            )

        plt.xlabel("Association p-value")
        plt.ylabel("Density")

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        plt.legend(prop={"size": 6}, frameon=False)

    def countplot_drugs(self):
        plot_df = (
            pd.Series(
                {
                    "All": self.assoc.drespo.shape[0],
                    "Unique": len(self.d_sets_name["all"]),
                    "Annotated": len(self.d_sets_name["annotated"]),
                    "Target tested": len(self.d_sets_name["tested"]),
                }
            )
            .rename("count")
            .reset_index()
        )

        plt.barh(plot_df.index, plot_df["count"], color=self.PAL_DTRACE[2], linewidth=0)

        for y, c in enumerate(plot_df["count"]):
            plt.text(
                c - 3,
                y,
                str(c),
                va="center",
                ha="right",
                fontsize=5,
                zorder=10,
                color="white",
            )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.yticks(plot_df.index, plot_df["index"])
        plt.xlabel("Number of drugs")
        plt.ylabel("")
        plt.title("")

    def countplot_drugs_significant(self):
        plot_df = (
            self.d_signif_ppi["target"]
            .value_counts()[reversed(self.PPI_ORDER)]
            .reset_index()
        )

        plt.barh(
            plot_df.index, plot_df["target"], color=self.PAL_DTRACE[2], linewidth=0
        )

        for y, c in enumerate(plot_df["target"]):
            plt.text(
                c - 3,
                y,
                str(c),
                va="center",
                ha="right",
                fontsize=5,
                zorder=10,
                color="white",
            )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.yticks(plot_df.index, plot_df["index"])
        plt.xlabel("Number of drugs")
        plt.ylabel("")
        plt.title("")

    def drugs_ppi(self, dtype="crispr", ax=None):
        if dtype == "crispr":
            df = self.assoc.by(
                self.assoc.lmm_drug_crispr, drug_name=self.d_sets_name["tested"]
            )

        elif dtype == "gexp":
            df = self.assoc.by(
                self.assoc.lmm_drug_gexp, drug_name=self.d_sets_name["tested"]
            )

        else:
            assert False, f"Dtype not supported: {dtype}"

        if ax is None:
            ax = plt.gca()

        QCplot.bias_boxplot(
            df.query(f"fdr < {self.fdr}"),
            x="target",
            y="fdr",
            notch=False,
            add_n=True,
            n_text_offset=5e-3,
            palette=self.PPI_PAL,
            hue_order=self.PPI_ORDER,
            order=self.PPI_ORDER,
            ax=ax,
        )

    def drugs_ppi_countplot(self, dtype="crispr", ax=None):
        if dtype == "crispr":
            df = self.assoc.by(
                self.assoc.lmm_drug_crispr, drug_name=self.d_sets_name["tested"]
            )

        elif dtype == "gexp":
            df = self.assoc.by(
                self.assoc.lmm_drug_gexp, drug_name=self.d_sets_name["tested"]
            )

        else:
            assert False, f"Dtype not supported: {dtype}"

        if ax is None:
            ax = plt.gca()

        plot_df = (
            df.query(f"fdr < {self.fdr}")["target"]
            .value_counts()
            .rename("count")
            .reset_index()
        )

        sns.barplot(
            "index",
            "count",
            data=plot_df,
            order=self.PPI_ORDER,
            palette=self.PPI_PAL,
            ax=ax,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

    def drugs_ppi_countplot_background(self, dtype="crispr"):
        if dtype == "crispr":
            df = self.assoc.by(
                self.assoc.lmm_drug_crispr, drug_name=self.d_sets_name["tested"]
            )

        elif dtype == "gexp":
            df = self.assoc.by(
                self.assoc.lmm_drug_gexp, drug_name=self.d_sets_name["tested"]
            )

        else:
            assert False, f"Dtype not supported: {dtype}"

        plot_df = df["target"].value_counts().rename("count").reset_index()

        sns.barplot(
            "index", "count", data=plot_df, order=self.PPI_ORDER, palette=self.PPI_PAL
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        plt.xlabel("Associated gene position in PPI")
        plt.ylabel("Number of associations")
        plt.title("All associations")

    def top_associations_barplot(self, ntop=50, n_cols=10):
        # Filter for signif associations
        df = self.assoc.by(self.assoc.lmm_drug_crispr, fdr=self.fdr).sort_values(["fdr", "pval"])
        df = df.groupby(["DRUG_NAME", "GeneSymbol"]).first()
        df = df.sort_values("fdr").reset_index()
        df = df.assign(logpval=-np.log10(df["pval"]).values)

        # Drug order
        order = list(df.groupby("DRUG_NAME")["fdr"].min().sort_values().index)[:ntop]

        # Build plot dataframe
        df_, xpos = [], 0
        for i, drug_name in enumerate(order):
            if i % n_cols == 0:
                xpos = 0

            df_drug = df[df["DRUG_NAME"] == drug_name].head(10)
            df_drug = df_drug.assign(xpos=np.arange(xpos, xpos + df_drug.shape[0]))
            df_drug = df_drug.assign(irow=int(np.floor(i / n_cols)))

            xpos += df_drug.shape[0] + 2

            df_.append(df_drug)

        df = pd.concat(df_).reset_index()

        # Plot
        n_rows = int(np.ceil(ntop / n_cols))

        f, axs = plt.subplots(
            n_rows,
            1,
            sharex="none", sharey="all",
            gridspec_kw=dict(hspace=0.0),
            figsize=(n_cols, n_rows * 1.7),
        )

        # Barplot
        for irow in set(df["irow"]):
            ax = axs[irow]
            df_irow = df[df["irow"] == irow]

            for t_type, c_idx in [("target != 'T'", 2), ("target == 'T'", 0)]:
                ax.bar(
                    df_irow.query(t_type)["xpos"].values,
                    df_irow.query(t_type)["logpval"].values,
                    color=self.PAL_DTRACE[c_idx],
                    align="center",
                    linewidth=0,
                )

            for k, v in (
                df_irow.groupby("DRUG_NAME")["xpos"]
                .min()
                .sort_values()
                .to_dict()
                .items()
            ):
                ax.text(
                    v - 1.2,
                    0.1,
                    textwrap.fill(k.split(" / ")[0], 15),
                    va="bottom",
                    fontsize=7,
                    zorder=10,
                    rotation="vertical",
                    color=self.PAL_DTRACE[2],
                )

            for g, p in df_irow[["GeneSymbol", "xpos"]].values:
                ax.text(
                    p,
                    0.1,
                    g,
                    ha="center",
                    va="bottom",
                    fontsize=5,
                    zorder=10,
                    rotation="vertical",
                    color="white",
                )

            for x, y, t, b in df_irow[["xpos", "logpval", "target", "beta"]].values:
                c = self.PAL_DTRACE[0] if t == "T" else self.PAL_DTRACE[2]

                ax.text(
                    x, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10
                )
                ax.text(
                    x,
                    -3,
                    f"{b:.1f}",
                    color=c,
                    ha="center",
                    fontsize=6,
                    rotation="vertical",
                    zorder=10,
                )

            ax.axes.get_xaxis().set_ticks([])
            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
            ax.set_ylabel("Drug association\n(-log10 p-value)")

    def drug_notarget_barplot(self, drug, genes):
        df = self.assoc.by(self.assoc.lmm_drug_crispr, drug_name=drug)
        df = df[df["GeneSymbol"].isin(genes)]
        df = df.groupby(["DRUG_NAME", "GeneSymbol"]).first()
        df = df.sort_values(["pval", "fdr"], ascending=False).reset_index()

        ax = plt.gca()

        ax.barh(
            df.query("target != 'T'").index,
            -np.log10(df.query("target != 'T'")["pval"]),
            0.8,
            color=self.PAL_DTRACE[2],
            align="center",
            zorder=1,
            linewidth=0,
        )

        ax.barh(
            df.query("target == 'T'").index,
            -np.log10(df.query("target == 'T'")["pval"]),
            0.8,
            color=self.PAL_DTRACE[0],
            align="center",
            zorder=1,
            linewidth=0,
        )

        for i, (y, t, f) in df[["pval", "target", "fdr"]].iterrows():

            ax.text(
                -np.log10(y) - 0.1,
                i,
                f"{t}{'*' if f < self.fdr else ''}",
                color="white",
                ha="right",
                va="center",
                fontsize=6,
                zorder=10,
            )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        ax.set_yticks(df.index)
        ax.set_yticklabels(df["GeneSymbol"])

        ax.set_xlabel("Drug association (-log10 p-value)")
        ax.set_title(drug)

        return ax

    def lmm_betas_clustermap(self, matrix_betas):
        matrix_betas_corr = matrix_betas.T.corr()

        row_cols = pd.Series(
            {d: self.get_drug_target_color(d[0]) for d in matrix_betas_corr.index}
        )
        col_cols = pd.Series(
            {d: self.get_drug_target_color(d[0]) for d in matrix_betas_corr.columns}
        )

        sns.clustermap(
            matrix_betas_corr,
            xticklabels=False,
            yticklabels=False,
            col_colors=col_cols,
            row_colors=row_cols,
            cmap="mako",
        )

    def lmm_betas_clustermap_legend(self):
        labels = {
            ";".join(self.DRUG_TARGETS_COLORS[c]): Line2D([0], [0], color=c, lw=4)
            for c in self.DRUG_TARGETS_COLORS
        }

        plt.legend(
            labels.values(), labels.keys(), bbox_to_anchor=(0.5, 1.0), frameon=False
        )

    def signif_per_screen(self):
        df = self.assoc.lmm_drug_crispr.groupby(self.assoc.dcols).first().reset_index()
        df = df[df["DRUG_NAME"].isin(self.d_sets_name["tested"])]

        df["signif"] = (df["fdr"] < self.fdr).astype(int)

        df = df.groupby("VERSION")["signif"].agg(["count", "sum"]).reset_index()
        df["perc"] = df["sum"] / df["count"] * 100

        plt.bar(df.index, df["count"], color=self.PAL_DTRACE[1], label="All")
        plt.bar(df.index, df["sum"], color=self.PAL_DTRACE[2], label="Signif.")

        for x, (y, p) in enumerate(df[["sum", "perc"]].values):
            plt.text(
                x,
                y + 1,
                f"{p:.1f}%",
                va="bottom",
                ha="center",
                fontsize=5,
                zorder=10,
                color=self.PAL_DTRACE[2],
            )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        plt.xticks(df.index, df["VERSION"])
        plt.ylabel("Number of drugs")
        plt.legend(prop={"size": 4}, frameon=False)

    def signif_genomic_markers(self):
        plot_df = pd.concat(
            [
                self.assoc.lmm_drug_crispr.groupby("DRUG_NAME")["fdr"]
                .min()
                .apply(lambda v: "Yes" if v < self.fdr else "No")
                .rename("crispr_fdr"),
                self.assoc.lmm_drug_genomic.groupby("DRUG_NAME")["fdr"]
                .min()
                .apply(lambda v: "Yes" if v < self.fdr else "No")
                .rename("genomic_fdr"),
            ],
            axis=1,
        ).reset_index()
        plot_df = plot_df[plot_df["DRUG_NAME"].isin(self.d_sets_name["tested"])]

        plot_df = pd.pivot_table(
            plot_df.reset_index(),
            index="crispr_fdr",
            columns="genomic_fdr",
            values="DRUG_NAME",
            aggfunc="count",
        )

        g = sns.heatmap(plot_df, annot=True, cbar=False, fmt=".0f", cmap="Greys")

        g.set_xlabel("Genomic marker")
        g.set_ylabel("CRISPR association")
        g.set_title("Drug association")

    def signif_upset(self):
        ess_genes = self.assoc.crispr_obj.import_sanger_essential_genes()

        plot_df = pd.concat(
            [
                self.assoc.lmm_drug_crispr.groupby("DRUG_NAME")["fdr"]
                .min()
                .apply(lambda v: v < self.fdr)
                .rename("crispr_fdr"),
                self.assoc.lmm_drug_genomic.groupby("DRUG_NAME")["fdr"]
                .min()
                .apply(lambda v: v < self.fdr)
                .rename("genomic_fdr"),
            ],
            axis=1,
        ).reset_index()
        plot_df = plot_df[plot_df["DRUG_NAME"].isin(self.d_sets_name["tested"])]
        plot_df["target_ess"] = plot_df["DRUG_NAME"].apply(
            lambda v: (len(self.d_targets[v].intersection(ess_genes)) > 0)
        )

        plot_df = plot_df.groupby(["crispr_fdr", "genomic_fdr", "target_ess"])[
            "DRUG_NAME"
        ].count()

        upsetplot.plot(plot_df)

    def pichart_drugs_significant(self):
        plot_df = self.d_signif_ppi["target"].value_counts().to_dict()
        plot_df["X"] = len(
            [
                d
                for d in self.d_sets_name["tested"]
                if d not in self.d_sets_name["significant"]
            ]
        )
        plot_df = pd.Series(plot_df)[self.PPI_ORDER + ["X"]]

        explode = [0, 0, 0, 0, 0, 0, 0, 0.1]

        plt.pie(
            plot_df,
            labels=plot_df.index,
            explode=explode,
            colors=list(self.PPI_PAL.values()),
            autopct="%1.1f%%",
            shadow=False,
            startangle=90,
            textprops={"fontsize": 7},
            wedgeprops=dict(linewidth=0),
        )

    def barplot_drugs_significant(self):
        plot_df = self.d_signif_ppi["target"].value_counts().to_dict()
        plot_df["X"] = len(
            [
                d
                for d in self.d_sets_name["tested"]
                if d not in self.d_sets_name["significant"]
            ]
        )
        plot_df = pd.Series(plot_df)[self.PPI_ORDER + ["X"]].reset_index().rename(columns={"index": "target", 0: "drugs"})

        _, ax = plt.subplots(1, 1, figsize=(2, 2))
        sns.barplot("target", "drugs", data=plot_df, palette=self.PPI_PAL, linewidth=0, ax=ax)
        for i, row in plot_df.iterrows():
            ax.text(
                i,
                row["drugs"],
                f"{(row['drugs'] / plot_df['drugs'].sum() * 100):.1f}%",
                va="bottom",
                ha="center",
                fontsize=5,
                zorder=10,
                color=self.PAL_DTRACE[2],
            )
        ax.tick_params(axis='x', which='both', bottom=False)
        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")
        ax.set_xlabel("Drug significant asociations\nshortest distance to target")
        ax.set_ylabel("Number of drugs (with target in PPI)")

    def signif_maxconcentration_scatter(self):
        # Build data-frame
        d_frist = self.assoc.lmm_drug_crispr.groupby(self.assoc.dcols).first()

        plot_df = pd.DataFrame(
            {
                d: {
                    "below": np.sum(
                        self.assoc.drespo.loc[d].dropna()
                        < np.log(self.assoc.drespo_obj.maxconcentration[d])
                    ),
                    "total": self.assoc.drespo.loc[d].dropna().shape[0],
                }
                for d in self.assoc.drespo.index
            }
        ).T
        plot_df = pd.concat([plot_df, d_frist], axis=1)

        plot_df["target"] = [
            t if f < self.fdr else "X" for f, t in plot_df[["fdr", "target"]].values
        ]
        plot_df["below_%"] = plot_df["below"] / plot_df["total"]
        plot_df["size"] = (
            MinMaxScaler().fit_transform(plot_df[["beta"]].abs())[:, 0] * 10 + 1
        )
        plot_df["fdr_log"] = -np.log10(plot_df["fdr"])

        #
        grid = sns.JointGrid("below_%", "fdr_log", data=plot_df, space=0)

        for ppid in reversed(self.PPI_ORDER + ["X"]):
            df = plot_df.query(f"(target == '{ppid}')")
            grid.ax_joint.scatter(
                df["below_%"],
                df["fdr_log"],
                s=df["size"],
                color=self.PPI_PAL[ppid],
                marker="o",
                label=ppid,
                edgecolor="white",
                lw=0.1,
            )

            grid.ax_marg_x.hist(
                df["below_%"], linewidth=0, bins=15, color=self.PPI_PAL[ppid], alpha=0.5
            )
            grid.ax_marg_y.hist(
                df["fdr_log"],
                linewidth=0,
                bins=15,
                color=self.PPI_PAL[ppid],
                orientation="horizontal",
                alpha=0.5,
            )

        grid.ax_joint.axhline(
            -np.log10(0.1), ls=":", lw=0.5, color=self.PAL_DTRACE[2], zorder=0
        )

        grid.ax_joint.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        grid.set_axis_labels(
            "Measurements lower than max Concentration\n(%)",
            "Drug lowest association FDR\n(min, -log10)",
        )

        grid.ax_joint.set_xlim(0, 1)

    def signif_fdr_scatter(self):
        plot_df = pd.concat(
            [
                self.assoc.lmm_drug_crispr.groupby(self.assoc.dcols)["fdr"]
                .min()
                .rename("crispr"),
                self.assoc.lmm_drug_genomic.groupby(self.assoc.dcols)["fdr"]
                .min()
                .rename("drug"),
            ],
            axis=1,
            sort=False,
        )

        x, y = -np.log10(plot_df["crispr"]), -np.log10(plot_df["drug"])

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.scatter(
            x, y, c=z, marker="o", edgecolor="", cmap="viridis_r", s=3, alpha=0.85
        )

        plt.axhline(-np.log10(0.1), ls=":", lw=0.5, color=self.PAL_DTRACE[2], zorder=0)
        plt.axvline(-np.log10(0.1), ls=":", lw=0.5, color=self.PAL_DTRACE[2], zorder=0)

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        plt.xlabel("Drug ~ CRISPR association FDR\n(-log10)")
        plt.ylabel("Drug ~ Genomic association FDR\n(-log10)")

    def drug_top_associations(self, drug, fdr_thres=None, ax=None):
        fdr_thres = self.fdr if fdr_thres is None else fdr_thres

        plot_df = self.assoc.by(
            self.assoc.lmm_drug_crispr, fdr=fdr_thres, drug_name=drug
        )

        plot_df = plot_df.reset_index(drop=True)
        plot_df = plot_df.groupby(["DRUG_NAME", "GeneSymbol"]).first()
        plot_df = plot_df.sort_values(["fdr", "pval"]).reset_index()
        plot_df["logpval"] = -np.log10(plot_df["pval"])

        #
        if ax is None:
            ax = plt.gca()

        df = plot_df.query("target != 'T'")
        ax.bar(
            df.index,
            df["logpval"],
            0.8,
            color=self.PAL_DTRACE[2],
            align="center",
            zorder=5,
            linewidth=0,
        )

        df = plot_df.query("target == 'T'")
        ax.bar(
            df.index,
            df["logpval"],
            0.8,
            color=self.PAL_DTRACE[0],
            align="center",
            zorder=5,
            linewidth=0,
        )

        for i, (g, p) in enumerate(plot_df[["GeneSymbol", "fdr"]].values):
            ax.text(
                i,
                0.1,
                f"{g}{'*' if p < self.fdr else ''}",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=10,
                rotation="vertical",
                color="white",
            )

        for i, (y, t, b) in enumerate(plot_df[["logpval", "target", "beta"]].values):
            c = self.PAL_DTRACE[0] if t == "T" else self.PAL_DTRACE[2]

            ax.text(i, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10)
            ax.text(
                i,
                -1,
                f"{b:.1f}",
                color=c,
                ha="center",
                fontsize=6,
                rotation="vertical",
                zorder=10,
            )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        ax.axes.get_xaxis().set_ticks([])

        ax.set_ylabel("Drug-gene association\n(-log10 p-value)")
        ax.set_title(f"{drug} associations")

        return plot_df

    def signif_volcano(self):
        plot_df = self.assoc.by(self.assoc.lmm_drug_crispr, fdr=self.fdr)

        plot_df["size"] = (
            MinMaxScaler().fit_transform(plot_df[["beta"]].abs())[:, 0] * 10 + 1
        )

        for t, df in plot_df.groupby("target"):
            plt.scatter(
                -np.log10(df["pval"]),
                df["beta"],
                s=df["size"],
                color=self.PPI_PAL[t],
                marker="o",
                label=t,
                edgecolor="white",
                lw=0.1,
                alpha=0.5,
            )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        plt.legend(
            frameon=False,
            prop={"size": 4},
            title="PPI distance",
            loc="center left",
            bbox_to_anchor=(1, 0.5),
        ).get_title().set_fontsize("4")

        plt.ylabel("Effect size (beta)")
        plt.xlabel("Association p-value (-log10)")
