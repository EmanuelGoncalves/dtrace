#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dtrace.DTracePlot import DTracePlot
from scipy.stats import gaussian_kde
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit


class TargetHit(DTracePlot):
    def __init__(self, target, assoc, fdr=0.1):
        super().__init__()

        self.dinfo = ["DRUG_ID", "DRUG_NAME", "VERSION"]

        self.fdr = fdr
        self.assoc = assoc
        self.target = target

        self.drugs = list(
            {
                tuple(d)
                for d in self.assoc.lmm_drug_crispr[
                    self.assoc.lmm_drug_crispr["DRUG_TARGETS"] == self.target
                ][self.dinfo].values
            }
        )

    def associations_beta_scatter(self):
        x, y = (
            self.assoc.lmm_combined["CRISPR_beta"],
            self.assoc.lmm_combined["GExp_beta"],
        )

        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.scatter(
            x, y, c=z, marker="o", edgecolor="", cmap="viridis_r", s=3, alpha=0.85
        )

        plt.axhline(-np.log10(0.1), ls=":", lw=0.5, color=self.PAL_DTRACE[2], zorder=0)
        plt.axvline(-np.log10(0.1), ls=":", lw=0.5, color=self.PAL_DTRACE[2], zorder=0)

        plt.xlabel("Drug ~ CRISPR association beta")
        plt.ylabel("Drug ~ Genomic association beta")

    def top_associations_barplot(self):
        # Filter for signif associations
        df = self.assoc.lmm_drug_crispr.query(
            f"(DRUG_TARGETS == '{self.target}')"
        ).sort_values("pval")
        df = df.groupby(["DRUG_NAME", "DRUG_ID"]).head(3)
        df = df.assign(logpval=-np.log10(df["pval"]).values)
        df = df.assign(label=[f"{n.split(' / ')[0]} (id={i})" for n, i in df[["DRUG_NAME", "DRUG_ID"]].values])

        # Drug order
        order = list(df.groupby("label")["pval"].min().sort_values().index)

        # Build plot dataframe
        df_, xpos = [], 0
        for i, drug_name in enumerate(order):
            df_drug = df[df["label"] == drug_name]
            df_drug = df_drug.assign(xpos=np.arange(xpos, xpos + df_drug.shape[0]))

            xpos = xpos + df_drug.shape[0] + 1

            df_.append(df_drug)

        df = pd.concat(df_).reset_index()

        # Plot
        fig, ax = plt.subplots(1, 1, dpi=300)

        plot_df = df.query("target != 'T'")
        ax.bar(
            plot_df["xpos"].values,
            plot_df["logpval"].values,
            0.8,
            color=self.PAL_DTRACE[2],
            align="center",
            zorder=5,
            linewidth=0,
        )

        plot_df = df.query("target == 'T'")
        ax.bar(
            plot_df["xpos"],
            plot_df["logpval"],
            0.8,
            color=self.PAL_DTRACE[0],
            align="center",
            zorder=5,
            linewidth=0,
        )

        for k, v in (
            df.groupby("label")["xpos"].min().sort_values().to_dict().items()
        ):
            ax.text(
                v - 1.2,
                0.1,
                k,
                va="bottom",
                fontsize=8,
                zorder=10,
                rotation="vertical",
                color=self.PAL_DTRACE[2],
            )

        for g, p in df[["GeneSymbol", "xpos"]].values:
            ax.text(
                p,
                0.1,
                g,
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=10,
                rotation="vertical",
                color="white",
            )

        for x, y, t, b in df[["xpos", "logpval", "target", "beta"]].values:
            c = self.PAL_DTRACE[0] if t == "T" else self.PAL_DTRACE[2]

            ax.text(x, y + 0.25, t, color=c, ha="center", fontsize=6, zorder=10)
            ax.text(
                x,
                -2.5,
                f"{b:.1f}",
                color=c,
                ha="center",
                fontsize=6,
                rotation="vertical",
                zorder=10,
            )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

        ax.axes.get_xaxis().set_ticks([])

    def plot_target_drugs_corr(self, data, gene, order=None):
        if order is None:
            order = [
                tuple(d)
                for d in self.assoc.lmm_drug_crispr.query(
                    f"(DRUG_TARGETS == '{self.target}') & (GeneSymbol == '{gene}')"
                )[self.dinfo].values
            ]

        fig, axs = plt.subplots(1, len(order), sharey="all", dpi=300)

        for i, d in enumerate(order):
            plot_df = pd.concat(
                [
                    data.drespo.loc[d].rename("drug"),
                    data.crispr.loc[gene].rename("crispr"),
                    data.samplesheet.samplesheet["institute"],
                ],
                axis=1,
                sort=False,
            ).dropna()

            for t, df in plot_df.groupby("institute"):
                axs[i].scatter(
                    x=df["drug"],
                    y=df["crispr"],
                    edgecolor="w",
                    lw=0.1,
                    s=5,
                    color=self.PAL_DTRACE[2],
                    marker=self.MARKERS[t],
                    label=t,
                    alpha=0.8,
                )

            sns.regplot(
                "drug",
                "crispr",
                data=plot_df,
                line_kws=dict(lw=1.0, color=self.PAL_DTRACE[0]),
                marker="",
                truncate=True,
                ax=axs[i],
            )

            #
            beta, fdr = (
                self.assoc.lmm_drug_crispr.query(f"GeneSymbol == '{gene}'")
                .set_index(self.dinfo)
                .loc[d, ["beta", "fdr"]]
                .values
            )
            annot_text = f"b={beta:.2g}, p={fdr:.1e}"
            axs[i].text(
                0.95,
                0.05,
                annot_text,
                fontsize=4,
                transform=axs[i].transAxes,
                ha="right",
            )

            #
            dmax = np.log(data.drespo_obj.maxconcentration[d])
            axs[i].axvline(dmax, ls="-", lw=0.3, c=self.PAL_DTRACE[1], zorder=0)

            #
            axs[i].axhline(-0.5, ls="-", lw=0.3, c=self.PAL_DTRACE[1], zorder=0)

            #
            axs[i].grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

            #
            axs[i].set_ylabel(f"{gene}\n(scaled log2 FC)" if i == 0 else "")
            axs[i].set_xlabel(f"Drug response\n(ln IC50)")
            axs[i].set_title(f"{d[1]} ({d[0]})")

        plt.subplots_adjust(bottom=0.15, wspace=0.05)
        plt.gcf().set_size_inches(1 * len(order), 1.0)

    def plot_drug_crispr_gexp(self, drug_targets):
        targets = pd.Series(
            [DTracePlot.PAL_DTRACE[i] for i in [0, 2, 3]], index=drug_targets
        )

        plot_df = self.assoc.lmm_combined[
            self.assoc.lmm_combined["CRISPR_DRUG_TARGETS"].isin(targets.index)
        ].reset_index()
        plot_df = plot_df[plot_df["GeneSymbol"] == plot_df["CRISPR_DRUG_TARGETS"]]

        ax = plt.gca()

        for target, df in plot_df.groupby("CRISPR_DRUG_TARGETS"):
            ax.scatter(
                df["CRISPR_beta"],
                df["GExp_beta"],
                label=target,
                color=targets[target],
                edgecolor="white",
                lw=0.3,
                zorder=1,
            )

            df_signif = df.query("(CRISPR_fdr < .1) & (GExp_fdr < .1)")
            df_signif_any = df.query("(CRISPR_fdr < .1) | (GExp_fdr < .1)")

            if df_signif.shape[0] > 0:
                ax.scatter(
                    df_signif["CRISPR_beta"],
                    df_signif["GExp_beta"],
                    color="white",
                    marker="$X$",
                    lw=0.3,
                    label=None,
                    zorder=1,
                )

            elif df_signif_any.shape[0] > 0:
                ax.scatter(
                    df_signif_any["CRISPR_beta"],
                    df_signif_any["GExp_beta"],
                    color="white",
                    marker="$/$",
                    lw=0.3,
                    label=None,
                    zorder=1,
                )

        ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="both")

        ax.legend(loc=3, frameon=False, prop={"size": 5}).get_title().set_fontsize("5")

        ax.set_xlabel("CRISPR beta")
        ax.set_ylabel("GExp beta")

        ax.set_title("LMM Drug response model")

    def lm_drug_train(self, y, x, drug, n_splits=1000, test_size=0.3):
        y = y[x.index].dropna()
        x = x.loc[y.index]

        df = []
        for train, test in ShuffleSplit(n_splits=n_splits, test_size=test_size).split(
            x, y
        ):
            lm = RidgeCV().fit(x.iloc[train], y.iloc[train])

            r2 = lm.score(x.iloc[test], y.iloc[test])

            df.append(list(drug) + [r2] + list(lm.coef_))

        return pd.DataFrame(df, columns=self.dinfo + ["r2"] + list(x.columns))

    def predict_drugresponse(self, data, features):
        xss = {
            "CRISPR+GEXP": pd.concat(
                [
                    data.crispr.loc[features].T.add_prefix("CRISPR_"),
                    data.gexp.loc[features].T.add_prefix("GExp_"),
                ],
                axis=1,
                sort=False,
            ).dropna(),
            "CRISPR": data.crispr.loc[features].T.add_prefix("CRISPR_").dropna(),
            "GEXP": data.gexp.loc[features].T.add_prefix("GExp_").dropna(),
        }

        drug_lms = []

        for ftype in xss:
            print(f"ftype = {ftype}")

            xs = xss[ftype]
            xs = pd.DataFrame(
                StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns
            )

            lm_df = pd.concat(
                [self.lm_drug_train(data.drespo.loc[d], xs, d) for d in self.drugs]
            )
            lm_df["ftype"] = ftype

            drug_lms.append(lm_df)

        drug_lms = pd.concat(drug_lms, sort=False)

        return drug_lms

    def predict_r2_barplot(self, drug_lms):
        order = list(
            drug_lms.query(f"ftype == 'CRISPR+GEXP'")
            .groupby(self.dinfo)["r2"]
            .median()
            .sort_values(ascending=False)
            .reset_index()["DRUG_NAME"]
        )

        pal = pd.Series(
            [DTracePlot.PAL_DTRACE[i] for i in [0, 2, 3]],
            index=["CRISPR", "CRISPR+GEXP", "GEXP"],
        )

        sns.barplot(
            "r2",
            "DRUG_NAME",
            "ftype",
            data=drug_lms,
            order=order,
            palette=pal,
            orient="h",
            errwidth=0.5,
            saturation=1.0,
            lw=0,
            hue_order=pal.index,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.xlabel("R-squared")
        plt.ylabel("")

        plt.legend(frameon=False, prop={"size": 5}).get_title().set_fontsize("5")

    def predict_feature_plot(self, drug_lms):
        plot_df = (
            drug_lms.drop(columns=["r2"])
            .groupby(self.dinfo + ["ftype"])
            .median()
            .reset_index()
        )
        plot_df = pd.melt(plot_df, id_vars=self.dinfo + ["ftype"]).dropna()
        plot_df["variable"] = [
            f"{i.split('_')[1]} ({i.split('_')[0]})" for i in plot_df["variable"]
        ]

        order = list(
            plot_df.groupby("variable")["value"]
            .median()
            .sort_values(ascending=False)
            .index
        )

        pal = pd.Series(
            [DTracePlot.PAL_DTRACE[i] for i in [0, 2, 3]],
            index=["CRISPR", "CRISPR+GEXP", "GEXP"],
        )

        sns.stripplot(
            "value",
            "variable",
            "ftype",
            data=plot_df,
            order=order,
            orient="h",
            edgecolor="white",
            linewidth=0.1,
            s=3,
            palette=pal,
            hue_order=pal.index,
        )

        plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

        plt.legend(frameon=False, prop={"size": 5}).get_title().set_fontsize("5")
        plt.xlabel("Median beta")
        plt.ylabel("")

    def drugresponse_boxplots(self, data, ctypes, hue_order, order, genes):
        # Build dataframe
        plot_df = self.assoc.build_df(
            drug=self.drugs,
            crispr=genes,
            crispr_discretise=True,
            sinfo=["cancer_type"],
        ).dropna()

        plot_df["cancer_type"] = plot_df["cancer_type"].apply(
            lambda v: v if v in ctypes else "Other"
        )

        # Color pallete
        pal = pd.Series(sns.color_palette("tab10", n_colors=len(hue_order)).as_hex(), index=hue_order)

        # Figure
        nrows = 2
        ncols = int(len(self.drugs) / nrows)

        fig, axs = plt.subplots(nrows, ncols, sharex="none", sharey="none", dpi=300)

        for i, d in enumerate(self.drugs):
            ax = axs[i % nrows, int(np.floor(i / nrows))]

            sns.boxplot(
                x=d,
                y="crispr",
                hue="cancer_type",
                data=plot_df,
                orient="h",
                saturation=1.0,
                showcaps=False,
                order=order,
                hue_order=hue_order,
                flierprops=self.FLIERPROPS,
                linewidth=0.3,
                palette=pal,
                ax=ax,
            )

            ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

            dmax = np.log(data.drespo_obj.maxconcentration[d])
            ax.axvline(dmax, linewidth=0.3, color=self.PAL_DTRACE[2], ls=":", zorder=0)

            ax.set_xlabel(f"{d[1]}, {d[0]}, {d[2]} (ln IC50)")
            ax.set_ylabel("")

            if int(np.floor(i / nrows)) != 0:
                ax.axes.get_yaxis().set_visible(False)

            ax.legend().remove()

        plt.legend(
            handles=[mpatches.Patch(color=v, label=i) for i, v in pal.iteritems()],
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            prop={"size": 6},
            frameon=False,
        )

        plt.subplots_adjust(wspace=0.05, hspace=0.3)

        plt.gcf().set_size_inches(ncols * 1.5, nrows * 1.5)
