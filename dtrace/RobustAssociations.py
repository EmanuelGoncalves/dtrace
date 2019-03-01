#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DTracePlot import DTracePlot


class RobustAssociations(DTracePlot):
    GENETIC_PAL = {"Mutation": "#ffde17", "CN loss": "#6ac4ea", "CN gain": "#177ba5"}
    GENETIC_ORDER = ["Mutation", "CN loss", "CN gain"]

    def __init__(self, assoc, fdr=0.1):
        self.fdr = fdr
        self.assoc = assoc

        self.assoc_count, self.assoc_count_ppi = self.get_associations_count()

        super().__init__()

    def get_associations(self, dtype, filter_nan=False):
        if dtype == "genomic":
            df = self.assoc.lmm_robust_genomic

            if filter_nan:
                df = df[
                    [
                        len(self.assoc.genomic_obj.mobem_feature_to_gene(i)) != 0
                        for i in df["feature"]
                    ]
                ]

        elif dtype == "gene-expression":
            df = self.assoc.lmm_robust_gexp

        else:
            assert False, "dtype not supported"

        return df

    def get_associations_count(self):
        cols, values = (
            ["DRUG_ID", "DRUG_NAME", "VERSION", "GeneSymbol"],
            ["feature", "target"],
        )

        filters = [
            ("Drug", f"(drug_fdr < {self.fdr})"),
            ("CRISPR", f"(crispr_fdr < {self.fdr})"),
            ("Both", f"(drug_fdr < {self.fdr}) & (crispr_fdr < {self.fdr})"),
        ]

        associations = [
            ("Genomic", self.assoc.lmm_robust_genomic),
            ("Gene-expression", self.assoc.lmm_robust_gexp),
        ]

        df_count, df_count_ppi = [], []

        for dtype, df in associations:
            for ftype, query in filters:
                df_pairs = df.query(query).groupby(cols)[values].agg(list)

                df_count.append(dict(dtype=dtype, ftype=ftype, count=df_pairs.shape[0]))

                ppi_dist = pd.Series([t for ts in df_pairs["target"] for t in ts])
                ppi_dist = ppi_dist.value_counts()

                for i, v in ppi_dist.iteritems():
                    df_count_ppi.append(dict(dtype=dtype, ftype=ftype, ppi=i, count=v))

            df_count.append(dict(dtype=dtype, ftype="Total", count=884))

        df_count, df_count_ppi = pd.DataFrame(df_count), pd.DataFrame(df_count_ppi)

        return df_count, df_count_ppi

    def genomic_histogram(self, ntop=40):
        plot_df = (
            self.assoc.genomic.drop(["msi_status"]).sum(1).rename("count").reset_index()
        )

        plot_df["genes"] = [
            self.assoc.genomic_obj.mobem_feature_to_gene(i) for i in plot_df["index"]
        ]
        plot_df = plot_df[plot_df["genes"].apply(len) != 0]
        plot_df["genes"] = plot_df["genes"].apply(lambda v: ";".join(v)).values

        plot_df["type"] = [
            self.assoc.genomic_obj.mobem_feature_type(i) for i in plot_df["index"]
        ]

        plot_df = plot_df.assign(
            name=["{} - {}".format(t, g) for t, g in plot_df[["type", "genes"]].values]
        )

        plot_df = plot_df.sort_values("count", ascending=False).head(ntop)

        # Plot
        sns.barplot(
            "count",
            "name",
            "type",
            data=plot_df,
            palette=self.GENETIC_PAL,
            hue_order=self.GENETIC_ORDER,
            dodge=False,
            saturation=1,
        )

        plt.xlabel("Number of occurrences")
        plt.ylabel("")

        plt.legend(frameon=False)

        plt.gcf().set_size_inches(2, 0.15 * ntop)

    def top_robust_features(self, ntop=30, dtype="genomic"):
        f, axs = plt.subplots(
            1, 2, sharex="none", sharey="none", gridspec_kw=dict(wspace=0.75), dpi=300
        )

        for i, d in enumerate(["drug", "crispr"]):
            ax = axs[i]
            beta, pval, fdr = f"{d}_beta", f"{d}_pval", f"{d}_fdr"

            feature = "DRUG_NAME" if d == "drug" else "GeneSymbol"

            plot_df = self.get_associations(dtype).query("feature != 'msi_status'")
            plot_df = plot_df.groupby([feature, "feature"])[beta, pval, fdr].first()
            plot_df = plot_df.reset_index()
            plot_df = plot_df.sort_values([fdr, pval])
            plot_df = plot_df.head(ntop)
            plot_df = plot_df.sort_values(beta, ascending=False)
            plot_df = plot_df.assign(y=range(plot_df.shape[0]))

            # Scatter
            if dtype == "genomic":
                plot_df = plot_df.assign(
                    type=[
                        self.assoc.genomic_obj.mobem_feature_type(i) for i in plot_df["feature"]
                    ]
                )

                for t in self.GENETIC_ORDER:
                    df = plot_df.query(f"type == '{t}'")
                    ax.scatter(df[beta], df["y"], c=self.GENETIC_PAL[t], label=t)

            else:
                ax.scatter(plot_df[beta], plot_df["y"], c=self.PAL_DTRACE[2])

            # Labels
            for fc, y, drug, genetic in plot_df[[beta, "y", feature, "feature"]].values:
                if dtype == "genomic":
                    g_genes = "; ".join(
                        self.assoc.genomic_obj.mobem_feature_to_gene(genetic)
                    )

                else:
                    g_genes = genetic

                xoffset = plot_df[beta].abs().max() * 0.2

                ax.text(
                    fc - xoffset,
                    y,
                    drug,
                    va="center",
                    fontsize=4,
                    zorder=10,
                    color="gray",
                    ha="right",
                )
                ax.text(
                    fc + xoffset,
                    y,
                    g_genes,
                    va="center",
                    fontsize=3,
                    zorder=10,
                    color="gray",
                    ha="left",
                )

            # Misc
            ax.axvline(0, lw=0.1, c=self.PAL_DTRACE[1])

            ax.set_xlabel("Effect size (beta)")
            ax.set_ylabel("")
            ax.set_title(
                "{} associations".format(d.capitalize() if d == "drug" else d.upper())
            )
            ax.axes.get_yaxis().set_ticks([])

            sns.despine(left=True, ax=ax)

        if dtype == "genomic":
            plt.legend(
                title="Genetic event",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )

        plt.gcf().set_size_inches(2.0 * axs.shape[0], ntop * 0.12)

    def robust_associations_barplot(self):
        #
        hue_order = ["Both", "Drug", "CRISPR", "Total"]
        pal = {
            "Both": "#fc8d62",
            "Drug": "#ababab",
            "CRISPR": "#656565",
            "Total": "#E1E1E1",
        }

        sns.catplot(
            "dtype",
            "count",
            "ftype",
            data=self.assoc_count,
            palette=pal,
            kind="bar",
            hue_order=hue_order,
            facet_kws={"despine": False}
        )

        plt.xlabel("")
        plt.ylabel("Drug-gene associations")

        plt.grid(lw=0.3, c=self.PAL_DTRACE[1], axis="y", alpha=5, zorder=0)

    def robust_associations_barplot_ppi(self):
        #
        order = ["T", "1", "2", "3", "4", "5+", "-"]
        hue_order = ["Both", "Drug", "CRISPR"]
        pal = {"Both": "#fc8d62", "Drug": "#ababab", "CRISPR": "#656565"}

        g = sns.catplot(
            "count",
            "ppi",
            "ftype",
            row="dtype",
            data=self.assoc_count_ppi,
            palette=pal,
            kind="bar",
            order=order,
            hue_order=hue_order,
        )

        g.set_titles("{row_name}")

        for ax in g.axes[:, 0]:
            ax.set_xlabel("Drug-gene associations")
            ax.set_ylabel("PPI distance")
            ax.grid(lw=0.3, c=self.PAL_DTRACE[1], axis="x", alpha=5, zorder=0)
