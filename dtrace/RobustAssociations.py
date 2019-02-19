#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from DTracePlot import DTracePlot
from natsort import natsorted
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from Associations import Association
from DataImporter import DrugResponse, PPI, GeneExpression
from sklearn.preprocessing import StandardScaler


class SingleLMMTSNE:
    DRUG_TARGETS_HUE = [
        ("#3182bd", [{"RAF1", "BRAF"}, {"MAPK1", "MAPK3"}, {"MAP2K1", "MAP2K2"}]),
        ("#e6550d", [{"PIK3CA", "PIK3CB"}, {"AKT1", "AKT2", "AKT3"}, {"MTOR"}]),
        ("#31a354", [{"EGFR"}, {"IGF1R"}]),
        ("#756bb1", [{"CHEK1", "CHEK2"}, {"ATR"}, {"WEE1", "TERT"}]),
        ("#e78ac3", [{"BTK"}, {"SYK"}]),
        ("#66c2a5", [{"PARP1"}]),
        ("#fdd10f", [{"BCL2", "BCL2L1"}]),
        ("#636363", [{"GLS"}]),
        ("#92d2df", [{"MCL1"}]),
        ("#dd9a00", [{"AURKA", "AURKB"}]),
        ("#bc80bd", [{"BRD2", "BRD4", "BRD3"}]),
        ("#983539", [{"JAK1", "JAK2", "JAK3"}]),
    ]

    DRUG_TARGETS_HUE = [
        (sns.light_palette(c, n_colors=len(s) + 1, reverse=True).as_hex()[:-1], s)
        for c, s in DRUG_TARGETS_HUE
    ]

    def __init__(
        self, lmm_dsingle, perplexity=15, learning_rate=250, n_iter=2000, fdr=0.1
    ):
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.fdr = fdr

        self.dtargets = DrugResponse().get_drugtargets()

        self.tsnes = self.drug_betas_tsne(lmm_dsingle)

    def drug_betas_tsne(self, lmm_dsingle):
        # Drugs into
        drugs = {tuple(i) for i in lmm_dsingle[DrugResponse.DRUG_COLUMNS].values}
        drugs_annot = {tuple(i) for i in drugs if i[0] in self.dtargets}
        drugs_screen = {v: {d for d in drugs if d[2] == v} for v in ["v17", "RS"]}

        # Build drug association beta matrix
        betas = pd.pivot_table(
            lmm_drug,
            index=DrugResponse.DRUG_COLUMNS,
            columns="GeneSymbol",
            values="beta",
        )
        betas = betas.loc[list(drugs)]

        # TSNE
        tsnes = []
        for s in drugs_screen:
            tsne_df = betas.loc[list(drugs_screen[s])]
            tsne_df = pd.DataFrame(
                StandardScaler().fit_transform(tsne_df.T).T,
                index=tsne_df.index,
                columns=tsne_df.columns,
            )

            tsne = TSNE(
                perplexity=self.perplexity,
                learning_rate=self.learning_rate,
                n_iter=self.n_iter,
                init="pca",
            ).fit_transform(tsne_df)

            tsne = pd.DataFrame(
                tsne, index=tsne_df.index, columns=["P1", "P2"]
            ).reset_index()
            tsne = tsne.assign(
                target=[
                    "Yes" if tuple(i) in drugs_annot else "No"
                    for i in tsne[DrugResponse.DRUG_COLUMNS].values
                ]
            )

            tsnes.append(tsne)

        tsnes = pd.concat(tsnes)
        tsnes = tsnes.assign(
            name=[
                ";".join(map(str, i[1:]))
                for i in tsnes[DrugResponse.DRUG_COLUMNS].values
            ]
        )

        # Annotate compound replicated
        rep_names = tsnes["name"].value_counts()
        rep_names = set(rep_names[rep_names > 1].index)
        tsnes = tsnes.assign(rep=[i if i in rep_names else "NA" for i in tsnes["name"]])

        # Annotate targets
        tsnes = tsnes.assign(
            targets=[
                ";".join(self.dtargets[i]) if i in self.dtargets else ""
                for i in tsnes[DrugResponse.DRUG_COLUMNS[0]]
            ]
        )

        # Annotate significant
        d_signif = {
            tuple(i)
            for i in lmm_drug.query(f"fdr < {self.fdr}")[
                DrugResponse.DRUG_COLUMNS
            ].values
        }
        tsnes = tsnes.assign(
            has_signif=[
                "Yes" if tuple(i) in d_signif else "No"
                for i in tsnes[DrugResponse.DRUG_COLUMNS].values
            ]
        )

        return tsnes

    def drug_beta_tsne(self, hueby):
        if hueby == "signif":
            pal = {"No": DTracePlot.PAL_DTRACE[2], "Yes": DTracePlot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes,
                col="VERSION",
                hue="has_signif",
                palette=pal,
                hue_order=["No", "Yes"],
                sharey=False,
                sharex=False,
                legend_out=True,
                despine=False,
                size=2,
                aspect=1,
            )

            g.map(plt.scatter, "P1", "P2", alpha=1.0, lw=0.3, edgecolor="white", s=10)
            g.map(
                plt.axhline,
                y=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )
            g.map(
                plt.axvline,
                x=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )

            g.add_legend(title="Significant", prop=dict(size=4), frameon=False)

        elif hueby == "target":
            pal = {"No": DTracePlot.PAL_DTRACE[2], "Yes": DTracePlot.PAL_DTRACE[0]}

            g = sns.FacetGrid(
                self.tsnes,
                col="VERSION",
                hue="target",
                palette=pal,
                hue_order=["Yes", "No"],
                sharey=False,
                sharex=False,
                legend_out=True,
                despine=False,
                size=2,
                aspect=1,
            )

            g.map(plt.scatter, "P1", "P2", alpha=1.0, lw=0.3, edgecolor="white", s=10)
            g.map(
                plt.axhline,
                y=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )
            g.map(
                plt.axvline,
                x=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )

            g.add_legend(title="Known target?", prop=dict(size=4), frameon=False)

        elif hueby == "replicates":
            rep_names = set(self.tsnes["rep"])

            pal_v17 = [n for n in rep_names if n.endswith(";v17")]
            pal_v17 = dict(
                zip(
                    *(
                        pal_v17,
                        sns.color_palette("tab20", n_colors=len(pal_v17)).as_hex(),
                    )
                )
            )

            pal_rs = [n for n in rep_names if n.endswith(";RS")]
            pal_rs = dict(
                zip(
                    *(pal_rs, sns.color_palette("tab20", n_colors=len(pal_rs)).as_hex())
                )
            )

            pal = {**pal_v17, **pal_rs}
            pal["NA"] = DTracePlot.PAL_DTRACE[1]

            g = sns.FacetGrid(
                self.tsnes,
                col="VERSION",
                hue="rep",
                palette=pal,
                sharey=False,
                sharex=False,
                legend_out=True,
                despine=False,
                size=2,
                aspect=1,
            )

            g.map(plt.scatter, "P1", "P2", alpha=1.0, lw=0.3, edgecolor="white", s=10)
            g.map(
                plt.axhline,
                y=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )
            g.map(
                plt.axvline,
                x=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )

            g.add_legend(title="", prop=dict(size=4), frameon=False)

        elif type(hueby) == list:
            sets = [i for l in hueby for i in l[1]]
            labels = [";".join(i) for l in hueby for i in l[1]]
            colors = [i for l in hueby for i in l[0]]

            pal = dict(zip(*(labels, colors)))
            pal["NA"] = DTracePlot.PAL_DTRACE[1]

            df = self.tsnes.assign(
                hue=[
                    [i for i, g in enumerate(sets) if g.intersection(t.split(";"))]
                    for t in self.tsnes["targets"]
                ]
            )
            df = self.tsnes.assign(
                hue=[labels[i[0]] if len(i) > 0 else "NA" for i in df["hue"]]
            )

            g = sns.FacetGrid(
                df.query("target == 'Yes'"),
                col="VERSION",
                hue="hue",
                palette=pal,
                sharey=False,
                sharex=False,
                legend_out=True,
                despine=False,
                size=2,
                aspect=1,
            )

            for i, s in enumerate(["v17", "RS"]):
                ax = g.axes.ravel()[i]
                df_plot = df.query("(target == 'No') & (VERSION == '{}')".format(s))
                ax.scatter(
                    df_plot["P1"],
                    df_plot["P2"],
                    color=DTracePlot.PAL_DTRACE[1],
                    marker="x",
                    lw=0.3,
                    s=5,
                    alpha=0.7,
                    label="No target info",
                )

            g.map(plt.scatter, "P1", "P2", alpha=1.0, lw=0.3, edgecolor="white", s=10)
            g.map(
                plt.axhline,
                y=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )
            g.map(
                plt.axvline,
                x=0,
                ls="-",
                lw=0.3,
                c=DTracePlot.PAL_DTRACE[1],
                alpha=0.2,
                zorder=0,
            )

            g.add_legend(
                title="",
                prop=dict(size=4),
                label_order=labels + ["NA"] + ["No info"],
                frameon=False,
            )

        g.set_titles("Screen = {col_name}")

    def drug_v17_id_batch(self):
        hue_order = ["[0, 200[", "[200, 1000[", "[1000, inf.["]

        plot_df = self.tsnes[self.tsnes["VERSION"] == "v17"]

        plot_df = plot_df.assign(
            id_discrete=pd.cut(
                plot_df["DRUG_ID"],
                bins=[0, 200, 1000, np.max(plot_df["DRUG_ID"])],
                labels=hue_order,
            )
        )

        discrete_pal = pd.Series(
            DTracePlot.PAL_DTRACE[:3], index=set(plot_df["id_discrete"])
        ).to_dict()

        g = DTracePlot.plot_corrplot_discrete(
            "P1",
            "P2",
            "id_discrete",
            plot_df,
            discrete_pal=discrete_pal,
            legend_title="DRUG_ID discretised",
            hue_order=hue_order,
        )

        g.set_axis_labels("", "")

        return g


class RobustLMMAnalysis:
    GENETIC_PAL = {"Mutation": "#ffde17", "CN loss": "#6ac4ea", "CN gain": "#177ba5"}
    GENETIC_ORDER = ["Mutation", "CN loss", "CN gain"]

    @classmethod
    def genomic_histogram(cls, datasets, ntop=40):
        # Build dataframe
        plot_df = (
            datasets.genomic.drop(["msi_status"]).sum(1).rename("count").reset_index()
        )

        plot_df["genes"] = [
            datasets.genomic_obj.mobem_feature_to_gene(i) for i in plot_df["index"]
        ]
        plot_df = plot_df[plot_df["genes"].apply(len) != 0]
        plot_df["genes"] = plot_df["genes"].apply(lambda v: ";".join(v)).values

        plot_df["type"] = [
            datasets.genomic_obj.mobem_feature_type(i) for i in plot_df["index"]
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
            palette=cls.GENETIC_PAL,
            hue_order=cls.GENETIC_ORDER,
            dodge=False,
            saturation=1,
        )

        plt.xlabel("Number of occurrences")
        plt.ylabel("")

        plt.legend(frameon=False)
        sns.despine()

        plt.gcf().set_size_inches(2, 0.15 * ntop)

    @classmethod
    def top_robust_features(cls, associations, ntop=30, dtype="genomic"):
        f, axs = plt.subplots(
            1, 2, sharex="none", sharey="none", gridspec_kw=dict(wspace=0.75)
        )

        for i, d in enumerate(["drug", "crispr"]):
            ax = axs[i]
            beta, pval, fdr = f"{d}_beta", f"{d}_pval", f"{d}_fdr"

            feature = "DRUG_NAME" if d == "drug" else "GeneSymbol"

            # Dataframe
            plot_df = associations.query("feature != 'msi_status'")
            plot_df = (
                plot_df.groupby([feature, "feature"])[beta, pval, fdr]
                .first()
                .reset_index()
            )

            if dtype == "genomic":
                plot_df = plot_df[
                    [
                        len(datasets.genomic_obj.mobem_feature_to_gene(i)) != 0
                        for i in plot_df["feature"]
                    ]
                ]
                plot_df = plot_df.assign(
                    type=[
                        datasets.genomic_obj.mobem_feature_type(i)
                        for i in plot_df["feature"]
                    ]
                )

            plot_df = (
                plot_df.sort_values([fdr, pval])
                .head(ntop)
                .sort_values(beta, ascending=False)
            )
            plot_df = plot_df.assign(y=range(plot_df.shape[0]))

            # Scatter
            if dtype == "genomic":
                for t in cls.GENETIC_ORDER:
                    df = plot_df.query(f"type == '{t}'")
                    ax.scatter(df[beta], df["y"], c=cls.GENETIC_PAL[t], label=t)
            else:
                ax.scatter(plot_df[beta], plot_df["y"], c=DTracePlot.PAL_DTRACE[2])

            # Labels
            for fc, y, drug, genetic in plot_df[[beta, "y", feature, "feature"]].values:
                if dtype == "genomic":
                    g_genes = "; ".join(
                        datasets.genomic_obj.mobem_feature_to_gene(genetic)
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
            ax.axvline(0, lw=0.1, c=DTracePlot.PAL_DTRACE[1])

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

    @staticmethod
    def robust_associations_barplot(fdr=0.1):
        cols, values = (
            ["DRUG_ID", "DRUG_NAME", "VERSION", "GeneSymbol"],
            ["feature", "target"],
        )
        filters = [
            ("Drug", f"(drug_fdr < {fdr})"),
            ("CRISPR", f"(crispr_fdr < {fdr})"),
            ("Both", f"(drug_fdr < {fdr}) & (crispr_fdr < {fdr})"),
        ]

        plot_df, plot_df_ppi = [], []
        for dtype, df in [
            ("Genomic", lmm_robust),
            ("Gene-expression", lmm_robust_gexp),
        ]:
            for ftype, query in filters:
                df_pairs = df.query(query).groupby(cols)[values].agg(list)

                plot_df.append(dict(dtype=dtype, ftype=ftype, count=df_pairs.shape[0]))

                for i, v in (
                    pd.Series([t for ts in df_pairs["target"] for t in ts])
                    .value_counts()
                    .iteritems()
                ):
                    plot_df_ppi.append(dict(dtype=dtype, ftype=ftype, ppi=i, count=v))

            plot_df.append(dict(dtype=dtype, ftype="Total", count=884))

        plot_df, plot_df_ppi = pd.DataFrame(plot_df), pd.DataFrame(plot_df_ppi)

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
            data=plot_df,
            palette=pal,
            kind="bar",
            hue_order=hue_order,
        )

        plt.xlabel("")
        plt.ylabel("Drug-gene associations")

        plt.grid(lw=0.3, c=DTracePlot.PAL_DTRACE[1], axis="y", alpha=5, zorder=0)

        plt.gcf().set_size_inches(2, 2)
        plt.savefig(
            "reports/robust_signif_association_barplot.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

        #
        order = ["T", "1", "2", "3", "4", "5+", "-"]
        hue_order = ["Both", "Drug", "CRISPR"]
        pal = {"Both": "#fc8d62", "Drug": "#ababab", "CRISPR": "#656565"}

        g = sns.catplot(
            "count",
            "ppi",
            "ftype",
            row="dtype",
            data=plot_df_ppi,
            palette=pal,
            kind="bar",
            order=order,
            hue_order=hue_order,
        )

        g.set_titles("{row_name}")

        for ax in g.axes[:, 0]:
            ax.set_xlabel("Drug-gene associations")
            ax.set_ylabel("PPI distance")
            ax.grid(lw=0.3, c=DTracePlot.PAL_DTRACE[1], axis="x", alpha=5, zorder=0)

        plt.gcf().set_size_inches(2, 3)
        plt.savefig(
            "reports/robust_signif_association_barplot_ppi.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")


if __name__ == "__main__":
    # - Import associations
    datasets = Association(dtype_drug="ic50")

    lmm_drug = pd.read_csv("data/drug_lmm_regressions_ic50.csv.gz")
    lmm_drug_gexp = pd.read_csv("data/drug_lmm_regressions_ic50_gexp.csv.gz")

    lmm_robust = pd.read_csv("data/drug_lmm_regressions_robust_ic50.csv.gz")
    lmm_robust_gexp = pd.read_csv("data/drug_lmm_regressions_robust_gexp_ic50.csv.gz")

    ppi = PPI().build_string_ppi(score_thres=900)
    ppi = PPI.ppi_corr(ppi, datasets.crispr)

    # - Drug betas TSNEs
    tsnes = SingleLMMTSNE(lmm_drug)

    tsnes.drug_v17_id_batch()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        "reports/drug_tsne_v17_ids_discrete.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    tsnes.drug_beta_tsne(hueby="signif")
    plt.suptitle("tSNE analysis of drug associations", y=1.05)
    plt.savefig(
        "reports/drug_associations_beta_tsne_signif.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    tsnes.drug_beta_tsne(hueby="target")
    plt.suptitle("tSNE analysis of drug associations", y=1.05)
    plt.savefig(
        "reports/drug_associations_beta_tsne.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    tsnes.drug_beta_tsne(hueby="replicates")
    plt.suptitle("tSNE analysis of drug associations", y=1.05)
    plt.savefig(
        "reports/drug_associations_beta_tsne_replicates.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    tsnes.drug_beta_tsne(hueby=SingleLMMTSNE.DRUG_TARGETS_HUE)
    plt.suptitle("tSNE analysis of drug associations", y=1.05)
    plt.savefig(
        "reports/drug_associations_beta_tsne_targets.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    # - Robust linear regressions
    RobustLMMAnalysis.genomic_histogram(datasets, ntop=40)
    plt.savefig(
        "reports/robust_mobems_countplot.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    RobustLMMAnalysis.top_robust_features(lmm_robust, ntop=30)
    plt.savefig(
        "reports/robust_top_associations.pdf", bbox_inches="tight", transparent=True
    )
    plt.close("all")

    RobustLMMAnalysis.top_robust_features(
        lmm_robust_gexp, ntop=30, dtype="gene-expression"
    )
    plt.savefig(
        "reports/robust_top_associations_gexp.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    RobustLMMAnalysis.robust_associations_barplot(fdr=0.1)

    #
    cols, values = (
        ["DRUG_ID", "DRUG_NAME", "VERSION", "feature"],
        ["GeneSymbol", "target"],
    )
    df_robust = lmm_robust.query("(drug_fdr < .1) & (crispr_fdr < .1)")
    df_robust_gexp = lmm_robust_gexp.query("(drug_fdr < .1) & (crispr_fdr < .1)")

    # - Genomic robust associations
    rassocs = [
        ("Olaparib", "FLI1", "EWSR1.FLI1_mut"),
        ("Dabrafenib", "BRAF", "BRAF_mut"),
        ("Nutlin-3a (-)", "MDM2", "TP53_mut"),
        ("Taselisib", "PIK3CA", "PIK3CA_mut"),
        ("MCL1_1284", "MCL1", "EZH2_mut"),
    ]

    # d, c, g = ('Linifanib', 'STAT5B', 'XRN1_mut')
    for d, c, g in rassocs:
        assoc = lmm_robust[
            (lmm_robust["DRUG_NAME"] == d)
            & (lmm_robust["GeneSymbol"] == c)
            & (lmm_robust["feature"] == g)
        ].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])

        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        plot_df = pd.concat(
            [
                datasets.drespo.loc[drug].rename("drug"),
                datasets.crispr.loc[c].rename("crispr"),
                datasets.genomic.loc[g].rename("genetic"),
                datasets.crispr_obj.institute.rename("Institute"),
            ],
            axis=1,
            sort=False,
        ).dropna()

        grid = DTracePlot.plot_corrplot_discrete(
            "crispr", "drug", "genetic", "Institute", plot_df
        )

        grid.ax_joint.axhline(
            y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
        )

        grid.set_axis_labels(f"{c} (scaled log2 FC)", f"{d} (ln IC50)")

        plt.suptitle(g, y=1.05, fontsize=8)

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(
            f"reports/robust_scatter_{d}_{c}_{g}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    # - Gene-expression robust associations
    rassocs = [
        ("MCL1_1284", "MCL1", "BCL2L1"),
        ("Linsitinib", "IGF1R", "IGF1R"),
        ("SN1041137233", "ERBB2", "ERBB2"),
        ("Nutlin-3a (-)", "MDM2", "BAX"),
        ("Venetoclax", "BCL2", "CDC42BPA"),
        ("AZD5582", "MAP3K7", "TNF"),
        ("SN1021632995", "MAP3K7", "TNF"),
        ("SN1043546339", "MAP3K7", "TNF"),
    ]

    # d, c, g = ('SN1043546339', 'MAP3K7', 'TNF')
    for d, c, g in rassocs:
        assoc = lmm_robust_gexp[
            (lmm_robust_gexp["DRUG_NAME"] == d)
            & (lmm_robust_gexp["GeneSymbol"] == c)
            & (lmm_robust_gexp["feature"] == g)
        ].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])

        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        plot_df = pd.concat(
            [
                datasets.drespo.loc[drug].rename("drug"),
                datasets.crispr.loc[c].rename("crispr"),
                datasets.gexp.loc[g].rename("gexp"),
                datasets.crispr_obj.institute.rename("Institute"),
                datasets.samplesheet.samplesheet["cancer_type"],
            ],
            axis=1,
            sort=False,
        ).dropna()

        #
        fig, axs = plt.subplots(1, 2, sharey="row", sharex="none")

        for i, dtype in enumerate(["crispr", "gexp"]):
            # Scatter
            for t, df in plot_df.groupby("Institute"):
                axs[i].scatter(
                    x=df[dtype],
                    y=df["drug"],
                    edgecolor="w",
                    lw=0.05,
                    s=10,
                    color=DTracePlot.PAL_DTRACE[2],
                    marker=DTracePlot.MARKERS[t],
                    label=t,
                    alpha=0.8,
                )

            # Reg
            sns.regplot(
                x=plot_df[dtype],
                y=plot_df["drug"],
                data=plot_df,
                color=DTracePlot.PAL_DTRACE[1],
                truncate=True,
                fit_reg=True,
                scatter=False,
                line_kws=dict(lw=1.0, color=DTracePlot.PAL_DTRACE[0]),
                ax=axs[i],
            )

            # Annotation
            cor, pval = pearsonr(plot_df[dtype], plot_df["drug"])
            annot_text = f"R={cor:.2g}, p={pval:.1e}"

            axs[i].text(
                0.95,
                0.05,
                annot_text,
                fontsize=4,
                transform=axs[i].transAxes,
                ha="right",
            )

            # Misc
            axs[i].axhline(
                y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
            )

            axs[i].set_ylabel(f"{d} (ln IC50)" if i == 0 else "")
            axs[i].set_xlabel(
                f"scaled log2 FC" if dtype == "crispr" else f"RNA-seq voom"
            )
            axs[i].set_title(c if dtype == "crispr" else g)

            # Legend
            axs[i].legend(prop=dict(size=4), frameon=False, loc=2)

        plt.subplots_adjust(wspace=0.05)
        plt.gcf().set_size_inches(3, 1.5)
        plt.savefig(
            f"reports/robust_scatter_gexp_{d}_{c}_{g}.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    # -
    # d, c, g = ('Venetoclax', 'BCL2', 'CDC42BPA')
    for d, c, g in [
        ("Venetoclax", "BCL2", "CDC42BPA"),
        ("SN1021632995", "MAP3K7", "TNF"),
    ]:
        assoc_all = lmm_robust_gexp[lmm_robust_gexp["DRUG_NAME"] == d]

        assoc = lmm_robust_gexp[
            (lmm_robust_gexp["DRUG_NAME"] == d)
            & (lmm_robust_gexp["GeneSymbol"] == c)
            & (lmm_robust_gexp["feature"] == g)
        ].iloc[0]

        drug = tuple(assoc[DrugResponse.DRUG_COLUMNS])
        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        plot_df = pd.concat(
            [
                datasets.drespo.loc[drug].rename("drug"),
                datasets.crispr.loc[c],
                datasets.gexp.loc[g],
                datasets.gexp_obj.rpkm.loc[g].rename(f"{g}_rpkm"),
                datasets.samplesheet.samplesheet[["institute", "cancer_type"]],
            ],
            axis=1,
            sort=False,
        ).dropna()
        plot_df[f"{g}_bin"] = (plot_df[g] < 0).astype(int)

        # Discrete
        grid = DTracePlot.plot_corrplot_discrete(
            c, "drug", f"{g}_bin", "institute", plot_df
        )

        grid.ax_joint.axhline(
            y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
        )

        grid.set_axis_labels(f"{c} (scaled log2 FC)", f"{d} (ln IC50)")

        plt.suptitle(f"{g} not expressed", y=1.05, fontsize=8)

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(
            f"reports/robust_scatter_{d}_{c}_gexp.pdf",
            bbox_inches="tight",
            transparent=True,
        )
        plt.close("all")

    #
    sensitive = list(plot_df.query(f"({g}_bin == 1) & ({c} < -0.75)").index)
    resistant = list(plot_df.query(f"({g}_bin == 1) & ({c} > -0.75)").index)

    # gene_x, gene_y = 'BCL2', 'CDC42BPA'
    for gene_x, gene_y in [("BTK", "CDC42BPA"), ("BCL2", "CDC42BPA")]:
        df = pd.concat(
            [
                datasets.crispr.loc[gene_x],
                datasets.gexp.loc[gene_y],
                datasets.samplesheet.samplesheet[["institute", "cancer_type"]],
            ],
            axis=1,
            sort=False,
        ).dropna()

        grid = DTracePlot().plot_corrplot(
            gene_x, gene_y, "institute", df, add_hline=False, add_vline=False
        )

        for c, lines in [
            (DTracePlot.PAL_DTRACE[0], sensitive),
            (DTracePlot.PAL_DTRACE[3], resistant),
        ]:
            grid.ax_joint.scatter(
                df.loc[lines, gene_x],
                df.loc[lines, gene_x],
                edgecolor="w",
                lw=0.05,
                s=10,
                color=c,
                marker="X",
                alpha=0.8,
            )

        grid.set_axis_labels(f"{gene_x}\nRNA-seq voom", f"{gene_y}\nRNA-seq voom")

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(
            f"reports/robust_gexp_scatter.pdf", bbox_inches="tight", transparent=True
        )
        plt.close("all")

    # -
    y_feature, x_feature = "STAG1", "STAG2_mut"

    plot_df = (
        pd.concat(
            [
                datasets.crispr.loc[y_feature],
                datasets.genomic.loc[x_feature].apply(lambda v: "Yes" if v else "No"),
            ],
            axis=1,
            sort=False,
        )
        .dropna()
        .sort_values(x_feature)
    )

    pal = dict(Yes=DTracePlot.PAL_DTRACE[1], No=DTracePlot.PAL_DTRACE[0])

    sns.boxplot(
        x=x_feature,
        y=y_feature,
        palette=pal,
        data=plot_df,
        linewidth=0.3,
        fliersize=1,
        notch=False,
        saturation=1.0,
        showcaps=False,
        boxprops=DTracePlot.BOXPROPS,
        whiskerprops=DTracePlot.WHISKERPROPS,
        flierprops=DTracePlot.FLIERPROPS,
        medianprops=dict(linestyle="-", linewidth=1.0),
    )

    plt.ylabel(f"{y_feature}\n(scaled log2 FC)")

    plt.gcf().set_size_inches(0.75, 1.5)
    plt.savefig(
        f"reports/robust_genomic_boxplot_{x_feature}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")

    # -
    gene_gexp, gene_crispr, gene_mut = "STAG2", "STAG1", "STAG2_mut"

    plot_df = pd.concat(
        [
            datasets.crispr.loc[gene_crispr],
            datasets.gexp.loc[gene_gexp],
            datasets.genomic.loc[gene_mut],
            datasets.crispr_obj.institute.rename("Institute"),
            datasets.samplesheet.samplesheet["cancer_type"],
        ],
        axis=1,
        sort=False,
    ).dropna()

    grid = DTracePlot.plot_corrplot_discrete(
        gene_crispr, gene_gexp, gene_mut, "Institute", plot_df
    )

    grid.set_axis_labels(
        f"{gene_crispr} (scaled log2 FC)", f"{gene_gexp} (RNA-seq voom)"
    )

    plt.suptitle(gene_mut, y=1.05, fontsize=8)

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"reports/robust_scatter_gexp_crispr_{gene_gexp}_{gene_crispr}_{gene_mut}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.close("all")
