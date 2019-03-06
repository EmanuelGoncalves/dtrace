#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from DataImporter import DrugResponse
from dtrace.DTracePlot import DTracePlot
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
            lmm_dsingle,
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
            for i in lmm_dsingle.query(f"fdr < {self.fdr}")[
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
