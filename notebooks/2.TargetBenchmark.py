# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from natsort import natsorted
from crispy.Utils import Utils
from scipy.stats import ttest_ind
from dtrace.DTraceUtils import rpath
from dtrace.DataImporter import Sample
from dtrace.Associations import Association
from dtrace.TargetBenchmark import TargetBenchmark


# ### Import data-sets and associations

assoc = Association(load_associations=True, load_ppi=True)

target = TargetBenchmark(assoc=assoc, fdr=0.1)


# ## Drug response and gene fitness associations

# Top associations between drug response and gene fitness

assoc.lmm_drug_crispr.head(15)


# Top associations between drug response and gene expression

assoc.lmm_drug_gexp.head(15)


# Volcano plot of the significant associations.

plt.figure(figsize=(3, 1.5), dpi=300)
target.signif_volcano()
plt.savefig(
    f"{rpath}/target_benchmark_volcano.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Top 50 most strongly correlated drugs

target.top_associations_barplot()
plt.savefig(
    f"{rpath}/target_benchmark_associations_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Representative examples of the drug-gene associations.

ss = Sample().samplesheet

dgs = [
    ("Alpelisib", "PIK3CA"),
    ("Nutlin-3a (-)", "MDM2"),
    ("MCL1_1284", "MCL1"),
    ("MCL1_1284", "MARCH5"),
    ("Venetoclax", "BCL2"),
    ("Volasertib", "PLK1"),
    ("Rigosertib", "PLK1"),
    ("Linsitinib", "CNPY2"),
    ("Cetuximab", "EGFR"),
    ("Olaparib", "PARP1"),
    ("Olaparib", "PARP2"),
    ("Capmatinib", "MET"),
    ("SGX-523", "MET"),
    ("AZD6094", "MET"),
    ("Savolitinib", "MET"),
    ("ASLAN002", "MET"),
    ("Tepotinib", "MET"),
    ("Crizotinib", "MET"),
    ("Foretinib", "MET"),
    ("Merestinib", "MET"),
    ("PD173074", "FGFR1"),
    ("AZD4547", "FGFR1"),
    ("AZD4547", "FGFR2"),
    ("FGFR_3831", "FGFR2"),
]

dg = ("Linsitinib", "IGF1R")
for dg in dgs:
    pair = assoc.by(assoc.lmm_drug_crispr, drug_name=dg[0], gene_name=dg[1]).iloc[0]

    drug = tuple(pair[assoc.dcols])

    dmax = np.log(assoc.drespo_obj.maxconcentration[drug])
    annot_text = f"Beta={pair['beta']:.2g}, FDR={pair['fdr']:.1e}"

    plot_df = assoc.build_df(drug=[drug], crispr=[dg[1]], sinfo=["institute"]).dropna()
    plot_df = plot_df.rename(columns={drug: "drug"})

    g = target.plot_corrplot(
        f"crispr_{dg[1]}",
        "drug",
        "institute",
        plot_df,
        annot_text=annot_text,
    )
    g.ax_joint.axhline(
        y=dmax, linewidth=0.3, color=target.PAL_DTRACE[2], ls=":", zorder=0
    )
    g.set_axis_labels(f"{dg[1]} (scaled log2 FC)", f"{dg[0]} (ln IC50)")
    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/association_drug_scatter_{dg[0]}_{dg[1]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Kinobeads drug-protein affinity measurements for 84 kinase inhibitors were obtained from an independent study [1] as
# aparent pKd (nM). These were ploted for the signifincant associations versus non-significant (Log-ratio test BH-FDR)
# found in our study.
#
#
# [1] Klaeger S, Heinzlmeir S, Wilhelm M, Polzer H, Vick B, Koenig P-A, Reinecke M, Ruprecht B, Petzoldt S, Meng C,
# Zecha J, Reiter K, Qiao H, Helm D, Koch H, Schoof M, Canevari G, Casale E, Depaolini SR, Feuchtinger A, et al. (2017)
# The target landscape of clinical kinase drugs. Science 358: eaan4368

plt.figure(figsize=(0.75, 2.0), dpi=300)
target.boxplot_kinobead()
plt.savefig(
    f"{rpath}/target_benchmark_kinobeads.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Association effect sizes with between drugs and their know targets

plt.figure(figsize=(2, 2), dpi=300)
target.beta_histogram()
plt.savefig(
    f"{rpath}/target_benchmark_beta_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()

# P-value histogram of the Drug-Genes associations highlighting Drug-Target associations.

plt.figure(figsize=(2, 1), dpi=300)
target.pval_histogram()
plt.savefig(
    f"{rpath}/target_benchmark_pval_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Distribution of the signifcant Drug-Gene associations across a protein-protein interaction network, with
# gene essentiality expression.

for dtype in ["crispr", "gexp"]:
    fig, axs = plt.subplots(2, 1, figsize=(1.5, 3), dpi=300)

    # Boxplot
    target.drugs_ppi(dtype, ax=axs[0])

    axs[0].set_xlabel("Associated gene position in PPI")
    axs[0].set_ylabel("Adj. p-value")
    axs[0].set_title("Significant associations\n(adj. p-value < 10%)")

    # Count plot
    target.drugs_ppi_countplot(dtype, ax=axs[1])

    axs[1].set_xlabel("Associated gene position in PPI")
    axs[1].set_ylabel("Number of associations")

    plt.savefig(
        f"{rpath}/target_benchmark_ppi_distance_{dtype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Background distribution of all Drug-Gene associations tested.

for dtype in ["crispr", "gexp"]:
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    target.drugs_ppi_countplot_background(dtype)
    plt.savefig(
        f"{rpath}/target_benchmark_ppi_distance_{dtype}_countplot_bkg.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# Breakdown numbers of (i) all the drugs screened, (ii) unique drugs, (iii) their annotation status, and (iv) those
# which at least one of the canonical targets were targeted with the CRISPR-Cas9 screen.

plt.figure(figsize=(2, 0.75), dpi=300)
target.countplot_drugs()
plt.savefig(
    f"{rpath}/target_benchmark_association_countplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Histogram of drugs with at least one significant association across the protein-protein network

plt.figure(figsize=(2, 1), dpi=300)
target.countplot_drugs_significant()
plt.savefig(
    f"{rpath}/target_benchmark_association_signif_countplot.pdf",
    bbox_inches="tight",
    transparent=True,
)


# Pie chart and barplot of significant associations per unique durgs ordered by distance in the PPI

plt.figure(figsize=(2, 2), dpi=300)
target.pichart_drugs_significant()
plt.savefig(
    f"{rpath}/target_benchmark_association_signif_piechart.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()

target.barplot_drugs_significant()
plt.savefig(
    f"{rpath}/target_benchmark_association_signif_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Number of significant associations found with drugs from the two different types of screening proceedures

plt.figure(figsize=(0.75, 1.5), dpi=300)
target.signif_per_screen()
plt.savefig(
    f"{rpath}/target_benchmark_significant_by_screen.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Heatmap counting the number of drugs which have a significant association with CRISPR and/or with a genomic marker

plt.figure(figsize=(1, 1), dpi=300)
target.signif_genomic_markers()
plt.savefig(
    f"{rpath}/target_benchmark_signif_genomic_heatmap.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Significant associations p-value (y-axis) spread across the number of times a drug displayed an IC50 lower than the
# maximum screened concentration.

for x in ["below_%", "min_resp"]:
    target.signif_maxconcentration_scatter(x_axis=x)
    plt.gcf().set_size_inches(2.5, 2.5)
    plt.savefig(
        f"{rpath}/target_benchmark_signif_scatter_maxconcentration_{x}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Drug-Gene CRISPR associations p-value (-log10) versus Drug-Genomic associations p-value (-log10).

plt.figure(figsize=(2, 2), dpi=300)
target.signif_fdr_scatter()
plt.savefig(
    f"{rpath}/target_benchmark_signif_fdr_scatter.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Top associations

drugs = ["AZD5582", "IAP_5620", "VE821", "AZD6738", "VE-822", "Cetuximab", "Ibrutinib"]

for d in ["Alpelisib", "AZD8186", "Buparlisib", "Omipalisib"]:
    plot_df, ax = target.drug_top_associations(d, fdr_thres=0.1)
    plt.gcf().set_size_inches(0.2 * plot_df.shape[0], 2)
    plt.savefig(
        f"{rpath}/target_benchmark_top_associations_barplot_{d}.pdf",
        bbox_inches="tight",
        transparent=True,
        dpi=600,
    )
    plt.show()


# PARP inhibitors (olaparib and talazoparib) associations

genes = ["STAG1", "LIG1", "FLI1", "PARP1", "PARP2", "PARP3", "PCGF5", "XRCC1", "RHNO1"]

for drug in ["Olaparib", "Talazoparib"]:
    plt.figure(figsize=(2, 1.5), dpi=300)
    target.drug_notarget_barplot(drug, genes)
    plt.savefig(
        f"{rpath}/target_benchmark_drug_notarget_{drug}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# Clustermap of drug association betas

betas_crispr = pd.pivot_table(
    assoc.lmm_drug_crispr.query("VERSION == 'GDSC2'"),
    index=["DRUG_ID", "DRUG_NAME"],
    columns="GeneSymbol",
    values="beta",
)


target.lmm_betas_clustermap(betas_crispr)
plt.gcf().set_size_inches(8, 8)
plt.savefig(
    f"{rpath}/target_benchmark_clustermap_betas_crispr.png",
    bbox_inches="tight",
    dpi=300,
)
plt.show()


plt.figure(figsize=(2, 2), dpi=300)
target.lmm_betas_clustermap_legend()
plt.axis("off")
plt.savefig(
    f"{rpath}/target_benchmark_clustermap_betas_crispr_legend.pdf", bbox_inches="tight"
)
plt.show()


# Drug association with gene-expression

dgs = [
    ("Nutlin-3a (-)", "MDM2"),
    ("Poziotinib", "ERBB2"),
    ("Afatinib", "ERBB2"),
    ("WEHI-539", "BCL2L1"),
]
for dg in dgs:
    pair = assoc.by(assoc.lmm_drug_gexp, drug_name=dg[0], gene_name=dg[1]).iloc[0]

    drug = tuple(pair[assoc.dcols])

    dmax = np.log(assoc.drespo_obj.maxconcentration[drug])
    annot_text = f"Beta={pair['beta']:.2g}, FDR={pair['fdr']:.1e}"

    plot_df = assoc.build_df(drug=[drug], gexp=[dg[1]]).dropna()
    plot_df = plot_df.rename(columns={drug: "drug"})
    plot_df["Institute"] = "Sanger"

    g = target.plot_corrplot(
        f"gexp_{dg[1]}", "drug", "Institute", plot_df, annot_text=annot_text
    )

    g.ax_joint.axhline(
        y=dmax, linewidth=0.3, color=target.PAL_DTRACE[2], ls=":", zorder=0
    )

    g.set_axis_labels(f"{dg[1]} (voom)", f"{dg[0]} (ln IC50)")

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(
        f"{rpath}/association_drug_gexp_scatter_{dg[0]}_{dg[1]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# CRISPR correlation profiles

for gene_x, gene_y in [("MARCH5", "MCL1"), ("SHC1", "EGFR")]:
    plot_df = assoc.build_df(crispr=[gene_x, gene_y], sinfo=["institute"]).dropna()

    g = target.plot_corrplot(
        f"crispr_{gene_x}", f"crispr_{gene_y}", "institute", plot_df, annot_text=""
    )

    g.set_axis_labels(f"{gene_x} (scaled log2 FC)", f"{gene_y} (scaled log2 FC)")

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/association_scatter_{gene_x}_{gene_y}.pdf",
        bbox_inches="tight",
        transparent=True,
    )
    plt.show()


# PPI weighted network

ppi_examples = [
    ("Nutlin-3a (-)", 0.4, 1, ["RPL37", "UBE3B"]),
    ("AZD3759", 0.3, 2, None),
    ("Cetuximab", 0.3, 2, None),
]
for d, t, o, e in ppi_examples:
    graph = assoc.ppi.plot_ppi(
        d,
        assoc.lmm_drug_crispr,
        assoc.ppi_string_corr,
        corr_thres=t,
        norder=o,
        fdr=0.1,
        exclude_nodes=e,
    )
    graph.write_pdf(f"{rpath}/association_ppi_{d}.pdf")


# Drug-target CRISPR variability between drug significant association

plot_df = assoc.lmm_drug_crispr.query("target == 'T'")
plot_df["crispr_std"] = assoc.crispr.loc[plot_df["GeneSymbol"]].std(1).values
plot_df["drug_std"] = (
    assoc.drespo.loc[[tuple(v) for v in plot_df[target.dinfo].values]].std(1).values
)
plot_df["signif"] = plot_df["fdr"].apply(lambda v: "Yes" if v < target.fdr else "No")

pal = {"No": target.PAL_DTRACE[1], "Yes": target.PAL_DTRACE[2]}

for dtype in ["crispr_std", "drug_std"]:
    plt.figure(figsize=(1.0, 1.5), dpi=600)

    ax = sns.boxplot(
        "signif",
        dtype,
        data=plot_df,
        palette=pal,
        linewidth=0.3,
        fliersize=1.5,
        flierprops=target.FLIERPROPS,
        showcaps=False,
        notch=True,
    )

    t, p = ttest_ind(
        plot_df.query(f"signif == 'Yes'")[dtype],
        plot_df.query(f"signif == 'No'")[dtype],
        equal_var=False,
    )

    ax.text(
        0.95,
        0.05,
        f"Welch's t-test p={p:.1e}",
        fontsize=4,
        transform=ax.transAxes,
        ha="right",
    )

    ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="y")

    if dtype == "crispr_std":
        ness_std = assoc.crispr.loc[Utils.get_non_essential_genes()].std(1).median()
        plt.axhline(ness_std, ls=":", lw=0.3, c="k", zorder=0)

        ess_std = assoc.crispr.loc[Utils.get_essential_genes()].std(1).median()
        plt.axhline(ess_std, ls=":", lw=0.3, c="k", zorder=0)

    plt.title("Drug ~ Gene association")
    plt.xlabel("Significant association\n(adj. p-value < 10%)")
    plt.ylabel(
        "Drug-target fold-change\nstandard deviation"
        if dtype == "crispr_std"
        else "Drug IC50 (ln)\nstandard deviation"
    )

    plt.savefig(
        f"{rpath}/target_benchmark_drug_signif_{dtype}_boxplot.pdf", bbox_inches="tight"
    )
    plt.show()


# Overview of drug taregt classes

dgroups = pd.Series(
    {
        "#ebf3fa": {"FGFR1", "FGFR2", "FGFR3", "FGFR4"},
        "#adcee5": {"RAF1", "BRAF"},
        "#6fa8d1": {"MAPK1", "MAPK3"},
        "#3182bd": {"MAP2K1", "MAP2K2"},

        "#f2a17a": {"PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG"},
        "#ec7b43": {"AKT1", "AKT2", "AKT3"},
        "#e6550d": {"MTOR"},

        "#6fc088": {"EGFR"},
        "#aedcbc": {"ERBB2"},
        "#edf7f0": {"IGF1R"},

        "#938bc2": {"ATR", "ATM"},
        "#756bb1": {"WEE1", "TERT"},

        "#eeaad3": {"BTK"},
        "#e78ac3": {"SYK"},

        "#cbeae0": {"TOP2A", "TOP1"},
        "#a9ddcc": {"BRD2", "BRD3", "BRD4"},
        "#87cfb9": {"HSP90AA1", "HSP90AB1"},
        "#66c2a5": {"PARP1", "PARP2"},

        "#fffee6": {"XIAP", "BIRC2", "BIRC3"},
        "#fefd9e": {"BCL2", "BCL2L1"},
        "#fefb57": {"MCL1"},

        "#636363": {"KIT"},
        "#aaaaaa": {"FLT1", "FLT3", "FLT4"},

        "#fff7e5": {"PLK1"},
        "#f6e0ac": {"ROCK1", "ROCK2"},
        "#eec872": {"CDK2", "CDK4", "CDK6", "CDK8", "CDK7", "CDK9", "CDK12"},
        "#e5b139": {"CHEK1", "CHEK2"},
        "#dd9a00": {"AURKA", "AURKB", "AURKC"},

        "#bc80bd": {"HDAC1", "HDAC2", "HDAC3", "HDAC6", "HDAC8"},
        "#f8eced": {"ABL1"},
        "#e0bec0": {"TNKS", "TNKS2"},
        "#c89092": {"PAK1"},
        "#b06265": {"MET"},
        "#983539": {"JAK1", "JAK2", "JAK3"},

        "#f2f2f2": {"No target"},
        "#d7d7d7": {"Other target"},
        "#bbbbbb": {"Multiple targets"},
    }
)


def get_drug_target_color(dname):
    if dname not in target.d_targets:
        return "#f2f2f2"

    dname_targets = [
        c
        for c in dgroups.index
        if len(target.d_targets[dname].intersection(dgroups[c])) > 0
    ]

    if len(dname_targets) == 0:
        return "#d7d7d7"

    elif len(dname_targets) == 1:
        return dname_targets[0]

    else:
        return "#bbbbbb"


plot_df = pd.DataFrame(
    [
        dict(
            drug_name=d,
            target=target.d_targets[d] if d in target.d_targets else np.nan,
            color=get_drug_target_color(d),
            group=";".join(dgroups[get_drug_target_color(d)]),
        )
        for d in target.d_sets_name["all"]
    ]
)
plot_df = plot_df.groupby(["group", "color"]).count()["drug_name"].reset_index()

plt.figure(figsize=(2.5, 5), dpi=600)
sns.barplot(
    "drug_name",
    "group",
    data=plot_df,
    order=[";".join(c) for c in dgroups if "Other target" not in c],
    linewidth=0,
    palette=plot_df.set_index("group")["color"].to_dict(),
    orient="h",
    saturation=1,
)
plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
plt.xlabel("Number of drugs")
plt.ylabel("")
plt.title("Most represented drug target classes")
plt.savefig(f"{rpath}/target_benchmark_drugs_group_barplot.pdf", bbox_inches="tight")
plt.close("all")


# Representation of drug target classes on significant associations

plot_df = assoc.lmm_drug_crispr.query("target == 'T'")

plot_df = pd.concat([
    plot_df.groupby("DRUG_NAME")["DRUG_TARGETS"].first().value_counts().rename("total"),
    plot_df.query("fdr < .1").groupby("DRUG_NAME")["DRUG_TARGETS"].first().value_counts().rename("signif"),
], axis=1, sort=False).fillna(0).query("total >= 3")
plot_df = plot_df.assign(perc=plot_df.eval("signif / total"))
plot_df = plot_df.sort_values("perc", ascending=False).reset_index()


plt.figure(figsize=(2.5, 4.5), dpi=600)

plt.scatter(
    plot_df["perc"],
    plot_df.index,
    c=plot_df["signif"],
    s=plot_df["signif"] + 2,
    marker="o",
    edgecolor="",
    cmap="viridis_r",
)

for i, (p, s, tt, t) in plot_df[["perc", "signif", "total", "index"]].iterrows():
    plt.text(
        (p - .015) if p == 1 else (p + 0.015),
        i,
        f"{t.replace(';', ', ')} (n={s:.0f}/{tt})",
        color=target.PAL_DTRACE[2],
        va="center",
        ha="right" if p == 1 else "left",
        fontsize=4,
        zorder=10
    )

plt.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")

plt.xlabel("Percentage of drugs in target class\nwith significant association with nominal target(s)")
plt.title("Representation of drug target classes")

plt.savefig(f"{rpath}/target_benchmark_drug_signif_enrichment_barplot.pdf", bbox_inches="tight")
plt.close("all")


# Drugs number of targets in ChEMBL

plot_df = pd.concat([
    target.chembl_ntargets.groupby(target.dinfo)["target_count"].min(),
    assoc.lmm_drug_crispr.groupby(target.dinfo)["pval"].min(),
], axis=1).dropna().reset_index()
plot_df["signif"] = plot_df["DRUG_NAME"].isin(target.d_sets_name["tested_corrected"]).apply(lambda v: "Yes" if v else "No")
plot_df["target_count_bin"] = plot_df["target_count"].apply(lambda v: Utils.bin_cnv(v, thresold=10))

order = ["Yes", "No"]

_, ax = plt.subplots(1, 1, figsize=(3, .75), dpi=600)

sns.boxplot(
    np.log(plot_df["target_count"]),
    plot_df["signif"],
    order=order,
    palette=target.PAL_YES_NO,
    notch=True,
    linewidth=0.3,
    fliersize=1.5,
    flierprops=target.FLIERPROPS,
    showcaps=False,
    orient="h",
    saturation=1,
    ax=ax,
)

groupn = plot_df.groupby("signif")["DRUG_NAME"].count()
ax.set_yticklabels([f"{i} (n={groupn[i]})" for i in order])

t, pval = ttest_ind(
    np.log(plot_df.query("signif == 'Yes'")["target_count"]),
    np.log(plot_df.query("signif == 'No'")["target_count"]),
    equal_var=False,
)
ax.set_title(f"Welch's t-test p-value={pval:.1g}")

ax.grid(True, ls="-", lw=0.1, alpha=1.0, zorder=0, axis="x")
ax.set_xlabel("Number of drug targets in ChEMBL (log)")
ax.set_ylabel("Drug-Target\nsignificant association")

plt.savefig(f"{rpath}/target_benchmark_chembl_boxplot.pdf", bbox_inches="tight")
plt.close("all")


# Copyright (C) 2019 Emanuel Goncalves
