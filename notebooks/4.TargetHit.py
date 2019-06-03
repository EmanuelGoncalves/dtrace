# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline
# %autosave 0
# %load_ext autoreload
# %autoreload 2

import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.TargetHit import TargetHit
from dtrace.DataImporter import DataPCA
from dtrace.DTracePlot import DTracePlot
from dtrace.Associations import Association
from dtrace.DTraceUtils import rpath, dpath
from dtrace.DTracePlot import MidpointNormalize
from sklearn.feature_selection import f_regression
from dtrace.DTraceEnrichment import DTraceEnrichment


# ### Import data-sets and associations

assoc = Association(dtype="ic50", load_associations=True, combine_lmm=True)


# ## MCL1 inhibitors associations

# Analysis of the significant associations between multiple MCL1 inhibitors (MCL1i) and MCL1 and MARCH5
# gene-essentiality.

hit = TargetHit("MCL1", assoc=assoc)


# ### Top associations with MCL1i

hit.top_associations_barplot()
plt.ylabel("Association p-value (-log10)")
plt.title("CRISPR associations with multiple MCL1 inhibitors")
plt.gcf().set_size_inches(5, 1.5)
plt.savefig(f"{rpath}/hit_topbarplot.pdf", bbox_inches="tight", transparent=True)


# ### Correlation plots of multiple MCL1i and MCL1/MARCH5

order = [
    tuple(d)
    for d in assoc.lmm_drug_crispr.query(
        f"(DRUG_TARGETS == 'MCL1') & (GeneSymbol == 'MCL1')"
    )[hit.dinfo].values
]
for g in ["MCL1", "MARCH5"]:
    hit.plot_target_drugs_corr(assoc, g, order=order)

    plt.savefig(
        f"{rpath}/hit_target_drugs_corr_{g}.pdf", bbox_inches="tight", transparent=True
    )


# ### BCL inhbitors association effects
#
# Associations effect sizes (betas) of inhibitors of BCL family members (MCL1, BCL2L1/BCL-XL and BCL2) drug-response
# with gene-essentiality and gene-expression.

plt.figure(figsize=(1.5, 1.5), dpi=300)
hit.plot_drug_crispr_gexp(["MCL1", "BCL2", "BCL2L1"])
plt.savefig(f"{rpath}/hit_BCLi_crispr~gexp.pdf", bbox_inches="tight", transparent=True)


# ### MCL1i drug-response predictive features
#
# A l2-regularised linear regression model with internal cross-validation for parameter optimisation
# [RidgeCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.
# RidgeCV) was used to estimate the predictive capacity [R-squared](https://scikit-learn.org/stable/modules/generated/
# sklearn.metrics.r2_score.html) of MCL1i drug-response and the features contribution.

# Both gene-essentiality and gene-expression measurements of BCL family members and regulators, defined in features
# variable, were considered.

features = [
    "MARCH5",
    "MCL1",
    "BCL2",
    "BCL2L1",
    "BCL2L11",
    "PMAIP1",
    "BAX",
    "BAK1",
    "BBC3",
    "BID",
    "BIK",
    "BAD",
]
drug_lms = hit.predict_drugresponse(assoc, features)

plt.figure(figsize=(1.5, 2.5), dpi=300)
hit.predict_r2_barplot(drug_lms)
plt.savefig(f"{rpath}/hit_rsqaured_barplot.pdf", bbox_inches="tight", transparent=True)

plt.figure(figsize=(1.5, 3), dpi=300)
hit.predict_feature_plot(drug_lms)
plt.savefig(
    f"{rpath}/hit_features_stripplot.pdf", bbox_inches="tight", transparent=True
)


# ### Gene-essentiality correlation plots

genes = [("MARCH5", "MCL1")]
for gene_x, gene_y in genes:
    plot_df = pd.concat(
        [
            assoc.crispr.loc[gene_x].rename(gene_x),
            assoc.crispr.loc[gene_y].rename(gene_y),
            assoc.crispr_obj.institute.rename("Institute"),
        ],
        axis=1,
        sort=False,
    ).dropna()

    g = DTracePlot().plot_corrplot(gene_x, gene_y, "Institute", plot_df, add_hline=True)

    g.set_axis_labels(f"{gene_x} (scaled log2 FC)", f"{gene_y} (scaled log2 FC)")

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(
        f"{rpath}/hit_scatter_{gene_x}_{gene_y}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# ## Stratification of MCL1i drug-response

# ### MCL1i inhibitors across all cell lines.

ctypes = ["Breast Carcinoma", "Colorectal Carcinoma", "Acute Myeloid Leukemia"]
genes = ["MCL1", "MARCH5"]
order = ["None", "MARCH5", "MCL1", "MCL1 + MARCH5"]
hue_order = [
    "Other",
    "Breast Carcinoma",
    "Colorectal Carcinoma",
    "Acute Myeloid Leukemia",
]


#

hit.drugresponse_boxplots(
    assoc, ctypes=ctypes, hue_order=hue_order, order=order, genes=genes
)
plt.savefig(
    f"{rpath}/hit_drugresponse_boxplot.pdf", bbox_inches="tight", transparent=True
)


# ### Drug-response of highly selective MCL1i (MCL1_1284 and AZD5991) in breast and colorectal carcinomas.

for drug in [(1956, "MCL1_1284", "RS"), (2235, "AZD5991", "RS")]:
    plot_df = assoc.build_df(
        drug=[drug], crispr=genes, crispr_discretise=True, sinfo=["cancer_type"]
    )
    plot_df["ctype"] = plot_df["cancer_type"].apply(
        lambda v: v if v in ctypes else "Other"
    )
    plot_df = plot_df.rename(columns={drug: "drug"})

    ctypes = ["Colorectal Carcinoma", "Breast Carcinoma"]

    fig, axs = plt.subplots(1, len(ctypes), sharey="all", sharex="all", dpi=300)

    for i, tissue in enumerate(ctypes):
        df = plot_df.query(f"ctype == '{tissue}'")

        g = DTracePlot().plot_multiple(
            "drug", "crispr", df, n_offset=1, n_fontsize=5, ax=axs[i]
        )

        sns.despine(ax=axs[i])

        dmax = np.log(assoc.drespo_obj.maxconcentration[drug])
        axs[i].axvline(
            dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
        )

        daml = plot_df.query("cancer_type == 'Acute Myeloid Leukemia'")["drug"].mean()
        axs[i].axvline(
            daml, linewidth=0.3, color=DTracePlot.PAL_DTRACE[0], ls=":", zorder=0
        )

        axs[i].set_xlabel(f"{drug[1]} (ln IC50, {drug[2]})")
        axs[i].set_ylabel("")

        axs[i].set_title(tissue)

    plt.gcf().set_size_inches(2 * len(ctypes), 0.75)
    plt.savefig(
        f"{rpath}/hit_drugresponse_boxplot_tissue_{drug[1]}.pdf",
        bbox_inches="tight",
        transparent=True,
    )


# ### MCL1 copy-number amplification

# MCL1 copy-number amplification association with drug-response of MCL1 gene-essentiality and inhibitor.

d, c = ("MCL1_1284", "MCL1")
drug = assoc.lmm_drug_crispr[
    (assoc.lmm_drug_crispr["DRUG_NAME"] == d)
    & (assoc.lmm_drug_crispr["GeneSymbol"] == c)
].iloc[0]

drug = tuple(drug[assoc.drespo_obj.DRUG_COLUMNS])
dmax = np.log(assoc.drespo_obj.maxconcentration[drug])

plot_df = assoc.build_df(
    drug=[drug], crispr=[c], cn=[hit.target], sinfo=["institute", "ploidy"]
)
plot_df = plot_df.rename(columns={drug: "drug"})
plot_df = plot_df.assign(
    amp=[assoc.cn_obj.is_amplified(c, p) for c, p in plot_df[["cn_MCL1", "ploidy"]].values]
)

grid = DTracePlot.plot_corrplot_discrete(
    f"crispr_{c}", "drug", f"amp", "institute", plot_df
)
grid.ax_joint.axhline(
    y=dmax, linewidth=0.3, color=DTracePlot.PAL_DTRACE[2], ls=":", zorder=0
)
grid.set_axis_labels(f"{c} (scaled log2 FC)", f"{d} (ln IC50)")
plt.suptitle("MCL1 amplification", y=1.05, fontsize=8)
plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig(
    f"{rpath}/hit_scatter_{d}_{c}_amp.pdf", bbox_inches="tight", transparent=True
)


#

genes = ["MCL1", "MARCH5"]
drug = (1956, "MCL1_1284", "RS")
drug2 = (2235, "AZD5991", "RS")
order = ["None", "MARCH5", "MCL1", "MCL1 + MARCH5"]

gsea_h = pd.read_csv(f"{dpath}/ssgsea/GExp_h.all.v6.2.symbols.gmt.csv.gz")

dmax = np.log(assoc.drespo_obj.maxconcentration[drug])


df = assoc.build_df(
    drug=[drug, drug2],
    crispr=genes,
    gexp=["BCL2L1", "MARCH5"],
    sinfo=["cancer_type", "growth", "model_name", "institute", "ploidy"],
    crispr_discretise=True
).dropna()
df["crispr"] = pd.Categorical(df["crispr"], order)


#

df_coread = (
    pd.concat(
        [
            df.query(f"cancer_type == 'Colorectal Carcinoma'"),
            assoc.samplesheet.load_coread_info(),
            gsea_h[gsea_h["gset"] == "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"].set_index("sample")["score"].rename("EMT")
        ],
        axis=1,
        sort=False,
    )
    .dropna(subset=[drug])
    .sort_values("crispr", ascending=False)
)
df_coread.to_excel(f"{dpath}/matrix_MCL1_COREAD.xlsx")
# df_coread = df_coread[df_coread.index.isin(assoc.gexp)]


#

df_brca = (
    pd.concat(
        [
            df.query(f"cancer_type == 'Breast Carcinoma'"),
            assoc.samplesheet.load_brca_info(),
        ],
        axis=1,
        sort=False,
    )
    .dropna(subset=[drug])
    .sort_values("crispr", ascending=False)
)
df_brca.to_excel(f"{dpath}/matrix_MCL1_BRCA.xlsx")


#

order = ["MCL1_1284", "AZD5991", "crispr_MCL1", "crispr_MARCH5", "gexp_BCL2L1"]

plot_df = df_coread.rename(columns={drug: "MCL1_1284", drug2: "AZD5991"})
plot_df = pd.melt(plot_df, id_vars=["Joint Classification"], value_vars=order).dropna()

f, axs = plt.subplots(
    len(order),
    1,
    sharex="none",
    sharey="none",
    dpi=300
)

for i, t in enumerate(order):
    g = sns.boxplot(
        x="value", y="Joint Classification", data=plot_df.query(f"variable == '{t}'"),
        orient="h",
        linewidth=0.3,
        fliersize=1,
        notch=False,
        saturation=1.0,
        showcaps=False,
        boxprops=DTracePlot.BOXPROPS,
        whiskerprops=DTracePlot.WHISKERPROPS,
        flierprops=DTracePlot.FLIERPROPS,
        medianprops=dict(linestyle="-", linewidth=1.0),
        ax=axs[i]
    )

    axs[i].set_xlabel("")
    axs[i].set_ylabel(t)

plt.gcf().set_size_inches(1.5, len(order))

plt.savefig(
    f"{rpath}/hit_cris_boxplots.pdf", bbox_inches="tight"
)
plt.show()


#

order = ["MCL1_1284", "AZD5991", "crispr_MCL1", "crispr_MARCH5", "gexp_BCL2L1"]

plot_df = df_brca.rename(columns={drug: "MCL1_1284", drug2: "AZD5991"})
plot_df = pd.melt(plot_df, id_vars=["PAM50"], value_vars=order).dropna()

f, axs = plt.subplots(
    len(order),
    1,
    sharex="none",
    sharey="none",
    dpi=300
)

for i, t in enumerate(order):
    g = sns.boxplot(
        x="value", y="PAM50", data=plot_df.query(f"variable == '{t}'"),
        orient="h",
        linewidth=0.3,
        fliersize=1,
        notch=False,
        saturation=1.0,
        showcaps=False,
        boxprops=DTracePlot.BOXPROPS,
        whiskerprops=DTracePlot.WHISKERPROPS,
        flierprops=DTracePlot.FLIERPROPS,
        medianprops=dict(linestyle="-", linewidth=1.0),
        ax=axs[i]
    )

    axs[i].set_xlabel("")
    axs[i].set_ylabel(t)

plt.gcf().set_size_inches(1.5, len(order))

plt.savefig(
    f"{rpath}/hit_pam50_boxplots.pdf", bbox_inches="tight"
)
plt.show()


#

methy_samples = list(set(assoc.methy).intersection(df_coread.index))
df_coread_methy_pca = DataPCA.perform_pca(assoc.methy[methy_samples])
df_coread_methy = pd.concat([df_coread, df_coread_methy_pca["column"]["pcs"]], axis=1, sort=False).loc[methy_samples]

pcs_pval_coread = [f"PC{i+1}" for i in range(10)]
pcs_pval_coread = pd.Series(
    f_regression(df_coread_methy[pcs_pval_coread], df_coread_methy[drug])[1], index=pcs_pval_coread
).sort_values()


#

plt.figure(figsize=(2.0, 1.5), dpi=300)
sc = plt.scatter(
    df_coread_methy["PC1"],
    df_coread_methy["PC4"],
    c=df_coread_methy[drug],
    cmap=DTracePlot.CMAP_DTRACE,
    s=10,
    norm=MidpointNormalize(midpoint=dmax),
)
cb = plt.colorbar(sc)
plt.show()


#

loadings_coread_methy = df_coread_methy_pca["column"]["pca"].components_
loadings_coread_methy = pd.DataFrame(
    loadings_coread_methy,
    index=[f"PC{i+1}" for i in range(10)],
    columns=assoc.methy[methy_samples].index,
).T


gsea = DTraceEnrichment(
    gmts=[
        "h.all.v6.2.symbols.gmt",
        "c2.cp.kegg.v6.2.symbols.gmt"
    ],
    verbose=1,
)

loadings_coread_methy_gsea = {}
for gmt in gsea.gmts:
    logging.getLogger("DTrace").info(gmt)

    loadings_coread_methy_gsea[gmt] = {}
    for pc in loadings_coread_methy:
        loadings_coread_methy_gsea[gmt][pc] = gsea.gsea_enrichments(loadings_coread_methy[pc], gmt)

loadings_coread_methy_gsea["h.all.v6.2.symbols.gmt"]["PC4"]
loadings_coread_methy_gsea["c2.cp.kegg.v6.2.symbols.gmt"]["PC4"]


# Copyright (C) 2019 Emanuel Goncalves
