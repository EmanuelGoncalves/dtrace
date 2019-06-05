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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.DTraceUtils import rpath
from dtrace.Associations import Association
from dtrace.TargetBenchmark import TargetBenchmark


# ### Import data-sets and associations

assoc = Association(dtype="ic50", load_simple_lm=True)
assoc_lmm = Association(dtype="ic50", load_simple_lm=False, load_associations=True)

target = TargetBenchmark(assoc=assoc, fdr=0.1)


# Top associations between drug-response and gene-essentiality

assoc.lmm_drug_crispr.head(15)


# Volcano plot of the significant associations

plt.figure(figsize=(1.5, 3), dpi=300)
target.signif_volcano()
plt.savefig(
    f"{rpath}/simplelm_target_benchmark_volcano.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Distribution of the signifcant Drug-Gene associations across a protein-protein interaction network

for dtype in ["crispr"]:

    fig, axs = plt.subplots(2, 1, figsize=(2.5, 5), dpi=300)

    # Boxplot
    target.drugs_ppi(dtype, ax=axs[0])

    axs[0].set_xlabel("Associated gene position in PPI")
    axs[0].set_ylabel("Bonferroni adj. p-value")
    axs[0].set_title("Significant associations\n(adj. p-value < 10%)")

    # Count plot
    target.drugs_ppi_countplot(dtype, ax=axs[1])

    axs[1].set_xlabel("Associated gene position in PPI")
    axs[1].set_ylabel("Number of associations")

    plt.savefig(
        f"{rpath}/simplelm_benchmark_ppi_distance_{dtype}.pdf",
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()


# Histogram of drugs with at least one significant association across the protein-protein network

plt.figure(figsize=(2, 1), dpi=300)
target.countplot_drugs_significant()
plt.savefig(
    f"{rpath}/simplelm_benchmark_association_signif_countplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Comparing significant associations simple LM with LMM

plot_df = pd.concat([
    assoc.lmm_drug_crispr.query("fdr < .1")["target"].value_counts().rename("lm"),
    assoc_lmm.lmm_drug_crispr.query("fdr < .1")["target"].value_counts().rename("lmm"),
], axis=1, sort=False).replace(np.nan, 0).astype(int)
plot_df = pd.melt(count_lm.reset_index(), id_vars="index")

plt.figure(figsize=(3, 2), dpi=300)
sns.barplot("index", "value", "variable", data=plot_df, order=target.PPI_ORDER, palette=dict(lmm="#fc8d62", lm="#2b8cbe"))
plt.legend(title="Model", frameon=False)
plt.xlabel("Relation of the gene to the drug targets in a PPI")
plt.ylabel("Number of associations")
plt.title("Significant CRISPR ~ Drug associations")
plt.savefig(
    f"{rpath}/simplelm_comparison_n_signif_assoc.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Pie chart of significant associations per unique durgs ordered by distance in the PPI

plt.figure(figsize=(2, 2), dpi=300)
target.pichart_drugs_significant()
plt.savefig(
    f"{rpath}/simplelm_benchmark_association_signif_piechart.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Significant associations p-value (y-axis) maximum screened concentration

target.signif_maxconcentration_scatter()
plt.gcf().set_size_inches(2.5, 2.5)
plt.savefig(
    f"{rpath}/simplelm_benchmark_signif_scatter_maxconcentration.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Copyright (C) 2019 Emanuel Goncalves
