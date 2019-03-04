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

import matplotlib.pyplot as plt
from dtrace import rpath
from dtrace.Associations import Association
from dtrace.TargetBenchmark import TargetBenchmark


# ### Import data-sets and associations

assoc = Association(dtype="ic50", load_simple_lm=True)

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
