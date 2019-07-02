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
from dtrace.DTraceUtils import rpath
from dtrace.Associations import Association
from dtrace.Preliminary import DrugPreliminary, CrisprPreliminary


# ### Import data-sets
dtype = "ic50"

assoc = Association(dtype=dtype)


# ## Principal Component Analysis (PCA)
#
# Import PCA results performed on the drug-response and CRISPR-Cas9 data-sets both per drug/gene and per samples. Note
# PCA is performed after running the first notebook (0.Associations).

pca_drug = assoc.drespo_obj.import_pca()[dtype]
pca_crispr = assoc.crispr_obj.import_pca()


# ## Growth-rate correlation analysis
#
# Correlation of cell lines growth rates (unperturbed) with drug-response (ln IC50).

g_corr = assoc.drespo_obj.perform_growth_corr(subset=assoc.samples)[dtype]


# Correlation of cell lines growth rates (unperturbed) with CRISPR-Cas9 (scaled log2 fold-change; median essential = -1)

c_corr = assoc.crispr_obj.perform_growth_corr(subset=assoc.samples)


# ## Strong viability responses
#
# Count for each compound the number of IC50s that are lower than 50% of the maximum concentration used for the
# respective compound

num_resp_drug = assoc.drespo_obj.perform_number_responses(
    resp_thres=0.5, subset=assoc.samples
)


# Count for each gene the number of fold-changes (scaled log2) that are lower than -0.5, i.e. have a viability impact
# lower than 50% of that observed in the set of known essential genes.

num_resp_crispr = assoc.crispr_obj.perform_number_responses(
    thres=-0.5, subset=assoc.samples
)


# # Drug-response


# Drug-response (IC50s) measurements across cell lines cumulative distribution

plt.figure(figsize=(2.5, 1.5), dpi=300)
DrugPreliminary.histogram_drug(assoc.drespo.count(1))
plt.savefig(
    f"{rpath}/preliminary_drug_histogram_drug.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Cumulative distribution of strong drug-response measurements. Strong response measurements are defined as IC50 < 50%
# Max. concentration

plt.figure(figsize=(3, 1.5), dpi=300)
DrugPreliminary.histogram_strong_response(num_resp_drug)
plt.xlabel("Number of cell lines screened")
plt.ylabel("Fraction of drugs")
plt.title(
    "Cumulative distribution of drug measurements lower than\n50% of the maximum screened concentration"
)
plt.savefig(
    f"{rpath}/preliminary_drug_response_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Cumulative distribution of samples with measurements across all compounds screened

plt.figure(figsize=(2.5, 1.5), dpi=300)
DrugPreliminary.histogram_sample(assoc.drespo.count(0))
plt.savefig(
    f"{rpath}/preliminary_drug_histogram_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Principal components of drugs

DrugPreliminary.pairplot_pca_by_rows(pca_drug)
plt.suptitle("PCA drug response (Drugs)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# Principal components of samples in the drug-response

DrugPreliminary.pairplot_pca_by_columns(pca_drug)
plt.suptitle("PCA drug response (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Principal components of samples in the drug-response coloured by cancer type

DrugPreliminary.pairplot_pca_samples_cancertype(
    pca_drug, assoc.samplesheet.samplesheet["cancer_type"]
)
plt.suptitle("PCA drug response (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot_cancertype.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Drug-response PCs correlation with growth-rates

plot_df = assoc.samplesheet.growth_corr(pca_drug["column"]["pcs"].T)

plt.figure(figsize=(1.5, 1.5), dpi=300)
DrugPreliminary.growth_corrs_pcs_barplot(plot_df)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_pcs_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Samples drug-response PC1 correlation with growth-rate

g = DrugPreliminary.corrplot_pcs_growth(
    pca_drug, assoc.samplesheet.samplesheet["growth"], "PC1"
)
g.fig.set_size_inches(1.5, 1.5)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Histogram of samples drug-response PC1 correlation with growth-rate

plt.figure(figsize=(2, 2), dpi=300)
DrugPreliminary.growth_correlation_histogram(g_corr)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Top correlated drugs with growth-rate

plt.figure(figsize=(2, 4), dpi=300)
DrugPreliminary.growth_correlation_top_drugs(g_corr)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot_top.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# # CRISPR-Cas9


# Cumulative distribution of strong gene essentiality measurements. Strong response measurements are defined as log2
# scaled fold-changes lower than 50% of that observed in known essential genes.

plt.figure(figsize=(3, 2), dpi=300)
DrugPreliminary.histogram_strong_response(num_resp_crispr)
plt.xlabel("Number of cell lines")
plt.ylabel("Fraction of genes")
plt.title(
    "Cumulative distribution of scaled fold-changes lower than\n50% of the effect observed in known essential genes"
)
plt.savefig(
    f"{rpath}/preliminary_crispr_response_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Principal components of the genes in the CRISPR-Cas9 data-set

plt.figure(figsize=(4, 4), dpi=300)
CrisprPreliminary.pairplot_pca_by_rows(pca_crispr, hue=None)
plt.suptitle("PCA CRISPR-Cas9 (Genes)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_pairplot.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
plt.show()


# Principal components of the samples in the CRISPR-Cas9 data-set

plt.figure(figsize=(4, 4), dpi=300)
CrisprPreliminary.pairplot_pca_by_columns(
    pca_crispr, hue="institute", hue_vars=assoc.samplesheet.samplesheet["institute"]
)
plt.suptitle("PCA CRISPR-Cas9 (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_pairplot_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Principal components of the samples in the CRISPR-Cas9 data-set coloured by cancer type

plt.figure(figsize=(4, 4), dpi=300)
CrisprPreliminary.pairplot_pca_samples_cancertype(
    pca_crispr, assoc.samplesheet.samplesheet["cancer_type"]
)
plt.suptitle("PCA CRISPR-Cas9 (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_pairplot_cancertype.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Drug-response PCs correlation with growth-rates

plt.figure(figsize=(1.5, 1.5), dpi=300)
plot_df = assoc.samplesheet.growth_corr(pca_crispr["column"]["pcs"].T)
CrisprPreliminary.growth_corrs_pcs_barplot(plot_df)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_growth_pcs_barplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# CRISPR samples principal component correlation with growth rates

CrisprPreliminary.corrplot_pcs_growth(
    pca_crispr, assoc.samplesheet.samplesheet["growth"], "PC4"
)
plt.gcf().set_size_inches(1.5, 1.5)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_growth_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# CRISPR gene principal component correlation with essentiality
#
# Correlation CRISPR genes PC1 with number of times a gene has a strong essentiality profile across the cell lines.
# Strong essentiality is defined as true if: scaled log2 FC < -0.5 (meaning 50% of the effect of known essential genes).

CrisprPreliminary.corrplot_pcs_essentiality(pca_crispr, num_resp_crispr, "PC1")
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_essentiality_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# Copyright (C) 2019 Emanuel Goncalves


