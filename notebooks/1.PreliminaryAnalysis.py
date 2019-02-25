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
import pandas as pd
import matplotlib.pyplot as plt
from dtrace import rpath, dpath
from Associations import Association
from dtrace.Preliminary import Preliminary, DrugPreliminary, CrisprPreliminary

# ### Import data-sets

datasets = Association()


# ## Principal Component Analysis (PCA)
#
# Perform PCA on the drug-response and CRISPR-Cas9 data-sets both per drug/gene and per samples.

pca_drug = Preliminary.perform_pca(datasets.drespo)
pca_crispr = Preliminary.perform_pca(datasets.crispr)

for n, pcas in [("drug", pca_drug), ("crispr", pca_crispr)]:
    for by in pcas:
        pcas[by]["pcs"].round(5).to_csv(f"{dpath}/{n}_pca_{by}_pcs.csv")


# ## Growth-rate correlation analysis
#
# Correlation of cell lines growth rates (unperturbed) with drug-response (ln IC50).

g_corr = datasets.drespo_obj.growth_corr(
    datasets.drespo, datasets.samplesheet.samplesheet["growth"]
)
g_corr.to_csv(f"{dpath}/drug_growth_correlation.csv", index=False)


# Correlation of cell lines growth rates (unperturbed) with CRISPR-Cas9 (scaled log2 fold-change; median essential = -1)

c_corr = datasets.crispr_obj.growth_corr(
    datasets.crispr, datasets.samplesheet.samplesheet["growth"]
)
c_corr.to_csv(f"{dpath}/crispr_growth_correlation.csv", index=False)


# ## Strong viability responses per compound
#
# Count for each compound the number of IC50s that are lower than 50% of the maximum concentration used for the
# respective compound

num_resp = pd.Series(
    {
        drug: (
            datasets.drespo.loc[drug].dropna()
            < np.log(datasets.drespo_obj.maxconcentration[drug] * 0.5)
        ).sum()
        for drug in datasets.drespo.index
    }
).reset_index()
num_resp.columns = ["DRUG_ID", "DRUG_NAME", "VERSION", "n_resp"]
num_resp.to_csv(f"{dpath}/drug_number_responses.csv", index=False)


# # Drug-response


# Drug-response (IC50s) measurements across cell lines cumulative distribution

DrugPreliminary.histogram_drug(datasets.drespo.count(1))
plt.gcf().set_size_inches(3, 1.5)
plt.savefig(
    f"{rpath}/preliminary_drug_histogram_drug.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.figure(figsize=(3, 2), dpi=300)
plt.show()


# Cumulative distribution of strong drug-response measurements. Strong response measurements are defined as IC50 < 50%
# Max. concentration

plt.figure(figsize=(3, 2), dpi=300)
DrugPreliminary.histogram_drug_response(num_resp)
plt.savefig(
    f"{rpath}/preliminary_drug_response_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Cumulative distribution of samples with measurements across all compounds screened

plt.figure(figsize=(3, 2), dpi=300)
DrugPreliminary.histogram_sample(datasets.drespo.count(0))
plt.savefig(
    f"{rpath}/preliminary_drug_histogram_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Principal components of drugs

plt.figure(figsize=(4, 4), dpi=300)
DrugPreliminary.pairplot_pca_by_rows(pca_drug)
plt.suptitle("PCA drug response (Drugs)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot.pdf", bbox_inches="tight", transparent=True
)
plt.show()


# ## Principal components of samples in the drug-response

plt.figure(figsize=(4, 4), dpi=300)
DrugPreliminary.pairplot_pca_by_columns(pca_drug)
plt.suptitle("PCA drug response (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Principal components of samples in the drug-response coloured by cancer type

plt.figure(figsize=(4, 4), dpi=300)
DrugPreliminary.pairplot_pca_samples_cancertype(
    pca_drug, datasets.samplesheet.samplesheet["cancer_type"]
)
plt.suptitle("PCA drug response (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_pairplot_cancertype.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Samples drug-response PC1 correlation with growth-rate

plt.figure(figsize=(2, 2), dpi=300)
DrugPreliminary.corrplot_pcs_growth(
    pca_drug, datasets.samplesheet.samplesheet["growth"], "PC1"
)
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Histogram of samples drug-response PC1 correlation with growth-rate

plt.figure(figsize=(2, 2), dpi=300)
DrugPreliminary.growth_correlation_histogram(g_corr)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot_histogram.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Top correlated drugs with growth-rate

plt.figure(figsize=(2, 4), dpi=300)
DrugPreliminary.growth_correlation_top_drugs(g_corr)
plt.savefig(
    f"{rpath}/preliminary_drug_pca_growth_corrplot_top.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# # CRISPR-Cas9


# ## Principal components of the genes in the CRISPR-Cas9 data-set

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


# ## Principal components of the samples in the CRISPR-Cas9 data-set

plt.figure(figsize=(4, 4), dpi=300)
CrisprPreliminary.pairplot_pca_by_columns(
    pca_crispr, hue="institute", hue_vars=datasets.samplesheet.samplesheet["institute"]
)
plt.suptitle("PCA CRISPR-Cas9 (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_pairplot_samples.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## Principal components of the samples in the CRISPR-Cas9 data-set coloured by cancer type

plt.figure(figsize=(4, 4), dpi=300)
CrisprPreliminary.pairplot_pca_samples_cancertype(
    pca_crispr, datasets.samplesheet.samplesheet["cancer_type"]
)
plt.suptitle("PCA CRISPR-Cas9 (Cell lines)", y=1.05, fontsize=9)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_pairplot_cancertype.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## CRISPR samples principal component correlation with growth rates

plt.figure(figsize=(2, 2), dpi=300)
CrisprPreliminary.corrplot_pcs_growth(
    pca_crispr, datasets.samplesheet.samplesheet["growth"], "PC4"
)
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_growth_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()


# ## CRISPR gene principal component correlation with essentiality
#
# Correlation CRISPR genes PC1 with number of times a gene has a strong essentiality profile across the cell lines.
# Strong essentiality is defined as true if: scaled log2 FC < -0.5 (meaning 50% of the effect of known essential genes).

plt.figure(figsize=(2, 2), dpi=300)
CrisprPreliminary.corrplot_pcs_essentiality(pca_crispr, datasets.crispr, "PC1")
plt.gcf().set_size_inches(2, 2)
plt.savefig(
    f"{rpath}/preliminary_crispr_pca_essentiality_corrplot.pdf",
    bbox_inches="tight",
    transparent=True,
)
plt.show()
