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
from dtrace.Associations import Association


# ### Import data-sets

# Association files are exported to "dtrace/data/" folder (dpath). Warning, due to the large number of tests executed
# a complete run this script takes over 3 hours (no parallelization, 3.1 GHz Intel Core i7). Executing can also be done
# using the main function of Association class.

assoc = Association(dtype="ic50")


# ## Linear drug-response associations

# These are the core functions to generate the linear associations with drug-response using other types of omics
# including CRISPR-Cas9 gene-essentiality, RNA-seq gene-expression, whole exome sequencing mutation status and DNA SNP6
# arrays copy-number status.

# Linear mixed models trained with [LIMIX](https://limix.readthedocs.io/en/stable/), these models use a covariance
# matrix (K) as random effects of the data-set used for the independent features (X). Additionally, covariates (M) are
# considered: (i) growth rate represented by PC1 of drug-response; (ii) dummy variables of growth properties of the cell
# lines, (adherent, suspension or semi-adherent); (iii) dummy variables for the institute of origin of the CRISPR-Cas9
# data-sets, (Sanger or Broad); and (iv) CRISPR-Cas9 samples essential genes recall capacity (area under the recall
# curve).


# ### Drug-response ~ CRISPR-Cas9

lmm_dsingle = assoc.lmm_single_associations()
lmm_dsingle.sort_values(["fdr", "pval"]).to_csv(
    assoc.lmm_drug_crispr_file, index=False, compression="gzip"
)


# ### Drug-response ~ CRISPR-Cas9

lmm_dgexp = assoc.lmm_gexp_drug()
lmm_dgexp.sort_values(["fdr", "pval"]).to_csv(
    assoc.lmm_drug_gexp_file, index=False, compression="gzip",
)


# ### Drug-response ~ CRISPR-Cas9

lmm_dgenomic = assoc.lmm_single_associations_genomic()
lmm_dgenomic.sort_values(["fdr", "pval"]).to_csv(
    assoc.lmm_drug_genomic_file, index=False, compression="gzip",
)


# ## Robust pharmacogenomic associations

# These functions test associations between drug-response/gene-essentiality and genomic/gene-expression for those pairs
# of drugs and gene which are significantly correlated (log-ratio test BH-FDR < 10%).

# (Drug,Gene) pairs that are both significantly correlated with a genomic feature (mutation, copy-number or gene-
# expression) are termed as robust pharmacogenomic associations, since these recapitulate an genomic association with
# two independent viability screens.


# ### (Drug-response; CRISPR-Cas9) ~ Genomic

lmm_robust = assoc.lmm_robust_association(lmm_dsingle, is_gexp=False)
lmm_robust.sort_values(["drug_fdr", "drug_pval"]).to_csv(
    assoc.lmm_robust_genomic_file, index=False, compression="gzip",
)

# ### (Drug-response; CRISPR-Cas9) ~ Gene-expression

lmm_robust_gexp = assoc.lmm_robust_association(lmm_dsingle, is_gexp=True)
lmm_robust_gexp.sort_values(["drug_fdr", "drug_pval"]).to_csv(
    assoc.lmm_robust_gexp_file, index=False, compression="gzip",
)