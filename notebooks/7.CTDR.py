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

from Associations import AssociationCTDR2

assoc = AssociationCTDR2(dtype="auc", pval_method="bonferroni")

lmm_dsingle = assoc.lmm_single_associations(verbose=1)

lmm_dsingle.to_csv(assoc.lmm_drug_crispr_file, index=False, compression="gzip")
