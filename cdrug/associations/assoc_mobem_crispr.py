#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from statsmodels.stats.multitest import multipletests


if __name__ == '__main__':

    # - Import linear regressions
    lm_df_mobems = pd.read_csv(lr_files.LR_BINARY_DRUG_MOBEMS)
    lm_df_crispr = pd.read_csv(lr_files.LR_BINARY_DRUG_CRISPR)

    # -
    crispr_genes = set(lm_df_crispr['GeneSymbol'])
    mobems_genes = pd.DataFrame([
        {'gene': g, 'feature': f} for f in set(lm_df_mobems['level_3']) for g in cdrug.mobem_feature_to_gene(f) if g in crispr_genes
    ])

    # -
    lm_df_mobems_subset = lm_df_mobems[lm_df_mobems['level_3'].isin(mobems_genes['feature'])]
    lm_df_crispr_subset = lm_df_crispr[lm_df_crispr['GeneSymbol'].isin(mobems_genes['gene'])]

    lm_df_mobems_subset = lm_df_mobems_subset.assign(lr_fdr=multipletests(lm_df_mobems_subset['lr_pval'])[1])
    lm_df_crispr_subset = lm_df_crispr_subset.assign(lr_fdr=multipletests(lm_df_crispr_subset['lr_pval'])[1])

    signif_crispr = lm_df_crispr_subset[lm_df_crispr_subset['lr_fdr'] < 0.1]
    signif_mobems = lm_df_mobems_subset[lm_df_mobems_subset['lr_fdr'] < 0.1]

    signif_crispr_assoc = [(d, g) for d, g in signif_crispr[['DRUG_ID_lib', 'GeneSymbol']].values]
    [np.any([(d, g) in signif_crispr_assoc for g in cdrug.mobem_feature_to_gene(fs)]) for d, fs in signif_mobems[['DRUG_ID_lib', 'level_3']].values]

