#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import pandas as pd
import cdrug.associations as lr_files
from crispy.regression.linear import lr
from sklearn.preprocessing import StandardScaler


def lm_drug(xs, ys, ws, scale_x=False):
    # Standardize xs
    if scale_x:
        xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

    # Regression
    res = lr(xs, ys, ws)

    # Export results
    res_df = pd.concat([res[i].unstack().rename(i) for i in res], axis=1).reset_index()

    return res_df


if __name__ == '__main__':
    # - Import
    mobems = cdrug.get_mobem()
    drespo = cdrug.get_drugresponse()
    crispr = cdrug.get_crispr(is_binary=True)
    crispr_logfc = cdrug.get_crispr(is_binary=False)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))

    # - Covariates
    covariates = cdrug.build_covariates(samples=samples).dropna()

    samples = list(set(covariates.index))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    mobems = cdrug.filter_mobem(mobems[samples])
    drespo = cdrug.filter_drugresponse(drespo[samples])
    crispr = cdrug.filter_crispr(crispr[samples], is_binary=True)
    crispr_logfc = crispr_logfc.loc[crispr.index, samples]

    print('#(Genomic features) = {}; #(Drugs) = {}; #(Genes) = {}'.format(len(set(mobems.index)), len(set(drespo.index)), len(set(crispr.index))))

    # - Linear Regression: Drug ~ Genomic (binary) + Covariates
    lm_df_mobems = lm_drug(mobems[samples].T, drespo[samples].T, covariates.loc[samples])

    # - Linear Regression: Drug ~ CRISPR (binary) + Covariates
    lm_df_crispr = lm_drug(crispr[samples].T, drespo[samples].T, covariates.loc[samples])

    # - Linear Regression: Drug ~ CRISPR (conitnuous) + Covariates
    lm_df_crispr_logfc = lm_drug(crispr_logfc[samples].T, drespo[samples].T, covariates.loc[samples], scale_x=True)

    # - Export
    lm_df_mobems.sort_values('lr_pval').to_csv(lr_files.LR_BINARY_DRUG_MOBEMS, index=False)
    lm_df_crispr.sort_values('lr_pval').to_csv(lr_files.LR_BINARY_DRUG_CRISPR, index=False)
    lm_df_crispr_logfc.sort_values('lr_pval').to_csv(lr_files.LR_DRUG_CRISPR, index=False)
