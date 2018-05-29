#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import pandas as pd
import drispr.associations as lr_files
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
    drespo = drispr.get_drugresponse()
    rnaseq = pd.read_csv(drispr.RNASEQ_VOOM, index_col=0)
    crispr = drispr.get_crispr(dtype='both')

    samples = list(set(drespo).intersection(rnaseq))

    # - Covariates
    covariates = drispr.build_covariates(samples=samples, add_growth=True).dropna()
    samples = list(set(covariates.index))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    drespo = drispr.filter_drugresponse(drespo[samples])
    crispr = drispr.filter_crispr(crispr)
    rnaseq = rnaseq.reindex(crispr.index).dropna()
    print('#(Drugs) = {}; #(Genes) = {}'.format(len(set(drespo.index)), len(set(rnaseq.index))))

    # - Linear Regression: Drug ~ GExp (conitnuous) + Covariates
    lm_df_crispr_logfc = lm_drug(rnaseq[samples].T, drespo[samples].T, covariates.loc[samples], scale_x=True)
    lm_df_crispr_logfc.sort_values('lr_pval').to_csv(lr_files.LR_DRUG_RNASEQ, index=False)
    print('[INFO] Done: Drug ~ GExp (conitnuous) + Covariates ({})'.format(lr_files.LR_DRUG_RNASEQ))
