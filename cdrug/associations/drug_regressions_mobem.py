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

    samples = list(set(mobems).intersection(drespo))

    # - Covariates
    covariates = cdrug.build_covariates(samples=samples, add_growth=True).dropna()
    covariates_no_growth = covariates.drop('growth_rate_median', axis=1)

    samples = list(set(covariates.index))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    mobems = cdrug.filter_mobem(mobems[samples])
    drespo = cdrug.filter_drugresponse(drespo[samples])
    print('#(Genomic features) = {}; #(Drugs) = {}'.format(len(set(mobems.index)), len(set(drespo.index))))

    # - Linear Regression: Drug ~ Genomic (binary) + Covariates
    lm_df_mobems = lm_drug(mobems[samples].T, drespo[samples].T, covariates.loc[samples])
    lm_df_mobems.sort_values('lr_pval').to_csv(lr_files.LR_BINARY_DRUG_MOBEMS_ALL, index=False)
    print('[INFO] Done: Drug ~ Genomic (binary) + Covariates')
