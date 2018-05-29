#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from drispr.associations import multipletests_per_drug


if __name__ == '__main__':
    # - Import
    mobems = drispr.get_mobem()
    drespo = drispr.get_drugresponse()

    crispr = drispr.get_crispr(dtype='both')
    crispr_logfc = drispr.get_crispr(dtype='logFC')
    crispr_logfc_scaled = drispr.scale_crispr(crispr_logfc)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Covariates
    covariates = drispr.build_covariates(samples=samples, add_growth=True).dropna()

    samples = list(set(covariates.index))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    mobems = drispr.filter_mobem(mobems[samples])
    drespo = drispr.filter_drugresponse(drespo[samples])

    crispr = drispr.filter_crispr(crispr[samples])
    crispr_logfc_scaled_filtered = crispr_logfc_scaled.loc[crispr.index, samples]
    print('#(Genomic features) = {}; #(Drugs) = {}; #(Genes) = {}'.format(len(set(mobems.index)), len(set(drespo.index)), len(set(crispr.index))))

    # -
    lmm_res = []
    for d in drespo.index:
        Y = drespo.loc[[d]].T.dropna()

        X = crispr_logfc_scaled_filtered[Y.index].T
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

        K = crispr_logfc_scaled[Y.index].T
        K = K.dot(K.T)
        K /= (K.values.diagonal().mean())

        M = covariates.loc[Y.index]

        lmm = scan(X, Y, K=K, M=M, lik='normal', verbose=False)

        df = pd.DataFrame(dict(beta=lmm.variant_effsizes.ravel(), pval=lmm.variant_pvalues.ravel(), GeneSymbol=X.columns))
        df = df.assign(DRUG_ID_lib=d[0])
        df = df.assign(DRUG_NAME=d[1])
        df = df.assign(VERSION=d[2])
        df = df.assign(len=Y.shape[0])

        lmm_res.append(df)

    lmm_res = pd.concat(lmm_res)
    print(lmm_res.sort_values('pval').head(10))

    lmm_res = multipletests_per_drug(lmm_res, field='pval')

    lmm_res.to_csv('data/drug_regressions_crispr_limix.csv')
