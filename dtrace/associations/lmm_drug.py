#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from dtrace.associations import multipletests_per_drug


def lmm_association(drug, y, x, M=None, expand_drug_id=True):
    # Build matrices
    Y = y.loc[[drug]].T.dropna()

    X = x[Y.index].T
    X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

    # Random effects matrix
    K = x[Y.index].T
    K = K.dot(K.T)
    K /= (K.values.diagonal().mean())

    # Covariates
    if M is not None:
        M = M.loc[Y.index]

    # Linear Mixed Model
    lmm = scan(X, Y, K=K, M=M, lik='normal', verbose=False)

    # Assemble output
    df = pd.DataFrame(dict(beta=lmm.variant_effsizes.ravel(), pval=lmm.variant_pvalues.ravel(), GeneSymbol=X.columns))

    if expand_drug_id:
        df = df.assign(DRUG_ID_lib=drug[0])
        df = df.assign(DRUG_NAME=drug[1])
        df = df.assign(VERSION=drug[2])
    else:
        df = df.assign(DRUG_ID=drug)

    df = df.assign(n_samples=Y.shape[0])

    return df


if __name__ == '__main__':
    # - Import
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    drespo = dtrace.filter_drugresponse(drespo[samples])

    crispr = dtrace.filter_crispr(crispr[samples])
    crispr_logfc = crispr_logfc.loc[crispr.index, samples]

    print(
        '#(Genomic) = {}; #(Drugs) = {}; #(Genes) = {}'.format(len(set(mobems.index)), len(set(drespo.index)), len(set(crispr.index)))
    )

    # - Linear Mixed Model
    lmm_res = pd.concat([lmm_association(d, drespo, crispr_logfc) for d in drespo.index])
    lmm_res = multipletests_per_drug(lmm_res, field='pval')
    print(lmm_res.sort_values('pval').head(60))

    # - Export
    lmm_res.to_csv(dtrace.LMM_ASSOCIATIONS, index=False)
