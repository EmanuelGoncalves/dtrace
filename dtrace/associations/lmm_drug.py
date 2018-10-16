#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from dtrace.associations import multipletests_per_drug
from dtrace.importer import DrugResponse, CRISPR, MOBEM


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
    mobems = MOBEM()
    crispr = CRISPR()
    drespo = DrugResponse()

    samples = list(set.intersection(
        set(mobems.get_data().columns),
        set(drespo.get_data().columns),
        set(crispr.get_data().columns)
    ))
    print(f'#(Samples) = {len(samples)}')

    # - Filter
    mobems = mobems.filter().get_data()
    drespo = drespo.filter(subset=samples).get_data(dtype='ic50')
    crispr = crispr.filter(subset=samples).get_data(dtype='logFC')
    print(f'#(Genomic) = {mobems.shape[0]}; #(Drugs) = {drespo.shape[0]}; #(Genes) = {crispr.shape[0]}')

    # - Linear Mixed Model
    lmm_res = pd.concat([lmm_association(d, drespo, crispr) for d in drespo.index])
    lmm_res = multipletests_per_drug(lmm_res, field='pval')
    print(lmm_res.sort_values('pval').head(60))

    # - Export
    lmm_res.to_csv(dtrace.LMM_ASSOCIATIONS, index=False)
