#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
from limix.qtl import scan
from dtrace.associations import DRUG_INFO_COLUMNS
from associations import multipletests_per_drug


def lmm_association(association, y1, y2, x, min_events):
    print('[INFO] LMM {}'.format(association))

    # Build drug measurement matrix
    Y1 = y1.loc[[tuple(association[:3])]].T.dropna()

    # Build CRISPR measurement matrix
    Y2 = y2.loc[[association[3]], Y1.index].T

    # Build genomic feature matrix
    X = x[Y1.index].T
    X = X.loc[:, X.sum() >= min_events]

    # Random effects matrix
    K = x[Y1.index].T
    K = K.dot(K.T)
    K /= (K.values.diagonal().mean())

    # Linear Mixed Model
    lmm_y1 = scan(X, Y1, K=K, lik='normal', verbose=False)
    lmm_y2 = scan(X, Y2, K=K, lik='normal', verbose=False)

    # Assemble output
    df = pd.DataFrame(dict(
        beta_drug=lmm_y1.variant_effsizes.ravel(),
        pval_drug=lmm_y1.variant_pvalues.ravel(),
        beta_crispr=lmm_y2.variant_effsizes.ravel(),
        pval_crispr=lmm_y2.variant_pvalues.ravel(),
        Genetic=X.columns,
        n_events=X.sum().values
    ))

    df = df.assign(DRUG_ID_lib=association[0])
    df = df.assign(DRUG_NAME=association[1])
    df = df.assign(VERSION=association[2])
    df = df.assign(GeneSymbol=association[3])
    df = df.assign(n_samples=Y1.shape[0])

    return df


if __name__ == '__main__':
    # - Import
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Import significant linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS).query('fdr < 0.05')

    # - Robust pharmacological regressions
    y1, y2, x = drespo[samples], crispr_logfc[samples], mobems[samples]

    lmm_robust = pd.concat([
        lmm_association(a, y1, y2, x, 3) for a in lmm_drug[DRUG_INFO_COLUMNS + ['GeneSymbol']].values
    ])

    # - Correct lmm p-values
    lmm_robust = multipletests_per_drug(lmm_robust, field='pval_drug', fdr_field='fdr_drug')
    lmm_robust = multipletests_per_drug(lmm_robust, field='pval_crispr', fdr_field='fdr_crispr')

    # - Export
    lmm_robust.to_csv(dtrace.LMM_ASSOCIATIONS_ROBUST, index=False)

