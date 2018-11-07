#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import itertools as it
from limix.qtl import scan
from limix.stats import lrt_pvalues
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from dtrace.importer import DrugResponse, CRISPR, MOBEM, Sample


class Association(object):
    def __init__(self, dtype_drug='ic50', dtype_crispr='logFC'):
        # Import
        self.samplesheet = Sample()
        self.mobems_obj = MOBEM()
        self.crispr_obj = CRISPR()
        self.drespo_obj = DrugResponse()

        self.samples = list(set.intersection(
            set(self.mobems_obj.get_data().columns),
            set(self.drespo_obj.get_data().columns),
            set(self.crispr_obj.get_data().columns)
        ))
        print(f'#(Samples)={len(self.samples)}')

        # Filter
        self.mobems = self.mobems_obj.filter().get_data()
        self.drespo = self.drespo_obj.filter(subset=self.samples).get_data(dtype=dtype_drug)
        self.crispr = self.crispr_obj.filter(subset=self.samples).get_data(dtype=dtype_crispr)
        print(f'#(Genomic)={self.mobems.shape[0]}; #(Drugs)={self.drespo.shape[0]}; #(Genes)={self.crispr.shape[0]}')

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= (K.values.diagonal().mean())
        return K

    @staticmethod
    def lmm_association_limix(drug, y, x, m=None, k=None, add_intercept=True):
        # Build matrices
        Y = y.loc[[drug]].T.dropna()
        # Y = pd.DataFrame(StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns)
        # TODO: Remove scaling of y to make effect sizes across drugs quantifiable

        X = x[Y.index].T
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

        # Random effects matrix
        if k is None:
            K = Association.kinship(x[Y.index].T)

        else:
            K = k.loc[Y.index, Y.index]

        # Covariates
        if m is not None:
            m = m[Y.index].T
            m = pd.DataFrame(StandardScaler().fit_transform(m), index=m.index, columns=m.columns)

        # Intercept
        if add_intercept and (m is not None):
            m['intercept'] = 1

        elif add_intercept and (m is None):
            m = pd.DataFrame(np.ones((Y.shape[0], 1)), index=Y.index, columns=['intercept'])

        else:
            m = pd.DataFrame(np.zeros((Y.shape[0], 1)), index=Y.index, columns=['zeros'])

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik='normal', verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def lmm_association(drug, y, x, m=None, k=None, expand_drug_id=True):
        lmm, params = Association.lmm_association_limix(drug, y, x, m, k)

        df = pd.DataFrame(
            dict(beta=lmm.variant_effsizes.ravel(), pval=lmm.variant_pvalues.ravel(), GeneSymbol=params['x'].columns)
        )

        if expand_drug_id:
            df = df.assign(DRUG_ID=drug[0])
            df = df.assign(DRUG_NAME=drug[1])
            df = df.assign(VERSION=drug[2])
        else:
            df = df.assign(DRUG_ID=drug)

        df = df.assign(n_samples=params['y'].shape[0])

        return df

    @staticmethod
    def lmm_robust_association(association, y1, y2, x, min_events):
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

        df = df.assign(DRUG_ID=association[0])
        df = df.assign(DRUG_NAME=association[1])
        df = df.assign(VERSION=association[2])
        df = df.assign(GeneSymbol=association[3])
        df = df.assign(n_samples=Y1.shape[0])

        return df

    @staticmethod
    def multipletests_per_drug(lr_associations, method='bonferroni', field='pval', fdr_field='fdr', index_cols=None):
        index_cols = DrugResponse.DRUG_INFO_COLUMNS if index_cols is None else index_cols

        d_unique = {tuple(i) for i in lr_associations[index_cols].values}

        df = lr_associations.set_index(index_cols)

        df = pd.concat([
            df.loc[i].assign(
                fdr=multipletests(df.loc[i, field], method=method)[1]
            ).rename(columns={'fdr': fdr_field}) for i in d_unique
        ]).reset_index()

        return df

    def get_drug_top(self, lmm_associations, drug, top_features):
        d_genes = lmm_associations\
            .query(f"DRUG_ID == {drug[0]} & DRUG_NAME == '{drug[1]}' & VERSION == '{drug[2]}'")\
            .sort_values('pval')\
            .head(top_features)['GeneSymbol']

        return pd.concat([
            self.drespo.loc[drug],
            self.crispr.loc[d_genes].T
        ], axis=1, sort=False).dropna().T

    def associations(self, method='bonferroni', min_events=3, fdr_thres=.1):
        # - Simple linear regression
        lmm_res = pd.concat([self.lmm_association(d, self.drespo, self.crispr) for d in self.drespo.index])
        lmm_res = self.multipletests_per_drug(lmm_res, field='pval', method=method)

        return lmm_res, None, None


if __name__ == '__main__':
    for dtype in ['ic50']:
        associations = Association(dtype_drug=dtype)
        assoc, assoc_multi, assoc_robust = associations.associations()

        assoc.sort_values('fdr').to_csv(f'data/drug_lmm_regressions_{dtype}.csv', index=False)
        # assoc_multi.sort_values('fdr').to_csv(f'data/drug_lmm_regressions_multiple_{dtype}.csv', index=False)
        # assoc_robust.sort_values('fdr_drug').to_csv(f'data/drug_lmm_regressions_robust_{dtype}.csv', index=False)
