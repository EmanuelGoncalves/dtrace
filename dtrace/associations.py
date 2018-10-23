#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from dtrace.importer import DrugResponse, CRISPR, MOBEM


class Association(object):
    def __init__(self, dtype_drug='ic50'):
        # Import
        self.mobems_obj = MOBEM()
        self.crispr_obj = CRISPR()
        self.drespo_obj = DrugResponse()

        samples = list(set.intersection(
            set(self.mobems_obj.get_data().columns),
            set(self.drespo_obj.get_data().columns),
            set(self.crispr_obj.get_data().columns)
        ))
        print(f'#(Samples)={len(samples)}')

        # Filter
        self.mobems = self.mobems_obj.filter().get_data()
        self.drespo = self.drespo_obj.filter(subset=samples).get_data(dtype=dtype_drug)
        self.crispr = self.crispr_obj.filter(subset=samples).get_data(dtype='logFC')
        print(f'#(Genomic)={self.mobems.shape[0]}; #(Drugs)={self.drespo.shape[0]}; #(Genes)={self.crispr.shape[0]}')

    @staticmethod
    def lmm_association(drug, y, x, m=None, expand_drug_id=True):
        # Build matrices
        Y = y.loc[[drug]].T.dropna()
        Y = pd.DataFrame(StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns)

        X = x[Y.index].T
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

        # Random effects matrix
        K = x[Y.index].T
        K = K.dot(K.T)
        K /= (K.values.diagonal().mean())

        # Covariates
        if m is not None:
            m = m[Y.index].T
            m = pd.DataFrame(StandardScaler().fit_transform(m), index=m.index, columns=m.columns)

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik='normal', verbose=False)

        # Assemble output
        df = pd.DataFrame(
            dict(beta=lmm.variant_effsizes.ravel(), pval=lmm.variant_pvalues.ravel(), GeneSymbol=X.columns))

        if expand_drug_id:
            df = df.assign(DRUG_ID_lib=drug[0])
            df = df.assign(DRUG_NAME=drug[1])
            df = df.assign(VERSION=drug[2])
        else:
            df = df.assign(DRUG_ID=drug)

        df = df.assign(n_samples=Y.shape[0])

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

        df = df.assign(DRUG_ID_lib=association[0])
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
            .query(f"DRUG_ID_lib == {drug[0]} & DRUG_NAME == '{drug[1]}' & VERSION == '{drug[2]}'")\
            .sort_values('pval')\
            .head(top_features)['GeneSymbol']

        return pd.concat([
            self.drespo.loc[drug],
            self.crispr.loc[d_genes].T
        ], axis=1, sort=False).dropna().T

    def associations(self, method='bonferroni', min_events=3, fdr_thres=.1, top_features=5):
        # - Simple linear regression
        lmm_res = pd.concat([self.lmm_association(d, self.drespo, self.crispr) for d in self.drespo.index])
        lmm_res = self.multipletests_per_drug(lmm_res, field='pval', method=method)

        # - Multi linear regression
        mlmm_res = []
        for drug in self.drespo.index:
            drug_df = self.get_drug_top(lmm_res, drug, top_features)

            drug_mlmm = pd.concat([self.lmm_association(
                drug,
                drug_df.iloc[[0]],
                drug_df.drop(drug_df.iloc[[0, i]].index),
                drug_df.iloc[[i]]
            ).assign(covar=drug_df.index[i]) for i in np.arange(1, top_features + 1)]).sort_values('pval')

            mlmm_res.append(drug_mlmm)
        mlmm_res = pd.concat(mlmm_res)
        mlmm_res = self.multipletests_per_drug(mlmm_res, field='pval', method=method)

        # - Robust pharmacological
        lmm_res_signif = lmm_res.query(f'fdr < {fdr_thres}')
        print(f'#(Significant associations) = {lmm_res_signif.shape[0]}')

        lmm_robust = pd.concat([
            self.lmm_robust_association(a, self.drespo, self.crispr, self.mobems, min_events)
            for a in lmm_res_signif[DrugResponse.DRUG_INFO_COLUMNS + ['GeneSymbol']].values
        ])

        # Correct lmm p-values
        lmm_robust = self.multipletests_per_drug(lmm_robust, field='pval_drug', fdr_field=f'fdr_drug')
        lmm_robust = self.multipletests_per_drug(lmm_robust, field='pval_crispr', fdr_field='fdr_crispr')

        return lmm_res, mlmm_res, lmm_robust


if __name__ == '__main__':
    for dtype in ['ic50', 'auc']:
        associations = Association(dtype_drug=dtype)
        assoc, assoc_multi, assoc_robust = associations.associations()

        assoc.sort_values('fdr').to_csv(f'data/drug_lmm_regressions_{dtype}.csv', index=False)
        assoc_multi.sort_values('fdr').to_csv(f'data/drug_lmm_regressions_multiple_{dtype}.csv', index=False)
        assoc_robust.sort_values('fdr_drug').to_csv(f'data/drug_lmm_regressions_robust_{dtype}.csv', index=False)
