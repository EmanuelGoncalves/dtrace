#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import scipy as sp
import numpy as np
import pandas as pd
import itertools as it
from limix.qtl import scan
from scipy import interpolate
from limix.stats import lrt_pvalues
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from Qvalue import QValue
from dtrace.DataImporter import DrugResponse, CRISPR, Genomic, Sample, PPI, GeneExpression, Proteomics, \
    PhosphoProteomics, CopyNumber, Apoptosis, RPPA, WES, RNAi, CRISPRComBat


class Association:
    def __init__(self, dtype_drug='ic50', pval_method='fdr_bh'):
        # Import
        self.dtype_drug = dtype_drug
        self.pval_method = pval_method

        self.samplesheet = Sample()

        self.ppi = PPI()

        self.crispr_obj = CRISPR()
        self.drespo_obj = DrugResponse()
        self.genomic_obj = Genomic()
        self.gexp_obj = GeneExpression()
        self.prot_obj = Proteomics()
        self.phospho_obj = PhosphoProteomics()
        self.cn_obj = CopyNumber()
        self.apoptosis_obj = Apoptosis()
        self.rppa_obj = RPPA()
        self.wes_obj = WES()
        self.rnai_obj = RNAi()
        self.crisprcb_obj = CRISPRComBat()

        self.samples = list(set.intersection(
            set(self.drespo_obj.get_data().columns),
            set(self.crispr_obj.get_data().columns)
        ))
        print(f'#(Samples)={len(self.samples)}')

        # Filter
        self.crispr = self.crispr_obj.filter(subset=self.samples, scale=True)
        self.drespo = self.drespo_obj.filter(subset=self.samples, dtype=dtype_drug)
        self.genomic = self.genomic_obj.filter(subset=self.samples, min_events=5)
        self.gexp = self.gexp_obj.filter(subset=self.samples)
        self.prot = self.prot_obj.filter(subset=self.samples)
        self.phospho = self.phospho_obj.filter(subset=self.samples)
        self.cn = self.cn_obj.filter(subset=self.samples)
        self.apoptosis = self.apoptosis_obj.filter(subset=self.samples)
        self.rppa = self.rppa_obj.filter(subset=self.samples)
        self.wes = self.wes_obj.filter(subset=self.samples)
        self.rnai = self.rnai_obj.filter(subset=self.samples)
        self.crisprcb = self.crisprcb_obj.filter(subset=self.samples).loc[self.crispr.index]
        print(f'#(Drugs)={self.drespo.shape[0]}; #(Genes)={self.crispr.shape[0]}; #(Genomic)={self.genomic.shape[0]}')

    def get_covariates(self):
        # Samples CRISPR QC (recall essential genes)
        crispr_ess_qc = self.crispr_obj.qc_ess.rename('recall_essential').rename('crispr_qc_ess')
        crispr_ess_qc = (crispr_ess_qc - crispr_ess_qc.mean()) / crispr_ess_qc.std()

        # CRISPR institute of origin PC
        crispr_insitute = pd.get_dummies(self.samplesheet.samplesheet['institute'])

        # Cell lines growth rate
        drug_growth = pd.read_csv('data/drug_pca_column_pcs.csv', index_col=0)['PC1'].rename('drug_growth')

        # Cell lines culture conditions
        culture = pd.get_dummies(self.samplesheet.samplesheet['growth_properties']).drop(columns=['Unknown'])

        # Merge covariates
        covariates = pd.concat([
            crispr_ess_qc, crispr_insitute, drug_growth, culture
        ], axis=1, sort=False).loc[self.samples]

        return covariates

    @staticmethod
    def estimate(pv, m=None, verbose=False, lowmem=False, pi0=None):
        assert (pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

        original_shape = pv.shape
        pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

        if m is None:
            m = float(len(pv))
        else:
            # the user has supplied an m
            m *= 1.0

        # if the number of hypotheses is small, just set pi0 to 1
        if len(pv) < 100 and pi0 is None:
            pi0 = 1.0
        elif pi0 is not None:
            pi0 = pi0
        else:
            # evaluate pi0 for different lambdas
            pi0 = []
            lam = sp.arange(0, 0.90, 0.01)
            counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
            for l in range(len(lam)):
                pi0.append(counts[l] / (m * (1 - lam[l])))

            pi0 = sp.array(pi0)

            # fit natural cubic spline
            tck = interpolate.splrep(lam, pi0, k=3)
            pi0 = interpolate.splev(lam[-1], tck)
            if verbose:
                print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

            if pi0 > 1:
                if verbose:
                    print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
                pi0 = 1.0

        assert (pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

        if lowmem:
            # low memory version, only uses 1 pv and 1 qv matrices
            qv = sp.zeros((len(pv),))
            last_pv = pv.argmax()
            qv[last_pv] = (pi0 * pv[last_pv] * m) / float(m)
            pv[last_pv] = -sp.inf
            prev_qv = last_pv
            for i in range(int(len(pv)) - 2, -1, -1):
                cur_max = pv.argmax()
                qv_i = (pi0 * m * pv[cur_max] / float(i + 1))
                pv[cur_max] = -sp.inf
                qv_i1 = prev_qv
                qv[cur_max] = min(qv_i, qv_i1)
                prev_qv = qv[cur_max]

        else:
            p_ordered = sp.argsort(pv)
            pv = pv[p_ordered]
            qv = pi0 * m / len(pv) * pv
            qv[-1] = min(qv[-1], 1.0)

            for i in range(len(pv) - 2, -1, -1):
                qv[i] = min(pi0 * m * pv[i] / (i + 1.0), qv[i + 1])

            # reorder qvalues
            qv_temp = qv.copy()
            qv = sp.zeros_like(qv)
            qv[p_ordered] = qv_temp

        # reshape qvalues
        qv = qv.reshape(original_shape)

        return qv

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= (K.values.diagonal().mean())
        return K

    @staticmethod
    def lmm_association_limix(y, x, m=None, k=None, add_intercept=True, lik='normal'):
        # Build matrices
        Y = y.dropna()
        Y = pd.DataFrame(StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns)

        X = x.loc[Y.index]
        X = pd.DataFrame(StandardScaler().fit_transform(X), index=X.index, columns=X.columns)

        # Random effects matrix
        if k is None:
            K = Association.kinship(x.loc[Y.index])

        else:
            K = k.loc[Y.index, Y.index]

        # Covariates
        if m is not None:
            m = m.loc[Y.index]
            # m = pd.DataFrame(StandardScaler().fit_transform(m), index=m.index, columns=m.columns)

        # Add intercept
        if add_intercept and (m is not None):
            m['intercept'] = 1

        elif add_intercept and (m is None):
            m = pd.DataFrame(np.ones((Y.shape[0], 1)), index=Y.index, columns=['intercept'])

        else:
            m = pd.DataFrame(np.zeros((Y.shape[0], 1)), index=Y.index, columns=['zeros'])

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik=lik, verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def multipletests_per_drug(associations, method, field='pval', fdr_field='fdr', index_cols=None):
        index_cols = DrugResponse.DRUG_COLUMNS if index_cols is None else index_cols

        d_unique = {tuple(i) for i in associations[index_cols].values}

        df = associations.set_index(index_cols)

        df = pd.concat([
            df.loc[i].assign(
                fdr=QValue(df.loc[i, field]).qvalue() if method == 'qvalue' else multipletests(df.loc[i, field], method=method)[1]
            ).rename(columns={'fdr': fdr_field}) for i in d_unique
        ]).reset_index()

        return df

    def annotate_drug_target(self, associations):
        d_targets = self.drespo_obj.get_drugtargets()

        associations['DRUG_TARGETS'] = [
            ';'.join(d_targets[d]) if d in d_targets else np.nan for d in associations['DRUG_ID']
        ]

        return associations

    @staticmethod
    def get_association(lmm_associations, drug, gene):
        return lmm_associations[
            (lmm_associations[DrugResponse.DRUG_COLUMNS[0]] == drug[0]) & \
            (lmm_associations[DrugResponse.DRUG_COLUMNS[1]] == drug[1]) & \
            (lmm_associations[DrugResponse.DRUG_COLUMNS[2]] == drug[2]) & \
            (lmm_associations['GeneSymbol'] == gene)
        ]

    @staticmethod
    def lmm_single_association(y, x, m=None, k=None, expand_drug_id=True):
        lmm, params = Association.lmm_association_limix(y, x, m, k)

        df = pd.DataFrame(
            dict(beta=lmm.variant_effsizes.values, pval=lmm.variant_pvalues.values, GeneSymbol=params['x'].columns)
        )

        drug = y.columns[0]

        if expand_drug_id:
            df = df.assign(DRUG_ID=drug[0])
            df = df.assign(DRUG_NAME=drug[1])
            df = df.assign(VERSION=drug[2])

        else:
            df = df.assign(DRUG_ID=drug)

        df = df.assign(n_samples=params['y'].shape[0])

        return df

    def lmm_single_associations(self):
        # - Kinship matrix (random effects)
        k = self.kinship(self.crispr.T)
        m = self.get_covariates()

        # - Single feature linear mixed regression
        # Association
        lmm_single = pd.concat([
            self.lmm_single_association(self.drespo.loc[[d]].T, self.crispr.T, k=k, m=m) for d in self.drespo.index
        ])

        # Multiple p-value correction
        lmm_single = self.multipletests_per_drug(lmm_single, field='pval', method=self.pval_method)

        # Annotate drug target
        lmm_single = self.annotate_drug_target(lmm_single)

        # Annotate association distance to target
        lmm_single = self.ppi.ppi_annotation(
            lmm_single, ppi_type='string', ppi_kws=dict(score_thres=900), target_thres=5
        )

        # Sort p-values
        lmm_single = lmm_single.sort_values(['fdr', 'pval'])

        return lmm_single

    def lmm_single_associations_genomic(self, min_events=5):
        X = self.genomic

        samples, drugs = set(X).intersection(self.samples), self.drespo.index
        print(f'Samples={len(samples)};Drugs={len(drugs)}')

        lmm_drug = []
        for drug in drugs:
            # Observations
            y = self.drespo.loc[[drug], samples].T.dropna()
            y = pd.DataFrame(StandardScaler().fit_transform(y), index=y.index, columns=y.columns)

            # Features
            x = X[y.index].T
            x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= (k.values.diagonal().mean())

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik='normal', verbose=False)

            # Export
            lmm_drug.append(pd.DataFrame(dict(
                beta=lmm.variant_effsizes.values, pval=lmm.variant_pvalues.values, feature=x.columns,
                DRUG_ID=drug[0], DRUG_NAME=drug[1], VERSION=drug[2]
            )))

        lmm_drug = pd.concat(lmm_drug).reset_index(drop=True)
        lmm_drug = self.multipletests_per_drug(lmm_drug, field='pval', fdr_field=f'fdr', method=self.pval_method)
        lmm_drug = lmm_drug.sort_values(['fdr', 'pval'])

        return lmm_drug

    def lmm_robust_association(self, lmm_dsingle, is_gexp=False, fdr_thres=.1, min_events=5):
        lmm_res_signif = lmm_dsingle.query(f'fdr < {fdr_thres}')

        genes = set(lmm_res_signif['GeneSymbol'])
        drugs = {tuple(d) for d in lmm_res_signif[DrugResponse.DRUG_COLUMNS].values}

        X = self.gexp if is_gexp else self.genomic
        samples = set(X).intersection(self.samples)
        print(f'Assoc={lmm_res_signif.shape[0]}; Genes={len(genes)}; Drugs={len(drugs)}; Samples={len(samples)}')

        # Drug associations
        lmm_drug = []
        for drug in drugs:
            # Observations
            y = self.drespo.loc[[drug], samples].T.dropna()
            y = pd.DataFrame(StandardScaler().fit_transform(y), index=y.index, columns=y.columns)

            # Features
            x = X[y.index].T
            if is_gexp:
                x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)
            else:
                x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= (k.values.diagonal().mean())

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik='normal', verbose=False)

            # Export
            lmm_drug.append(pd.DataFrame(dict(
                beta=lmm.variant_effsizes.values, pval=lmm.variant_pvalues.values, feature=x.columns,
                DRUG_ID=drug[0], DRUG_NAME=drug[1], VERSION=drug[2]
            )))

        lmm_drug = pd.concat(lmm_drug).reset_index(drop=True)
        lmm_drug = self.multipletests_per_drug(lmm_drug, field='pval', fdr_field=f'fdr', method=self.pval_method)
        lmm_drug = lmm_drug.sort_values('fdr')

        # CRISPR associations
        lmm_crispr = []
        for gene in genes:
            # Observations
            y = self.crispr.loc[[gene], samples].T.dropna()
            y = pd.DataFrame(StandardScaler().fit_transform(y), index=y.index, columns=y.columns)

            # Features
            x = X[y.index].T
            if is_gexp:
                x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)
            else:
                x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= (k.values.diagonal().mean())

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik='normal', verbose=False)

            # Export
            lmm_crispr.append(pd.DataFrame(dict(
                beta=lmm.variant_effsizes.values, pval=lmm.variant_pvalues.values, feature=x.columns, gene=gene
            )))

        lmm_crispr = pd.concat(lmm_crispr).reset_index(drop=True)
        lmm_crispr = self.multipletests_per_drug(lmm_crispr, field='pval', fdr_field='fdr', index_cols=['gene'], method=self.pval_method)
        lmm_crispr = lmm_crispr.sort_values('fdr')

        # Expand <Drug, CRISPR> significant associations
        lmm_robust = []
        for i in lmm_res_signif.index:
            row = lmm_res_signif.loc[i]
            drug, gene = tuple(row[self.drespo_obj.DRUG_COLUMNS].values), row['GeneSymbol']

            pair_associations = pd.concat([
                lmm_drug.set_index(self.drespo_obj.DRUG_COLUMNS + ['feature']).loc[drug].add_prefix('drug_'),
                lmm_crispr.set_index(['gene', 'feature']).loc[gene].add_prefix('crispr_')
            ], axis=1, sort=False).reset_index().rename(columns={'index': 'feature'})
            pair_associations.index = [tuple(list(drug) + [gene]) for i in pair_associations.index]

            row = row.to_frame().T
            row.index = [tuple(list(drug) + [gene]) for i in row.index]

            lmm_robust.append(
                row.merge(pair_associations, how='outer', left_index=True, right_index=True)
            )

        lmm_robust = pd.concat(lmm_robust).reset_index(drop=True)

        return lmm_robust

    def get_drug_top(self, lmm_associations, drug, top_features):
        d_genes = lmm_associations\
            .query(f"DRUG_ID == {drug[0]} & DRUG_NAME == '{drug[1]}' & VERSION == '{drug[2]}'")\
            .sort_values('pval')\
            .head(top_features)['GeneSymbol']

        return pd.concat([
            self.drespo.loc[drug],
            self.crispr.loc[d_genes].T
        ], axis=1, sort=False).dropna().T

    def lmm_multiple_association(self, lmm_associations, top_features=10):
        mlmm_res = []
        # drug = (1411, 'SN1041137233', 'v17')

        for drug in self.drespo.index:
            # Top associated genes
            drug_df = self.get_drug_top(lmm_associations, drug, top_features)

            # Kinship and covariates
            k = self.kinship(self.crispr[drug_df.columns].T)
            m = self.get_covariates().loc[drug_df.columns]

            # Combination of features with the top feature
            features = list(drug_df.index[1:])
            combinations = list(it.combinations(features, 2))
            combinations = [c for c in combinations if features[0] in c]

            # Null model with top feature
            lmm_best, lmm_best_params = Association.lmm_association_limix(
                y=drug_df.loc[[drug]].T,
                x=drug_df.loc[[features[0]]].T,
                m=pd.concat([m, drug_df.loc[[features[0]]].T], axis=1).dropna(),
                k=k
            )

            # Test the feature combinations
            for c in combinations:
                # Alternative model with both features
                lmm_alternative, lmm_alt_params = Association.lmm_association_limix(
                    y=drug_df.loc[[drug]].T,
                    x=drug_df.loc[[features[0]]].T,
                    m=pd.concat([m, drug_df.loc[list(c)].T], axis=1).dropna(),
                    k=k
                )

                # Log-ratio test
                lrt_pval = lrt_pvalues(lmm_best.null_lml, lmm_alternative.null_lml, len(c) - 1)

                # Add results
                mlmm_res.append(dict(
                    features='+'.join(c),
                    betas=';'.join(lmm_alternative.null_covariate_effsizes[list(c)].apply(lambda v: f'{v:.5f}')),
                    pval=lrt_pval,
                    DRUG_ID=drug[0],
                    DRUG_NAME=drug[1],
                    VERSION=drug[2]
                ))

        mlmm_res = pd.DataFrame(mlmm_res).sort_values('pval')
        mlmm_res = self.multipletests_per_drug(mlmm_res, field='pval', fdr_field='fdr')

        return mlmm_res

    def lmm_gexp_drug(self):
        # Samples with gene-expression
        samples = list(self.gexp)

        # Kinship matrix (random effects)
        k = self.kinship(self.gexp.T)
        m = self.get_covariates().loc[samples]

        # Association
        lmm_dgexp = pd.concat([
            self.lmm_single_association(
                self.drespo.loc[[d], samples].T, self.gexp.T.loc[samples], k=k.loc[samples, samples], m=m.loc[samples]
            ) for d in self.drespo.index
        ])

        # Multiple p-value correction
        lmm_dgexp = self.multipletests_per_drug(lmm_dgexp, field='pval', method=self.pval_method)

        # Annotate drug target
        lmm_dgexp = self.annotate_drug_target(lmm_dgexp)

        # Annotate association distance to target
        lmm_dgexp = self.ppi.ppi_annotation(
            lmm_dgexp, ppi_type='string', ppi_kws=dict(score_thres=900), target_thres=5
        )

        # Sort p-values
        lmm_dgexp = lmm_dgexp.sort_values(['fdr', 'pval'])

        return lmm_dgexp

    def lmm_gexp_crispr(self, crispr_genes):
        # Samples with gene-expression
        samples = list(self.gexp)

        # Kinship matrix (random effects)
        k = self.kinship(self.gexp.T)
        m = self.get_covariates().loc[samples]

        # Association
        lmm_gexp_crispr = pd.concat([
            self.lmm_single_association(
                self.crispr.loc[[g], samples].T, self.gexp.T.loc[samples], k=k.loc[samples, samples], m=m.loc[samples],
                expand_drug_id=False
            ) for g in crispr_genes
        ])

        # Multiple p-value correction
        lmm_gexp_crispr = self.multipletests_per_drug(
            lmm_gexp_crispr, field='pval', method=self.pval_method, index_cols=['DRUG_ID']
        )

        # Sort p-values
        lmm_gexp_crispr = lmm_gexp_crispr.sort_values(['fdr', 'pval'])

        return lmm_gexp_crispr


if __name__ == '__main__':
    dtype = 'ic50'

    assoc = Association(dtype_drug=dtype)

    lmm_dsingle = assoc.lmm_single_associations()
    lmm_dsingle\
        .sort_values(['pval', 'fdr'])\
        .to_csv(f'data/drug_lmm_regressions_{dtype}.csv.gz', index=False, compression='gzip')

    lmm_dgexp = assoc.lmm_gexp_drug()
    lmm_dgexp\
        .sort_values(['pval', 'fdr'])\
        .to_csv(f'data/drug_lmm_regressions_{dtype}_gexp.csv.gz', index=False, compression='gzip')

    lmm_dgenomic = assoc.lmm_single_associations_genomic()
    lmm_dgenomic\
        .sort_values(['pval', 'fdr'])\
        .to_csv(f'data/drug_lmm_regressions_{dtype}_genomic.csv.gz', index=False, compression='gzip')

    lmm_robust = assoc.lmm_robust_association(lmm_dsingle, is_gexp=False)
    lmm_robust\
        .sort_values(['drug_pval', 'drug_fdr'])\
        .to_csv(f'data/drug_lmm_regressions_robust_{dtype}.csv.gz', index=False, compression='gzip')

    lmm_robust_gexp = assoc.lmm_robust_association(lmm_dsingle, is_gexp=True)
    lmm_robust_gexp\
        .sort_values(['drug_pval', 'drug_fdr'])\
        .to_csv(f'data/drug_lmm_regressions_robust_gexp_{dtype}.csv.gz', index=False, compression='gzip')

    lmm_multi = assoc.lmm_multiple_association(lmm_dsingle)
    lmm_multi \
        .sort_values(['pval', 'fdr']) \
        .to_csv(f'data/drug_lmm_regressions_multiple_{dtype}.csv.gz', index=False, compression='gzip')

    genes = ['MARCH5', 'MCL1', 'BCL2', 'BCL2L1', 'BCL2L11', 'BAX', 'PMAIP1', 'CYCS']
    lmm_gexp_crispr = assoc.lmm_gexp_crispr(crispr_genes=genes)
    lmm_gexp_crispr \
        .sort_values(['pval', 'fdr']) \
        .to_csv(f'data/crispr_lmm_regressions_gexp.csv.gz', index=False, compression='gzip')

