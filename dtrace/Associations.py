#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import itertools as it
import DataImporter as DataImporter
from limix.qtl import scan
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from statsmodels.stats.multitest import multipletests


dpath = pkg_resources.resource_filename("dtrace", "data/")


class Association:
    """
    Main class to test linear associations bewteen data-sets (e.g. drug-response and CRISPR-Cas9 knockout viability
    measurements).

    """

    def __init__(
        self,
        dtype="ic50",
        pval_method="bonferroni",
        load_associations=False,
        load_robust=False,
        load_ppi=False,
        load_simple_lm=False,
        ppi_thres=900,
        combine_lmm=False,
    ):
        """
        :param dtype: Drug-response of a drug in a cell line was represented either as an IC50 (ic50) or
            area-under the drug-response curve (auc).

        :param pval_method: Multiple hypothesis adjustment method. Any option available in multipletests
            (statsmodels.stats.multitest).

        :param load_associations: Load associations (this implies the associations were already tested).

        """

        self.dtype = dtype
        self.ppi_thres = ppi_thres
        self.pval_method = pval_method
        self.dcols = DataImporter.DrugResponse.DRUG_COLUMNS
        self.ppi_order = ["T", "1", "2", "3", "4", "5+", "-"]

        # Import data-sets
        self.crispr_obj = DataImporter.CRISPR()
        self.drespo_obj = DataImporter.DrugResponse()
        self.genomic_obj = DataImporter.Genomic()
        self.gexp_obj = DataImporter.GeneExpression()
        self.cn_obj = DataImporter.CopyNumber()
        self.wes_obj = DataImporter.WES()
        self.methy_obj = DataImporter.Methylation()

        self.samples = list(
            set.intersection(
                set(self.drespo_obj.get_data().columns),
                set(self.crispr_obj.get_data().columns),
            )
        )

        logging.getLogger("DTrace").info(f"#(Samples)={len(self.samples)}")

        # Import PPI and samplesheet
        self.ppi = DataImporter.PPI()
        self.samplesheet = DataImporter.Sample()

        # Filter
        self.crispr = self.crispr_obj.filter(subset=self.samples, scale=True)
        self.drespo = self.drespo_obj.filter(subset=self.samples, dtype=self.dtype)
        self.genomic = self.genomic_obj.filter(subset=self.samples, min_events=5)
        self.gexp = self.gexp_obj.filter(subset=self.samples)
        self.cn = self.cn_obj.filter(subset=self.samples)
        self.wes = self.wes_obj.filter(
            subset=self.samples, as_matrix=True, min_events=5
        )
        self.methy = self.methy_obj.filter(subset=self.samples)

        logging.getLogger("DTrace").info(
            f"#(Drugs)={self.drespo.shape[0]}; "
            f"#(Genes)={self.crispr.shape[0]}; "
            f"#(Genomic)={self.genomic.shape[0]}; "
        )

        # Association files
        self.lmm_drug_crispr_file = (
            f"{dpath}/drug_lmm_regressions_{self.dtype}_crispr.csv.gz"
        )
        self.lmm_drug_gexp_file = (
            f"{dpath}/drug_lmm_regressions_{self.dtype}_gexp.csv.gz"
        )
        self.lmm_drug_genomic_file = (
            f"{dpath}/drug_lmm_regressions_{self.dtype}_genomic.csv.gz"
        )

        self.lmm_robust_gexp_file = (
            f"{dpath}/drug_lmm_regressions_robust_{self.dtype}_gexp.csv.gz"
        )
        self.lmm_robust_genomic_file = (
            f"{dpath}/drug_lmm_regressions_robust_{self.dtype}_genomic.csv.gz"
        )
        self.lmm_robust_wes_file = (
            f"{dpath}/drug_lmm_regressions_robust_{self.dtype}_wes.csv.gz"
        )

        # Association file with no covariates nor random effects
        self.lmm_drug_crispr_NO_mk_file = (
            f"{dpath}/drug_lmm_regressions_{self.dtype}_crispr_NO_mk.csv.gz"
        )

        # Load associations
        if load_associations:
            self.lmm_drug_crispr = pd.read_csv(self.lmm_drug_crispr_file)
            self.lmm_drug_gexp = pd.read_csv(self.lmm_drug_gexp_file)
            self.lmm_drug_genomic = pd.read_csv(self.lmm_drug_genomic_file)

        # Load robust associations
        if load_robust:
            self.lmm_robust_gexp = pd.read_csv(self.lmm_robust_gexp_file)
            self.lmm_robust_genomic = pd.read_csv(self.lmm_robust_genomic_file)
            self.lmm_robust_wes = pd.read_csv(self.lmm_robust_wes_file)

        # Load PPI
        if load_ppi:
            self.ppi_string = self.ppi.build_string_ppi(score_thres=self.ppi_thres)
            self.ppi_string_corr = self.ppi.ppi_corr(self.ppi_string, self.crispr)

        # Load associations estimated with simple linear regressions
        if load_simple_lm:
            self.lmm_drug_crispr = pd.read_csv(self.lmm_drug_crispr_NO_mk_file)

        # Combine Drug ~ CRISPR and Drug ~ GExp associations
        if combine_lmm:
            cols = ["DRUG_ID", "DRUG_NAME", "VERSION", "GeneSymbol"]
            self.lmm_combined = pd.concat(
                [
                    self.lmm_drug_crispr.set_index(cols).add_prefix("CRISPR_"),
                    self.lmm_drug_gexp.set_index(cols).add_prefix("GExp_"),
                ],
                axis=1,
                sort=False,
            ).dropna()

    def build_association_matrix(
        self, associations=None, index=None, columns=None, values=None
    ):
        associations = self.lmm_drug_crispr if associations is None else associations
        index = ["DRUG_ID", "DRUG_NAME", "VERSION"] if index is None else index
        columns = "GeneSymbol" if columns is None else columns
        values = "beta" if values is None else values

        assoc_matrix = pd.pivot_table(
            associations, index=index, columns=columns, values=values
        )

        return assoc_matrix

    def get_covariates(self):
        # Samples CRISPR QC (recall essential genes)
        crispr_ess_qc = self.crispr_obj.qc_ess.rename("recall_essential").rename(
            "crispr_qc_ess"
        )
        crispr_ess_qc = (crispr_ess_qc - crispr_ess_qc.mean()) / crispr_ess_qc.std()

        # CRISPR institute of origin PC
        crispr_insitute = pd.get_dummies(self.samplesheet.samplesheet["institute"])

        # Cell lines growth rate
        drug_growth = self.drespo_obj.import_pca()[self.dtype]["column"]["pcs"]["PC1"]

        # Cell lines culture conditions
        culture = pd.get_dummies(
            self.samplesheet.samplesheet["growth_properties"]
        ).drop(columns=["Unknown"])

        # Merge covariates
        covariates = pd.concat(
            [crispr_ess_qc, crispr_insitute, drug_growth, culture], axis=1, sort=False
        ).loc[self.samples]

        return covariates

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K

    @staticmethod
    def lmm_association_limix(y, x, m=None, k=None, add_intercept=True, lik="normal", filter_std=True):
        # Build matrices
        Y = y.dropna()
        Y = pd.DataFrame(
            StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
        )

        X = x.loc[Y.index]
        X = pd.DataFrame(
            StandardScaler().fit_transform(X), index=X.index, columns=X.columns
        )

        # Random effects matrix
        if k is None:
            K = Association.kinship(x.loc[Y.index])

        else:
            K = k.loc[Y.index, Y.index]

        # Covariates
        if m is not None:
            m = m.loc[Y.index]

            if filter_std:
                m = m.loc[:, m.std() > 0]

        # Add intercept
        if add_intercept and (m is not None):
            m["intercept"] = 1

        elif add_intercept and (m is None):
            m = pd.DataFrame(
                np.ones((Y.shape[0], 1)), index=Y.index, columns=["intercept"]
            )

        else:
            m = pd.DataFrame(
                np.zeros((Y.shape[0], 1)), index=Y.index, columns=["zeros"]
            )

        # Linear Mixed Model
        lmm = scan(X, Y, K=K.values, M=m, lik=lik, verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def multipletests_per_drug(
        associations, method, field="pval", fdr_field="fdr", index_cols=None
    ):
        index_cols = (
            DataImporter.DrugResponse.DRUG_COLUMNS if index_cols is None else index_cols
        )

        d_unique = {tuple(i) for i in associations[index_cols].values}

        df = associations.set_index(index_cols)

        df = pd.concat(
            [
                df.loc[i]
                .assign(fdr=multipletests(df.loc[i, field], method=method)[1])
                .rename(columns={"fdr": fdr_field})
                for i in d_unique
            ]
        ).reset_index()

        return df

    @staticmethod
    def get_association(lmm_associations, drug, gene):
        return lmm_associations[
            (lmm_associations[DataImporter.DrugResponse.DRUG_COLUMNS[0]] == drug[0])
            & (lmm_associations[DataImporter.DrugResponse.DRUG_COLUMNS[1]] == drug[1])
            & (lmm_associations[DataImporter.DrugResponse.DRUG_COLUMNS[2]] == drug[2])
            & (lmm_associations["GeneSymbol"] == gene)
        ]

    @staticmethod
    def lmm_single_association(y, x, m=None, k=None, expand_drug_id=True, verbose=0):
        cols_rename = dict(
            effect_name="GeneSymbol", pv20="pval", effsize="beta", effsize_se="beta_se"
        )

        drug = y.columns[0]

        if verbose:
            v_label = f"Drug {drug[1]} from screen {drug[2]} with ID {drug[0]}" if expand_drug_id else drug
            logging.getLogger("DTrace").info(
                f"[lmm_single_association] {v_label}"
            )

        lmm, params = Association.lmm_association_limix(y, x, m, k)

        effect_sizes = lmm.effsizes["h2"].query("effect_type == 'candidate'")
        effect_sizes = effect_sizes.set_index("test").drop(
            columns=["trait", "effect_type"]
        )

        pvalues = lmm.stats["pv20"]

        res = pd.concat([effect_sizes, pvalues], axis=1, sort=False)
        res = res.reset_index(drop=True).rename(columns=cols_rename)

        if expand_drug_id:
            res = res.assign(DRUG_ID=drug[0])
            res = res.assign(DRUG_NAME=drug[1])
            res = res.assign(VERSION=drug[2])

        else:
            res = res.assign(DRUG_ID=drug)

        res = res.assign(samples=params["y"].shape[0])
        res = res.assign(ncovariates=params["m"].shape[1])

        return res

    def lmm_single_associations(
        self, verbose=0, add_covariates=True, add_random_effects=True
    ):
        # - Kinship matrix (random effects)
        k = self.kinship(self.crispr.T) if add_random_effects else None
        m = self.get_covariates() if add_covariates else None

        # - Single feature linear mixed regression
        # Association
        lmm_single = pd.concat(
            [
                self.lmm_single_association(
                    self.drespo.loc[[d]].T, self.crispr.T, k=k, m=m, verbose=verbose
                )
                for d in self.drespo.index
            ]
        )

        # Multiple p-value correction
        lmm_single = self.multipletests_per_drug(
            lmm_single, field="pval", method=self.pval_method
        )

        # Annotate drug target
        lmm_single = self.annotate_drug_target(lmm_single)

        # Annotate association distance to target
        lmm_single = self.ppi.ppi_annotation(
            lmm_single, ppi_type="string", ppi_kws=dict(score_thres=900), target_thres=5
        )

        # Sort p-values
        lmm_single = lmm_single.sort_values(["fdr", "pval"])

        return lmm_single

    def lmm_single_associations_genomic(self, min_events=5):
        X = self.genomic

        samples, drugs = set(X).intersection(self.samples), self.drespo.index
        print(f"Samples={len(samples)};Drugs={len(drugs)}")

        lmm_drug = []
        for drug in drugs:
            # Observations
            y = self.drespo.loc[[drug], samples].T.dropna()
            y = pd.DataFrame(
                StandardScaler().fit_transform(y), index=y.index, columns=y.columns
            )

            # Features
            x = X[y.index].T
            x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= k.values.diagonal().mean()

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik="normal", verbose=False)

            # Export
            lmm_drug.append(
                pd.DataFrame(
                    dict(
                        beta=lmm.variant_effsizes.values,
                        pval=lmm.variant_pvalues.values,
                        feature=x.columns,
                        DRUG_ID=drug[0],
                        DRUG_NAME=drug[1],
                        VERSION=drug[2],
                    )
                )
            )

        lmm_drug = pd.concat(lmm_drug).reset_index(drop=True)
        lmm_drug = self.multipletests_per_drug(
            lmm_drug, field="pval", fdr_field=f"fdr", method=self.pval_method
        )
        lmm_drug = lmm_drug.sort_values(["fdr", "pval"])

        return lmm_drug

    def get_xdata(self, xtype):
        if xtype == "genomic":
            return self.genomic

        elif xtype == "wes":
            return self.wes

        elif xtype == "gexp":
            return self.gexp

        else:
            assert False, f"{xtype} not defined"

    def lmm_robust_association(
        self, lmm_dsingle, xtype="genomic", fdr_thres=0.1, min_events=5, verbose=0
    ):
        if verbose > 0:
            logging.getLogger("DTrace").info(f"Robust associations with {xtype}")

        lmm_res_signif = lmm_dsingle.query(f"fdr < {fdr_thres}")

        genes = set(lmm_res_signif["GeneSymbol"])
        drugs = {tuple(d) for d in lmm_res_signif[self.dcols].values}

        X = self.get_xdata(xtype)
        samples = set(X).intersection(self.samples)
        logging.getLogger("DTrace").info(
            f"Assoc={lmm_res_signif.shape[0]}; Genes={len(genes)}; Drugs={len(drugs)}; Samples={len(samples)}"
        )

        # Drug associations
        lmm_drug = []
        for drug in drugs:
            # Observations
            y = self.drespo.loc[[drug], samples].T.dropna()
            y = pd.DataFrame(
                StandardScaler().fit_transform(y), index=y.index, columns=y.columns
            )

            # Features
            x = X[y.index].T

            if xtype == "gexp":
                x = pd.DataFrame(
                    StandardScaler().fit_transform(x), index=x.index, columns=x.columns
                )

            else:
                x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= k.values.diagonal().mean()

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik="normal", verbose=False)

            # Export
            lmm_drug.append(
                pd.DataFrame(
                    dict(
                        beta=lmm.variant_effsizes.values,
                        pval=lmm.variant_pvalues.values,
                        feature=x.columns,
                        DRUG_ID=drug[0],
                        DRUG_NAME=drug[1],
                        VERSION=drug[2],
                    )
                )
            )

        lmm_drug = pd.concat(lmm_drug).reset_index(drop=True)
        lmm_drug = self.multipletests_per_drug(
            lmm_drug, field="pval", fdr_field=f"fdr", method=self.pval_method
        )
        lmm_drug = lmm_drug.sort_values("fdr")

        # CRISPR associations
        lmm_crispr = []
        for gene in genes:
            # Observations
            y = self.crispr.loc[[gene], samples].T.dropna()
            y = pd.DataFrame(
                StandardScaler().fit_transform(y), index=y.index, columns=y.columns
            )

            # Features
            x = X[y.index].T
            if xtype == "gexp":
                x = pd.DataFrame(
                    StandardScaler().fit_transform(x), index=x.index, columns=x.columns
                )

            else:
                x = x.loc[:, x.sum() >= min_events]

            # Random effects matrix
            k = x.loc[y.index]
            k = k.dot(k.T)
            k /= k.values.diagonal().mean()

            # Linear Mixed Model
            lmm = scan(x, y, K=k, lik="normal", verbose=False)

            # Export
            lmm_crispr.append(
                pd.DataFrame(
                    dict(
                        beta=lmm.variant_effsizes.values,
                        pval=lmm.variant_pvalues.values,
                        feature=x.columns,
                        gene=gene,
                    )
                )
            )

        lmm_crispr = pd.concat(lmm_crispr).reset_index(drop=True)
        lmm_crispr = self.multipletests_per_drug(
            lmm_crispr,
            field="pval",
            fdr_field="fdr",
            index_cols=["gene"],
            method=self.pval_method,
        )
        lmm_crispr = lmm_crispr.sort_values("fdr")

        # Expand <Drug, CRISPR> significant associations
        lmm_robust = []
        for i in lmm_res_signif.index:
            row = lmm_res_signif.loc[i]
            drug, gene = (
                tuple(row[self.drespo_obj.DRUG_COLUMNS].values),
                row["GeneSymbol"],
            )

            pair_associations = (
                pd.concat(
                    [
                        lmm_drug.set_index(self.drespo_obj.DRUG_COLUMNS + ["feature"])
                        .loc[drug]
                        .add_prefix("drug_"),
                        lmm_crispr.set_index(["gene", "feature"])
                        .loc[gene]
                        .add_prefix("crispr_"),
                    ],
                    axis=1,
                    sort=False,
                )
                .reset_index()
                .rename(columns={"index": "feature"})
            )
            pair_associations.index = [
                tuple(list(drug) + [gene]) for i in pair_associations.index
            ]

            row = row.to_frame().T
            row.index = [tuple(list(drug) + [gene]) for i in row.index]

            lmm_robust.append(
                row.merge(
                    pair_associations, how="outer", left_index=True, right_index=True
                )
            )

        lmm_robust = pd.concat(lmm_robust).reset_index(drop=True)

        return lmm_robust

    @staticmethod
    def lm_outofsample(y, x, n_splits, test_size):
        y = y[x.index].dropna()
        y = pd.DataFrame(
            StandardScaler().fit_transform(y.to_frame()), index=y.index
        ).iloc[:, 0]

        x = x.loc[y.index]
        x = pd.DataFrame(
            StandardScaler().fit_transform(x), index=x.index, columns=x.columns
        )

        df = []
        for train, test in ShuffleSplit(n_splits=n_splits, test_size=test_size).split(
            x, y
        ):
            lm = RidgeCV().fit(x.iloc[train], y.iloc[train])

            r2 = lm.score(x.iloc[test], y.iloc[test])

            df.append(dict(r2=r2, beta=lm.coef_[0]))

        return pd.DataFrame(df)

    def lmm_multiple_association(
        self, lmm_associations, top_features=3, n_splits=1000, test_size=0.3, verbose=1
    ):
        mlmm_res = []
        # drug = (1411, 'SN1041137233', 'v17')
        for drug in self.drespo.index:
            if verbose > 0:
                print("Drug =", drug)

            # Top associated genes
            drug_df = self.get_drug_top(lmm_associations, drug, top_features).T

            # Combination of features with the top feature
            features = list(drug_df.iloc[:, 1:])
            combinations = list(it.combinations(features, 2))

            # Test single feature models
            y = drug_df.iloc[:, 0]

            lm_features = {
                f: self.lm_outofsample(
                    y=y, x=drug_df[[f]], n_splits=n_splits, test_size=test_size
                )
                for f in features
            }

            # Test the feature combinations
            for c in combinations:
                for atype in ["+", "*"]:
                    if atype == "+":
                        c_feature = (drug_df[c[0]] + drug_df[c[1]]).to_frame()
                    elif atype == "*":
                        c_feature = (drug_df[c[0]] * drug_df[c[1]]).to_frame()
                    else:
                        assert False, f"Combination type not supported: {atype}"

                    lm_combined = self.lm_outofsample(
                        y=y, x=c_feature, n_splits=n_splits, test_size=test_size
                    )

                    c_lm = pd.concat(
                        [
                            lm_combined.add_suffix("_combined"),
                            lm_features[c[0]].add_suffix("_feature1"),
                            lm_features[c[1]].add_suffix("_feature2"),
                        ],
                        axis=1,
                        sort=False,
                    )

                    c_lm = (
                        c_lm.assign(combined=atype.join(c))
                        .assign(feature1=c[0])
                        .assign(feature2=c[1])
                        .assign(atype=atype)
                        .assign(DRUG_ID=drug[0])
                        .assign(DRUG_NAME=drug[1])
                        .assign(VERSION=drug[2])
                    )

                    mlmm_res.append(c_lm)

        mlmm_res = pd.concat(mlmm_res)

        return mlmm_res

    def lmm_gexp_drug(self):
        # Samples with gene-expression
        samples = list(self.gexp)

        # Kinship matrix (random effects)
        k = self.kinship(self.gexp.T)
        m = self.get_covariates().loc[samples]

        # Association
        lmm_dgexp = pd.concat(
            [
                self.lmm_single_association(
                    self.drespo.loc[[d], samples].T,
                    self.gexp.T.loc[samples],
                    k=k.loc[samples, samples],
                    m=m.loc[samples],
                )
                for d in self.drespo.index
            ]
        )

        # Multiple p-value correction
        lmm_dgexp = self.multipletests_per_drug(
            lmm_dgexp, field="pval", method=self.pval_method
        )

        # Annotate drug target
        lmm_dgexp = self.annotate_drug_target(lmm_dgexp)

        # Annotate association distance to target
        lmm_dgexp = self.ppi.ppi_annotation(
            lmm_dgexp, ppi_type="string", ppi_kws=dict(score_thres=900), target_thres=5
        )

        # Sort p-values
        lmm_dgexp = lmm_dgexp.sort_values(["fdr", "pval"])

        return lmm_dgexp

    def lmm_gexp_crispr(self, crispr_genes):
        # Samples with gene-expression
        samples = list(self.gexp)

        # Kinship matrix (random effects)
        k = self.kinship(self.gexp.T)
        m = self.get_covariates().loc[samples]

        # Association
        lmm_gexp_crispr = pd.concat(
            [
                self.lmm_single_association(
                    self.crispr.loc[[g], samples].T,
                    self.gexp.T.loc[samples],
                    k=k.loc[samples, samples],
                    m=m.loc[samples],
                    expand_drug_id=False,
                )
                for g in crispr_genes
            ]
        )

        # Multiple p-value correction
        lmm_gexp_crispr = self.multipletests_per_drug(
            lmm_gexp_crispr,
            field="pval",
            method=self.pval_method,
            index_cols=["DRUG_ID"],
        )

        # Sort p-values
        lmm_gexp_crispr = lmm_gexp_crispr.sort_values(["fdr", "pval"])

        return lmm_gexp_crispr

    def annotate_drug_target(self, associations):
        d_targets = self.drespo_obj.get_drugtargets()

        associations["DRUG_TARGETS"] = [
            ";".join(d_targets[d]) if d in d_targets else np.nan
            for d in associations["DRUG_ID"]
        ]

        return associations

    def get_drug_top(self, associations, drug, top_features):
        d_genes = self.by(
            associations, drug_id=drug[0], drug_name=drug[1], drug_version=drug[2]
        )
        d_genes = d_genes.sort_values("pval").head(top_features)["GeneSymbol"]

        d_genes_top = pd.concat(
            [self.drespo.loc[drug], self.crispr.loc[d_genes].T], axis=1, sort=False
        )
        d_genes_top = d_genes_top.dropna().T

        return d_genes_top

    @staticmethod
    def by(
        associations,
        drug_id=None,
        drug_name=None,
        drug_version=None,
        gene_name=None,
        target=None,
        fdr=None,
        fdr_reverse=False,
        pval=None,
        pval_reverse=False,
        feature=None,
    ):
        """
        Utility function to select associations based on and operations between multiple parameters.

        :param associations:
        :param drug_id:
        :param drug_name:
        :param drug_version:
        :param gene_name:
        :param target:
        :param fdr:
        :param fdr_reverse:
        :param pval:
        :param pval_reverse:
        :param feature: For Robust Associations ONLY
        :return:
        """

        df = associations.copy()

        if fdr is not None:
            if fdr_reverse:
                df = df[df["fdr"] >= fdr]
            else:
                df = df[df["fdr"] < fdr]

        if pval is not None:
            if pval_reverse:
                df = df[df["pval"] >= pval]
            else:
                df = df[df["pval"] < pval]

        if drug_id is not None:
            if (type(drug_id) == list) or (type(drug_id) == set):
                df = df[df["DRUG_ID"].isin(drug_id)]
            else:
                df = df[df["DRUG_ID"] == drug_id]

        if drug_name is not None:
            if (type(drug_name) == list) or (type(drug_name) == set):
                df = df[df["DRUG_NAME"].isin(drug_name)]
            else:
                df = df[df["DRUG_NAME"] == drug_name]

        if drug_version is not None:
            if (type(drug_version) == list) or (type(drug_version) == set):
                df = df[df["VERSION"].isin(drug_version)]
            else:
                df = df[df["VERSION"] == drug_version]

        if gene_name is not None:
            if (type(gene_name) == list) or (type(gene_name) == set):
                df = df[df["GeneSymbol"].isin(gene_name)]
            else:
                df = df[df["GeneSymbol"] == gene_name]

        if target is not None:
            if (type(target) == list) or (type(target) == set):
                df = df[df["target"].isin(target)]
            else:
                df = df[df["target"] == target]

        if feature is not None:
            if (type(feature) == list) or (type(feature) == set):
                df = df[df["feature"].isin(feature)]
            else:
                df = df[df["feature"] == feature]

        return df

    def build_df(
        self,
        drug=None,
        crispr=None,
        gexp=None,
        genomic=None,
        wes=None,
        cn=None,
        sinfo=None,
        bin_to_string=False,
        crispr_discretise=False,
    ):
        """
        Utility function to build data-frames containing multiple types of measurements.

        :param drug:
        :param crispr:
        :param gexp:
        :param genomic:
        :param wes:
        :param cn:
        :param sinfo:
        :param bin_to_string:
        :param crispr_discretise:
        :return:
        """

        def bin_to_string_fun(v):
            if np.isnan(v):
                return np.nan
            elif v == 1:
                return "Yes"
            else:
                return "No"

        df = []

        if drug is not None:
            df.append(self.drespo.loc[drug].T)

        if crispr is not None:
            df.append(self.crispr.loc[crispr].T.add_prefix("crispr_"))

            if crispr_discretise:
                df.append(
                    self.discretise_essentiality(crispr, self.crispr)
                    .rename("crispr")
                    .to_frame()
                )

        if gexp is not None:
            df.append(self.gexp.loc[gexp].T.add_prefix("gexp_"))

        if wes is not None:
            wes_df = self.wes.loc[wes].T.add_prefix("wes_")

            if bin_to_string:
                wes_df = wes_df.applymap(bin_to_string_fun)

            df.append(wes_df)

        if cn is not None:
            df.append(self.cn.loc[cn].T.add_prefix("cn_"))

        if genomic is not None:
            genomic_df = self.genomic.loc[genomic].T

            if bin_to_string:
                genomic_df = genomic_df.applymap(bin_to_string_fun)

            df.append(genomic_df)

        if sinfo is not None:
            df.append(self.samplesheet.samplesheet[sinfo])

        df = pd.concat(df, axis=1, sort=False)

        return df

    @staticmethod
    def discretise_essentiality(gene_list, dmatrix, threshold=-0.5):
        discrete = {
            s: " + ".join([g for g in gene_list if dmatrix.loc[g, s] < threshold])
            for s in dmatrix
        }

        discrete = pd.Series(discrete).replace("", "None")

        return discrete


class AssociationCTDR2:
    def __init__(
        self,
        dtype="auc",
        pval_method="bonferroni",
        ppi_thres=900,
        load_lmm_crispr=False,
        load_ppi=False,
    ):
        self.dtype = dtype
        self.ppi_thres = ppi_thres
        self.pval_method = pval_method

        # Import data-sets
        self.crispr_obj = DataImporter.Ceres()
        self.drespo_obj = DataImporter.CTDR2()

        self.samples = list(
            set.intersection(
                set(self.drespo_obj.get_data().columns),
                set(self.crispr_obj.get_data().columns),
            )
        )

        logging.getLogger("DTrace").info(f"#(Samples)={len(self.samples)}")

        # Import PPI and samplesheet
        self.ppi = DataImporter.PPI(drug_targets=self.drespo_obj.get_drugtargets())

        # Filter
        self.crispr = self.crispr_obj.filter(subset=self.samples)
        self.drespo = self.drespo_obj.filter(subset=self.samples, dtype=self.dtype)

        logging.getLogger("DTrace").info(
            f"#(Drugs)={self.drespo.shape[0]}; " f"#(Genes)={self.crispr.shape[0]}; "
        )

        # Association files
        self.lmm_drug_crispr_file = (
            f"{dpath}/drug_lmm_regressions_{self.dtype}_crispr_ctdr2.csv.gz"
        )

        if load_lmm_crispr:
            self.lmm_drug_crispr = pd.read_csv(self.lmm_drug_crispr_file)

        # Load PPI
        if load_ppi:
            self.ppi_string = self.ppi.build_string_ppi(score_thres=self.ppi_thres)
            self.ppi_string_corr = self.ppi.ppi_corr(self.ppi_string, self.crispr)

    def get_covariates(self):
        # Cell lines culture conditions
        culture = pd.get_dummies(
            self.drespo_obj.samplesheet.groupby("DepMap_ID")["culture_media"].first()
        )

        # Cell lines growth rate
        drug_growth = self.drespo_obj.import_pca()[self.dtype]["column"]["pcs"]["PC1"]

        # Merge covariates
        covariates = pd.concat([drug_growth, culture], axis=1, sort=False).loc[self.samples]

        return covariates

    def annotate_drug(self, associations):
        d_targets = self.drespo_obj.get_drugtargets()

        associations["DRUG_TARGETS"] = [
            ";".join(d_targets[d]) if d in d_targets else np.nan
            for d in associations["DRUG_ID"]
        ]

        associations["DRUG_NAME"] = self.drespo_obj.drugsheet.loc[
            associations["DRUG_ID"], "cpd_name"
        ].values
        associations["broad_cpd_id"] = self.drespo_obj.drugsheet.loc[
            associations["DRUG_ID"].values, "broad_cpd_id"
        ].values
        associations["cpd_status"] = self.drespo_obj.drugsheet.loc[
            associations["DRUG_ID"].values, "cpd_status"
        ].values
        associations["target_or_activity_of_compound"] = self.drespo_obj.drugsheet.loc[
            associations["DRUG_ID"].values, "target_or_activity_of_compound"
        ].values

        return associations

    def lmm_single_associations(
        self, verbose=0, add_covariates=True, add_random_effects=True
    ):
        # - Kinship matrix (random effects)
        k = (
            Association.kinship(self.crispr[self.samples].T)
            if add_random_effects
            else None
        )
        m = self.get_covariates() if add_covariates else None

        # - Single feature linear mixed regression
        # Association
        lmm_single = pd.concat(
            [
                Association.lmm_single_association(
                    y=self.drespo.loc[[d], self.samples].T,
                    x=self.crispr[self.samples].T,
                    k=k,
                    m=m,
                    verbose=verbose,
                    expand_drug_id=False,
                )
                for d in self.drespo.index
            ]
        )

        # Multiple p-value correction
        lmm_single = Association.multipletests_per_drug(
            lmm_single, field="pval", method=self.pval_method, index_cols=["DRUG_ID"]
        )

        # Annotate drug target
        lmm_single = self.annotate_drug(lmm_single)

        # Annotate association distance to target
        lmm_single = self.ppi.ppi_annotation(
            lmm_single, ppi_type="string", ppi_kws=dict(score_thres=900), target_thres=5
        )

        # Sort p-values
        lmm_single = lmm_single.sort_values(["fdr", "pval"])

        return lmm_single
