#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import numpy as np
import pandas as pd
import pkg_resources
import dtrace.DataImporter as DataImporter
from limix.qtl import scan
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


dpath = pkg_resources.resource_filename("dtrace", "data/")


class Association:
    """
    Main class to test linear associations bewteen data-sets (e.g. drug response and CRISPR-Cas9 knockout viability
    measurements).

    """

    def __init__(
        self,
        pval_method="fdr_bh",
        load_associations=False,
        load_robust=False,
        load_ppi=False,
        ppi_thres=900,
        combine_lmm=False,
    ):
        """
        :param pval_method: Multiple hypothesis adjustment method. Any option available in multipletests
            (statsmodels.stats.multitest).

        :param load_associations: Load associations (this implies the associations were already tested).

        """

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
        self.drespo = self.drespo_obj.filter(subset=self.samples)
        self.genomic = self.genomic_obj.filter(subset=self.samples, min_events=5)
        self.gexp = self.gexp_obj.filter(subset=self.samples)
        self.cn = self.cn_obj.filter(subset=self.samples)

        logging.getLogger("DTrace").info(
            f"#(Drugs)={self.drespo.shape[0]}; "
            f"#(Genes)={self.crispr.shape[0]}; "
            f"#(Genomic)={self.genomic.shape[0]}; "
        )

        # Association files
        self.lmm_drug_crispr_file = f"{dpath}/drug_lmm_regressions_crispr.csv.gz"
        self.lmm_drug_gexp_file = f"{dpath}/drug_lmm_regressions_gexp.csv.gz"
        self.lmm_drug_genomic_file = f"{dpath}/drug_lmm_regressions_genomic.csv.gz"

        self.lmm_robust_gexp_file = f"{dpath}/drug_lmm_regressions_robust_gexp.csv.gz"
        self.lmm_robust_genomic_file = (
            f"{dpath}/drug_lmm_regressions_robust_genomic.csv.gz"
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
            self.lmm_robust_genomic = self.lmm_robust_genomic.rename(columns=dict(genomic="x_feature"))

        # Load PPI
        if load_ppi:
            self.ppi_string = self.ppi.build_string_ppi(score_thres=self.ppi_thres)
            self.ppi_string_corr = self.ppi.ppi_corr(self.ppi_string, self.crispr)

        # Combine Drug ~ CRISPR and Drug ~ GExp associations
        if combine_lmm:
            cols = DataImporter.DrugResponse.DRUG_COLUMNS + ["GeneSymbol"]
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
        index = DataImporter.DrugResponse.DRUG_COLUMNS if index is None else index
        columns = "GeneSymbol" if columns is None else columns
        values = "beta" if values is None else values

        assoc_matrix = pd.pivot_table(
            associations, index=index, columns=columns, values=values
        )

        return assoc_matrix

    def get_covariates(self):
        # CRISPR institute of origin PC
        crispr_insitute = pd.get_dummies(self.samplesheet.samplesheet["institute"])

        # Cell lines growth rate
        drug_growth = self.drespo_obj.import_pca()["column"]["pcs"]["PC1"]

        # Cell lines culture conditions
        culture = pd.get_dummies(
            self.samplesheet.samplesheet["growth_properties"]
        ).drop(columns=["Unknown"])

        # Merge covariates
        covariates = pd.concat(
            [crispr_insitute, drug_growth, culture], axis=1, sort=False
        ).loc[self.samples]

        return covariates

    @staticmethod
    def kinship(k):
        K = k.dot(k.T)
        K /= K.values.diagonal().mean()
        return K

    @staticmethod
    def lmm_association_limix(
        y,
        x,
        m=None,
        k=None,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        filter_std=True,
        min_events=5,
    ):
        # Build Y
        Y = y.dropna()

        if transform_y == "scale":
            Y = pd.DataFrame(
                StandardScaler().fit_transform(Y), index=Y.index, columns=Y.columns
            )

        elif transform_y == "rank":
            Y = Y.rank(axis=1)

        # Build X
        X = x.loc[Y.index]
        X = X.loc[:, X.std() > 0]

        if transform_x == "scale":
            X = pd.DataFrame(
                StandardScaler().fit_transform(X), index=X.index, columns=X.columns
            )

        elif transform_x == "rank":
            X = X.rank(axis=1)

        elif transform_x == "min_events":
            X = X.loc[:, X.sum() >= min_events]

        # Random effects matrix
        if k is None:
            K = None

        elif k is False:
            K = Association.kinship(x.loc[Y.index]).values

        else:
            K = k.loc[Y.index, Y.index].values

        # Covariates + Intercept
        if m is not None:
            m = m.loc[Y.index]

            if filter_std:
                m = m.loc[:, m.std() > 0]

            m = m.assign(intercept=1)

        # Linear Mixed Model
        lmm = scan(X, Y, K=K, M=m, lik=lik, verbose=False)

        return lmm, dict(x=X, y=Y, k=K, m=m)

    @staticmethod
    def lmm_single_association(
        y,
        x,
        m=None,
        k=None,
        expand_id=True,
        lik="normal",
        transform_y="scale",
        transform_x="scale",
        filter_std=True,
        verbose=0,
    ):
        cols_rename = dict(
            effect_name="GeneSymbol", pv20="pval", effsize="beta", effsize_se="beta_se"
        )

        y_id = y.columns[0]

        if verbose:
            v_label = (
                f"Drug {y_id[1]} from screen {y_id[2]} with ID {y_id[0]}"
                if expand_id
                else y_id
            )
            logging.getLogger("DTrace").info(f"[lmm_single_association] {v_label}")

        lmm, params = Association.lmm_association_limix(
            y=y,
            x=x,
            m=m,
            k=k,
            lik=lik,
            transform_y=transform_y,
            transform_x=transform_x,
            filter_std=filter_std,
        )

        effect_sizes = lmm.effsizes["h2"].query("effect_type == 'candidate'")
        effect_sizes = effect_sizes.set_index("test").drop(
            columns=["trait", "effect_type"]
        )

        pvalues = lmm.stats["pv20"]

        res = pd.concat([effect_sizes, pvalues], axis=1, sort=False)
        res = res.reset_index(drop=True).rename(columns=cols_rename)

        if expand_id:
            res = res.assign(DRUG_ID=y_id[0])
            res = res.assign(DRUG_NAME=y_id[1])
            res = res.assign(VERSION=y_id[2])

        else:
            res = res.assign(Y_ID=y_id)

        res = res.assign(samples=params["y"].shape[0])
        res = res.assign(ncovariates=0 if m is None else params["m"].shape[1])

        return res

    def lmm_single_associations(
        self, add_covariates=True, add_random_effects=True, x_dtype="crispr", verbose=0
    ):
        # - Samples
        if x_dtype == "gexp":
            samples = list(self.gexp)

        elif x_dtype == "genomic":
            samples = list(self.genomic)

        else:
            samples = self.samples

        # - y
        y = self.drespo[samples].T

        # - x
        if x_dtype == "gexp":
            x = self.gexp[samples].T

        elif x_dtype == "genomic":
            x = self.genomic[samples].T

        else:
            x = self.crispr[samples].T

        # - Kinship matrix (random effects)
        k = self.kinship(x).loc[samples, samples] if add_random_effects else None

        # - Covariates
        m = self.get_covariates().loc[samples] if add_covariates else None

        # - Single feature linear mixed regression
        # Association
        lmm_single = pd.concat(
            [
                self.lmm_single_association(
                    y[[d]],
                    x,
                    k=k,
                    m=m,
                    transform_x="scale" if x_dtype != "genomic" else "min_events",
                    verbose=verbose,
                )
                for d in y
            ]
        )

        # Multiple p-value correction
        lmm_single = self.multipletests_per_drug(
            lmm_single, field="pval", method=self.pval_method
        )

        # Annotate drug target
        lmm_single = self.annotate_drug_target(lmm_single)

        # Annotate association distance to target
        if x_dtype != "genomic":
            lmm_single = self.ppi.ppi_annotation(
                lmm_single, ppi_type="string", ppi_kws=dict(score_thres=900), target_thres=5
            )

        # X data type
        lmm_single = lmm_single.assign(x_dtype=x_dtype)

        # Sort p-values
        lmm_single = lmm_single.sort_values(["fdr", "pval"])

        return lmm_single

    def lmm_robust_associations(
        self,
        lmm_dsingle,
        add_covariates=True,
        add_random_effects=True,
        x_dtype="genomic",
        fdr_thres=0.1,
        verbose=0,
    ):
        if verbose > 0:
            logging.getLogger("DTrace").info(f"Robust associations with {x_dtype}")

        # - Get significant <Drug, CRISPR> pairs
        pairs = lmm_dsingle.query(f"fdr < {fdr_thres}")
        drugs = list({tuple(d) for d in pairs[self.dcols].values})
        genes = set(pairs["GeneSymbol"])

        # - Samples
        if x_dtype == "gexp":
            samples = list(self.gexp)

        else:
            samples = list(self.genomic)

        # - yy
        y_drug = self.drespo.loc[drugs, samples].T
        y_crispr = self.crispr.loc[genes, samples].T

        # - x
        if x_dtype == "gexp":
            x = self.gexp[samples].T

        else:
            x = self.genomic[samples].T

        # - Kinship matrix (random effects)
        k = self.kinship(x).loc[samples, samples] if add_random_effects else None

        # - Covariates
        m = self.get_covariates().loc[samples] if add_covariates else None

        logging.getLogger("DTrace").info(
            f"Assoc={pairs.shape[0]}; Genes={len(genes)}; Drugs={len(drugs)}; Samples={len(samples)}"
        )

        # - Drug associations ((DRUG_ID, DRUG_NAME, VERSION), x_feature)
        lmm_drug = pd.concat(
            [
                self.lmm_single_association(
                    y_drug[[d]],
                    x,
                    k=k,
                    m=m,
                    transform_x="scale" if x_dtype == "gexp" else "min_events",
                    verbose=verbose,
                )
                for d in y_drug
            ]
        ).rename(columns={"GeneSymbol": "x_feature"})
        lmm_drug = lmm_drug.set_index(self.drespo_obj.DRUG_COLUMNS)

        # - CRISPR associations (Y_ID, x_feature)
        lmm_crispr = pd.concat(
            [
                self.lmm_single_association(
                    y_crispr[[g]],
                    x,
                    k=k,
                    m=m,
                    transform_x="scale" if x_dtype == "gexp" else "min_events",
                    expand_id=False,
                    verbose=verbose,
                )
                for g in y_crispr
            ]
        ).rename(columns={"GeneSymbol": "x_feature"})
        lmm_crispr = lmm_crispr.set_index("Y_ID")

        # - Expand <Drug, CRISPR> associations
        lmm_robust = []
        for i in pairs.index:
            row = pairs.loc[i]

            drug = tuple(row[self.drespo_obj.DRUG_COLUMNS].values)
            gene = row["GeneSymbol"]

            pair_associations = (
                pd.concat(
                    [
                        lmm_drug.loc[drug].set_index("x_feature").add_prefix("drug_"),
                        lmm_crispr.loc[gene]
                        .set_index("x_feature")
                        .add_prefix("crispr_"),
                    ],
                    axis=1,
                    sort=False,
                )
                .dropna()
                .reset_index()
                .rename(columns={"index": x_dtype})
            )

            pair_associations.insert(0, "GeneSymbol", gene)
            pair_associations.insert(0, "VERSION", drug[2])
            pair_associations.insert(0, "DRUG_NAME", drug[1])
            pair_associations.insert(0, "DRUG_ID", drug[0])

            lmm_robust.append(pair_associations)

        lmm_robust = pd.concat(lmm_robust).reset_index(drop=True)

        lmm_robust = self.multipletests_per_drug(
            lmm_robust,
            field="drug_pval",
            fdr_field=f"drug_fdr",
            index_cols=self.drespo_obj.DRUG_COLUMNS + ["GeneSymbol"],
            method=self.pval_method,
        )

        lmm_robust = self.multipletests_per_drug(
            lmm_robust,
            field="crispr_pval",
            fdr_field=f"crispr_fdr",
            index_cols=self.drespo_obj.DRUG_COLUMNS + ["GeneSymbol"],
            method=self.pval_method,
        )

        lmm_robust = self.annotate_drug_target(lmm_robust)

        lmm_robust = self.ppi.ppi_annotation(
            lmm_robust, ppi_type="string", ppi_kws=dict(score_thres=900), target_thres=5
        )

        return lmm_robust

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
        x_feature=None,
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

        if x_feature is not None:
            if (type(x_feature) == list) or (type(x_feature) == set):
                df = df[df["x_feature"].isin(x_feature)]
            else:
                df = df[df["x_feature"] == x_feature]

        return df

    def build_df(
        self,
        drug=None,
        crispr=None,
        gexp=None,
        genomic=None,
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
