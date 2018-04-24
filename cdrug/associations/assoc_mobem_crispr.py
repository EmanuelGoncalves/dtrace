#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.associations import multipletests_per_drug
from statsmodels.stats.multitest import multipletests


THRES_FDR, THRES_BETA = .1, 0.25


def get_significant_crispr_associations(associations, thres_fdr, thres_beta):
    df = associations[(associations['beta'].abs() > thres_beta) & (associations['lr_fdr'] < thres_fdr)]
    df = {(d, g) for d, g in df[['DRUG_ID_lib', 'GeneSymbol']].values}
    return df


def is_feature_in(drug, feature, signif_associations):
    """
    drug, feature = 1021, 'gain.cnaPANCAN344..MYCN.'

    :param feature:
    :return:
    """
    feature_genes = cdrug.mobem_feature_to_gene(feature)

    if len(feature_genes) != 0:
        is_in = len({(drug, gene) for gene in feature_genes}.intersection(signif_associations)) > 0

    else:
        is_in = np.nan

    return is_in


def annotate_significant(df, signif):
    df = df.assign(crispr_signif=[is_feature_in(d, f, signif) for d, f in df[['DRUG_ID_lib', 'level_3']].values])
    return df


if __name__ == '__main__':
    # - Import linear regressions
    lr_mobem = pd.read_csv(lr_files.LR_BINARY_DRUG_MOBEMS_ALL)
    lr_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # - Compute FDR per drug
    lr_mobem = multipletests_per_drug(lr_mobem)
    lr_crispr = multipletests_per_drug(lr_crispr)

    # - Remove Genetic Feature without any gene mapped
    crispr_genes = set(lr_crispr['GeneSymbol'])
    lr_mobem = lr_mobem[[len(cdrug.mobem_feature_to_gene(i).intersection(crispr_genes)) > 0 for i in lr_mobem['level_3']]]

    # - CRISPR significant (Drug, Gene) associations
    signif_crispr = get_significant_crispr_associations(lr_crispr, THRES_FDR, THRES_BETA)

    # -
    lr_mobem = annotate_significant(lr_mobem, signif_crispr)
    print(lr_mobem[['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'level_3', 'lr_fdr', 'crispr_signif']].sort_values('lr_fdr'))

    # -
    df = lr_mobem[(lr_mobem['beta'].abs() > THRES_BETA) & (lr_mobem['lr_fdr'] < THRES_FDR)]
    df.groupby(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'level_3'])['crispr_signif'].max().sum()
