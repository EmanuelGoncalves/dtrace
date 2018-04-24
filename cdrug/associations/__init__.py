#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from cdrug.assemble.assemble_ppi import build_biogrid_ppi
from cdrug import get_drugtargets, dist_drugtarget_genes, DRUG_INFO_COLUMNS

# - CONTINUOUS CRISPR ASSOCIATIONS
LR_DRUG_CRISPR = 'data/drug_regressions_crispr.csv'
LR_DRUG_CRISPR_NOGROWTH = 'data/drug_regressions_crispr_NOGROWTH.csv'

LR_DRUG_CRISPR_NOSCALE = 'data/drug_regressions_crispr_NOSCALE.csv'
LR_DRUG_CRISPR_NOSCALE_NOGROWTH = 'data/drug_regressions_crispr_NOSCALE_NOGROWTH.csv'

# - CONTINUOUS RNASEQ ASSOCIATIONS
LR_DRUG_RNASEQ = 'data/drug_regressions_rnaseq.csv'

# - BINARY ASSOCIATIONS
LR_BINARY_DRUG_MOBEMS = 'data/drug_regressions_binary_mobems.csv'
LR_BINARY_DRUG_MOBEMS_ALL = 'data/drug_regressions_binary_mobems_all.csv'

LR_BINARY_DRUG_CRISPR = 'data/drug_regressions_binary_crispr.csv'


def multipletests_per_drug(lr_associations, method='bonferroni'):
    d_unique = {(d_id, d_name, d_version) for d_id, d_name, d_version in lr_associations[DRUG_INFO_COLUMNS].values}

    df = lr_associations.set_index(DRUG_INFO_COLUMNS)

    df = pd.concat([
        df.loc[(d_id, d_name, d_version)].assign(
            lr_fdr=multipletests(df.loc[(d_id, d_name, d_version), 'lr_pval'], method=method)[1]
        ) for d_id, d_name, d_version in d_unique
    ]).reset_index()

    return df


def ppi_annotation(df, int_type, exp_type, target_thres=4):
    # PPI annotation
    ppi = build_biogrid_ppi(int_type=int_type, exp_type=exp_type)

    # Drug target
    d_targets = get_drugtargets()

    # Calculate distance between drugs and CRISPR genes in PPI
    dist_d_g = dist_drugtarget_genes(d_targets, set(df['GeneSymbol']), ppi)

    # Annotate drug regressions
    df = df.assign(
        target=[
            dist_d_g[d][g] if d in dist_d_g and g in dist_d_g[d] else np.nan for d, g in df[['DRUG_ID_lib', 'GeneSymbol']].values
        ]
    )

    # Discrete annotation of targets
    df = df.assign(target_thres=['Target' if i == 0 else ('%d' % i if i < target_thres else '>={}'.format(target_thres)) for i in df['target']])

    # Preserve the non-mapped drugs
    df.loc[df['target'].apply(np.isnan), 'target_thres'] = np.nan

    return df


def count_ppis(lr_associations):
    d_columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']

    df = lr_associations.set_index(d_columns).dropna()

    d_unique = {(d_id, d_name, d_version) for d_id, d_name, d_version in df.index}

    d_targets = {(d_id, d_name, d_version): df.loc[(d_id, d_name, d_version), 'target_thres'].value_counts() for d_id, d_name, d_version in d_unique}

    d_targets = pd.DataFrame(d_targets).T

    return d_targets
