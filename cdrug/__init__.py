#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import iqr
from cdrug.assemble.assemble_ppi import STRING_PICKLE, BIOGRID_PICKLE

# META DATA
SAMPLESHEET_FILE = 'data/meta/samplesheet.csv'
DRUGSHEET_FILE = 'data/meta/drug_samplesheet_updated.txt'

# GENE LISTS
HART_ESSENTIAL = 'data/gene_sets/curated_BAGEL_essential.csv'
HART_NON_ESSENTIAL = 'data/gene_sets/curated_BAGEL_nonEssential.csv'

# GROWTH RATE
GROWTHRATE_FILE = 'data/gdsc/growth/growth_rate.csv'

# CRISPR
CRISPR_GENE_FILE = 'data/meta/_00_Genes_for_panCancer_assocStudies.csv'
CRISPR_GENE_FC_CORRECTED = 'data/crispr/CRISPRcleaned_logFCs.tsv'
CRISPR_GENE_BAGEL = 'data/crispr/BayesianFactors.tsv'
CRISPR_GENE_BINARY = 'data/crispr/binaryDepScores.tsv'

# DRUG-RESPONSE
DRUG_RESPONSE_FILE = 'data/drug_ic50_merged_matrix.csv'

DRUG_RESPONSE_V17 = 'data/drug/screening_set_384_all_owners_fitted_data_20180308.csv'
DRUG_RESPONSE_VRS = 'data/drug/rapid_screen_1536_all_owners_fitted_data_20180308.csv'

# Palettes
PAL_DBGD = ['#37454B', '#F2C500']
PAL_TAB20C = sns.color_palette('tab20c', n_colors=20).as_hex()
PAL_SET2 = sns.color_palette('Set2', n_colors=8).as_hex()

PAL_DRUG_VERSION = dict(RS=PAL_SET2[1], v17=PAL_SET2[7])

# Number of mutations
WES_COUNT = 'data/gdsc/WES_variants.csv'

# Methylation
METHYLATION_GENE_PROMOTER = 'data/gdsc/methylation/methy_beta_gene_promoter.csv'

# MOBEM
MOBEM = 'data/PANCAN_mobem.csv'

# Gene-expression
RNASEQ_VOOM = 'data/gdsc/gene_expression/merged_voom_preprocessed.csv'

# Set plotting aesthetics
sns_rc = {
    'axes.linewidth': .3,
    'xtick.major.width': .3, 'ytick.major.width': .3,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'in', 'ytick.direction': 'in'
}
sns.set(style='ticks', context='paper', rc=sns_rc, font_scale=.75)


def import_drug_list(filter_web_pub=True, drug_list_file=None, sep='\t', index_col=0):
    drug_list_file = DRUGSHEET_FILE if drug_list_file is None else drug_list_file

    ds = pd.read_csv(drug_list_file, sep=sep, index_col=index_col)

    if filter_web_pub:
        ds = ds[[w == 'Y' or p == 'Y' for w, p in ds[['Web Release', 'Suitable for publication']].values]]

    return ds


def scale_crispr(df, essential=None, non_essential=None, metric=np.median):
    if essential is None:
        essential = set(pd.read_csv(HART_ESSENTIAL)['gene'])

    if non_essential is None:
        non_essential = set(pd.read_csv(HART_NON_ESSENTIAL)['gene'])

    assert len(essential.intersection(df.index)) != 0, 'DataFrame has no index overlapping with essential list'
    assert len(non_essential.intersection(df.index)) != 0, 'DataFrame has no index overlapping with non essential list'

    essential_metric = metric(df.reindex(essential).dropna(), axis=0)
    non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

    df = df.subtract(non_essential_metric).divide(non_essential_metric - essential_metric)

    return df


def crispr_genes(file=None, samples_thres=5, type_thres=3):
    file = CRISPR_GENE_FILE if file is None else file

    c_genes = pd.read_csv(file, index_col=0)

    c_genes = c_genes[c_genes.drop('n. vulnerable cell lines', axis=1).sum(1) <= type_thres]

    c_genes = c_genes[c_genes['n. vulnerable cell lines'] >= samples_thres]

    return set(c_genes.index)


def filter_crispr(df, essential=None, essential_thres=90, value_thres=1.5, value_nevents=5):
    if not isinstance(essential, set):
        essential = pd.read_csv(CRISPR_GENE_FILE, index_col=0)['n. vulnerable cell lines']
        essential = set(essential[essential > essential_thres].index)

    assert len(essential) != 0, 'Essential genes list is empty'

    df = df.drop(essential, errors='ignore', axis=0)

    df = df[(df.abs() >= value_thres).sum(1) >= value_nevents]

    return df


def filter_drug_response(df, percentage_measurements=0.85, ic50_samples=5, iqr_thres=1):
    df = df[df.count(1) > df.shape[1] * percentage_measurements]

    df = df.loc[(df < df.mean().mean()).sum(1) >= ic50_samples]

    df = df.loc[[iqr(values, nan_policy='omit') > iqr_thres for idx, values in df.iterrows()]]

    return df


def filter_mobem(df, n_events=5):
    df = df[df.sum(1) >= n_events]
    return df


def drug_targets(file=None):
    file = DRUGSHEET_FILE if file is None else file

    d_targets = pd.read_csv(file, index_col=0, sep='\t')['Target Curated'].dropna().to_dict()

    d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}

    return d_targets


def check_in_list(gene_list, test_list=None):
    if test_list is None:
        test_list = {'TP53', 'PTEN', 'EGFR', 'ERBB2', 'IGF1R', 'MDM2', 'MDM4', 'BRAF', 'MAPK1'}

    list_difference = test_list.difference(gene_list)

    assert len(list_difference) == 0, 'Genes missing: {}'.format(';'.join(list_difference))

    return True
