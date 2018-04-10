#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd
import seaborn as sns
from cdrug.assemble.assemble_ppi import STRING_PICKLE, BIOGRID_PICKLE

# - META DATA
SAMPLESHEET_FILE = 'data/meta/samplesheet.csv'
DRUGSHEET_FILE = 'data/meta/drug_samplesheet_updated.txt'

# - GENE LISTS
HART_ESSENTIAL = 'data/gene_sets/curated_BAGEL_essential.csv'
HART_NON_ESSENTIAL = 'data/gene_sets/curated_BAGEL_nonEssential.csv'

# - GROWTH RATE
GROWTHRATE_FILE = 'data/gdsc/growth/growth_rate.csv'

# - CRISPR
CRISPR_GENE_FILE = 'data/meta/_00_Genes_for_panCancer_assocStudies.csv'
CRISPR_GENE_FC_CORRECTED = 'data/crispr/CRISPRcleaned_logFCs.tsv'
CRISPR_GENE_BAGEL = 'data/crispr/BayesianFactors.tsv'
CRISPR_GENE_BINARY = 'data/crispr/binaryDepScores.tsv'

# - DRUG-RESPONSE
DRUG_RESPONSE_FILE = 'data/drug_ic50_merged_matrix.csv'

DRUG_RESPONSE_V17 = 'data/drug/screening_set_384_all_owners_fitted_data_20180308.csv'
DRUG_RESPONSE_VRS = 'data/drug/rapid_screen_1536_all_owners_fitted_data_20180308.csv'

# - NUMBER OF MUTATIONS
WES_COUNT = 'data/gdsc/WES_variants.csv'

# - METHYLATION
METHYLATION_GENE_PROMOTER = 'data/gdsc/methylation/methy_beta_gene_promoter.csv'

# - MOBEM
MOBEM = 'data/PANCAN_mobem.csv'

# - GENE-EXPRESSION
RNASEQ_VOOM = 'data/gdsc/gene_expression/merged_voom_preprocessed.csv'

# - PALETTES
PAL_DBGD = ['#37454B', '#F2C500']
PAL_TAB20C = sns.color_palette('tab20c', n_colors=20).as_hex()
PAL_SET2 = sns.color_palette('Set2', n_colors=8).as_hex() + ['#333333']

PAL_DRUG_VERSION = dict(RS=PAL_SET2[1], v17=PAL_SET2[8])

# - PLOTTING AESTHETICS
SNS_RC = {
    'axes.linewidth': .3,
    'xtick.major.width': .3,
    'ytick.major.width': .3,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
}

sns.set(style='ticks', context='paper', rc=SNS_RC, font_scale=.75)


# - GETS
def get_drugsheet():
    return pd.read_csv(DRUGSHEET_FILE, sep='\t', index_col=0)


def get_samplesheet(index='Cell Line Name'):
    return pd.read_csv(SAMPLESHEET_FILE).dropna(subset=[index]).set_index(index)


def get_mobem(drop_factors=True):
    ss = get_samplesheet(index='COSMIC ID')

    mobem = pd.read_csv(MOBEM, index_col=0)
    mobem = mobem.set_index(ss.loc[mobem.index, 'Cell Line Name'], drop=True)
    mobem = mobem.T

    if drop_factors:
        mobem = mobem.drop(['TISSUE_FACTOR', 'MSI_FACTOR', 'MEDIA_FACTOR'])

    return mobem


def get_drugresponse():
    d_response = pd.read_csv(DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    return d_response


def get_crispr(is_binary=False):
    crispr = pd.read_csv(CRISPR_GENE_BINARY if is_binary else CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    return crispr


def get_growth():
    growth = pd.read_csv(GROWTHRATE_FILE, index_col=0)
    return growth


# -
def mobem_feature_to_gene(f):
    if f.endswith('_mut'):
        genes = set([f.split('_')[0]])

    if f.startswith('gain.') or f.startswith('loss.'):
        genes = {g for fs in f.split('..') if not (fs.startswith('gain.') or fs.startswith('loss.')) for g in fs.split('.') if g != ''}

    return genes


def filter_drugresponse(d_response, min_meas=0.85):
    df = d_response[d_response.count(1) > d_response.shape[1] * min_meas]
    return df


def filter_mobem(mobem, min_events=3):
    df = mobem[mobem.sum(1) >= min_events]
    return df


def filter_crispr(crispr, is_binary=False, min_events=3):
    if is_binary:
        df = crispr[crispr.sum(1) >= min_events]

    else:
        raise NotImplementedError

    return df


def build_covariates(variables=None, add_growth=True, samples=None):
    variables = ['Cancer Type'] if variables is None else variables

    ss = get_samplesheet()

    covariates = pd.get_dummies(ss[variables])

    if add_growth:
        covariates = pd.concat([covariates, get_growth()['growth_rate_median']], axis=1)

    if samples is not None:
        covariates = covariates.loc[samples]

    covariates = covariates.loc[:, covariates.sum() != 0]

    return covariates


def is_same_drug(drug_id_1, drug_id_2):
    """
    From 2 drug IDs check if Drug ID 1 has Name or Synonyms in common with Drug ID 2.

    :param drug_id_1:
    :param drug_id_2:
    :return: Bool
    """

    drug_list = get_drugsheet()

    for i, d in enumerate([drug_id_1, drug_id_2]):
        assert d in drug_list.index, 'Drug ID {} not in drug list'.format(i)

    drug_names = {d: get_drug_names(d, drug_list) for d in [drug_id_1, drug_id_2]}

    return len(drug_names[drug_id_1].intersection(drug_names[drug_id_2])) > 0


def get_drug_names(drug_id):
    """
    From a Drug ID get drug Name and Synonyms.

    :param drug_id:
    :return:
    """

    drug_list = get_drugsheet()

    if drug_id not in drug_list.index:
        print('{} Drug ID not in drug list'.format(drug_id))
        return None

    drgu_name = [drug_list.loc[drug_id, 'Name']]

    drug_synonyms = drug_list.loc[drug_id, 'Synonyms']
    drug_synonyms = [] if str(drug_synonyms).lower() == 'nan' else drug_synonyms.split(', ')

    return set(drgu_name + drug_synonyms)
