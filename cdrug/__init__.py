#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
from cdrug.assemble.assemble_ppi import STRING_PICKLE, BIOGRID_PICKLE, build_biogrid_ppi

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
CRISPR_MAGECK_DEP_FDR = 'data/crispr/MAGeCK_depFDRs.tsv'
CRISPR_MAGECK_ENR_FDR = 'data/crispr/MAGeCK_enrFDRs.tsv'

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

PAL_BIN = {1: PAL_SET2[1], 0: PAL_SET2[8]}
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

# - DRUG INFO COLUMNS
DRUG_INFO_COLUMNS = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']


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


def get_crispr(dtype='logFC', fdr_thres=0.05):
    """
    CRISPR-Cas9 scores as log fold-changes (CN corrected) or binary matrices marking (1) significant
    depletions (dtype = 'depletions'), enrichments (dtype = 'enrichments') or both (dtype = 'both').

    :param dtype: String (default = 'logFC')
    :param fdr_thres: Float (default = 0.1)
    :return: pandas.DataFrame
    """

    dep_fdr = pd.read_csv(CRISPR_MAGECK_DEP_FDR, index_col=0, sep='\t').dropna()
    enr_fdr = pd.read_csv(CRISPR_MAGECK_ENR_FDR, index_col=0, sep='\t').dropna()

    if dtype == 'both':
        crispr = ((dep_fdr < fdr_thres) | (enr_fdr < fdr_thres)).astype(int)

    elif dtype == 'depletions':
        crispr = (dep_fdr < fdr_thres).astype(int)

    elif dtype == 'enrichments':
        crispr = (enr_fdr < fdr_thres).astype(int)

    else:
        crispr = pd.read_csv(CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()

    return crispr


def get_growth():
    growth = pd.read_csv(GROWTHRATE_FILE, index_col=0)
    return growth


def get_drugtargets():
    ds = get_drugsheet()

    d_targets = ds['Target Curated'].dropna().to_dict()

    d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}

    return d_targets


def get_essential_genes():
    return set(pd.read_csv(HART_ESSENTIAL)['gene'])


def get_nonessential_genes():
    return set(pd.read_csv(HART_NON_ESSENTIAL)['gene'])


# - DATA-SETS FILTER FUNCTIONS
def filter_drugresponse(d_response, min_events=3, min_meas=0.85):
    """
    Filter Drug-response (ln IC50) data-set to consider only drugs with measurements across
    at least min_meas (deafult=0.85) of the total cell lines measured and drugs with an IC50
    lower than the global average in at least min_events (default = 3) cell lines.

    :param d_response:
    :param min_events:
    :param min_meas:
    :return:
    """
    df = d_response[d_response.count(1) > (d_response.shape[1] * min_meas)]

    df = df[(df < df.median().median()).sum(1) >= min_events]

    return df


def filter_mobem(mobem, min_events=3):
    df = mobem[mobem.sum(1) >= min_events]
    return df


def filter_crispr(crispr, min_events=3, fdr_thres=0.05):
    """
    Filter CRISPR-Cas9 data-set to consider only genes that show a significant depletion or
    enrichment, MAGeCK depletion/enrichment FDR < fdr_thres (default = 0.05), in at least
    min_events (default = 3) cell lines.

    :param crispr:
    :param min_events:
    :param fdr_thres:
    :return:
    """
    signif_genes = get_crispr(dtype='both', fdr_thres=fdr_thres)
    signif_genes = signif_genes[signif_genes.sum(1) >= min_events]

    df = crispr.loc[signif_genes.index]

    return df


# - DATA-SETS PROCESSING FUNCTIONS
def scale_crispr(df, essential=None, non_essential=None, metric=np.median):
    """
    Min/Max scaling of CRISPR-Cas9 log-FC by median (default) Essential and Non-Essential.

    :param df: Float pandas.DataFrame
    :param essential: set(String)
    :param non_essential: set(String)
    :param metric: np.Median (default)
    :return: Float pandas.DataFrame
    """

    if essential is None:
        essential = get_essential_genes()

    if non_essential is None:
        non_essential = get_nonessential_genes()

    assert len(essential.intersection(df.index)) != 0, 'DataFrame has no index overlapping with essential list'
    assert len(non_essential.intersection(df.index)) != 0, 'DataFrame has no index overlapping with non essential list'

    essential_metric = metric(df.reindex(essential).dropna(), axis=0)
    non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

    df = df.subtract(non_essential_metric).divide(non_essential_metric - essential_metric)

    return df


# -
def mobem_feature_to_gene(f):
    """
    Extract Gene Symbol of the MOBEM copy-number and mutation features.

    :param f:
    :return:
    """
    if f.endswith('_mut'):
        genes = {f.split('_')[0]}

    elif f.startswith('gain.') or f.startswith('loss.'):
        genes = {g for fs in f.split('..') if not (fs.startswith('gain.') or fs.startswith('loss.')) for g in fs.split('.') if g != ''}

    else:
        raise ValueError('{} is not a valid MOBEM feature.'.format(f))

    return genes


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


def dist_drugtarget_genes(drug_targets, genes, ppi):
    genes = genes.intersection(set(ppi.vs['name']))
    assert len(genes) != 0, 'No genes overlapping with PPI provided'

    dmatrix = {}

    for drug in drug_targets:
        drug_genes = drug_targets[drug].intersection(genes)

        if len(drug_genes) != 0:
            dmatrix[drug] = dict(zip(*(genes, np.min(ppi.shortest_paths(source=drug_genes, target=genes), axis=0))))

    return dmatrix
