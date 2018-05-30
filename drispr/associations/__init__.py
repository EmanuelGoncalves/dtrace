#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from drispr import get_drugtargets, dist_drugtarget_genes, DRUG_INFO_COLUMNS


def multipletests_per_drug(lr_associations, method='bonferroni', field='lr_pval'):
    d_unique = {(d_id, d_name, d_version) for d_id, d_name, d_version in lr_associations[DRUG_INFO_COLUMNS].values}

    df = lr_associations.set_index(DRUG_INFO_COLUMNS)

    df = pd.concat([
        df.loc[(d_id, d_name, d_version)].assign(
            fdr=multipletests(df.loc[(d_id, d_name, d_version), field], method=method)[1]
        ) for d_id, d_name, d_version in d_unique
    ]).reset_index()

    return df


def ppi_corr(ppi, m_corr, m_corr_thres=None):
    """
    Annotate PPI network based on Pearson correlation between the vertices of each edge using
    m_corr data-frame and m_corr_thres (Pearson > m_corr_thress).

    :param ppi:
    :param m_corr:
    :param m_corr_thres:
    :return:
    """
    # Subset PPI network
    ppi = ppi.subgraph([i.index for i in ppi.vs if i['name'] in m_corr.index])

    # Edge correlation
    crispr_pcc = np.corrcoef(m_corr.loc[ppi.vs['name']].values)
    ppi.es['corr'] = [crispr_pcc[i.source, i.target] for i in ppi.es]

    # Sub-set by correlation between vertices of each edge
    if m_corr_thres is not None:
        ppi = ppi.subgraph_edges([i.index for i in ppi.es if abs(i['corr']) > m_corr_thres])

    print(ppi.summary())

    return ppi


def ppi_annotation(df, ppi_type, ppi_kws, target_thres=4):
    # PPI annotation
    ppi = ppi_type(**ppi_kws)

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
