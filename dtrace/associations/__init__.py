#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


DRUG_INFO_COLUMNS = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']


def multipletests_per_drug(lr_associations, method='bonferroni', field='lr_pval'):
    d_unique = {(d_id, d_name, d_version) for d_id, d_name, d_version in lr_associations[DRUG_INFO_COLUMNS].values}

    df = lr_associations.set_index(DRUG_INFO_COLUMNS)

    df = pd.concat([
        df.loc[(d_id, d_name, d_version)].assign(
            fdr=multipletests(df.loc[(d_id, d_name, d_version), field], method=method)[1]
        ) for d_id, d_name, d_version in d_unique
    ]).reset_index()

    return df


def corr_drugtarget_gene(lmm_drug):
    # Get CRISPR
    crispr = dtrace.get_crispr(dtype='logFC', scale=True)

    # List all drugs and genes used for associations
    drugs, genes = set(lmm_drug['DRUG_ID_lib']), set(lmm_drug['GeneSymbol'])

    # Assemble drug targets for which associations where tested
    d_targets = dtrace.get_drugtargets()
    d_targets = {d: d_targets[d].intersection(genes) for d in d_targets if d in drugs}
    d_targets = {d: d_targets[d] for d in d_targets if len(d_targets[d]) > 0}

    # List all tested targets
    targets = {g for d in d_targets for g in d_targets[d]}

    # Correlation matrix
    c_genes = list(targets.union(genes))
    c_corr = pd.DataFrame(np.corrcoef(crispr.loc[c_genes].values), index=c_genes, columns=c_genes).to_dict()

    def get_drug_gene_corr(d, g):
        if (d not in d_targets) or (g not in c_corr):
            return np.nan

        elif g in d_targets[d]:
            return 1.

        else:
            return max([c_corr[t][g] for t in d_targets[d]], key=abs)

    lmm_drug = lmm_drug.assign(corr=[get_drug_gene_corr(d, g) for d, g in lmm_drug[['DRUG_ID_lib', 'GeneSymbol']].values])

    return lmm_drug


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


def dist_drugtarget_genes(drug_targets, genes, ppi):
    genes = genes.intersection(set(ppi.vs['name']))
    assert len(genes) != 0, 'No genes overlapping with PPI provided'

    dmatrix = {}

    for drug in drug_targets:
        drug_genes = drug_targets[drug].intersection(genes)

        if len(drug_genes) != 0:
            dmatrix[drug] = dict(zip(*(genes, np.min(ppi.shortest_paths(source=drug_genes, target=genes), axis=0))))

    return dmatrix


def ppi_dist_to_string(d, target_thres):
    if d == 0:
        res = 'T'

    elif d == np.inf:
        res = '-'

    elif d < target_thres:
        res = str(int(d))

    else:
        res = '>={}'.format(target_thres)

    return res


def ppi_annotation(df, ppi_type, ppi_kws, target_thres=4):
    df_genes, df_drugs = set(df['GeneSymbol']), set(df['DRUG_ID_lib'])

    # PPI annotation
    ppi = ppi_type(**ppi_kws)

    # Drug target
    d_targets = dtrace.get_drugtargets()
    d_targets = {k: d_targets[k] for k in df_drugs if k in d_targets}

    # Calculate distance between drugs and CRISPRed genes in PPI
    dist_d_g = dist_drugtarget_genes(d_targets, df_genes, ppi)

    # Annotate drug regressions
    def drug_gene_annot(d, g):
        if d not in d_targets:
            res = '-'

        elif g in d_targets[d]:
            res = 'T'

        elif d not in dist_d_g or g not in dist_d_g[d]:
            res = '-'

        else:
            res = ppi_dist_to_string(dist_d_g[d][g], target_thres)

        return res

    df = df.assign(target=[drug_gene_annot(d, g) for d, g in df[['DRUG_ID_lib', 'GeneSymbol']].values])

    return df


def count_ppis(lr_associations):
    d_columns = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']

    df = lr_associations.set_index(d_columns).dropna()

    d_unique = {(d_id, d_name, d_version) for d_id, d_name, d_version in df.index}

    d_targets = {(d_id, d_name, d_version): df.loc[(d_id, d_name, d_version), 'target_thres'].value_counts() for d_id, d_name, d_version in d_unique}

    d_targets = pd.DataFrame(d_targets).T

    return d_targets
