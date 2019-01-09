#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import random
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import analysis.ctrp as ctrp
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import auc
from statsmodels.stats.weightstats import ztest
from dtrace.Associations import DRUG_INFO_COLUMNS
from scipy.stats import pearsonr, gaussian_kde, spearmanr, rankdata


def gsea(dataset, signature, permutations=0):
    # Sort data-set by values
    _dataset = list(zip(*sorted(dataset.items(), key=operator.itemgetter(1), reverse=False)))
    genes, expression = _dataset[0], _dataset[1]

    # Signature overlapping the data-set
    _signature = set(signature).intersection(genes)

    # Check signature overlap
    e_score, p_value = np.NaN, np.NaN
    hits, running_hit = [], []
    if len(_signature) != 0:

        # ---- Calculate signature enrichment score
        n, sig_size = len(genes), len(_signature)
        nh = n - sig_size
        nr = sum([abs(dataset[g]) for g in _signature])

        e_score = __es(genes, expression, _signature, nr, nh, n, hits, running_hit)

        # ---- Calculate statistical enrichment
        # Generate random signatures sampled from the data-set genes
        if permutations > 0:
            count = 0

            for i in range(permutations):
                r_signature = random.sample(genes, sig_size)

                r_nr = sum([abs(dataset[g]) for g in r_signature])

                r_es = __es(genes, expression, r_signature, r_nr, nh, n)

                if (r_es >= e_score >= 0) or (r_es <= e_score < 0):
                    count += 1

            # If no permutation was above the Enrichment score the p-value is lower than 1 divided by the number of permutations
            p_value = 1 / permutations if count == 0 else count / permutations

        else:
            p_value = np.nan

    return e_score, p_value, hits, running_hit


def __es(genes, expression, signature, nr, nh, n, hits=None, running_hit=None):
    hit, miss, es, r = 0, 0, 0, 0
    for i in range(n):
        if genes[i] in signature:
            hit += abs(expression[i]) / nr

            if hits is not None:
                hits.append(1)

        else:
            miss += 1 / nh

            if hits is not None:
                hits.append(0)

        r = hit - miss

        if running_hit is not None:
            running_hit.append(r)

        if abs(r) > abs(es):
            es = r

    return es


def gkn(values):
    kernel = gaussian_kde(values)
    kernel = pd.Series({k: np.log(kernel.integrate_box_1d(-1e7, v) / kernel.integrate_box_1d(v, 1e7)) for k, v in values.to_dict().items()})
    return kernel


def recall_curve(rank, index_set, min_events=None):
    x = rank.sort_values().dropna()

    # Observed cumsum
    y = x.index.isin(index_set)

    if (min_events is not None) and (sum(y) < min_events):
        return None

    y = np.cumsum(y) / sum(y)

    # Rank fold-changes
    x = rankdata(x) / x.shape[0]

    # Calculate AUC
    xy_auc = auc(x, y)

    return x, y, xy_auc


if __name__ == '__main__':
    # - Imports
    essential = dtrace.get_essential_genes()
    nessential = dtrace.get_nonessential_genes()

    # GDSC
    drespo = dtrace.get_drugresponse()
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    # Samplesheet
    ctrp_samples = ctrp.import_ctrp_samplesheet()

    # Compounds
    ctrp_compounds = ctrp.import_ctrp_compound_sheet()

    # Drug AUCs
    ctrp_aucs = ctrp.import_ctrp_aucs()

    # Gene CERES scores
    depmap_ceres = ctrp.get_ceres()

    # - Overlap
    samples = list(set(ctrp_aucs).intersection(depmap_ceres))

    # - Drug signatures
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)

    d_signatures = pd.pivot_table(lmm_drug.query('beta > 0.25'), index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')
    d_signatures = d_signatures[d_signatures.count(1) >= 10].dropna(how='all', axis=1)

    # -
    d_signatures = pd.pivot_table(lmm_drug, index=DRUG_INFO_COLUMNS, columns='GeneSymbol', values='beta')

    d = (1168, 'Erlotinib', 'RS')

    d_escore = {}
    for s in samples:
        print(f'# {s}')

        sig = d_signatures.loc[d]
        sig = sig[sig > 0.2]

        score, pval = ztest(
            depmap_ceres[s].reindex(sig.index).dropna(),
            depmap_ceres[s].reindex(nessential).drop(sig.index, errors='ignore').dropna()
        )

        # score = recall_curve(depmap_ceres[s], set())[2]

        d_escore[s] = -np.log10(pval) if score < 0 else np.log10(pval)

    d_escore = pd.Series(d_escore)

    #
    plot_df = pd.concat([
        d_escore.rename('score'),
        ctrp_aucs.loc[365].rename('auc')
    ], axis=1, sort=False).dropna()

    sns.jointplot('score', 'auc', data=plot_df)
    plt.show()
