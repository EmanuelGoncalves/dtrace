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
from cdrug.plot.corrplot import plot_corrplot_discrete


THRES_FDR, THRES_BETA = .1, 0.5


if __name__ == '__main__':
    # - Import
    mobems = cdrug.get_mobem()
    drespo = cdrug.get_drugresponse()

    crispr = cdrug.get_crispr(dtype='logFC')

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # - Import linear regressions
    lr_mobem = pd.read_csv(lr_files.LR_BINARY_DRUG_MOBEMS)
    lr_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)
    lr_crispr_mobem = pd.read_csv(lr_files.LR_CRISPR_MOBEMS)

    # Compute FDR per drug
    lr_mobem = multipletests_per_drug(lr_mobem)
    lr_crispr = multipletests_per_drug(lr_crispr)
    lr_crispr_mobem = lr_crispr_mobem.assign(lr_fdr=multipletests(lr_crispr_mobem['lr_pval'], method='bonferroni')[1])

    #
    lr_mobem_ = lr_mobem[(lr_mobem['beta'].abs() > THRES_BETA) & (lr_mobem['lr_fdr'] < THRES_FDR)]
    lr_crispr_ = lr_crispr[(lr_crispr['beta'].abs() > THRES_BETA) & (lr_crispr['lr_fdr'] < THRES_FDR)]
    lr_crispr_mobem_ = lr_crispr_mobem[(lr_crispr_mobem['beta'].abs() > THRES_BETA) & (lr_crispr_mobem['lr_fdr'] < THRES_FDR)]

    # -
    lr_crispr_signif = lr_crispr_\
        .assign(crispr_genetic=lr_crispr_['GeneSymbol'].isin(lr_crispr_mobem_['GeneSymbol']).astype(int))\
        .assign(drug_genetic=lr_crispr_['DRUG_ID_lib'].isin(lr_mobem_['DRUG_ID_lib']).astype(int))\
        .sort_values('lr_fdr')

    # -
    df = lr_crispr_signif[lr_crispr_signif[['crispr_genetic', 'drug_genetic']].sum(1) == 2]

    # Corr plot discrete
    idx = 898462

    d_id, d_name, d_screen, gene = lr_crispr_signif.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    drug_genetic = lr_mobem_[lr_mobem_['DRUG_NAME'] == d_name]
    crispr_genetic = lr_crispr_mobem_[lr_crispr_mobem_['GeneSymbol'] == gene]

    genomic = [i for i in drug_genetic['level_3'] for j in crispr_genetic['level_1'] if i == j][0]
    print('Drug: {}; Gene: {}; Genetic: {}'.format(d_name, gene, genomic))

    plot_df = pd.concat([
        crispr.loc[gene].rename(gene),
        drespo.loc[(d_id, d_name, d_screen)].rename(d_name),
        mobems.loc[genomic].rename(genomic)
    ], axis=1).dropna()

    plot_corrplot_discrete(gene, d_name, genomic, plot_df)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/crispr_drug_robust_corrplot.pdf', bbox_inches='tight')
    plt.close('all')
