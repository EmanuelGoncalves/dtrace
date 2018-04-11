#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.plot.corrplot import plot_corrplot
from statsmodels.stats.multitest import multipletests
from cdrug.assemble.assemble_ppi import build_biogrid_ppi


def dist_drugtarget_genes(drug_targets, genes, ppi):
    genes = genes.intersection(set(ppi.vs['name']))
    assert len(genes) != 0, 'No genes overlapping with PPI provided'

    dmatrix = {}

    for drug in drug_targets:
        drug_genes = drug_targets[drug].intersection(genes)

        if len(drug_genes) != 0:
            dmatrix[drug] = dict(zip(*(genes, np.min(ppi.shortest_paths(source=drug_genes, target=genes), axis=0))))

    return dmatrix


def drug_gene_corrplot(idx):
    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x, y = '{} CRISPR'.format(gene), '{} {} Drug'.format(d_name, d_screen)

    plot_df = pd.concat([
        crispr.loc[gene].rename(x), drespo.loc[(d_id, d_name, d_screen)].rename(y)
    ], axis=1).dropna()

    plot_corrplot(x, y, plot_df, add_hline=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/crispr_drug_corrplot.pdf', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # PPI annotation
    ppi = build_biogrid_ppi(int_type={'physical'})

    # Drug target
    d_targets = cdrug.get_drugtargets()

    # Drug response
    drespo = cdrug.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = cdrug.get_crispr(is_binary=False)

    # - Calculate distance between drugs and CRISPR genes in PPI
    dist_d_g = dist_drugtarget_genes(d_targets, set(lm_df_crispr['GeneSymbol']), ppi)

    # - Calculate FDR
    lm_df_crispr = lm_df_crispr.assign(lr_fdr=multipletests(lm_df_crispr['lr_pval'])[1])

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    # d, g = 1047, 'TP53'
    lm_df_crispr = lm_df_crispr.assign(
        target=[
            dist_d_g[d][g] if d in dist_d_g and g in dist_d_g[d] else np.nan for d, g in lm_df_crispr[['DRUG_ID_lib', 'GeneSymbol']].values
        ]
    )
    print(lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < 0.1)].sort_values('lr_fdr'))

    # -
    # Plot Drug ~ CRISPR corrplot
    drug_gene_corrplot(344)
