#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.plot.corrplot import plot_corrplot
from statsmodels.stats.multitest import multipletests
from cdrug.assemble.assemble_ppi import build_biogrid_ppi
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score


def drug_gene_corrplot(idx):
    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x, y = '{}'.format(gene), '{} {}'.format(d_name, d_screen)

    plot_df = pd.concat([
        crispr.loc[gene].rename(x), drespo.loc[(d_id, d_name, d_screen)].rename(y)
    ], axis=1).dropna()

    plot_corrplot(x, y, plot_df, add_hline=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/crispr_drug_corrplot.pdf', bbox_inches='tight')
    plt.close('all')


def dist_drugtarget_genes(drug_targets, genes, ppi):
    genes = genes.intersection(set(ppi.vs['name']))
    assert len(genes) != 0, 'No genes overlapping with PPI provided'

    dmatrix = {}

    for drug in drug_targets:
        drug_genes = drug_targets[drug].intersection(genes)

        if len(drug_genes) != 0:
            dmatrix[drug] = dict(zip(*(genes, np.min(ppi.shortest_paths(source=drug_genes, target=genes), axis=0))))

    return dmatrix


def ppi_annotation(df, target_thres=4):
    # PPI annotation
    ppi = build_biogrid_ppi(int_type={'physical'})

    # Drug target
    d_targets = cdrug.get_drugtargets()

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

    return df


def target_enrichment(df, betas=None, pvalue=None):
    if betas is None:
        betas = np.arange(0, .6, .1)

    if pvalue is None:
        pvalue = [1e-4, 1e-3, 1e-2, 1e-1, .15, .2]

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [cdrug.PAL_SET2[1]] + sns.light_palette(cdrug.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()
    order_pal = dict(zip(*(order, order_color)))

    aucs = []
    for b in np.arange(0, .6, .1):
        for p in [1e-4, 1e-3, 1e-2, 1e-1, .15, .2]:
            plot_df = df[(df['beta'].abs() > b) & (df['lr_fdr'] < p)]

            for t in order:
                y_true, y_score = (plot_df['target_thres'] == t).astype(int), 1 - plot_df['lr_fdr']

                aucs.append({'beta': b, 'pval': p, 'thres': t, 'score': roc_auc_score(y_true, y_score), 'type': 'auc'})
                aucs.append({'beta': b, 'pval': p, 'thres': t, 'score': average_precision_score(y_true, y_score), 'type': 'precision'})
    aucs = pd.DataFrame(aucs)

    sns.boxplot('type', 'score', 'thres', aucs, palette=order_pal)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/ppi_signif_roc_boxplots.pdf', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # Drug response
    drespo = cdrug.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = cdrug.get_crispr(dtype='logFC')

    # Gene-expression
    gexp = pd.read_csv('data/gdsc/gene_expression/merged_voom_preprocessed.csv', index_col=0)

    samples = list(set(drespo).intersection(crispr).intersection(gexp))

    # - Calculate FDR
    lm_df_crispr = lm_df_crispr.assign(lr_fdr=multipletests(lm_df_crispr['lr_pval'])[1])

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = ppi_annotation(lm_df_crispr)
    print(lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < 0.1)].sort_values('lr_fdr'))

    # - Plot Drug ~ CRISPR corrplot
    drug_gene_corrplot(344)

    #
    order = ['Target', '1', '2', '3', '>=4']
    order_color = [cdrug.PAL_SET2[1]] + sns.light_palette(cdrug.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()
    order_pal = dict(zip(*(order, order_color)))

    df = lm_df_crispr[lm_df_crispr['lr_fdr'] < .25]

    for o in order:
        sns.distplot(df[df['target_thres'] == o]['beta'], hist=False, color=order_pal[o], kde_kws=dict(cut=0, shade=True))

    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/ppi_signif_roc_hists.pdf', bbox_inches='tight')
    plt.close('all')

    df = []
    for b in np.arange(0, .6, .1):
        for p in [1e-4, 1e-3, 1e-2, 1e-1, .15, .2]:
            plot_df = lm_df_crispr[(lm_df_crispr['beta'].abs() > b) & (lm_df_crispr['lr_fdr'] < p)]

            for t in order:
                fpr, tpr, _ = roc_curve((plot_df['target_thres'] == t).astype(int), 1 - plot_df['lr_fdr'])

                df.append({'beta': b, 'pval': p, 'thres': t, 'auc': auc(fpr, tpr)})

    df = pd.DataFrame(df)
    sns.boxplot('thres', 'auc', data=df, order=order)
    plt.show()

    plot_df = lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < 0.1)]
    plot_df = plot_df.assign(thres=['Target' if i == 0 else ('%d' % i if i < 4 else '>=4') for i in plot_df['target']])

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [cdrug.PAL_SET2[1]] + sns.light_palette(cdrug.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()

    ax = plt.gca()
    for t, c in zip(*(order, order_color)):
        fpr, tpr, _ = roc_curve((plot_df['thres'] == t).astype(int), 1 - plot_df['lr_fdr'])
        ax.plot(fpr, tpr, label='%s=%.2f (AUC)' % (t, auc(fpr, tpr)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Protein-protein interactions\nDrug ~ CRISPR (FDR<{}%, |b|>{})'.format(.1 * 100, .25))

    ax.legend(loc=4, prop={'size': 8})

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/ppi_signif_roc.pdf', bbox_inches='tight')
    plt.close('all')
