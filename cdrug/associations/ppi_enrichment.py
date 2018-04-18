#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cdrug.associations as lr_files
from cdrug.plot.corrplot import plot_corrplot
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score


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


def plot_drug_associations_barplot(plot_df, order, ppi_text_offset=0.075, drug_name_offset=1., ylim_offset=1.1, fdr_line=0.05):
    # Group Drug ~ Gene associations
    df = plot_df.groupby(['DRUG_NAME', 'GeneSymbol']).first().reset_index().sort_values('lr_fdr')

    # Pick top 10 associations for each drug
    df = df.groupby('DRUG_NAME').head(10).set_index('DRUG_NAME')

    df__, xpos = [], 0
    for drug_name in order:
        df_ = df.loc[[drug_name]]
        df_ = df_.assign(y=-np.log10(df_['lr_fdr']))

        df_ = df_.assign(xpos=np.arange(xpos, xpos + df_.shape[0]))
        xpos += (df_.shape[0] + 2)

        df__.append(df_)

    df = pd.concat(df__).reset_index()

    # Significant line
    if fdr_line is not None:
        plt.axhline(-np.log10(0.05), ls='--', lw=.1, c=cdrug.PAL_BIN[0], alpha=.3, zorder=0)

    # Barplot
    plt.bar(df.query('target != 0')['xpos'], df.query('target != 0')['y'], .8, color=cdrug.PAL_BIN[0], align='center', zorder=5)
    plt.bar(df.query('target == 0')['xpos'], df.query('target == 0')['y'], .8, color=cdrug.PAL_BIN[1], align='center', zorder=5)

    # Distance to target text
    for x, y, t in df[['xpos', 'y', 'target']].values:
        l = '-' if np.isnan(t) or np.isposinf(t) else ('T' if t == 0 else str(int(t)))
        plt.text(x, y - (df['y'].max() * ppi_text_offset), l, color='white', ha='center', fontsize=6, zorder=10)

    plt.ylim(ymax=df['y'].max() * ylim_offset)

    # Name drugs
    for k, v in df.groupby('DRUG_NAME')['xpos'].mean().sort_values().to_dict().items():
        plt.text(v, df['y'].max() * drug_name_offset, textwrap.fill(k, 15), ha='center', fontsize=6, zorder=10)

    plt.xticks(df['xpos'], df['GeneSymbol'], rotation=90, fontsize=5)
    plt.ylabel('Log-ratio FDR (-log10)')
    plt.title('Top significant Drug ~ CRISPR associations')

    plt.gcf().set_size_inches(12., 2.)
    plt.savefig('reports/drug_associations_barplot.pdf', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # Drug response
    drespo = cdrug.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = cdrug.get_crispr(dtype='logFC')

    samples = list(set(drespo).intersection(crispr))

    # - Calculate FDR
    lm_df_crispr = lm_df_crispr.assign(lr_fdr=multipletests(lm_df_crispr['lr_pval'], method='fdr_bh')[1])

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = cdrug.ppi_annotation(lm_df_crispr, exp_type={'Affinity Capture-MS', 'Affinity Capture-Western'}, int_type={'physical'})
    print(lm_df_crispr[(lm_df_crispr['beta'].abs() > .5) & (lm_df_crispr['lr_fdr'] < 0.1)])

    # - Top associations
    plot_df = lm_df_crispr[(lm_df_crispr['beta'].abs() > .5) & (lm_df_crispr['lr_fdr'] < .1)]

    order = list(plot_df.groupby('DRUG_NAME')['lr_fdr'].min().sort_values().index)[:10]

    plot_drug_associations_barplot(plot_df, order)

    # - Plot Drug ~ CRISPR corrplot
    idx = 733

    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x, y = '{}'.format(gene), '{} {}'.format(d_name, d_screen)

    plot_df = pd.concat([
        crispr.loc[gene].rename(x), drespo.loc[(d_id, d_name, d_screen)].rename(y)
    ], axis=1).dropna()

    plot_corrplot(x, y, plot_df, add_hline=True, lowess=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/crispr_drug_corrplot.pdf', bbox_inches='tight')
    plt.close('all')

    # -
    order = ['Target', '1', '2', '3', '>=4']
    order_color = [cdrug.PAL_SET2[1]] + sns.light_palette(cdrug.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()
    order_pal = dict(zip(*(order, order_color)))

    for t in order:
        plt.hist(
            lm_df_crispr.query("target_thres == '{}'".format(t))['f_stat'], normed=1, histtype='step', cumulative=True, color=order_pal[t], bins=50, label=t
        )
    plt.legend()
    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/ppi_cumsum_hist.pdf', bbox_inches='tight')
    plt.close('all')

    #
    # plot_df = lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < 0.2)]
    plot_df = lm_df_crispr[lm_df_crispr['beta'].abs() > .5].dropna()

    ax = plt.gca()
    for t, c in zip(*(order, order_color)):
        fpr, tpr, _ = roc_curve((plot_df['target_thres'] == t).astype(int), plot_df['lr'])
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
