#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from analysis.plot.corrplot import plot_corrplot
from drispr.associations import multipletests_per_drug, ppi_annotation
from drispr.assemble.assemble_ppi import build_string_ppi
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score


ORDER = ['Target', '1', '2', '>=3']
ORDER_COLOR = [drispr.PAL_SET2[1]] + sns.light_palette(drispr.PAL_SET2[8], len(ORDER) - 1, reverse=True).as_hex()
ORDER_PAL = dict(zip(*(ORDER, ORDER_COLOR)))


def target_enrichment(df, betas=None, pvalue=None):
    if betas is None:
        betas = np.arange(0, .6, .1)

    if pvalue is None:
        pvalue = [1e-4, 1e-3, 1e-2, 1e-1, .15, .2]

    order = ['Target', '1', '2', '3', '>=4']
    order_color = [drispr.PAL_SET2[1]] + sns.light_palette(drispr.PAL_SET2[8], len(order) - 1, reverse=True).as_hex()
    order_pal = dict(zip(*(order, order_color)))

    aucs = []
    for b in np.arange(0, .6, .1):
        for p in [1e-4, 1e-3, 1e-2, 1e-1, .15, .2]:
            plot_df = df[(df['beta'].abs() > b) & (df['fdr'] < p)]

            for t in order:
                y_true, y_score = (plot_df['target_thres'] == t).astype(int), 1 - plot_df['fdr']

                aucs.append({'beta': b, 'pval': p, 'thres': t, 'score': roc_auc_score(y_true, y_score), 'type': 'auc'})
                aucs.append({'beta': b, 'pval': p, 'thres': t, 'score': average_precision_score(y_true, y_score), 'type': 'precision'})
    aucs = pd.DataFrame(aucs)

    sns.boxplot('type', 'score', 'thres', aucs, palette=order_pal)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig('reports/ppi_signif_roc_boxplots.pdf', bbox_inches='tight')
    plt.close('all')


def plot_drug_associations_barplot(plot_df, order, ppi_text_offset=0.075, drug_name_offset=1., ylim_offset=1.1, fdr_line=0.05):
    # Group Drug ~ Gene associations
    df = plot_df.groupby(['DRUG_NAME', 'GeneSymbol']).first().reset_index().sort_values('fdr')

    # Pick top 10 associations for each drug
    df = df.groupby('DRUG_NAME').head(10).set_index('DRUG_NAME')

    df__, xpos = [], 0
    for drug_name in order:
        df_ = df.loc[[drug_name]]
        df_ = df_.assign(y=-np.log10(df_['fdr']))

        df_ = df_.assign(xpos=np.arange(xpos, xpos + df_.shape[0]))
        xpos += (df_.shape[0] + 2)

        df__.append(df_)

    df = pd.concat(df__).reset_index()

    # Significant line
    if fdr_line is not None:
        plt.axhline(-np.log10(fdr_line), ls='--', lw=.5, c=drispr.PAL_BIN[0], alpha=.3, zorder=0)

    # Barplot
    plt.bar(df.query('target != 0')['xpos'], df.query('target != 0')['y'], .8, color=drispr.PAL_BIN[0], align='center', zorder=5)
    plt.bar(df.query('target == 0')['xpos'], df.query('target == 0')['y'], .8, color=drispr.PAL_BIN[1], align='center', zorder=5)

    # Distance to target text
    for x, y, t in df[['xpos', 'y', 'target']].values:
        l = '-' if np.isnan(t) or np.isposinf(t) else ('T' if t == 0 else str(int(t)))
        plt.text(x, y - (df['y'].max() * ppi_text_offset), l, color='white', ha='center', fontsize=6, zorder=10)

    plt.ylim(ymax=df['y'].max() * ylim_offset)

    # Name drugs
    for k, v in df.groupby('DRUG_NAME')['xpos'].mean().sort_values().to_dict().items():
        plt.text(v, df['y'].max() * drug_name_offset, textwrap.fill(k, 15), ha='center', fontsize=6, zorder=10)

    plt.grid(True, color=drispr.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0, axis='y')

    plt.xticks(df['xpos'], df['GeneSymbol'], rotation=90, fontsize=5)
    plt.ylabel('Log-ratio FDR (-log10)')
    plt.title('Top significant Drug ~ CRISPR associations')

    plt.gcf().set_size_inches(12., 2.)
    plt.savefig('reports/drug_associations_barplot.pdf', bbox_inches='tight')
    plt.close('all')


def plot_count_associations(lm_res_df, fdr_thres, beta_thres, min_nevents=5):
    df = lm_res_df[(lm_res_df['beta'].abs() > beta_thres) & (lm_res_df['fdr'] < fdr_thres)].copy()

    df = df.groupby('DRUG_NAME')['fdr'].count().sort_values(ascending=False).reset_index()

    df.columns = ['drug', 'counts']

    if min_nevents is not None:
        df = df[df['counts'] >= min_nevents]

    ax = sns.barplot('counts', 'drug', data=df, color=drispr.PAL_BIN[0], linewidth=.8, orient='h')

    ax.xaxis.grid(True, color=drispr.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

    plt.ylabel('')
    plt.xlabel('#(associations)')
    plt.title('Significant associations (FDR<{}%, |b|>{})'.format(fdr_thres * 100, beta_thres))

    plt.gcf().set_size_inches(2, 8)
    plt.savefig('reports/drug_associations_count.pdf', bbox_inches='tight')
    plt.close('all')


def plot_drug_corr(idx):
    # Specific drug association
    d_id, d_name, d_screen, gene = lm_df_crispr.loc[idx, ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION', 'GeneSymbol']].values

    x, y = '{}'.format(gene), '{} {}'.format(d_name, d_screen)

    plot_df = pd.concat([
        crispr_scaled.loc[gene].rename(x), drespo.loc[(d_id, d_name, d_screen)].rename(y), ss['Cancer Type']
    ], axis=1).dropna().sort_values(x)

    plot_corrplot(x, y, plot_df, add_hline=True, lowess=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/crispr_drug_corrplot.pdf', bbox_inches='tight')
    plt.close('all')

    # Top drug associations
    plot_df = lm_df_crispr[lm_df_crispr['DRUG_NAME'] == d_name].head()

    d_id, d_screen = lm_df_crispr.loc[plot_df.index[0], ['DRUG_ID_lib', 'VERSION']].values
    genes = plot_df['GeneSymbol'].values

    plot_df = pd.concat([
        drespo.loc[(d_id, d_name, d_screen), samples].rename(d_name), crispr.loc[genes, samples].T
    ], axis=1).dropna()

    plot_df = pd.melt(plot_df.reset_index(), id_vars=['CELL_LINE_NAME', d_name])

    g = sns.FacetGrid(plot_df, col='variable', size=2, legend_out=True, despine=False, sharey=False, sharex=True)

    g = g.map(sns.regplot, d_name, 'value', color=drispr.PAL_BIN[0], line_kws=dict(lw=1., color=drispr.PAL_SET2[1]), scatter_kws=dict(edgecolor='w', lw=.3, s=12))

    g.set_titles('{col_name}')

    plt.gcf().set_size_inches(8, 2)
    plt.savefig('reports/drug_top_corrplots.pdf', bbox_inches='tight')
    plt.close('all')


def plot_arocs(df, thres_fdr, thres_beta):
    plot_df = df[(df['beta'].abs() > thres_beta) & (df['fdr'] < thres_fdr)].dropna()

    ax = plt.gca()
    for t, c in ORDER_PAL.items():
        fpr, tpr, _ = roc_curve((plot_df['target_thres'] == t).astype(int), 1 - plot_df['fdr'])
        ax.plot(fpr, tpr, label='AROC({}) = {:.2f}'.format(t, auc(fpr, tpr)), lw=1., c=c)

    ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('Enrichment for PPI (FDR<{}%, |b|>{})'.format(thres_fdr * 100, thres_beta))

    ax.legend(loc=4, prop={'size': 8})

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/ppi_signif_roc.pdf', bbox_inches='tight')
    plt.close('all')


def aurc(df, outfile=None, thres_label='target_thres', rank_label='pval', min_events=5, legend_size=6, title='', legend_title=''):
    aucs = {}

    ax = plt.gca() if outfile is not None else None

    for t, c in ORDER_PAL.items():
        index_set = set(df[df[thres_label] == t].index)

        if len(index_set) >= min_events:
            # Build data-frame
            x = df[rank_label].sort_values().dropna()

            # Observed cumsum
            y = x.index.isin(index_set)
            y = np.cumsum(y) / sum(y)

            # Rank fold-changes
            x = st.rankdata(x) / x.shape[0]

            # Calculate AUC
            f_auc = auc(x, y)
            aucs[t] = f_auc

            # Plot
            if outfile is not None:
                ax.plot(x, y, label='{}: AURC={:.2f}'.format(t, f_auc), lw=1., c=c)

    if outfile is not None:
        # Random
        ax.plot((0, 1), (0, 1), 'k--', lw=.3, alpha=.5)

        # Limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Labels
        ax.set_xlabel('Ranked')
        ax.set_ylabel('Recall')

        ax.legend(loc=4)

        ax.set_title(title)
        legend = ax.legend(loc=4, title=legend_title, prop={'size': legend_size})
        legend.get_title().set_fontsize('{}'.format(legend_size))

        # plt.show()
        plt.gcf().set_size_inches(3, 3)
        plt.savefig(outfile, bbox_inches='tight', dpi=600)
        plt.close('all')

    return ax, aucs


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ss = drispr.get_samplesheet()

    # Linear regressions
    lm_df_crispr = pd.read_csv('data/drug_regressions_crispr_limix.csv')
    # lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # Drug response
    drespo = drispr.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = drispr.get_crispr(dtype='logFC')
    crispr_scaled = drispr.scale_crispr(crispr)
    crispr_binary = drispr.get_crispr('depletions')

    samples = list(set(drespo).intersection(crispr))

    # - Compute FDR per drug
    lm_df_crispr = multipletests_per_drug(lm_df_crispr, field='pval')

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = ppi_annotation(
        lm_df_crispr, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3,
    )
    print(lm_df_crispr[(lm_df_crispr['beta'].abs() > .2) & (lm_df_crispr['fdr'] < 0.05)].sort_values('fdr'))

    # - Top associations
    plot_df = lm_df_crispr[(lm_df_crispr['beta'].abs() > .2) & (lm_df_crispr['fdr'] < .1)]
    order = list(plot_df.groupby('DRUG_NAME')['fdr'].min().sort_values().index)[:10]
    plot_drug_associations_barplot(plot_df, order, fdr_line=.1)

    # - Count signif assoc per drug
    plot_count_associations(lm_df_crispr, 0.1, .2, min_nevents=3)

    # - Plot Drug ~ CRISPR corrplot
    plot_drug_corr(2113601)

    # - AROC enrichment
    plot_arocs(lm_df_crispr, .1, .2)

    lm_d_signif = lm_df_crispr.dropna(subset=['target']).groupby(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION'])['fdr'].min()
    lm_d_signif = set(lm_d_signif[lm_d_signif < .1].index)

    df = lm_df_crispr[[(did, dname, dscreen) in lm_d_signif for did, dname, dscreen in lm_df_crispr[['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']].values]]

    aurc(df, outfile='reports/ppi_signif_aurc.pdf', rank_label='pval')
