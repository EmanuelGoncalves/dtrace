#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import drispr.associations as lr_files
from drispr.plot.corrplot import plot_corrplot
from drispr.associations import multipletests_per_drug, ppi_annotation
from drispr.assemble.assemble_ppi import build_biogrid_ppi, build_string_ppi
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


THRES_FDR, THRES_BETA = .1, 0.25


def drug_count_barplot(df):
    plot_df = pd.Series({
        'All': len({(i, n, v) for i, n, v in df[drispr.DRUG_INFO_COLUMNS].values}),
        'All unique': len({n for i, n, v in df[drispr.DRUG_INFO_COLUMNS].values}),
        'W/Target': len({(i, n, v) for i, n, v in df.dropna(subset=['target'])[drispr.DRUG_INFO_COLUMNS].values}),
        'W/Target unique': len({n for i, n, v in df.dropna(subset=['target'])[drispr.DRUG_INFO_COLUMNS].values})
    }).rename('count').to_frame().reset_index()

    ax = sns.barplot('count', 'index', data=plot_df, color=drispr.PAL_BIN[0], linewidth=.8, orient='h')
    ax.set_xlim(0, 1000)
    ax.xaxis.grid(True, color=drispr.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=.5, zorder=0)

    plt.ylabel('Drug annotation')
    plt.xlabel('#(drugs)')

    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/drug_target_count_barplot.pdf', bbox_inches='tight')
    plt.close('all')


def drug_beta_histogram(plot_df):
    sns.distplot(plot_df['beta'], color=drispr.PAL_BIN[0], kde_kws=dict(cut=0, lw=1, zorder=1, alpha=.8), hist_kws=dict(alpha=.4, zorder=1), label='All', bins=30)
    sns.distplot(plot_df.query('target == 0')['beta'], color=drispr.PAL_BIN[1], kde_kws=dict(cut=0, lw=1, zorder=3), hist_kws=dict(alpha=.6, zorder=3), label='Target', bins=30)

    beta_median = plot_df.query('fdr < 0.05 & target == 0')['beta'].median()
    plt.axvline(beta_median, c=drispr.PAL_BIN[1], lw=.3, ls='--')
    plt.text(beta_median, 0, 'FDR < 5%', fontdict=dict(color='white', fontsize=3), rotation=90, ha='right', va='bottom')

    # plt.grid(True, color=drispr.PAL_SET2[7], linestyle='-', linewidth=.1, alpha=1., zorder=0, axis='x')
    plt.axvline(0, c=drispr.PAL_SET2[7], lw=.3, ls='-', zorder=0)

    plt.ylabel('Density')

    plt.legend()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/drug_target_pval_histogram.pdf', bbox_inches='tight')
    plt.close('all')


def recapitulated_drug_targets_barplot(df, thres_fdr, thres_beta):
    plot_df = df.dropna(subset=['target'])
    plot_df = pd.Series({
        'Non-significant': len({n for i, n, v in plot_df[drispr.DRUG_INFO_COLUMNS].values}),
        'Significant': len({n for i, n, v in plot_df.query('fdr < {} & beta > {} & target < 1'.format(thres_fdr, thres_beta))[drispr.DRUG_INFO_COLUMNS].values}),
    }).rename('count')

    for i, l in enumerate(plot_df.index):
        plt.bar(0, plot_df.loc[l], color=drispr.PAL_BIN[i], label=l)

    signif_prc = plot_df['Significant'] / plot_df['Non-significant']
    plt.text(0, plot_df.loc['Significant'] - 5, '{:.1f}%'.format(signif_prc*100), fontdict=dict(color='white', fontsize=10), ha='center', va='top')

    plt.axes().get_xaxis().set_visible(False)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Target CRISPR-Cas9:\n(FDR<{}% and $\\beta$ > {})'.format(int(thres_fdr * 100), thres_beta))
    plt.ylabel('Number of drugs with targets annotated')
    plt.title('Drug targets recapitulated\nwith CRISPR-Cas9')

    plt.gcf().set_size_inches(.75, 3)
    plt.savefig('reports/drug_target_signif_barplot.pdf', bbox_inches='tight')
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Samplesheet
    ss = drispr.get_samplesheet()

    # Linear regressions
    # lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)
    lm_df_crispr = pd.read_csv('data/drug_regressions_crispr_limix.csv')

    # Drug response
    drespo = drispr.get_drugresponse()

    # CIRSPR CN corrected logFC
    crispr = drispr.get_crispr(dtype='logFC')
    crispr_scaled = drispr.scale_crispr(crispr)

    samples = list(set(drespo).intersection(crispr))

    # - Compute FDR per drug
    lm_df_crispr = multipletests_per_drug(lm_df_crispr, field='pval')

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = ppi_annotation(
        lm_df_crispr, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3,
    )

    # - Count number of drugs
    drug_count_barplot(lm_df_crispr)

    # - Drug CRISPR associations betas distributions
    drug_beta_histogram(lm_df_crispr.dropna())

    # - Drugs that have a target significantly associated
    lm_d_signif = lm_df_crispr.dropna(subset=['target']).groupby(['DRUG_ID_lib', 'DRUG_NAME', 'VERSION'])['fdr'].min()
    lm_d_signif = set(lm_d_signif[lm_d_signif < .05].index)

    df = lm_df_crispr[[(did, dname, dscreen) in lm_d_signif for did, dname, dscreen in lm_df_crispr[['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']].values]]

    recapitulated_drug_targets_barplot(df, .05, 0)

    # -
    betas = pd.pivot_table(lm_df_crispr, index=['DRUG_ID_lib', 'DRUG_NAME', 'VERSION'], columns='GeneSymbol', values='beta')

    drugsheet = drispr.get_drugsheet()

    def classify_drug_pairs(d1, d2):
        d1_id, d1_name, d1_screen = d1.split(' ; ')
        d2_id, d2_name, d2_screen = d2.split(' ; ')

        d1_id, d2_id = int(d1_id), int(d2_id)

        dsame = drispr.is_same_drug(d1_id, d2_id, drugsheet=drugsheet) if d1_id in drugsheet.index and d2_id in drugsheet.index else d1_name == d2_name

        if dsame & (d1_screen == d2_screen):
            return 'Drug & Screen'
        elif dsame:
            return 'Drug'
        else:
            return '-'

    betas_corr = betas.T.corr()
    betas_corr = betas_corr.where(np.triu(np.ones(betas_corr.shape), 1).astype(np.bool))
    betas_corr.index = [' ; '.join(map(str, i)) for i in betas_corr.index]
    betas_corr.columns = [' ; '.join(map(str, i)) for i in betas_corr.columns]
    betas_corr = betas_corr.unstack().dropna()
    betas_corr = betas_corr.reset_index()
    betas_corr.columns = ['drug_1', 'drug_2', 'r']
    betas_corr = betas_corr.assign(
        dtype=[classify_drug_pairs(d1, d2) for d1, d2 in betas_corr[['drug_1', 'drug_2']].values]
    )

    order = ['-', 'Drug', 'Drug & Screen']
    pal = dict(zip(*(order, sns.light_palette(drispr.PAL_SET2[8], len(order) + 1).as_hex()[1:])))

    sns.boxplot('r', 'dtype', data=betas_corr, orient='h', order=order, palette=pal, fliersize=1, notch=True, linewidth=.3)
    plt.xlabel('Pearson\'s R')
    plt.ylabel('Same')
    plt.gcf().set_size_inches(3, 1)
    plt.savefig('reports/drugs_betas_cor.pdf', bbox_inches='tight')
    plt.close('all')
