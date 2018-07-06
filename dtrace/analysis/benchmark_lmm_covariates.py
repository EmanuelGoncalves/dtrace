#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from dtrace.associations.lmm_drug import lmm_association
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.analysis.drug_associations import boxplot_kinobead, drug_aurc
from dtrace.associations import multipletests_per_drug, ppi_annotation, DRUG_INFO_COLUMNS, corr_drugtarget_gene


def corrplot_betas(lmm_w_cov, lmm_n_cov):
    plot_df = pd.concat([
        lmm_w_cov.set_index(DRUG_INFO_COLUMNS + ['GeneSymbol'])['beta'].rename('With covariates'),
        lmm_n_cov.set_index(DRUG_INFO_COLUMNS + ['GeneSymbol'])['beta'].rename('Without covariates'),
    ], axis=1)

    g = sns.jointplot(
        'Without covariates', 'With covariates', data=plot_df, kind='scatter', space=0, color=PAL_DTRACE[2],
        marginal_kws=dict(kde=False), joint_kws=dict(edgecolor='w', lw=.1, s=4), annot_kws=dict(template='R={val:.2g}', loc=2)
    )

    g.ax_joint.axhline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)
    g.ax_joint.axvline(0, ls='-', lw=0.1, c=PAL_DTRACE[1], zorder=0)


def count_associations(lmm_w_cov, lmm_n_cov, fdr_line=0.05):
    df_w_cov = lmm_w_cov[lmm_w_cov['fdr'] < fdr_line]
    df_w_cov = df_w_cov.groupby(DRUG_INFO_COLUMNS)['fdr'].count().rename('count').reset_index()

    df_n_cov = lmm_n_cov[lmm_n_cov['fdr'] < fdr_line]
    df_n_cov = df_n_cov.groupby(DRUG_INFO_COLUMNS)['fdr'].count().rename('count').reset_index()

    plot_df = pd.concat([
        df_w_cov.set_index(DRUG_INFO_COLUMNS)['count'].rename('With covariates'),
        df_n_cov.set_index(DRUG_INFO_COLUMNS)['count'].rename('Without covariates'),
    ], axis=1).fillna(0).astype(int).reset_index()

    plot_df = pd.pivot_table(
        plot_df, index='With covariates', columns='Without covariates', aggfunc='count', values='VERSION', fill_value=0
    )

    g = sns.heatmap(plot_df, cmap=sns.light_palette(PAL_DTRACE[2], as_cmap=True), square=True, cbar=False, annot=True, annot_kws=dict(fontsize=4), linewidths=.3)
    plt.title('Number of significant associations (per drug)')
    plt.setp(g.get_yticklabels(), rotation=0)


def beta_covariates_histogram(lmm_w_cov, lmm_n_cov):
    kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
    hist_kws = dict(alpha=.4, zorder=1)

    label_order = ['All', 'Target']

    f, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    for i, (n, df) in enumerate([('With covariates', lmm_w_cov), ('Without covariates', lmm_n_cov)]):
        sns.distplot(df['beta'], color=PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws, label=label_order[0], bins=30, ax=axs[i])
        sns.distplot(df.query("target == 'T'")['beta'], color=PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws, label=label_order[1], bins=30, ax=axs[i])

        sns.despine(right=True, top=True, ax=axs[i])

        axs[i].axvline(0, c=PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        axs[i].set_xlabel('Association beta')
        axs[i].set_ylabel('Density')

        axs[i].set_title(n)

        axs[i].legend(prop={'size': 6}, loc=2)


def drug_targets_difference(lmm_w_cov, lmm_n_cov, ntop=30):
    plot_df = pd.concat([
        lmm_w_cov.query("target == 'T'").set_index(DRUG_INFO_COLUMNS + ['GeneSymbol'])['beta'].rename('With covariates'),
        lmm_n_cov.query("target == 'T'").set_index(DRUG_INFO_COLUMNS + ['GeneSymbol'])['beta'].rename('Without covariates'),
    ], axis=1)

    plot_df = (plot_df['Without covariates'] - plot_df['With covariates']).rename('Difference')

    plot_df = plot_df.loc[plot_df.abs().sort_values(ascending=False).head(ntop).index]
    plot_df = plot_df.sort_values(ascending=False)
    plot_df = plot_df.rename('diff').reset_index().assign(y=range(plot_df.shape[0]))

    # Plot
    plt.scatter(plot_df['diff'], plot_df['y'], c=PAL_DTRACE[2])

    for d_id, d_n, d_v, g, d, y in plot_df[DRUG_INFO_COLUMNS + ['GeneSymbol', 'diff', 'y']].values:
        label = '{}, {} [{}, {}]'.format(d_n, g, d_id, d_v)

        plt.text(d - .025, y, label, va='center', fontsize=4, zorder=10, color='gray', ha='right')

    plt.axvline(0, lw=.1, c=PAL_DTRACE[1])

    plt.xlabel('Delta beta\n(Without covariates - With covariates)')
    plt.ylabel('')
    plt.title('Drug, Gene associations')
    plt.axes().get_yaxis().set_ticks([])

    sns.despine(left=True)

    plt.gcf().set_size_inches(2., ntop * .12)

    plt.savefig('reports/covariates_lmm_count_beta_difference.pdf', bbox_inches='tight')
    plt.close('all')


def selectivity_difference(lmm_w_cov, lmm_n_cov):
    dfs = [(lmm_w_cov, 'With covariates'), (lmm_n_cov, 'Without covariates')]

    f, axs = plt.subplots(1, len(dfs), sharex=True, sharey=True, gridspec_kw=dict(wspace=.5))

    for i in range(len(dfs)):
        boxplot_kinobead(dfs[i][0], ax=axs[i])
        axs[i].set_title(dfs[i][1])
        axs[i].set_xlabel('Drug has a significant\nassociation')

    plt.gcf().set_size_inches(1 * len(dfs), 2)


def aroc_difference(lmm_w_cov, lmm_n_cov):
    dfs = [(lmm_w_cov, 'With covariates'), (lmm_n_cov, 'Without covariates')]

    f, axs = plt.subplots(1, len(dfs), sharex=True, sharey=True, gridspec_kw=dict(wspace=.2))

    for i in range(len(dfs)):
        drug_aurc(dfs[i][0], ax=axs[i])
        axs[i].set_title(dfs[i][1])
        axs[i].set_xlabel('Drug has a significant\nassociation')

    plt.gcf().set_size_inches(2 * len(dfs), 2)


if __name__ == '__main__':
    # - Import
    # Genetic
    mobems = dtrace.get_mobem()

    # Drug-response
    drespo = dtrace.get_drugresponse()

    # CRISPR
    crispr = dtrace.get_crispr(dtype='both')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    # Growth rate
    growth = pd.read_csv(dtrace.GROWTHRATE_FILE, index_col=0)['growth_rate_median'].dropna()

    # Samplesheet
    ctype = dtrace.get_samplesheet().dropna(subset=['Cancer Type'])['Cancer Type']

    # - Overlap
    samples = list(set.intersection(set(mobems), set(drespo), set(crispr), set(growth.index), set(ctype.index)))
    print('#(Samples) = {}'.format(len(samples)))

    # - Filter
    drespo = dtrace.filter_drugresponse(drespo[samples])

    crispr = dtrace.filter_crispr(crispr[samples])
    crispr_logfc = crispr_logfc.loc[crispr.index, samples]

    print(
        '#(Genomic) = {}; #(Drugs) = {}; #(Genes) = {}'.format(len(set(mobems.index)), len(set(drespo.index)), len(set(crispr.index)))
    )

    # - Covariates
    covariates = pd.concat([growth[samples], pd.get_dummies(ctype[samples])], axis=1)

    # - Linear Mixed Model
    lmm_w_cov = pd.concat([lmm_association(drug=d, y=drespo, x=crispr_logfc, M=covariates) for d in drespo.index])
    lmm_w_cov = multipletests_per_drug(lmm_w_cov, field='pval')
    print(lmm_w_cov.sort_values('pval').head(60))

    lmm_n_cov = pd.concat([lmm_association(drug=d, y=drespo, x=crispr_logfc, M=None) for d in drespo.index])
    lmm_n_cov = multipletests_per_drug(lmm_n_cov, field='pval')
    print(lmm_n_cov.sort_values('pval').head(60))

    # - Target + PPI annotation
    lmm_w_cov = ppi_annotation(lmm_w_cov, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)
    lmm_n_cov = ppi_annotation(lmm_n_cov, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)

    lmm_w_cov = corr_drugtarget_gene(lmm_w_cov)
    lmm_n_cov = corr_drugtarget_gene(lmm_n_cov)

    # - Plot
    # Correlation betas
    corrplot_betas(lmm_w_cov, lmm_n_cov)
    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/covariates_lmm_betas_jointplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Count significant associations
    count_associations(lmm_w_cov, lmm_n_cov)
    plt.gcf().set_size_inches(2., 2.)
    plt.savefig('reports/covariates_lmm_count_signif.pdf', bbox_inches='tight')
    plt.close('all')

    # Betas histogram
    beta_covariates_histogram(lmm_w_cov, lmm_n_cov)
    plt.gcf().set_size_inches(4, 2)
    plt.savefig('reports/covariates_lmm_betas_histogram.pdf', bbox_inches='tight')
    plt.close('all')

    # Selectivity difference
    selectivity_difference(lmm_w_cov, lmm_n_cov)
    plt.savefig('reports/covariates_lmm_associations_kinobeads.pdf', bbox_inches='tight')
    plt.close('all')

    # AROCs difference
    aroc_difference(lmm_w_cov, lmm_n_cov)
    plt.savefig('reports/covariates_lmm_aurc.pdf', bbox_inches='tight')
    plt.close('all')
