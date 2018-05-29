#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import drispr
import textwrap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import drispr.associations as lr_files
from sklearn.decomposition import PCA
from drispr.plot.corrplot import plot_corrplot
from drispr.associations import multipletests_per_drug, ppi_annotation
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score


if __name__ == '__main__':
    # - Imports
    # Linear regressions
    lm_df_crispr = pd.read_csv(lr_files.LR_DRUG_CRISPR)

    # - Compute FDR per drug
    lm_df_crispr = multipletests_per_drug(lm_df_crispr)

    # - Annotate regressions with Drug -> Target -> Protein (in PPI)
    lm_df_crispr = ppi_annotation(lm_df_crispr, exp_type={'Affinity Capture-MS', 'Affinity Capture-Western'}, int_type={'physical'}, target_thres=3)

    # - Create unique ID for drug
    lm_df_crispr = lm_df_crispr.assign(DrugId=['|'.join(map(str, i)) for i in lm_df_crispr[drispr.DRUG_INFO_COLUMNS].values])

    # - Build betas matrix
    lm_betas = pd.pivot_table(lm_df_crispr, index='GeneSymbol', columns='DrugId', values='beta')

    # - Drug with significant associations
    drugs = list(set(lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < .1)]['DrugId']))
    genes = list(set(lm_df_crispr[(lm_df_crispr['beta'].abs() > .25) & (lm_df_crispr['lr_fdr'] < .1)]['GeneSymbol']))

    # -
    df = lm_betas.loc[genes, drugs]
    df = df.subtract(df.mean(), axis=1)
    df = df.T

    # PCA
    b_pca = PCA(n_components=10).fit(df)
    b_pcs = pd.DataFrame(b_pca.transform(df), index=df.index, columns=map(lambda v: 'PC{}'.format(v + 1), range(10)))
    b_expvar = pd.Series(b_pca.explained_variance_ratio_, index=map(lambda v: 'PC{}'.format(v + 1), range(10)))
    print(b_expvar)

    b_pcs = b_pcs.assign(version=list(map(lambda v: v.split('|')[2], b_pcs.index)))
    b_pcs = b_pcs.assign(target=b_pcs.index.isin(set(lm_df_crispr.query('target == 0')['DrugId'])).astype(int))

    g = sns.PairGrid(
        b_pcs, vars=['PC1', 'PC2', 'PC3'], despine=False, size=1, hue='version', palette=drispr.PAL_DRUG_VERSION
    )

    g = g.map_diag(plt.hist)

    g = g.map_offdiag(plt.scatter, s=3, edgecolor='white', lw=.1)
    g = g.add_legend()

    for i, ax in enumerate(g.axes):
        vexp = b_expvar['PC{}'.format(i + 1)]
        ax[0].set_ylabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

    for i, ax in enumerate(g.axes[2]):
        vexp = b_expvar['PC{}'.format(i + 1)]
        ax.set_xlabel('PC{} ({:.1f}%)'.format(i + 1, vexp * 100))

    plt.savefig('reports/lr_pca_pairplot.pdf', bbox_inches='tight')
    plt.close('all')

    # - TSNE
    perplexity, learning_rate, n_iter = 30, 300, 500

    tsne = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter).fit_transform(df)
    tsne = pd.DataFrame(tsne, index=df.index, columns=['P1', 'P2'])
    tsne = tsne.assign(version=list(map(lambda v: v.split('|')[2], b_pcs.index)))
    tsne = tsne.assign(target=tsne.index.isin(set(lm_df_crispr.query('target == 0')['DrugId'])).astype(int))

    #
    for t in set(tsne['version']):
        plot_df = tsne.query("version == '{}'".format(t))
        plt.scatter(plot_df['P1'], plot_df['P2'], c=drispr.PAL_DRUG_VERSION[t], label=t, edgecolors='white', lw=.3, alpha=.8, s=10)

    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

    plt.legend()
    plt.title('t-SNE projection of drug associations')
    plt.text(plt.xlim()[0], plt.ylim()[0], 'Perplexity={}; Learning Rate={}'.format(perplexity, learning_rate), fontsize=4, ha='left', va='bottom')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/lr_tsne_pairplot.pdf', bbox_inches='tight')
    plt.close('all')

