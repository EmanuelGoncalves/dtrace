#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def filter_drug_response(df):
    df = df[df.count(1) > df.shape[1] * .85]
    df = df.fillna(df.mean())
    return df


def perform_pca(df, n_components=10):
    d_pca = PCA(n_components=n_components).fit(df)
    d_pca_pcs = pd.DataFrame(d_pca.transform(df), index=df.index, columns=map(lambda x: 'PC{}'.format(x), range(1, n_components + 1)))
    return d_pca, d_pca_pcs


def plot_var_explained(pca, title, outfile):
    plot_df = pd.Series(
        pca.explained_variance_ratio_,
        index=map(lambda x: 'PC{}'.format(x), range(1, len(pca.explained_variance_ratio_) + 1))
    ).sort_values(ascending=True).multiply(100)

    ypos = list(map(lambda x: x[0], enumerate(plot_df.index)))

    plt.barh(ypos, plot_df, .8, color=bipal_dbgd[0], align='center')

    plt.yticks(ypos)
    plt.yticks(ypos, plot_df.index)

    plt.xlabel('Explained variance (%)')
    plt.ylabel('Principal component')
    plt.title(title)

    plt.gcf().set_size_inches(2, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')

    return plot_df


def plot_pca_jointplot(pc, feature, varexp, outfile):
    plot_df = pd.concat([pc, feature], axis=1).dropna()

    g = sns.jointplot(pc.name, feature.name, data=plot_df, space=0, color=bipal_dbgd[0], joint_kws={'edgecolor': 'white', 'alpha': .8})

    g.set_axis_labels('Drug response PC 1 ({0:.1f}%)'.format(varexp), 'Growth rate')

    plt.gcf().set_size_inches(3, 3)
    plt.savefig(outfile, bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    # Drug response
    d_response = pd.read_csv('data/gdsc/drug_single/drug_ic50_merged_matrix.csv', index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # CRISPR
    crispr = pd.read_csv('data/gdsc/crispr/crisprcleanr_gene.csv', index_col=0).dropna()

    # - Filter Drug response
    d_response = filter_drug_response(d_response[list(set(d_response).intersection(crispr).intersection(crispr).intersection(grate.index))])

    # - Drug response PCA
    d_pca, d_pca_pcs = perform_pca(d_response.T)

    d_pca_varexp = plot_var_explained(d_pca, 'Drug response PCA', 'reports/drug/drug_pca_expvar.png')

    plot_pca_jointplot(d_pca_pcs['PC1'], grate['growth_rate_median'], d_pca_varexp['PC1'], 'reports/drug/drug_pca_growth.png')

    # - Drug response correlation with growth
    dg_cor = d_response.T.corrwith(grate.loc[d_response.columns, 'growth_rate_median']).sort_values().rename('cor').reset_index()

    # Histogram
    sns.distplot(dg_cor.query("VERSION == 'RS'")['cor'], label='RS', color=bipal_dbgd[0], bins=10, kde=False, kde_kws={'lw': 1.})
    sns.distplot(dg_cor.query("VERSION == 'v17'")['cor'], label='v17', color=bipal_dbgd[1], bins=10, kde=False, kde_kws={'lw': 1.})

    plt.title('Drug-response correlation with growth rate')
    plt.xlabel("Pearson's r")
    plt.ylabel('Count')
    plt.legend(loc='upper left')

    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/drug/drug_growth_corr_hist.png', bbox_inches='tight', dpi=600)
    plt.close('all')
