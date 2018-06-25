#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import ShuffleSplit
from dtrace.associations import DRUG_INFO_COLUMNS
from dtrace.analysis import PAL_DTRACE, MidpointNormalize


def growht_by_cancer_type(growth, ctype):
    # Build dataframe
    plot_df = pd.concat([growth, ctype], axis=1)

    # Plot
    order = list(plot_df.groupby('Cancer Type')['growth_rate_median'].mean().sort_values().index)

    sns.boxplot('growth_rate_median', 'Cancer Type', data=plot_df, orient='h', color=PAL_DTRACE[1], linewidth=.3, fliersize=1.5, order=order)
    sns.swarmplot('growth_rate_median', 'Cancer Type', data=plot_df, orient='h', color=PAL_DTRACE[1], linewidth=.3, size=2, order=order)

    sns.despine(top=True, right=True)

    plt.xlabel('Growth rate\n(median day 1 / day 4)')
    plt.ylabel('Cancer type')


def r2_scatter(drug_r2):
    cmap = sns.light_palette(PAL_DTRACE[2], as_cmap=True, reverse=True)

    ax = plt.gca()

    ax.scatter(drug_r2['r2_ctype'], drug_r2['r2_growth'], c=PAL_DTRACE[2], s=3, lw=0., zorder=2, alpha=.9)
    sns.kdeplot(drug_r2['r2_ctype'], drug_r2['r2_growth'], n_levels=30, cmap=cmap, linewidths=.5, zorder=3, alpha=.2, ax=ax)

    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x_lim, y_lim, 'k--', lw=.3)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Cancer type covariate (R2)')
    ax.set_ylabel('Growth covariate (R2)')
    ax.set_title('Drug response prediction')


def predict_drug_response(drespo, growth, ctype, n_splits=10, test_size=.2):
    drug_r2 = []
    for d in drespo.index:
        print(d)

        y = drespo.loc[d, samples].dropna()

        x_growth = growth.loc[y.index].to_frame()
        x_ctype = pd.get_dummies(ctype.loc[y.index])

        #
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        #
        for train_idx, test_idx in cv.split(x_ctype):
            # Growth
            lm_growth = ElasticNetCV().fit(x_growth.iloc[train_idx], y.iloc[train_idx])
            r2_growth = lm_growth.score(x_growth.iloc[test_idx], y.iloc[test_idx])

            # Cancer type
            lm_ctype = ElasticNetCV().fit(x_ctype.iloc[train_idx], y.iloc[train_idx])
            r2_ctype = lm_ctype.score(x_ctype.iloc[test_idx], y.iloc[test_idx])

            #
            drug_r2.append(dict(
                r2_growth=r2_growth, r2_ctype=r2_ctype,
                DRUG_ID_lib=d[0], DRUG_NAME=d[1], VERSION=d[2]
            ))
    drug_r2 = pd.DataFrame(drug_r2).groupby(DRUG_INFO_COLUMNS).median()

    return drug_r2


def top_predicted_drugs(drug_r2, ntop=20, xoffset=0.01):
    f, axs = plt.subplots(1, drug_r2.shape[1], sharex=True, sharey=False)

    cmap = sns.light_palette(PAL_DTRACE[2], as_cmap=True)

    for i, ax in enumerate(axs.ravel()):
        c = drug_r2.columns[i]

        # Dataframe
        plot_df = drug_r2.loc[drug_r2[c].sort_values(ascending=False).head(ntop).index, c]
        plot_df = plot_df.sort_values().reset_index()
        plot_df = plot_df.assign(y=range(plot_df.shape[0]))

        # Plot
        ax.scatter(plot_df[c], plot_df['y'], c=plot_df[c], cmap=cmap)

        for fc, y, n in plot_df[[c, 'y', 'DRUG_NAME']].values:
            ax.text(fc - xoffset, y, n, va='center', fontsize=5, zorder=10, color='gray', ha='right')

        ax.set_xlabel('Drug R2 (median)')
        ax.set_ylabel('')
        ax.set_title('Growth rate' if c == 'r2_growth' else 'Cancer Type')
        ax.axes.get_yaxis().set_ticks([])

    plt.gcf().set_size_inches(2. * axs.shape[0], 20 * .1)

    plt.suptitle('ElasticNet prediction of drug response', y=1.05, fontsize=10)
    plt.savefig('reports/covariates_elasticnet_top_predicted.pdf', bbox_inches='tight', dpi=600)
    plt.close('all')


if __name__ == '__main__':
    # - Imports
    # Drug-response
    drespo = dtrace.get_drugresponse()

    # CRISPR-Cas9
    crispr = dtrace.get_crispr(dtype='both')

    # Growth rate
    growth = pd.read_csv(dtrace.GROWTHRATE_FILE, index_col=0)
    growth = growth['growth_rate_median'].dropna()

    # Samplesheet
    ss = dtrace.get_samplesheet().dropna(subset=['Cancer Type'])

    samples = list(set(drespo).intersection(crispr).intersection(growth.index).intersection(ss.index))
    print('Samples={}'.format(len(samples)))

    # - Filter drug response
    drespo = dtrace.filter_drugresponse(drespo[samples])

    # - ElasticNet prediction of drug-response
    drug_r2 = predict_drug_response(drespo[samples], growth[samples], ss.loc[samples, 'Cancer Type'])
    drug_r2.to_csv('data/drug_covariates_r2_score.csv')

    # - Plot
    # Relation between growth rate and cancer type
    growht_by_cancer_type(growth[samples], ss.loc[samples, 'Cancer Type'])
    plt.gcf().set_size_inches(1, 3)
    plt.savefig('reports/covariates_growth_per_tissue.pdf', bbox_inches='tight')
    plt.close('all')

    # Performance R2 scatter
    r2_scatter(drug_r2)
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/covariates_elasticnet_scatter.pdf', bbox_inches='tight', dpi=600)
    plt.close('all')

    # Top predicted drugs per covariate
    top_predicted_drugs(drug_r2)
    plt.savefig('reports/covariates_elasticnet_top_predicted.pdf', bbox_inches='tight', dpi=600)
    plt.close('all')
