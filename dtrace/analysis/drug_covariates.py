#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace.analysis import PAL_DTRACE
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import ShuffleSplit
from dtrace.Associations import DRUG_INFO_COLUMNS


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


def pred_scatter(pred):
    plot_df = pd.pivot_table(pred, index=DRUG_INFO_COLUMNS, columns='covariate', values='r2')

    order = list(plot_df.median().sort_values(ascending=False).index)

    cmap = sns.light_palette(PAL_DTRACE[2], as_cmap=True, reverse=True)

    g = sns.PairGrid(plot_df[order], despine=False)

    g = g.map_diag(plt.hist, color=PAL_DTRACE[2], linewidth=0)
    g = g.map_upper(plt.scatter, color=PAL_DTRACE[2], s=3, lw=0., zorder=2, alpha=.9)
    g = g.map_lower(sns.kdeplot, color=PAL_DTRACE[2], n_levels=30, cmap=cmap, linewidths=.3, zorder=3)

    plt.suptitle('Drug response prediction - R2 score', y=1.05)


def top_predicted_drugs(pred, ntop=20, xoffset=0.01):
    plot_df = pd.pivot_table(pred, index=DRUG_INFO_COLUMNS, columns='covariate', values='r2')
    order = ['Growth', 'Type', 'Burden', 'Ploidy']

    # Plot
    cmap = sns.light_palette(PAL_DTRACE[2], as_cmap=True)

    f, axs = plt.subplots(1, plot_df.shape[1], sharex=True, sharey=False, gridspec_kw=dict(wspace=.1))
    for i, covariate in enumerate(order):
        ax = axs[i]

        # Dataframe
        df = plot_df.loc[plot_df[covariate].sort_values(ascending=False).head(ntop).index, covariate]
        df = df.sort_values().reset_index()
        df = df.assign(y=range(df.shape[0]))

        # Plot
        ax.scatter(df[covariate], df['y'], c=df[covariate], cmap=cmap)

        for fc, y, n in df[[covariate, 'y', 'DRUG_NAME']].values:
            ax.text(fc - xoffset, y, n, va='center', fontsize=4, zorder=10, color='gray', ha='right')

        ax.set_xlabel('Drug R2 (median)')
        ax.set_ylabel('')
        ax.set_title(covariate)
        ax.axes.get_yaxis().set_ticks([])

        sns.despine(left=True, ax=ax)

    plt.gcf().set_size_inches(2.5 * axs.shape[0], 20 * .1)

    plt.suptitle('Linear regression of drug response', y=1.05, fontsize=10)


def predict_drug_response(drespo, covariates, discrete_covariates=['Type'], n_splits=3, test_size=.2):
    pred = []

    for drug in drespo.index:
        # Drug response data
        y = drespo.loc[drug, samples].dropna()

        # Cross-validation
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size)

        for i, (train_idx, test_idx) in enumerate(cv.split(y)):

            for covariate in covariates:
                print('Drug: {}; Split: {}; Covariate: {}'.format(','.join(map(str, drug)), i + 1, covariate))

                x = covariates.loc[y.index, [covariate]]

                # Covariate
                if covariate in discrete_covariates:
                    x = pd.get_dummies(x)

                # Build + train model
                lm = ElasticNetCV().fit(x.iloc[train_idx], y.iloc[train_idx])

                # Evalutate model
                r2 = lm.score(x.iloc[test_idx], y.iloc[test_idx])

                #
                pred.append(dict(
                    r2=r2, covariate=covariate,
                    DRUG_ID_lib=drug[0], DRUG_NAME=drug[1], VERSION=drug[2]
                ))

    pred = pd.DataFrame(pred).groupby(DRUG_INFO_COLUMNS + ['covariate']).median()

    pred = pred.reset_index()

    return pred


if __name__ == '__main__':
    # - Imports
    # Drug-response
    drespo = dtrace.get_drugresponse()

    # CRISPR-Cas9
    crispr = dtrace.get_crispr(dtype='both')

    # Growth rate
    growth = pd.read_csv(dtrace.GROWTHRATE_FILE, index_col=0)
    growth = growth['growth_rate_median'].dropna()

    # Mutation burden
    mburden = pd.read_csv(dtrace.MUTATION_BURDERN, index_col=0)['burden']

    # Ploidy
    ploidy = pd.read_csv(dtrace.PLOIDY, index_col=0)['ploidy']

    # Samplesheet
    ss = dtrace.get_samplesheet().dropna(subset=['Cancer Type'])
    ctype = ss['Cancer Type']

    samples = list(set.intersection(set(drespo), set(crispr), set(growth.index), set(ss.index), set(ploidy.index), set(mburden.index)))
    print('Samples={}'.format(len(samples)))

    # - Filter drug response
    drespo = dtrace.filter_drugresponse(drespo[samples])

    # - ElasticNet prediction of drug-response
    covariates = pd.concat([
        growth[samples].rename('Growth'),
        ctype[samples].rename('Type'),
        ploidy[samples].rename('Ploidy'),
        mburden[samples].rename('Burden')
    ], axis=1)

    pred = predict_drug_response(drespo[samples], covariates, n_splits=30, test_size=.2)
    pred.to_csv('data/drug_covariates_r2_score.csv')

    # - Plot
    # Relation between growth rate and cancer type
    growht_by_cancer_type(growth[samples], ss.loc[samples, 'Cancer Type'])
    plt.gcf().set_size_inches(1, 3)
    plt.savefig('reports/covariates_growth_per_tissue.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Performance R2 scatter
    pred_scatter(pred)
    plt.gcf().set_size_inches(3, 3)
    plt.savefig('reports/covariates_elasticnet_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # Top predicted drugs per covariate
    top_predicted_drugs(pred)
    plt.savefig('reports/covariates_elasticnet_top_predicted.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
