#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pydot
import dtrace
import numpy as np
import pandas as pd
import seaborn as sns
import itertools as it
import scipy.stats as st
import matplotlib.pyplot as plt
from dtrace import get_drugtargets
from statsmodels.distributions import ECDF
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from analysis.plot.corrplot import plot_corrplot
from statsmodels.stats.multitest import multipletests
from dtrace.analysis import PAL_DTRACE, MidpointNormalize
from dtrace.assemble.assemble_ppi import build_string_ppi
from dtrace.Associations import ppi_annotation, DRUG_INFO_COLUMNS


def log_likelihood(y_true, y_pred):
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = (1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l


def r_squared(y_true, y_pred):
    sse = np.power(y_true - y_pred, 2).sum()
    sst = np.power(y_true - y_true.mean(), 2).sum()
    r = 1 - sse / sst
    return r


def logratio(full_model, small_model, dof=1):
    lr = 2 * (full_model - small_model)
    lr_pval = st.chi2(dof).sf(lr)
    return lr_pval


def linear_regression(x, y):
    x_ = StandardScaler().fit_transform(x)

    lm = LinearRegression().fit(x_, y)

    y_pred = lm.predict(x)

    res = dict(
        llm=log_likelihood(y, y_pred),
        r=r_squared(y, y_pred),
        name=' + '.join(x.columns),
        n_features=x.shape[1],
    )

    res['beta1'] = lm.coef_[0]

    if len(lm.coef_) > 1:
        res['beta2'] = lm.coef_[1]
    else:
        res['beta2'] = np.nan

    return res


def get_drug_top(d_id, d_name, d_screen, top_n, essential_genes):
    d_genes = lmm_drug\
        .query(f"DRUG_ID_lib == {d_id} & DRUG_NAME == '{d_name}' & VERSION == '{d_screen}'")

    if essential_genes is not None:
        d_genes = d_genes[~d_genes['GeneSymbol'].isin(essential_genes)]

    d_genes = d_genes.sort_values('pval').head(top_n)['GeneSymbol']

    df = pd.concat([
        drespo.loc[(d_id, d_name, d_screen)],
        crispr_logfc.loc[d_genes].T
    ], axis=1, sort=False).dropna()

    return df


def lm_feature_combination(d_id, d_name, d_screen, n_features=2, top_n=5, essential_genes=None):
    d_df = get_drug_top(d_id, d_name, d_screen, top_n=top_n, essential_genes=essential_genes)

    lms = []
    for n in np.arange(1, n_features + 1):
        for features in it.combinations(d_df.columns[1:], n):
            print(f'Features: {features}')

            y = d_df.iloc[:, 0]
            x = d_df[list(features)].loc[y.index]

            lm = linear_regression(x, y)

            lms.append(lm)

    lms = pd.DataFrame(lms).sort_values('llm', ascending=False)

    return lms


def lm_feature_combination_lrt(d_lms):
    outputs = []
    for idx, values in d_lms.iterrows():
        best_single = d_lms[d_lms['name'].isin(values['name'].split(' + '))].iloc[0]

        lrt = logratio(values['llm'], best_single['llm'])

        outputs.append(dict(
            lrt_pval=lrt,
            best_single=best_single['name'],
            delta_r=values['r'] - best_single['r']
        ))

    return pd.concat([d_lms, pd.DataFrame(outputs, index=d_lms.index)], axis=1).query('n_features > 1')


if __name__ == '__main__':
    # - Imports
    samplesheet = dtrace.get_samplesheet()

    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='bagel')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    # Drug max screened concentration
    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = lmm_drug[['+' not in i for i in lmm_drug['DRUG_NAME']]]
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)

    # BROAD DepMap pan essential genes
    pancore_ceres = set(pd.read_csv(dtrace.CERES_PANCANCER).iloc[:, 0].apply(lambda v: v.split(' ')[0]))

    # Drug targets
    d_targets = get_drugtargets()

    # - Log-ratio tests of drug multiple featured model
    drugs = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}

    drugs_lrts = []
    for d_id, d_name, d_screen in drugs:
        d_lms = lm_feature_combination(
            d_id, d_name, d_screen, n_features=2, top_n=5, essential_genes=pancore_ceres
        )

        d_lms = lm_feature_combination_lrt(d_lms)

        d_lms[DRUG_INFO_COLUMNS[0]] = d_id
        d_lms[DRUG_INFO_COLUMNS[1]] = d_name
        d_lms[DRUG_INFO_COLUMNS[2]] = d_screen

        drugs_lrts.append(d_lms)

    drugs_lrts = pd.concat(drugs_lrts).reset_index(drop=True).sort_values('lrt_pval')
    drugs_lrts['lrt_fdr'] = multipletests(drugs_lrts['lrt_pval'], method='fdr_bh')[1]
    drugs_lrts = drugs_lrts.set_index(DRUG_INFO_COLUMNS + ['name'])

    # - Plot joint models
    d_id, d_name, d_screen, d_model = (1916, 'AZD5363', 'RS', 'WRN + KRAS')

    dmax = np.log(d_maxc.loc[(d_id, d_name, d_screen), 'max_conc_micromolar'])

    plot_df = get_drug_top(d_id, d_name, d_screen, top_n=10, essential_genes=pancore_ceres)
    plot_df = plot_df.rename(columns={(d_id, d_name, d_screen): 'drug'})
    plot_df = pd.concat([plot_df, samplesheet['Cancer Type']], axis=1, sort=False).dropna()
    plot_df['Response'] = (plot_df['drug'] < (0.5 * dmax)).astype(int).replace({0: 'No', 1: 'Yes'})

    # Order
    order = plot_df['Cancer Type'].value_counts()
    plot_df = plot_df.replace({'Cancer Type': {i: 'Other' for i in order[order < 20].index}})

    order = list(order[order >= 20].index) + ['Other']
    pal = sns.color_palette('tab20', n_colors=len(order) - 1).as_hex() + [PAL_DTRACE[1]]
    pal = dict(zip(*(order, pal)))

    #
    x, y = d_model.split(' + ')

    sns.scatterplot(
        x, y, hue='Cancer Type', style='Response', data=plot_df, palette=pal, legend='full',
        markers=['.', 'X'], linewidth=.1, style_order=['No', 'Yes'], hue_order=order
    )

    plt.axhline(0, ls=':', lw=0.1, c=PAL_DTRACE[1], zorder=0)
    plt.axvline(0, ls=':', lw=0.1, c=PAL_DTRACE[1], zorder=0)

    plt.xlabel(f'{x} CRISPR-Cas9 logFC')
    plt.ylabel(f'{y} CRISPR-Cas9 logFC')

    lrt_fdr = drugs_lrts.loc[(d_id, d_name, d_screen, d_model), 'lrt_fdr']
    plt.title(f'{d_name} ({d_id} {d_screen})\nModel: {d_model}; FDR: {lrt_fdr:.2e}')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig(f'reports/double_feature_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
