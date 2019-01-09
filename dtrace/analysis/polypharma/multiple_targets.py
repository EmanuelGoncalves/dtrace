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
from dtrace.analysis import PAL_DTRACE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from analysis.plot.corrplot import plot_corrplot
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


if __name__ == '__main__':
    # - Imports
    # Data-sets
    mobems = dtrace.get_mobem()
    drespo = dtrace.get_drugresponse()

    crispr = dtrace.get_crispr(dtype='bagel')
    crispr_logfc = dtrace.get_crispr(dtype='logFC', scale=True)

    samples = list(set(mobems).intersection(drespo).intersection(crispr))
    print('#(Samples) = {}'.format(len(samples)))

    wes = dtrace.get_wes()
    gexp = dtrace.get_geneexpression()

    # Drug max screened concentration
    d_maxc = pd.read_csv(dtrace.DRUG_RESPONSE_MAXC, index_col=[0, 1, 2])

    # Drugsheet
    ds = dtrace.get_drugsheet()

    # Drug target
    d_targets = get_drugtargets()

    # Linear regressions
    lmm_drug = pd.read_csv(dtrace.LMM_ASSOCIATIONS)
    lmm_drug = lmm_drug[['+' not in i for i in lmm_drug['DRUG_NAME']]]
    lmm_drug = ppi_annotation(lmm_drug, ppi_type=build_string_ppi, ppi_kws=dict(score_thres=900), target_thres=3)

    # -
    # Count number of drugs
    df_genes = set(lmm_drug['GeneSymbol'])

    fdr_thres = .1

    d_all = {tuple(i) for i in lmm_drug[DRUG_INFO_COLUMNS].values}
    d_annot = {tuple(i) for i in d_all if i[0] in d_targets}
    d_tested = {tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0}
    d_tested_signif = {tuple(i) for i in lmm_drug.query(f'fdr < {fdr_thres}')[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested}
    d_tested_correct = {tuple(i) for i in lmm_drug.query(f"fdr < {fdr_thres} & target == 'T'")[DRUG_INFO_COLUMNS].values if tuple(i) in d_tested_signif}

    # -
    pancore_ceres = set(pd.read_csv(dtrace.CERES_PANCANCER).iloc[:, 0].apply(lambda v: v.split(' ')[0]))

    df = lmm_drug[~lmm_drug['GeneSymbol'].isin(pancore_ceres)]
    df = df[df['DRUG_ID_lib'].isin(d_targets.keys())]
    df = df.query("target != 'T'").sort_values('beta')

    # -
    d_id, d_name, d_screen = 2125, 'Mcl1_7350', 'RS'

    genes = ['EHMT2']
    lines = list(set(wes[wes['Gene'].isin(genes)]['SAMPLE']))

    plot_df = drespo.loc[(d_id, d_name, d_screen)].dropna().rename('drug').to_frame()
    plot_df = plot_df.assign(mut=plot_df.index.isin(lines).astype(int))

    sns.boxplot('mut', 'drug', data=plot_df, notch=True, palette=PAL_DTRACE)
    plt.show()

    # -
    d_targ = d_targets[d_id]

    d_name, d_screen = 'Trametinib', 'RS'

    # d_targ = {'EGFR', 'ERBB2', 'ERBB3', 'KRAS'}
    # d_targ = {'PIK3CB', 'RICTOR', 'AKT1'}

    df = pd.concat([
        drespo.loc[(d_id, d_name, d_screen)],
        crispr_logfc.loc[d_targ].T
    ], axis=1, sort=False).dropna()

    dmax = np.log(d_maxc.loc[(d_id, d_name, d_screen), 'max_conc_micromolar'])

    d_lms = []
    for n in [1, 2]:
        for features in it.combinations(d_targ, n):
            for interaction in [False, True]:
                y = df.iloc[:, 0]

                if n == 1 and interaction:
                    continue

                if n > 1 and interaction:
                    x = df[list(features)]

                    x_int = '*'.join(features)
                    x[x_int] = x.eval(x_int)

                    name = ' + '.join(x.columns)

                    x = StandardScaler().fit_transform(x)

                else:
                    x = df[list(features)].loc[y.index]

                    name = ' + '.join(x.columns)

                    x = StandardScaler().fit_transform(x)

                lm = LinearRegression().fit(x, y)

                # Predict
                y_pred = lm.predict(x)

                # Log likelihood
                llm = log_likelihood(y, y_pred)

                # R-squared
                r = r_squared(y, y_pred)

                d_lms.append((n, name, lm, llm, r, lm.coef_))

    print(list(enumerate(d_lms)))

    #
    model_full, model_small = d_lms[15][3], d_lms[14][3]
    lr = 2 * (model_full - model_small)
    lr_pval = st.chi2(1).sf(lr)
    print(lr_pval)

    # Plot
    plot_corrplot('MAP2K2', (d_id, d_name, d_screen), df, add_hline=True, lowess=False)

    plt.axhline(dmax, lw=.3, color=PAL_DTRACE[2], ls=':', zorder=0)
    plt.axvline(-.5, lw=.3, color=PAL_DTRACE[2], ls=':', zorder=0)

    plt.gcf().set_size_inches(2., 2.)
    plt.savefig(f'reports/targets_lm.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
