#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from DTracePlot import DTracePlot
from Associations import Association
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
import matplotlib.ticker as plticker
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit


def correlation(x, y, dtype):
    ctype = 'pearson' if dtype in ['Mutation'] else 'spearman'

    df = pd.concat([x.rename('x'), y.rename('y')], axis=1, sort=False).dropna()

    if ctype == 'pearson':
        cor, pval = pearsonr(df['x'], df['y'])

    else:
        cor, pval = spearmanr(df['x'], df['y'])

    return dict(len=df.shape[0], cor=cor, pval=pval, dtype=dtype, feature=y.name, ctype=ctype)


def pred_scatterplot(y_true, y_pred, annot_text, data):
    plot_df = pd.concat([
        y_true.rename('observed'),
        y_pred.rename('predicted'),
        data.samplesheet.samplesheet['institute']
    ], axis=1, sort=False).dropna()

    #
    g = DTracePlot.plot_corrplot(
        'observed', 'predicted', 'institute', plot_df, add_vline=False, add_hline=False, annot_text=annot_text,
        fit_reg=False
    )

    #
    g.set_axis_labels(f'{y_true.name}\nobserved CRISPR log2 FC', f'{y_pred.name}\npredicted CRISPR log2 FC')

    #
    xlim = g.ax_joint.get_xlim()
    ylim = g.ax_joint.get_ylim()

    xy_min, xy_max = min(xlim[0], ylim[0]), max(xlim[1], ylim[1])

    g.ax_joint.set_xlim(xy_min, xy_max)
    g.ax_joint.set_ylim(xy_min, xy_max)

    #
    (x0, x1), (y0, y1) = g.ax_joint.get_xlim(), g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ls='--', lw=.3, zorder=0, c=DTracePlot.PAL_DTRACE[1])

    #
    g.ax_joint.xaxis.set_major_locator(plticker.MultipleLocator(base=1.))
    g.ax_joint.yaxis.set_major_locator(plticker.MultipleLocator(base=1.))

    return g


if __name__ == '__main__':
    # - Import
    data = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_gexp = pd.read_csv('data/drug_lmm_regressions_ic50_gexp.csv.gz')
    lmm_cgexp = pd.read_csv('data/crispr_lmm_regressions_gexp.csv.gz')

    # -
    #
    wes_matrix = data.wes.copy()
    wes_matrix['value'] = 1

    wes_matrix = pd.pivot_table(wes_matrix, index='Gene', columns='model_id', values='value', fill_value=0)
    wes_matrix = wes_matrix[wes_matrix.sum(1) > 5]

    #
    genes = list(set(data.prot.index).intersection(data.gexp.index))
    samples = list(set(data.prot).intersection(data.gexp))
    print(f'Genes={len(genes)}; Samples={len(samples)}')

    y_ = {}
    for g in genes:
        print(g)

        y = data.prot.loc[g, samples].dropna()
        x = data.gexp.loc[g, y.index]

        lm = sm.OLS(y, sm.add_constant(x)).fit()
        y_[g] = lm.resid

    prot_ = pd.DataFrame(y_).T

    #
    dataframes = [
        ('Mutation', wes_matrix), ('Gexp', data.gexp), ('RNAi', data.rnai), ('Genomic', data.genomic),
        ('RPPA', data.rppa), ('CN', data.cn), ('Proteomics', data.prot), ('Phospho', data.phospho),
        ('Apoptosis', data.apoptosis), ('Degradation', prot_)
    ]

    x, corrs = data.crispr.loc['MARCH5'], []
    for dtype, df in dataframes:
        print(f'Correlation: {dtype}')
        corrs.append([
            correlation(x, df.iloc[i], dtype) for i in range(df.shape[0])
        ])

    corrs = pd.DataFrame([e for l in corrs for e in l]).sort_values('pval')
    print(corrs)

    # -
    samples = list(data.gexp)

    gene = 'MARCH5'

    y = data.crispr.loc[gene, samples]

    x = data.gexp.loc[lmm_cgexp.query(f"(DRUG_ID == '{gene}') & (pval < .15)")['GeneSymbol'], samples].T
    x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

    y_scores = []
    for train, test in ShuffleSplit(n_splits=100, test_size=.4).split(x, y):
        lm = Ridge().fit(x.iloc[train], y.iloc[train])

        r2 = lm.score(x.iloc[test], y.iloc[test])

        y_scores.append([r2] + list(lm.coef_))

    y_scores = pd.DataFrame(y_scores, columns=['r2'] + list(x.columns))
    print(f"Mean r2: {y_scores['r2'].mean()}; n features: {y_scores.shape[1] - 1}")

    #
    lm = Ridge().fit(x, y)
    y_pred = pd.Series(lm.predict(x), index=x.index)

    annot = f"Median R-squared = {y_scores['r2'].median():.2f} (±{y_scores['r2'].std():.2f})"

    pred_scatterplot(y, y_pred, annot, data=data)
    plt.gcf().set_size_inches(2, 2)
    plt_name = f'reports/biomarker_{gene}_pred_scatter.pdf'
    plt.savefig(plt_name, bbox_inches='tight', transparent=True)
    plt.close('all')
