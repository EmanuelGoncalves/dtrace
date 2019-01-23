#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.ticker as plticker
from scipy.stats import gaussian_kde
from sklearn.svm import LinearSVC
from DTracePlot import DTracePlot
from Associations import Association
from scipy.stats import spearmanr, pearsonr
from DTraceEnrichment import DTraceEnrichment
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import f_regression, f_classif
from sklearn.linear_model import Ridge, ElasticNetCV, Lasso, RidgeCV, LinearRegression


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

    lmm_cgexp = pd.read_csv('data/crispr_lmm_regressions_gexp.csv.gz')

    # -
    tissue = data.samplesheet.samplesheet
    tissue = list(tissue[tissue['cancer_type'].isin(['Breast Carcinoma', 'Colorectal Carcinoma'])].index)

    samples = list(set(data.gexp))

    gene = 'MARCH5'

    y = data.crispr.loc[gene, samples]

    x = data.gexp[samples].T
    x = pd.DataFrame(StandardScaler().fit_transform(x), index=x.index, columns=x.columns)

    #
    fselection = lmm_cgexp.query(f"DRUG_ID == '{gene}'").set_index('GeneSymbol')['pval']
    # fselection = pd.Series(f_regression(x, y)[0], index=x.columns)
    x = x[fselection[fselection < .1].index]

    #
    y_scores, y_features = [], []
    for train, test in ShuffleSplit(n_splits=200, test_size=.3).split(x):
        lm = RidgeCV().fit(x.iloc[train], y.iloc[train])
        r2 = lm.score(x.iloc[test], y.iloc[test])

        y_scores.append([r2] + list(lm.coef_))
        y_features.append(lm.coef_)

    y_scores = pd.DataFrame(y_scores, columns=['r2'] + list(x.columns))
    y_features = pd.DataFrame(y_features, columns=x.columns)
    print(f"Mean r2: {y_scores['r2'].mean():.2f}; n features: {y_scores.shape[1] - 1}")
    y_features.median().sort_values()

    #
    ssgsea = DTraceEnrichment()

    gmt_file = 'c5.bp.v6.2.symbols.gmt'

    values = lmm_cgexp.query(f"DRUG_ID == '{gene}'").set_index('GeneSymbol')['beta']

    enr = ssgsea.gsea_enrichments(values, gmt_file)
    enr.query('len >= 5')
    enr.loc[[i for i in enr.index if 'CYTOCHROME_C' in i]]

    ssgsea.plot(values, gmt_file, 'GO_POSITIVE_REGULATION_OF_RELEASE_OF_CYTOCHROME_C_FROM_MITOCHONDRIA')
    plt.show()

    ssgsea.get_signature(
        gmt_file, 'GO_REGULATION_OF_GENE_SILENCING'
    )

    #
    y_pred, y_features = {}, {}

    for s in samples:
        lm = LinearRegression().fit(x.drop(index=[s]), y.drop(s))

        y_pred[s] = lm.predict(x.loc[[s]])[0]
        y_features[s] = lm.coef_

    y_pred = pd.Series(y_pred)
    y_features = pd.DataFrame(y_features, index=x.columns)
    y_features.median(1).sort_values()

    #
    annot = f"Median R-squared = {y_scores['r2'].median():.2f} (±{y_scores['r2'].std():.2f})"

    pred_scatterplot(y, y_pred, annot, data=data)
    plt.gcf().set_size_inches(2, 2)
    plt_name = f'reports/biomarker_{gene}_pred_scatter.pdf'
    plt.savefig(plt_name, bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    plot_df = pd.concat([
        data.crispr.loc['MARCH5'],
        data.samplesheet.samplesheet['gender']
    ], axis=1, sort=False).dropna()

    sns.boxplot('gender', 'MARCH5', data=plot_df)
    plt.show()
