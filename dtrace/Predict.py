#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from DTracePlot import DTracePlot
from Associations import Association
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit


DRUG_INFO = ['DRUG_ID', 'DRUG_NAME', 'VERSION']


def features_barplot(df):
    f, axs = plt.subplots(len(drugs), 1, sharex='col')

    for i, ((d_id, d_name, d_ver), plot_df) in enumerate(df.groupby(DRUG_INFO)):
        #
        x_mean = list(plot_df.groupby(DRUG_INFO).median().iloc[0])
        x_mean.insert(1, 0)

        x_std = list(plot_df.groupby(DRUG_INFO).std().iloc[0])
        x_std.insert(1, 0)

        y = list(range(len(x_std)))

        labels = ['R-squared' if i == 'r2' else i.replace('_', ' ') for i in plot_df.groupby(DRUG_INFO).median()]
        labels.insert(1, '')

        colors = [DTracePlot.PAL_DTRACE[2]] * len(y)

        #
        axs[i].barh(y, x_mean, color=colors, xerr=x_std, ecolor=DTracePlot.PAL_DTRACE[0])

        axs[i].axvline(0, ls='-', lw=.3, c=DTracePlot.PAL_DTRACE[2], zorder=0)
        axs[i].axhline(y[1], ls='--', lw=.3, c=DTracePlot.PAL_DTRACE[2], zorder=0)
        axs[i].grid(ls='-', lw=.3, alpha=.8, c=DTracePlot.PAL_DTRACE[1], zorder=0, axis='x')

        axs[i].set_yticks(y)

        axs[i].set_ylabel(f'{d_name}\n({d_id}, {d_ver})', fontsize=7)
        axs[i].yaxis.set_label_position("right")

        axs[i].set_yticklabels(labels, fontsize=5)
        axs[i].xaxis.set_tick_params(labelsize=5)

        axs[i].xaxis.set_major_locator(plticker.MultipleLocator(base=.2))

    plt.subplots_adjust(hspace=0.05)

    axs[0].set_title('Drug-response prediction')

    return f, axs


def pred_scatterplot(y, x, annot_text):
    y = y[x.index].dropna()
    x = x.loc[y.index]

    lm = Ridge().fit(x, y)

    y_pred = pd.Series(lm.predict(x), index=y.index)

    plot_df = pd.concat([
        y.rename('observed'),
        y_pred.rename('predicted'),
        data.samplesheet.samplesheet['institute']
    ], axis=1, sort=False).dropna()

    #
    g = DTracePlot.plot_corrplot(
        'observed', 'predicted', 'institute', plot_df, add_vline=False, add_hline=False, annot_text=annot_text,
        fit_reg=False
    )

    #
    dmax = np.log(data.drespo_obj.maxconcentration[drug])
    g.ax_joint.axhline(dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)
    g.ax_joint.axvline(dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

    g.set_axis_labels(f'{drug[1]}\nobserved drug-response IC50', f'{drug[1]}\npredicted drug-response IC50')

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
    g.ax_joint.xaxis.set_major_locator(plticker.MultipleLocator(base=2.5))
    g.ax_joint.yaxis.set_major_locator(plticker.MultipleLocator(base=2.5))

    return g


def lm_drug_train(y, x, drug, n_splits=1000, test_size=.3):
    y = y[x.index].dropna()
    x = x.loc[y.index]

    df = []
    for train, test in ShuffleSplit(n_splits=n_splits, test_size=test_size).split(x, y):
        lm = Ridge().fit(x.iloc[train], y.iloc[train])

        r2 = lm.score(x.iloc[test], y.iloc[test])

        df.append(list(drug) + [r2] + list(lm.coef_))

    return pd.DataFrame(df, columns=DRUG_INFO + ['r2'] + list(x.columns))


if __name__ == '__main__':
    # - Import
    data = Association(dtype_drug='ic50')
    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')

    # -
    d_target = 'MCL1'

    gene_products = ['MARCH5', 'MCL1', 'BCL2', 'BCL2L1']

    drugs = list({
        tuple(i) for i in lmm_drug[lmm_drug['DRUG_TARGETS'] == d_target][DRUG_INFO].values
    })

    for ftype in ['CRISPR+GEXP', 'CRISPR', 'GEXP']:
        print(f'ftype = {ftype}')

        if ftype == 'CRISPR+GEXP':
            xs = pd.concat([
                data.crispr.loc[gene_products].T.add_prefix('CRISPR_'),
                data.gexp.loc[gene_products].T.add_prefix('GExp_')
            ], axis=1, sort=False).dropna()

        elif ftype == 'CRISPR':
            xs = data.crispr.loc[gene_products].T.add_prefix('CRISPR_').dropna()

        elif ftype == 'GEXP':
            xs = data.gexp.loc[gene_products].T.add_prefix('GExp_').dropna()

        else:
            assert True, 'xs not defined'

        xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

        #
        lm_df = pd.concat([lm_drug_train(data.drespo.loc[d], xs, d) for d in drugs])

        #
        features_barplot(lm_df)

        plt.gcf().set_size_inches(2, 1 * len(drugs))
        plt.savefig(f'reports/predict_{d_target}_{ftype}_feature_barplot.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

        # -
        # drug = (2235, 'AZD5991', 'RS')
        for drug in drugs:
            drug_r2_mean = lm_df.groupby(DRUG_INFO)['r2'].median().loc[drug]
            drug_r2_std = lm_df.groupby(DRUG_INFO)['r2'].std().loc[drug]
            drug_annot = f"Median R-squared = {drug_r2_mean:.2f} (±{drug_r2_std:.2f})"

            pred_scatterplot(data.drespo.loc[drug], xs, drug_annot)
            plt.gcf().set_size_inches(2, 2)
            plt_name = f'reports/predict_{d_target}_{ftype}_{drug[0]}_{drug[1]}_{drug[2]}_pred_scatter.pdf'
            plt.savefig(plt_name, bbox_inches='tight', transparent=True)
            plt.close('all')
