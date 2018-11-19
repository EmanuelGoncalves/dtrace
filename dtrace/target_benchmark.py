#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from natsort import natsorted
from crispy.qc_plot import QCplot
from associations import Association
from sklearn.linear_model import ElasticNetCV
from scipy.stats.distributions import hypergeom
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score


class TargetBenchmark:
    def __init__(self):
        self.datasets = Association(dtype_drug='ic50')

        self.lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')

        self.drugs = self.get_benchmarkable_drugs()
        self.lmm_drug_target = self.lmm_drug[self.lmm_drug['DRUG_ID'].isin(self.drugs)].copy()

        self.target_scores = self.calculate_target_precision()

    def get_benchmarkable_drugs(self):
        d_targets = self.datasets.drespo_obj.get_drugtargets()

        d_targets = {
            d[0]: d_targets[d[0]]
            for d in self.datasets.drespo.index
            if d[0] in d_targets and len(d_targets[d[0]].intersection(self.datasets.crispr.index)) > 0
        }

        return d_targets

    def calculate_target_precision(self):
        cls_scores = []

        for target in set(self.lmm_drug_target['target']):
            y_true = (self.lmm_drug_target['target'] == target).astype(int)

            M = self.lmm_drug_target.shape[0]
            n = self.lmm_drug_target.query(f"target == '{target}'").shape[0]

            for pval_thres in [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, .25]:
                y_pred = (self.lmm_drug_target['fdr'] < pval_thres).astype(int)

                x = self.lmm_drug_target.query(f"(fdr < {pval_thres}) & (target == '{target}')").shape[0]
                N = self.lmm_drug_target.query(f"fdr < {pval_thres}").shape[0]

                cls_scores.append(dict(
                    f1=f1_score(y_true, y_pred),
                    recall=recall_score(y_true, y_pred),
                    precision=precision_score(y_true, y_pred),
                    fdr=pval_thres,
                    target=target,
                    hypergeom=hypergeom.sf(x, M, n, N)
                ))

        cls_scores = pd.DataFrame(cls_scores)

        return cls_scores

    def beta_histogram(self):
        kde_kws = dict(cut=0, lw=1, zorder=1, alpha=.8)
        hist_kws = dict(alpha=.4, zorder=1, linewidth=0)

        sns.distplot(
            self.lmm_drug.query("target != 'T'")['beta'], color=Plot.PAL_DTRACE[2], kde_kws=kde_kws, hist_kws=hist_kws,
            label='All', bins=30
        )

        sns.distplot(
            self.lmm_drug.query("target == 'T'")['beta'], color=Plot.PAL_DTRACE[0], kde_kws=kde_kws, hist_kws=hist_kws,
            label='Target', bins=30
        )

        sns.despine(right=True, top=True)

        plt.axvline(0, c=Plot.PAL_DTRACE[1], lw=.3, ls='-', zorder=0)

        plt.xlabel('Association beta')
        plt.ylabel('Density')

        plt.legend(prop={'size': 6}, loc=2, frameon=False)

    def recapitulated_drug_targets_barplot(self, fdr=.1):
        dcols = self.datasets.drespo_obj.DRUG_COLUMNS

        d_targets = self.datasets.drespo_obj.get_drugtargets()

        df_genes = set(self.lmm_drug['GeneSymbol'])

        # Count number of drugs
        d_all = {
            tuple(i) for i in self.lmm_drug[dcols].values
        }

        d_annot = {
            tuple(i) for i in d_all if i[0] in d_targets
        }

        d_tested = {
            tuple(i) for i in d_annot if len(d_targets[i[0]].intersection(df_genes)) > 0
        }

        d_tested_signif = {
            tuple(i) for i in self.lmm_drug.query(f'fdr < {fdr}')[dcols].values if tuple(i) in d_tested
        }

        d_tested_correct = {
            tuple(i) for i in self.lmm_drug.query(f"fdr < {fdr} & target == 'T'")[dcols].values if
        tuple(i) in d_tested_signif
        }

        # Build dataframe
        plot_df = pd.DataFrame(dict(
            names=['All', 'w/Target', 'w/Tested target', 'w/Signif tested target', 'w/Correct target'],
            count=list(map(len, [d_all, d_annot, d_tested, d_tested_signif, d_tested_correct]))
        )).sort_values('count', ascending=True)
        plot_df = plot_df.assign(y=range(plot_df.shape[0]))

        # Plot
        plt.barh(plot_df['y'], plot_df['count'], color=Plot.PAL_DTRACE[2], linewidth=0)

        sns.despine(right=True, top=True)

        for c, y in plot_df[['count', 'y']].values:
            plt.text(c + 3, y, str(c), va='center', fontsize=5, zorder=10, color=Plot.PAL_DTRACE[2])

        plt.yticks(plot_df['y'], plot_df['names'])
        plt.xlabel('Number of drugs')
        plt.ylabel('')

    def plot_target_scores(self, ymetric='precision'):
        hue_order = natsorted(list(set(self.lmm_drug_target['target'])))

        pal = pd.Series(
            QCplot.get_palette_continuous(len(hue_order) - 2, Plot.PAL_DTRACE[2]) + [Plot.PAL_DTRACE[3], Plot.PAL_DTRACE[0]],
            index=hue_order
        )

        plot_df = self.target_scores.copy()
        plot_df['signif'] = (plot_df['hypergeom'] < 0.05).astype(int)

        sns.scatterplot(
            'fdr', ymetric, 'target', style='signif', data=plot_df, hue_order=hue_order, palette=pal
        )

        plt.grid(axis='y', lw=.3, ls='--', color=Plot.PAL_DTRACE[1], zorder=0)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title='Association')

        plt.xlabel('FDR threshold')
        plt.ylabel(ymetric.capitalize())

        sns.despine()


if __name__ == '__main__':
    target_bench = TargetBenchmark()

    target_bench.plot_target_scores()
    plt.gcf().set_size_inches(4, 2)
    plt.savefig('reports/target_benchmark_precision_barplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    target_bench.beta_histogram()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/target_benchmark_beta_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    target_bench.recapitulated_drug_targets_barplot(fdr=.1)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/target_benchmark_count_signif.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
