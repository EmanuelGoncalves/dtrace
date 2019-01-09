#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from DTracePlot import DTracePlot
from Associations import Association
from DataImporter import DrugResponse
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # - Import
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_cgexp = pd.read_csv(f'data/drug_lmm_regressions_gexp_ic50.csv.gz')

    # -
    gene_assoc, gene_extra = 'MARCH5', 'MCL1'

    drugs = [(1956, 'MCL1_1284', 'RS'), (2354, 'MCL1_8070', 'RS'), (1946, 'MCL1_5526', 'RS'), (2127, 'Mcl1_6386', 'RS')]

    for drug in drugs:
        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        for tissue in ['Colorectal Carcinoma', 'Breast Carcinoma']:
            print(drug, tissue)

            # Data-frame
            plot_df = pd.concat([
                datasets.drespo.loc[drug].rename('drug'),

                datasets.crispr.loc[gene_assoc].rename(f'CRISPR {gene_assoc}'),
                datasets.crispr.loc[gene_extra].rename(f'CRISPR {gene_extra}'),

                datasets.crispr_obj.institute.rename('Institute'),

                datasets.samplesheet.samplesheet['model_name'],
                datasets.samplesheet.samplesheet['cancer_type'],
            ], axis=1, sort=False).dropna()

            # Tissue
            plot_df = plot_df.query(f"cancer_type == '{tissue}'")

            # Aggregate essentiality
            cbin = pd.concat([
                plot_df[f'CRISPR {g}'].apply(lambda v: g if v < -.5 else '') for g in [gene_assoc, gene_extra]
            ], axis=1)
            plot_df['essentiality'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values

            # -
            g = DTracePlot().plot_multiple('drug', 'essentiality', 'Institute', plot_df, n_offset=1.2)

            sns.despine()

            plt.axvline(dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

            plt.xlabel(f'{drug[1]} (ln IC50, {drug[2]})')
            plt.ylabel('Essential genes')

            plt.title(tissue)

            plt.gcf().set_size_inches(2, 1)
            plt.savefig(
                f"reports/boxplot_{tissue}_{drug[1]}_{gene_assoc}_{gene_extra}.pdf",
                bbox_inches='tight', transparent=True
            )
            plt.close('all')

            # -
            if tissue == 'Colorectal Carcinoma':
                plot_df = pd.concat([plot_df, datasets.apoptosis.T], axis=1, sort=False).dropna().drop('SIDM00834')
                plot_df = pd.melt(plot_df, id_vars='essentiality', value_vars=list(datasets.apoptosis.T))

                order = ['None', 'MARCH5', 'MCL1', 'MARCH5 + MCL1']

                pal = pd.Series(DTracePlot.get_palette_continuous(len(order), DTracePlot.PAL_DTRACE[2]), index=order)

                sns.boxplot(
                    'variable', 'value', 'essentiality', data=plot_df, showcaps=False, hue_order=order, palette=pal
                )

                plt.gcf().set_size_inches(3, 2)
                plt.savefig(f"reports/boxplot_apoptosis_{tissue}_{drug[1]}_{gene_assoc}_{gene_extra}.pdf", bbox_inches='tight', transparent=True)
                plt.close('all')
