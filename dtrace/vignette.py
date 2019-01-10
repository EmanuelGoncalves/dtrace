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
from DTraceEnrichment import DTraceEnrichment
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


if __name__ == '__main__':
    # - Import
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_cgexp = pd.read_csv(f'data/drug_lmm_regressions_gexp_ic50.csv.gz')

    # -
    gene_assoc, gene_extra = 'MARCH5', 'MCL1'

    drugs = [(1956, 'MCL1_1284', 'RS'), (2354, 'MCL1_8070', 'RS'), (1946, 'MCL1_5526', 'RS'), (2127, 'Mcl1_6386', 'RS')]

    # drug = (1956, 'MCL1_1284', 'RS')
    for drug in drugs:
        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        # Data-frame
        plot_df = pd.concat([
            datasets.drespo.loc[drug].rename('drug'),

            datasets.crispr.loc[gene_assoc].rename(f'CRISPR {gene_assoc}'),
            datasets.crispr.loc[gene_extra].rename(f'CRISPR {gene_extra}'),

            datasets.crispr_obj.institute.rename('Institute'),

            datasets.samplesheet.samplesheet['model_name'],
            datasets.samplesheet.samplesheet['cancer_type'],
        ], axis=1, sort=False).dropna()

        # Aggregate essentiality
        cbin = pd.concat([
            plot_df[f'CRISPR {g}'].apply(lambda v: g if v < -.5 else '') for g in [gene_assoc, gene_extra]
        ], axis=1)
        plot_df['essentiality'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values

        # tissue = 'Colorectal Carcinoma' tissue = 'Breast Carcinoma'
        for tissue in ['Colorectal Carcinoma', 'Breast Carcinoma']:
            df = plot_df.query(f"cancer_type == '{tissue}'")
            print(drug, tissue)

            # -
            g = DTracePlot().plot_multiple('drug', 'essentiality', 'Institute', df, n_offset=1.1)

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
            resistant = list(df[df['essentiality'] != f'{gene_assoc} + {gene_extra}'].index)
            sensitive = list(df[df['essentiality'] == f'{gene_assoc} + {gene_extra}'].index)

            gexp_delta = (datasets.gexp.loc[:, sensitive].median(1) - datasets.gexp.loc[:, resistant].median(1)).sort_values()

            # -
            gmt_file = 'c5.bp.v6.2.symbols.gmt'

            #
            # values = datasets.crispr[df.index].T.corrwith(datasets.crispr.loc['MARCH5', df.index])
            values = lmm_drug[lmm_drug['DRUG_NAME'] == drug[1]].set_index('GeneSymbol')
            ssgsea = DTraceEnrichment().gsea_enrichments(values['beta'], gmt_file)

            #
            signature = DTraceEnrichment().get_signature(gmt_file, 'GO_RESPIRATORY_CHAIN_COMPLEX_IV_ASSEMBLY')

            #
            top_sigs_genes = pd.Series([g for s in ssgsea.index[-5:] for g in DTraceEnrichment().get_signature(gmt_file, s)])

            #
            x = datasets.gexp.loc[signature, df.index].dropna(how='all', axis=1).dropna().T
            y = df.loc[x.index, 'drug']

            pred_scores = []

            cv = ShuffleSplit(n_splits=30, test_size=.1)
            for train, test in cv.split(x, y):
                lm = Ridge().fit(x.iloc[train], y.iloc[train])

                score = lm.score(x.iloc[test], y.iloc[test])
                pred_scores.append(score)

                print(score)

