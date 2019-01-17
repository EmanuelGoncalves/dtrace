#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from limix.qtl import scan
from scipy.stats import spearmanr
from DTracePlot import DTracePlot
from Associations import Association
from DataImporter import DrugResponse
from DTraceEnrichment import DTraceEnrichment
from sklearn.model_selection import ShuffleSplit
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet


if __name__ == '__main__':
    # - Import
    data = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_gexp = pd.read_csv('data/drug_lmm_regressions_ic50_gexp.csv.gz')

    # -
    gene_assoc, gene_extra = 'MARCH5', 'MCL1'

    drugs = [
        (1956, 'MCL1_1284', 'RS'), (2354, 'MCL1_8070', 'RS'), (1946, 'MCL1_5526', 'RS'), (2127, 'Mcl1_6386', 'RS'),
        (1720, 'AZD5991', 'RS'), (2235, 'AZD5991', 'RS')
    ]

    # drug = (1956, 'MCL1_1284', 'RS')
    for drug in drugs:
        dmax = np.log(data.drespo_obj.maxconcentration[drug])

        # Data-frame
        plot_df = pd.concat([
            data.drespo.loc[drug].rename('drug'),

            data.crispr.loc[gene_assoc].rename(f'CRISPR {gene_assoc}'),
            data.crispr.loc[gene_extra].rename(f'CRISPR {gene_extra}'),

            data.crispr_obj.institute.rename('Institute'),

            data.samplesheet.samplesheet['model_name'],
            data.samplesheet.samplesheet['cancer_type'],
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
            g = DTracePlot().plot_multiple('drug', 'essentiality', df, n_offset=1.1)

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

            fcs = {}
            # [('RPPA', datasets.rppa), ('Gexp', datasets.gexp), ('CRISPR', datasets.rppa)]
            for name, data in [('RPPA', data.rppa)]:
                fcs[name] = {}

                for i in data.index:
                    data_resistant = data.loc[i, sensitive].dropna()
                    data_sensitive = data.loc[i, resistant].dropna()

                    if len(data_resistant) >= 3 and len(data_sensitive) >= 3:
                        fcs[name][i] = data_resistant.median() - data_sensitive.median()

            fcs = pd.DataFrame(fcs).sort_values('RPPA')

            # -
            gmt_file = 'c2.cp.kegg.v6.2.symbols.gmt'

            #
            # values = datasets.crispr[df.index].T.corrwith(datasets.crispr.loc['MARCH5', df.index])
            values = lmm_drug[lmm_drug['DRUG_NAME'] == drug[1]].set_index('GeneSymbol')
            ssgsea = DTraceEnrichment().gsea_enrichments(values['beta'], gmt_file)

            #
            signature = DTraceEnrichment().get_signature(gmt_file, 'KEGG_REGULATION_OF_AUTOPHAGY')

            #
            top_sigs_genes = pd.Series([g for s in ssgsea.index[-5:] for g in DTraceEnrichment().get_signature(gmt_file, s)])

            #
            x = data.prot.loc[signature, df.index].dropna(how='all', axis=1).dropna().T
            y = df.loc[x.index, 'drug']

            pred_scores = []

            cv = ShuffleSplit(n_splits=30, test_size=.15)
            for train, test in cv.split(x, y):
                lm = Ridge().fit(x.iloc[train], y.iloc[train])

                score = lm.score(x.iloc[test], y.iloc[test])
                pred_scores.append(score)

                print(score)
    # -
    bcl_abs = [
        'Bad_pS112', 'Bak_Caution', 'Bap1.c.4', 'Bax', 'Bcl.2', 'Bcl.xL', 'Bid_Caution', 'Bim.CST2933.', 'Bim.EP1036.',
        'PARP_cleaved_Caution', 'BCL2A1', 'Bim', 'Mcl.1'
    ]

    proteins = ['MARCH5', 'MCL1', 'BCL2', 'BCL2L1', 'BCL2L11']

    plot_df = pd.concat([
        data.drespo.loc[drugs].T,

        data.crispr_obj.institute.rename('Institute'),

        data.samplesheet.samplesheet['cancer_type'],
        data.samplesheet.samplesheet['model_name'],
        data.samplesheet.samplesheet['growth'],
        pd.get_dummies(data.samplesheet.samplesheet['msi_status']),

        data.crispr.loc['WRN'].rename('CRISPR_WRN'),
        data.crispr.loc['MCL1'].rename('CRISPR_MCL1'),
        data.crispr.loc['BCL2L1'].rename('CRISPR_BCL2L1'),
        data.crispr.loc['MARCH5'].rename('CRISPR_MARCH5'),
        data.crispr.loc['PMAIP1'].rename('CRISPR_PMAIP1'),
        data.crispr.loc['BCL2L11'].rename('CRISPR_BCL2L11'),

        data.crispr.T.eval('MCL1 - BCL2L1').rename('CRISPR MCL1/BCL2L1 ratio'),
        data.crispr.T.eval('MCL1 - MARCH5').rename('CRISPR MCL1/MARCH5 ratio'),
        data.crispr.T.eval('PMAIP1 - MARCH5').rename('CRISPR PMAIP1/MARCH5 ratio'),

        data.gexp.loc['MCL1'].rename('Gexp_MCL1'),
        data.gexp.loc['BCL2L1'].rename('Gexp_BCL2L1'),
        data.gexp.loc['MARCH5'].rename('Gexp_MARCH5'),
        data.gexp.loc['PMAIP1'].rename('Gexp_PMAIP1'),
        data.gexp.loc['BCL2L11'].rename('Gexp_BCL2L11'),

        data.gexp.T.eval('MCL1 / BCL2L1').rename('Gexp MCL1/BCL2L1 ratio'),
        data.gexp.T.eval('MCL1 / MARCH5').rename('Gexp MCL1/MARCH5 ratio'),
        data.gexp.T.eval('PMAIP1 / MARCH5').rename('Gexp PMAIP1/MARCH5 ratio'),

        # data.cn.loc['MCL1'].rename('CN_MCL1'),
        #
        # data.rppa.loc[bcl_abs].T.add_prefix('RPPA '),
        # (data.rppa.loc['Mcl.1'] / data.rppa.loc['Bcl.xL']).rename('RPPA MCL1/BCL2L1 ratio'),
        #
        # data.apoptosis.T,

        data.prot.loc[proteins].T.add_prefix('Prot '),

        data.prot.T.eval('MCL1 / BCL2L1').rename('Prot MCL1/BCL2L1 ratio'),
    ], axis=1, sort=False)

    cbin = pd.concat([
        data.crispr.loc[g, plot_df.index].apply(lambda v: g if v < -.5 else '') for g in [gene_assoc, gene_extra]
    ], axis=1)
    plot_df['ess'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values
    plot_df['ess_num'] = plot_df['ess'].map({'None': 0, 'MCL1': 1, 'MARCH5': 1, 'MARCH5 + MCL1': 2})

    # plot_df = plot_df[plot_df['cancer_type'].isin(['Colorectal Carcinoma', 'Breast Carcinoma'])]
    # plot_df = plot_df[plot_df['cancer_type'].isin(['Lung Adenocarcinoma'])]
    # plot_df = plot_df[plot_df['cancer_type'].isin(['Lung Adenocarcinoma', 'Small Cell Lung Carcinoma'])]

    #
    feature = 'CRISPR_MCL1'
    drug = (1956, 'MCL1_1284', 'RS')
    dmax = np.log(data.drespo_obj.maxconcentration[drug])
    dmax_thres = np.log(data.drespo_obj.maxconcentration[drug] * 0.5)

    #
    g = DTracePlot.plot_corrplot(feature, drug, 'Institute', plot_df, add_hline=True)

    g.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

    g.set_axis_labels(f'{feature} (scaled log2 FC)', f'{drug[1]} (ln IC50)')

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(f'reports/vignette_drug_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    # resistant = list(plot_df[plot_df['ess'] != f'{gene_assoc} + {gene_extra}'].index)
    # sensitive = list(plot_df[plot_df['ess'] == f'{gene_assoc} + {gene_extra}'].index)
    resistant = list(plot_df[((plot_df[drug] > dmax) & (plot_df['CRISPR_MCL1'] < -0.5))].index)
    sensitive = list(plot_df[((plot_df[drug] < dmax_thres) & (plot_df['CRISPR_MCL1'] < -0.5))].index)

    fcs = {}
    for name, data in [('RPPA', data.rppa), ('Gexp', data.gexp), ('CRISPR', data.crispr)]:
        fcs[name] = {}

        for i in data.index:
            data_resistant = data.loc[i, sensitive].dropna()
            data_sensitive = data.loc[i, resistant].dropna()

            if len(data_resistant) > 3 and len(data_sensitive) > 3:
                fcs[name][i] = data_resistant.median() - data_sensitive.median()

    fcs = pd.DataFrame(fcs).sort_values('RPPA')

    fcs.dropna(subset=['Gexp']).sort_values('Gexp')

    #
    name, data, index = 'RPPA', data.rppa, 'Caveolin.1'

    df = data.loc[index].reset_index()
    df.columns = ['model_id', index]
    df['ess'] = ['Resistant' if index in resistant else ('Sensitive' if index in sensitive else 'None') for index in df['model_id']]

    sns.boxplot(index, 'ess', data=df, color=DTracePlot.PAL_DBGD[2], orient='h', notch=True)
    plt.gcf().set_size_inches(2, 1)
    plt.savefig(f'reports/fcs_boxplots_{name}.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    gmt_file = 'c5.bp.v6.2.symbols.gmt'

    ssgsea = DTraceEnrichment().gsea_enrichments(fcs['Gexp'].dropna(), gmt_file)

    #
    signature = DTraceEnrichment().get_signature(gmt_file, 'GO_ALKALOID_METABOLIC_PROCESS')

    #
    x = pd.concat([
        data.gexp.loc[signature, plot_df.index].dropna(how='all', axis=1).dropna().T,
        # datasets.crispr.loc[signature, plot_df.index].dropna(how='all', axis=1).dropna().T
    ], axis=1, sort=False).dropna()

    y = plot_df.loc[x.index, [drug]].iloc[:, 0].dropna()
    x = x.loc[y.index]

    pred_scores = []

    cv = ShuffleSplit(n_splits=100, test_size=.3)
    for train, test in cv.split(x, y):
        lm = Ridge().fit(x.iloc[train], y.iloc[train])

        score = lm.score(x.iloc[test], y.iloc[test])
        pred_scores.append(score)

    print(f'Mean score: {np.median(pred_scores)}')

    #
    # samples = list(datasets.samplesheet.samplesheet.query("cancer_type == 'Breast Carcinoma'").index)
    samples = data.samplesheet.samplesheet
    samples = set(samples[samples['cancer_type'].isin(['Colorectal Carcinoma'])].index)
    samples = list(samples.intersection(data.crispr).intersection(data.prot))

    y = data.crispr.loc[['MARCH5'], samples].T.dropna()
    x = data.prot[y.index].dropna().T

    k = Association.kinship(x)

    m = pd.concat([data.get_covariates().loc[y.index], data.prot.loc['HNF4A', y.index]], axis=1).loc[y.index]

    df = data.lmm_single_association(y=y, x=x, m=m, k=k, expand_drug_id=False)
    df['fdr'] = multipletests(df['pval'], method='bonferroni')[1]
    print(df.sort_values(['pval', 'fdr']).head(60))
