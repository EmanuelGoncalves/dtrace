#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
import statsmodels.api as sm
from associations import Association
from importer import DrugResponse
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    # - Import
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_cgexp = pd.read_csv(f'data/drug_lmm_regressions_gexp_ic50.csv.gz')

    # -
    drug, gene_assoc, gene_extra = (1956, 'MCL1_1284', 'RS'), 'MARCH5', 'MCL1'
    # drug, gene_assoc, gene_extra = (1786, 'AZD4547', 'RS'), 'FGFR2', 'FGFR1'

    plot_df = pd.concat([
        datasets.drespo.loc[drug].rename('drug'),

        datasets.crispr.loc[gene_assoc].rename(gene_assoc),
        datasets.crispr.loc[gene_extra].rename(gene_extra),

        datasets.crispr_obj.institute.rename('Institute'),

        datasets.samplesheet.samplesheet['model_name'],
        datasets.samplesheet.samplesheet['cancer_type']
    ], axis=1, sort=False).dropna()

    cbin = pd.concat([plot_df[g].apply(lambda v: g if v < -.5 else '') for g in [gene_assoc, gene_extra]], axis=1)
    plot_df['essentiality'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values

    plot_df['TP53_mut'] = datasets.genomic.loc['TP53_mut'].reindex(plot_df.index)

    for g in [gene_assoc, gene_extra]:
        plot_df[f'{g}_cn'] = datasets.cn.loc[g].reindex(plot_df.index).values

    for g in [gene_assoc, gene_extra, 'FIS1', 'DNM1L', 'MFN1', 'MFN2', 'MIEF2', 'MIEF1', 'FUNDC1', 'HNF4A']:
        plot_df[f'{g}_gexp'] = datasets.gexp.loc[g].reindex(plot_df.index).values

    for p in ['FIS1', 'DNM1L', 'MFN1', 'MFN2', 'MIEF2', 'MIEF1', 'FUNDC1', 'HNF4A']:
        plot_df[f'{p}_prot'] = datasets.prot.loc[p].reindex(plot_df.index).values

    #
    dg_lmm = datasets.get_association(lmm_drug, drug, gene_assoc)
    annot_text = f"Beta={dg_lmm.iloc[0]['beta']:.2g}, FDR={dg_lmm.iloc[0]['fdr']:.1e}"

    dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

    plot_df['sensitive'] = (plot_df['drug'] < dmax).astype(int).values

    #
    g = Plot().plot_corrplot(gene_assoc, gene_extra, 'Institute', plot_df, add_hline=True, annot_text=annot_text)

    g.set_axis_labels(f'{gene_assoc} (scaled log2 FC)', f'{gene_extra} (scaled log2 FC)')

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(f'reports/vignette_corr_scatter_{gene_assoc}_{gene_extra}.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    g = Plot().plot_multiple('drug', 'essentiality', 'Institute', plot_df)

    sns.despine()

    plt.axvline(dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    plt.xlabel(f'{drug[1]} (ln IC50, {drug[2]})')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3, 1.5)
    plt.savefig(f"reports/vignette_multiple_boxplot_{gene_assoc}_{gene_extra}.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    order = list(plot_df.groupby('cancer_type')['drug'].mean().sort_values().index)

    sns.boxplot(
        'drug', 'cancer_type', 'essentiality', data=plot_df, saturation=1., showcaps=False, order=order,
        flierprops=Plot.FLIERPROPS, whiskerprops=Plot.WHISKERPROPS, boxprops=Plot.BOXPROPS
    )

    plt.axvline(dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    plt.gcf().set_size_inches(3, 10)
    plt.savefig(f"reports/association_multiple_scatter_cancer_type.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    df = datasets.crispr[[i.startswith('MARCH') for i in datasets.crispr.index]].corr()

    row_colors = plot_df['sensitive'].astype(str).map(dict(zip(plot_df['sensitive'].astype(str).unique(), "rbg")))

    sns.clustermap(df, row_colors=row_colors)

    plt.show()

    #
    x, order = 'FUNDC1', ['None', gene_assoc, gene_extra, f'{gene_assoc} + {gene_extra}']

    g = Plot().plot_multiple(f'{x}_gexp', 'essentiality', 'Institute', plot_df, order=order)

    plt.xlabel(f'{x} (RNA-seq voom)')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_gexp_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    'FIS1', 'DNM1L', 'MFN1', 'MFN2', 'MIEF2', 'MIEF1', 'FUNDC1'

    x = 'FUNDC1'

    order = ['None', gene_assoc, gene_extra, f'{gene_assoc} + {gene_extra}']

    Plot().plot_multiple(f'{x}_prot', 'essentiality', 'Institute', plot_df, order=order)

    plt.xlabel(f'{x} (Proteomics log2 FC)')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_prot_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    df = plot_df[[f'{x}_prot', f'{x}_gexp']].dropna()

    lm = sm.OLS(df[f'{x}_prot'], sm.add_constant(df[[f'{x}_gexp']])).fit()
    print(lm.summary())

    plot_df[f'{x}_prot_residual'] = lm.resid.reindex(plot_df.index).values

    Plot().plot_multiple(f'{x}_prot_residual', 'essentiality', 'Institute', plot_df, order=order)

    plt.xlabel(f'{x} (Proteomics residuals log2 FC)')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_prot_residual_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')
