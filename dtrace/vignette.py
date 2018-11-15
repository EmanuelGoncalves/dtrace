#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import Plot
from associations import Association

if __name__ == '__main__':
    # - Import
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_robust = pd.read_csv('data/drug_lmm_regressions_robust_ic50.csv.gz')

    # -
    drug, gene_assoc, gene_extra = (1946, 'MCL1_5526', 'RS'), 'MARCH5', 'MCL1'

    plot_df = pd.concat([
        datasets.drespo.loc[drug].rename('drug'),

        datasets.crispr.loc[gene_assoc].rename(gene_assoc),
        datasets.crispr.loc[gene_extra].rename(gene_extra),

        datasets.crispr_obj.institute.rename('Institute'),

        datasets.samplesheet.samplesheet['model_name'],
        datasets.samplesheet.samplesheet['cancer_type']
    ], axis=1, sort=False).dropna()

    cbin = pd.concat([plot_df[g].apply(lambda v: g if v < -1 else '') for g in [gene_assoc, gene_extra]], axis=1)
    plot_df['essentiality'] = cbin.apply(lambda v: ' + '.join([i for i in v if i != '']), axis=1).replace('', 'None').values

    for g in [gene_assoc, gene_extra]:
        plot_df[f'{g}_gexp'] = datasets.gexp.loc[g].reindex(plot_df.index).values

    for p in ['FIS1', 'DNM1L', 'MFN1', 'MFN2']:
        plot_df[f'{p}_prot'] = datasets.prot.loc[p].reindex(plot_df.index).values

    #
    dg_lmm = datasets.get_association(lmm_drug, drug, gene_assoc)
    annot_text = f"Beta={dg_lmm.iloc[0]['beta']:.2g}, FDR={dg_lmm.iloc[0]['fdr']:.1e}"

    dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

    #
    g = Plot().plot_corrplot(gene_assoc, 'drug', 'Institute', plot_df, add_hline=True, annot_text=annot_text)

    g.ax_joint.axhline(y=dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    g.set_axis_labels(f'{gene_assoc} (scaled log2 FC)', f'{drug[1]} (ln IC50, {drug[2]})')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/association_drug_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    g = Plot().plot_multiple('drug', 'essentiality', 'Institute', plot_df)

    plt.axvline(dmax, linewidth=.3, color=Plot.PAL_DTRACE[2], ls=':', zorder=0)

    plt.xlabel(f'{drug[1]} (ln IC50, {drug[2]})')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    x, order = gene_assoc, ['None', gene_assoc, gene_extra, f'{gene_assoc} + {gene_extra}']

    g = Plot().plot_multiple(f'{x}_gexp', 'essentiality', 'Institute', plot_df, order=order)

    plt.xlabel(f'{x} (RNA-seq voom)')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_gexp_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')

    #
    x, order = 'FIS1', ['None', gene_assoc, gene_extra, f'{gene_assoc} + {gene_extra}']

    g = Plot().plot_multiple(f'{x}_prot', 'essentiality', 'Institute', plot_df, order=order)

    plt.xlabel(f'{x} (Proteomics log2 FC)')
    plt.ylabel('Essential genes')

    plt.gcf().set_size_inches(3., 1)
    plt.savefig(f"reports/association_multiple_prot_scatter.pdf", bbox_inches='tight', transparent=True)
    plt.close('all')
