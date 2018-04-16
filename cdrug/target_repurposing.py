#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from crispy.regression.linear import lr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def lm_drug_crispr(xs, ys, ws):
    print('CRISPR genes: %d, Drug: %d' % (len(set(xs.columns)), len(set(ys.columns))))

    # # Standardize xs
    # xs = pd.DataFrame(StandardScaler().fit_transform(xs), index=xs.index, columns=xs.columns)

    # Regression
    res = lr(xs, ys, ws)

    # - Export results
    res_df = pd.concat([res[i].unstack().rename(i) for i in res], axis=1).reset_index()

    res_df = res_df.assign(f_fdr=multipletests(res_df['f_pval'], method='fdr_bh')[1])
    res_df = res_df.assign(lr_fdr=multipletests(res_df['lr_pval'], method='fdr_bh')[1])

    return res_df


def plot_corrplot(plot_df):
    g = sns.jointplot(
        'x', 'y', data=plot_df, kind='reg', space=0, color=cdrug.PAL_DBGD[0], annot_kws=dict(stat='r'),
        marginal_kws={'kde': False}, joint_kws={'scatter_kws': {'edgecolor': 'w', 'lw': .3, 's': 12}, 'line_kws': {'lw': 1.}}
    )

    g.ax_joint.axhline(0, ls='-', lw=0.1, c=cdrug.PAL_DBGD[0])
    g.ax_joint.axvline(0, ls='-', lw=0.1, c=cdrug.PAL_DBGD[0])

    return g


if __name__ == '__main__':
    # - Import sample data
    # Samplesheet
    ss = pd.read_csv(cdrug.SAMPLESHEET_FILE, index_col=0).dropna(subset=['Cancer Type'])

    # Growth rate
    growth = pd.read_csv(cdrug.GROWTHRATE_FILE, index_col=0)

    # - Import data
    d_sheet = pd.read_csv(cdrug.DRUGSHEET_FILE, sep='\t', index_col=0)
    suitable_drugs = set(d_sheet[(d_sheet['Web Release'] == 'Y') & (d_sheet['Suitable for publication'] == 'Y')].index)

    # - Import drug repurposing
    d_repur_map = pd.read_csv('data/repo_drug_map.txt', sep='\t', index_col=0).dropna()['Drug ID']

    d_repur = pd.read_csv('data/repo_list.txt', sep='\t')
    d_repur = pd.DataFrame([{
        'tissue': t, 'genomic': f, 'gene': g, 'drug': d, 'drug_id': d_repur_map[d]
    } for t, f, g, ds, _ in d_repur.values for d in ds.split('|') if d in d_repur_map.index])

    # CRISPR gene-level corrected fold-changes
    crispr_bagel = pd.read_csv(cdrug.CRISPR_GENE_BAGEL, index_col=0, sep='\t').dropna()
    crispr_binary = pd.read_csv(cdrug.CRISPR_GENE_BINARY, index_col=0, sep='\t').dropna()

    # Drug response
    d_response = pd.read_csv(cdrug.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr_bagel).intersection(ss.index).intersection(growth.index))
    d_response, crispr_scaled, crispr_binary = d_response[samples], crispr_bagel[samples], crispr_binary[samples]
    print('Samples: %d' % len(samples))

    # - Linear regression: drug ~ crispr + tissue
    lr_df = []

    # drug_name, drug_id, gene, feature, tissue = 'VORINOSTAT', 1012.0, 'HDAC1', 'PIK3R1_mut', 'Ovarian Carcinoma'
    for drug_name, drug_id, gene, feature, tissue in d_repur.values:

        if drug_id in d_response.index and gene in crispr_binary.index:
            tissue_samples = set(ss[ss['Cancer Type'] == tissue].index).intersection(samples)

            x = crispr_binary.loc[gene, tissue_samples]

            for (d_name, d_screen), y in d_response.loc[drug_id, tissue_samples].iterrows():
                df = pd.concat([x.rename('x'), y.rename('y')], axis=1).dropna()

                if df.shape[0] > 2 and df['x'].sum() > 2:

                    stat, pval = ttest_ind(df.query('x == 0')['x'], df.query('x == 1')['y'], equal_var=False)

                    delta_ic50 = df.query('x == 1')['y'].mean() - df.query('x == 0')['y'].mean()

                    res = {
                        'tissue': tissue, 'drug_id': drug_id, 'drug_name': d_name,
                        'gene': gene, 'genomic': feature, 'version': d_screen,
                        'stat': stat, 'pval': pval, 'essential': df['x'].sum(0),
                        'delta_ic50': delta_ic50
                    }

                    lr_df.append(res)

                    print('# -- Drug repurposing')
                    print(drug_name, drug_id, gene, feature, tissue)
                    print('#(Essential cell lines) = {}'.format(sum(x)))
                    print(d_name, d_screen)

                print('\n')

    lr_df = pd.DataFrame(lr_df)
    lr_df = lr_df.assign(fdr=multipletests(lr_df['pval'], method='fdr_bh')[1])

    # - Export table
    lr_df.query('fdr < 0.05').sort_values('delta_ic50').to_csv('data/repo_associations.txt', sep='\t', index=False)
    lr_df.query('fdr < 0.05').sort_values('delta_ic50').to_clipboard(index=False)

    # - Import
    lr_df = pd.read_csv('data/repo_associations.txt', sep='\t')
    print(lr_df.query('fdr < 0.05').sort_values('delta_ic50'))

    # - Plot Drug ~ CRISPR corrplot
    idx = 29
    d_id, d_name, d_screen, gene, tissue, fdr, genomic = lr_df.loc[idx, ['drug_id', 'drug_name', 'version', 'gene', 'tissue', 'fdr', 'genomic']].values

    tissue_samples = set(ss[ss['Cancer Type'] == tissue].index).intersection(samples)

    pal, order = dict(zip(*(['No', 'Yes'], cdrug.PAL_DBGD))), ['No', 'Yes']

    plot_df = pd.concat([
        crispr_binary.loc[gene, tissue_samples].rename('x'),
        d_response.loc[(d_id, d_name, d_screen), tissue_samples].rename('y')
    ], axis=1).dropna()
    plot_df = plot_df.replace({'x': {0: 'No', 1: 'Yes'}})

    sns.boxplot('x', 'y', data=plot_df, palette=pal, order=order, fliersize=0)
    sns.swarmplot('x', 'y', data=plot_df, palette=pal, order=order, linewidth=.1, edgecolor='white', alpha=.8, size=3)

    plt.axhline(0, c=cdrug.PAL_DBGD[0], lw=.1, alpha=.8)
    plt.xlabel('{} (BAGEL essential)'.format(gene))
    plt.ylabel('{} ({} ln IC50; {})'.format(d_name, d_screen, d_sheet.loc[d_id, 'Target name']))
    plt.title('{} \n {} \n FDR = {:.2}'.format(tissue, genomic, fdr))

    plt.gcf().set_size_inches(1., 3.)
    plt.savefig('reports/drug_repo_boxplots.png', bbox_inches='tight', dpi=600)
    plt.close('all')
