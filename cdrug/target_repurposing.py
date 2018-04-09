#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import cdrug
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
    crispr = pd.read_csv(cdrug.CRISPR_GENE_FC_CORRECTED, index_col=0, sep='\t').dropna()
    crispr_scaled = cdrug.scale_crispr(crispr)
    # crispr_scaled = pd.read_csv(cdrug.CRISPR_GENE_BINARY, sep='\t', index_col=0).dropna().astype(int)

    # Drug response
    d_response = pd.read_csv(cdrug.DRUG_RESPONSE_FILE, index_col=[0, 1, 2], header=[0, 1])
    d_response.columns = d_response.columns.droplevel(0)

    # - Overlap
    samples = list(set(d_response).intersection(crispr_scaled).intersection(ss.index).intersection(growth.index))
    d_response, crispr_scaled = d_response[samples], crispr_scaled[samples]
    print('Samples: %d' % len(samples))

    # - Linear regression: drug ~ crispr + tissue
    lr_df = []
    for d, d_id, g, f, t in d_repur.values:
        if d_id in d_response.index and g in crispr_scaled.index:
            print(d, d_id, g, f, t)

            y = d_response.loc[d_id, samples].T.dropna()
            x = crispr_scaled.loc[g, y.index].rename(g).to_frame()

            lm_res = lr(x, y)

            for d_name, s_version in lm_res['beta'].columns:
                res = {'tissue': t, 'drug_id': d_id, 'drug_name': d_name, 'gene': g, 'genomic': f, 'version': s_version}

                for s in ['beta', 'f_pval', 'r2']:
                    res[s] = lm_res[s].loc[g, (d_name, s_version)]

                lr_df.append(res)

    lr_df = pd.DataFrame(lr_df)
    lr_df = lr_df.assign(f_fdr=multipletests(lr_df['f_pval'], method='fdr_bh')[1])
    print(lr_df.sort_values('f_fdr'))

    # - Plot Drug ~ CRISPR corrplot
    idx = 43
    d_id, d_name, d_screen, gene, tissue = lr_df.loc[idx, ['drug_id', 'drug_name', 'version', 'gene', 'tissue']].values

    plot_df = pd.concat([x.rename('x'), y.rename('y'), ss['Cancer Type']], axis=1).dropna()

    g = plot_corrplot(plot_df)
    sns.regplot(
        x='x', y='y', data=plot_df[plot_df['Cancer Type'] == tissue], color=cdrug.PAL_DBGD[1], truncate=True, fit_reg=False, ax=g.ax_joint,
        scatter_kws={'s': 20, 'edgecolor': 'w', 'linewidth': .1, 'alpha': .8}, label=tissue
    )

    g.set_axis_labels('{} (log10 FC)'.format(gene), '{} (ln IC50)'.format(d_name))

    plt.legend()

    plt.gcf().set_size_inches(3., 3.)
    plt.savefig('reports/crispr_drug_corrplot.png', bbox_inches='tight', dpi=600)
    plt.close('all')
