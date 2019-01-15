import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from DTracePlot import DTracePlot
from Associations import Association


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


if __name__ == '__main__':
    # - Import
    datasets = Association(dtype_drug='ic50')

    lmm_drug = pd.read_csv('data/drug_lmm_regressions_ic50.csv.gz')
    lmm_gexp = pd.read_csv('data/drug_lmm_regressions_ic50_gexp.csv.gz')

    # -
    lmm = pd.concat([
        lmm_drug.set_index(['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GeneSymbol']).add_prefix('CRISPR_'),
        lmm_gexp.set_index(['DRUG_ID', 'DRUG_NAME', 'VERSION', 'GeneSymbol']).add_prefix('GExp_'),
    ], axis=1, sort=False).dropna()

    lmm_signif = lmm.query('CRISPR_fdr < 0.1 & GExp_fdr < 0.1')

    # -
    ax = plt.gca()

    ax.hexbin(lmm['CRISPR_beta'], lmm['GExp_beta'], cmap='Blues', gridsize=60, bins='log')

    plt.scatter(
        lmm_signif['CRISPR_beta'], lmm_signif['GExp_beta'], s=5, color='#de2d26', edgecolors='white', lw=.3, marker='o'
    )

    ax.axhline(0, c='white', lw=.3, ls='-')
    ax.axvline(0, c='white', lw=.3, ls='-')

    ax.set_xlabel('CRISPR beta')
    ax.set_ylabel('GExp beta')
    ax.set_title('LMM Drug-response model')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/lmm_gexp_crispr_hexbin.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # -
    targets = pd.Series([DTracePlot.PAL_DTRACE[i] for i in [0, 2, 3]], index=['MCL1', 'BCL2', 'BCL2L1'])

    plot_df = lmm[lmm['CRISPR_DRUG_TARGETS'].isin(targets.index)].reset_index()
    plot_df = plot_df[plot_df['GeneSymbol'] == plot_df['CRISPR_DRUG_TARGETS']]

    ax = plt.gca()

    for target, df in plot_df.groupby('CRISPR_DRUG_TARGETS'):
        ax.scatter(
            df['CRISPR_beta'], df['GExp_beta'], label=target, color=targets[target], edgecolor='white', lw=.3, zorder=1
        )

        df_signif = df.query('(CRISPR_fdr < .1) & (GExp_fdr < .1)')
        df_signif_any = df.query('(CRISPR_fdr < .1) | (GExp_fdr < .1)')

        if df_signif.shape[0] > 0:
            ax.scatter(df_signif['CRISPR_beta'], df_signif['GExp_beta'], color='white', marker='$X$', lw=.3, label=None, zorder=1)

        elif df_signif_any.shape[0] > 0:
            ax.scatter(df_signif_any['CRISPR_beta'], df_signif_any['GExp_beta'], color='white', marker='$/$', lw=.3, label=None, zorder=1)

    ax.axhline(0, ls='-', lw=0.1, c=DTracePlot.PAL_DTRACE[1], zorder=0)
    ax.axvline(0, ls='-', lw=0.1, c=DTracePlot.PAL_DTRACE[1], zorder=0)

    ax.legend(loc=3, frameon=False, prop={'size': 6})

    ax.set_xlabel('CRISPR beta')
    ax.set_ylabel('GExp beta')

    ax.set_title('LMM Drug-response model')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/lmm_scatter_BCL_inhibitors.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # -
    gene = 'MCL1'

    drugs = {tuple(i) for i in lmm_drug[lmm_drug['DRUG_TARGETS'] == 'MCL1'][['DRUG_ID', 'DRUG_NAME', 'VERSION']].values}

    f, axs = plt.subplots(len(drugs), 1, sharex='all')

    for i, drug in enumerate(drugs):
        plot_df = pd.concat([
            datasets.cn.loc[gene].rename('Copy-number'),
            datasets.drespo.loc[drug].rename('Drug-response')
        ], axis=1, sort=False).dropna()

        axs[i].scatter(
            rand_jitter(plot_df['Copy-number']), plot_df['Drug-response'], color=DTracePlot.PAL_DTRACE[2], edgecolor='white', lw=.3,
            s=10, alpha=.7
        )

        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        axs[i].axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

        axs[i].set_ylabel(f'{drug[1]}\n({drug[0]} ln IC50)')

        if i == (len(drugs) - 1):
            axs[i].set_xlabel(f'{gene} (copy-number)')

        axs[i].xaxis.set_major_locator(plticker.MultipleLocator(base=2.))

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    plt.gcf().set_size_inches(3, 1.5 * len(drugs))
    plt.savefig(f'reports/mcl_cn_drug_scatter.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # -
    drug, feature = (2323, 'A-1155463', 'RS'), 'BCL2L1'

    plot_df = pd.concat([
        datasets.drespo.loc[drug].rename(drug[1]),
        datasets.crispr_obj.institute.rename('Institute'),

        datasets.samplesheet.samplesheet['cancer_type'],
        datasets.samplesheet.samplesheet['model_name'],

        datasets.crispr.loc[feature].rename(f'CRISPR_{feature}'),
        datasets.gexp.loc[feature].rename(f'Gexp_{feature}'),
    ], axis=1, sort=False).dropna()

    for x in [f'CRISPR_{feature}', f'Gexp_{feature}']:
        dmax = np.log(datasets.drespo_obj.maxconcentration[drug])

        g = DTracePlot.plot_corrplot(x, drug[1], 'Institute', plot_df, annot_text='', add_hline=False, add_vline=False)

        g.ax_joint.axhline(y=dmax, linewidth=.3, color=DTracePlot.PAL_DTRACE[2], ls=':', zorder=0)

        g.set_axis_labels(x, f'{drug[1]} (ln IC50)')

        plt.gcf().set_size_inches(1.5, 1.5)
        plt.savefig(f'reports/lmm_scatter_{x}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

    g = DTracePlot.plot_corrplot(f'CRISPR_{feature}', f'Gexp_{feature}', 'Institute', plot_df, annot_text='', add_hline=False, add_vline=False)

    g.set_axis_labels(f'CRISPR {feature}', f'Gexp {feature}')

    plt.gcf().set_size_inches(1.5, 1.5)
    plt.savefig(f'reports/lmm_scatter_{feature}.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')
