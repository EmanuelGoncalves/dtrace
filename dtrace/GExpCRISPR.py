import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from DTracePlot import DTracePlot
from Associations import Association


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
    lmm_signif[['CRISPR_beta', 'GExp_beta']]

    # -
    ax = plt.gca()

    ax.hexbin(lmm['CRISPR_beta'], lmm['GExp_beta'], cmap='viridis', gridsize=50, bins='log')

    plt.scatter(
        lmm_signif['CRISPR_beta'], lmm_signif['GExp_beta'], s=15, color=DTracePlot.PAL_DBGD[1], edgecolors='white',
        lw=.5, marker='x'
    )

    ax.axhline(0, c='white', lw=.3, ls='-')
    ax.axvline(0, c='white', lw=.3, ls='-')

    ax.set_xlabel('CRISPR beta')
    ax.set_ylabel('GExp beta')
    ax.set_title('LMM Drug models')

    plt.gcf().set_size_inches(2, 2)
    plt.savefig(f'reports/lmm_gexp_crispr_hexbin.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    #
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
