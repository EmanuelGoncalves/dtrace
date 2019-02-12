#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Associations import Association
from TargetBenchmark import TargetBenchmark
from Preliminary import Preliminary, CrisprPreliminary


def lmm_single_associations(assoc):
    # - Kinship matrix (random effects)
    k = assoc.kinship(assoc.crisprcb.T)
    m = assoc.get_covariates()

    # - Single feature linear mixed regression
    # Association
    lmm_single = pd.concat([
        assoc.lmm_single_association(assoc.drespo.loc[[d]].T, assoc.crisprcb.T, k=k, m=m) for d in assoc.drespo.index
    ])

    # Multiple p-value correction
    lmm_single = assoc.multipletests_per_drug(lmm_single, field='pval', method=assoc.pval_method)

    # Annotate drug target
    lmm_single = assoc.annotate_drug_target(lmm_single)

    # Annotate association distance to target
    lmm_single = assoc.ppi.ppi_annotation(
        lmm_single, ppi_type='string', ppi_kws=dict(score_thres=900), target_thres=5
    )

    # Sort p-values
    lmm_single = lmm_single.sort_values(['fdr', 'pval'])

    return lmm_single


if __name__ == '__main__':
    # - LMM tests
    assoc = Association()

    lmm_dsingle = lmm_single_associations(assoc)

    # - Preliminary
    pca_crispr = Preliminary.perform_pca(assoc.crisprcb)

    #
    CrisprPreliminary.pairplot_pca_by_rows(pca_crispr, hue=None)
    plt.suptitle('PCA CRISPR-Cas9 (Genes)', y=1.05, fontsize=9)
    plt.savefig('reports/combat_pca_pairplot.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close('all')

    CrisprPreliminary.pairplot_pca_by_columns(
        pca_crispr, hue='institute', hue_vars=assoc.samplesheet.samplesheet['institute']
    )
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/combat_pca_pairplot_samples.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.pairplot_pca_samples_cancertype(pca_crispr, assoc.samplesheet.samplesheet['cancer_type'])
    plt.suptitle('PCA CRISPR-Cas9 (Cell lines)', y=1.05, fontsize=9)
    plt.savefig('reports/combat_pca_pairplot_cancertype.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.corrplot_pcs_growth(pca_crispr, assoc.samplesheet.samplesheet['growth'], 'PC4')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/combat_pca_growth_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    CrisprPreliminary.corrplot_pcs_essentiality(pca_crispr, assoc.crisprcb, 'PC1')
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/combat_pca_essentiality_corrplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    # - Target benchmark
    trg = TargetBenchmark(fdr=.1, lmm_drug=lmm_dsingle)

    #
    trg.signif_essential_heatmap()
    plt.gcf().set_size_inches(1, 1)
    plt.savefig('reports/combat_signif_essential_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.signif_per_screen()
    plt.gcf().set_size_inches(0.75, 1.5)
    plt.savefig('reports/combat_significant_by_screen.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.signif_genomic_markers()
    plt.gcf().set_size_inches(1, 1)
    plt.savefig('reports/combat_signif_genomic_heatmap.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.countplot_drugs()
    plt.gcf().set_size_inches(2, 0.75)
    plt.savefig('reports/combat_association_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.countplot_drugs_significant()
    plt.gcf().set_size_inches(2, 1)
    plt.savefig('reports/combat_association_signif_countplot.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.pichart_drugs_significant()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/combat_association_signif_piechart.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.boxplot_kinobead()
    plt.gcf().set_size_inches(2.5, .75)
    plt.savefig(f'reports/combat_kinobeads.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.beta_histogram()
    plt.gcf().set_size_inches(2, 2)
    plt.savefig('reports/combat_beta_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    trg.pval_histogram()
    plt.gcf().set_size_inches(3, 2)
    plt.savefig('reports/combat_pval_histogram.pdf', bbox_inches='tight', transparent=True)
    plt.close('all')

    for dtype in ['crispr']:
        trg.drugs_ppi(dtype)
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(f'reports/combat_ppi_distance_{dtype}.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')

        trg.drugs_ppi_countplot(dtype)
        plt.gcf().set_size_inches(2.5, 2.5)
        plt.savefig(f'reports/combat_ppi_distance_{dtype}_countplot.pdf', bbox_inches='tight', transparent=True)
        plt.close('all')
