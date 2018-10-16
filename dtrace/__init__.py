#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import pandas as pd


# - DATA
# NUMBER OF MUTATIONS
WES_COUNT = 'data/WES_variants.csv'

# METHYLATION
METHYLATION_GENE_PROMOTER = 'data/gdsc/methylation/methy_beta_gene_promoter.csv'

# GENE-EXPRESSION
RNASEQ_VOOM = 'data/gdsc/gene_expression/merged_voom_preprocessed.csv'

# MUTATION BURDEN
MUTATION_BURDERN = 'data/mutation_burden.csv'

# COPY-NUMBER
COPYNUMBER = 'data/crispy_copy_number_gene_snp.csv'

# - ASSOCIATIONS
DRUG_BETAS_CORR = 'data/drug_beta_correlation.csv'

# - tSNE components
DRUG_BETAS_TSNE = 'data/drug_beta_tsne.csv'


# - GETS
def get_wes():
    df = pd.read_csv(WES_COUNT)

    return df


def get_copynumber(round=True):
    df = pd.read_csv(COPYNUMBER, index_col=0)

    if round:
        df = df.dropna()
        df = df.astype(int)

    return df


def get_geneexpression():
    df = pd.read_csv(RNASEQ_VOOM, index_col=0)
    return df


def read_gmt(file_path):
    with open(file_path) as f:
        signatures = {l.split('\t')[0]: set(l.strip().split('\t')[2:]) for l in f.readlines()}

    return signatures
