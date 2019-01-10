#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from crispy import Utils, SSGSEA
from statsmodels.stats.multitest import multipletests


class DTraceEnrichment:
    def __init__(self, data_dir='data/pathways/'):
        self.data_dir = data_dir

    @staticmethod
    def read_gmt(file_path):
        with open(file_path) as f:
            signatures = {l.split('\t')[0]: set(l.strip().split('\t')[2:]) for l in f.readlines()}
        return signatures

    @staticmethod
    def gsea(values, signature, permutations):
        return SSGSEA.gsea(values.to_dict(), signature, permutations=permutations)

    def gsea_enrichments(self, values, gmt_file, permutations=0, padj_method='fdr_bh'):
        geneset = self.read_gmt(f'{self.data_dir}/{gmt_file}')

        ssgsea_geneset = []
        for gset in geneset:
            print(f'[INFO] Gene-set: {gset}')

            gset_len = len({i for i in geneset[gset] if i in values.index})

            e_score, p_value, _, _ = self.gsea(values, geneset[gset], permutations=permutations)

            ssgsea_geneset.append(dict(gset=gset, e_score=e_score, p_value=p_value, len=gset_len))

        gsea_hallmarks = pd.DataFrame(ssgsea_geneset).set_index('gset').sort_values('e_score')

        if permutations > 0:
            gsea_hallmarks['fdr'] = multipletests(gsea_hallmarks['p_value'], method=padj_method)[1]

        return gsea_hallmarks

    def get_signature(self, gmt_file, signature):
        geneset = self.read_gmt(f'{self.data_dir}/{gmt_file}')

        assert signature in geneset, f'{signature} not in {gmt_file}'

        return geneset[signature]
