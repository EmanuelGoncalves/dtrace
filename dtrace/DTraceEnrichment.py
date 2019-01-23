#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import pandas as pd
from crispy import SSGSEA, GSEAplot
from statsmodels.stats.multitest import multipletests


class DTraceEnrichment:
    def __init__(self, data_dir='data/pathways/'):
        self.data_dir = data_dir

        self.gmts = {
            f: self.read_gmt(f'{self.data_dir}/{f}') for f in os.listdir(self.data_dir) if f.endswith('.gmt')
        }

    def __assert_gmt_file(self, gmt_file):
        assert gmt_file in self.gmts, f'{gmt_file} not in gmt files: {self.gmts.keys()}'

    def __assert_signature(self, gmt_file, signature):
        self.__assert_gmt_file(gmt_file)

        assert signature in self.gmts[gmt_file], f'{signature} not in {gmt_file}'

    @staticmethod
    def read_gmt(file_path):
        with open(file_path) as f:
            signatures = {l.split('\t')[0]: set(l.strip().split('\t')[2:]) for l in f.readlines()}
        return signatures

    @staticmethod
    def gsea(values, signature, permutations):
        return SSGSEA.gsea(values.to_dict(), signature, permutations=permutations)

    def gsea_enrichments(self, values, gmt_file, permutations=0, padj_method='fdr_bh', verbose=0):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        ssgsea_geneset = []
        for gset in geneset:
            if verbose > 0:
                print(f'[INFO] Gene-set: {gset}')

            gset_len = len({i for i in geneset[gset] if i in values.index})

            e_score, p_value, _, _ = self.gsea(values, geneset[gset], permutations=permutations)

            ssgsea_geneset.append(dict(gset=gset, e_score=e_score, p_value=p_value, len=gset_len))

        gsea_hallmarks = pd.DataFrame(ssgsea_geneset).set_index('gset').sort_values('e_score')

        if permutations > 0:
            gsea_hallmarks['fdr'] = multipletests(gsea_hallmarks['p_value'], method=padj_method)[1]

        return gsea_hallmarks

    def get_signature(self, gmt_file, signature):
        self.__assert_signature(gmt_file, signature)
        return self.gmts[gmt_file][signature]

    def plot(self, values, gmt_file, signature, permutations=0, vertical_lines=False, shade=False):
        if type(signature) == str:
            signature = self.get_signature(gmt_file, signature)

        e_score, p_value, hits, running_hit = self.gsea(values, signature, permutations=permutations)

        ax = GSEAplot.plot_gsea(
            hits, running_hit, dataset=values.to_dict(), vertical_lines=vertical_lines, shade=shade
        )

        return ax
