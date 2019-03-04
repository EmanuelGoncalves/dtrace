#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import os
import logging
import numpy as np
import pandas as pd
from dtrace import logger, dpath
from crispy import SSGSEA, GSEAplot
from scipy.stats.distributions import hypergeom
from statsmodels.stats.multitest import multipletests


class DTraceEnrichment:
    """
    Gene enrichment analysis class.

    """

    def __init__(self):
        self.data_dir = f"{dpath}/pathways/"

        self.gmts = {
            f: self.read_gmt(f"{self.data_dir}/{f}")
            for f in os.listdir(self.data_dir)
            if f.endswith(".gmt")
        }

    def __assert_gmt_file(self, gmt_file):
        assert gmt_file in self.gmts, f"{gmt_file} not in gmt files: {self.gmts.keys()}"

    def __assert_signature(self, gmt_file, signature):
        self.__assert_gmt_file(gmt_file)
        assert signature in self.gmts[gmt_file], f"{signature} not in {gmt_file}"

    @staticmethod
    def read_gmt(file_path):
        with open(file_path) as f:
            signatures = {
                l.split("\t")[0]: set(l.strip().split("\t")[2:]) for l in f.readlines()
            }
        return signatures

    @staticmethod
    def gsea(values, signature, permutations):
        return SSGSEA.gsea(values.to_dict(), signature, permutations=permutations)

    def gsea_enrichments(
        self, values, gmt_file, permutations=0, padj_method="fdr_bh", verbose=0, min_len=None
    ):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        if verbose > 0 and type(values) == pd.Series:
            logger.log(logging.INFO, f"Values={values.name}")

        ssgsea_geneset = []
        for gset in geneset:
            if verbose > 0:
                logger.log(logging.INFO, f"Gene-set={gset}")

            gset_len = len({i for i in geneset[gset] if i in values.index})

            e_score, p_value, _, _ = self.gsea(
                values, geneset[gset], permutations=permutations
            )

            ssgsea_geneset.append(
                dict(gset=gset, e_score=e_score, p_value=p_value, len=gset_len)
            )

        gsea_hallmarks = (
            pd.DataFrame(ssgsea_geneset).set_index("gset").sort_values("e_score")
        )

        if permutations > 0:
            gsea_hallmarks["fdr"] = multipletests(
                gsea_hallmarks["p_value"], method=padj_method
            )[1]

        if min_len is not None:
            gsea_hallmarks = gsea_hallmarks.query(f"len >= {min_len}")

        return gsea_hallmarks

    def get_signature(self, gmt_file, signature):
        self.__assert_signature(gmt_file, signature)
        return self.gmts[gmt_file][signature]

    def plot(
        self,
        values,
        gmt_file,
        signature,
        permutations=0,
        vertical_lines=False,
        shade=False,
    ):
        if type(signature) == str:
            signature = self.get_signature(gmt_file, signature)

        e_score, p_value, hits, running_hit = self.gsea(
            values, signature, permutations=permutations
        )

        ax = GSEAplot.plot_gsea(
            hits,
            running_hit,
            dataset=values.to_dict(),
            vertical_lines=vertical_lines,
            shade=shade,
        )

        return ax

    @staticmethod
    def hypergeom_test(signature, background, sublist):
        """
        Performs hypergeometric test

        Arguements:
                signature: {string} - Signature IDs
                background: {string} - Background IDs
                sublist: {string} - Sub-set IDs

        # hypergeom.sf(x, M, n, N, loc=0)
        # M: total number of objects,
        # n: total number of type I objects
        # N: total number of type I objects drawn without replacement

        """
        pvalue = hypergeom.sf(
            len(sublist.intersection(signature)),
            len(background),
            len(background.intersection(signature)),
            len(sublist),
        )

        intersection = len(sublist.intersection(signature))

        return pvalue, intersection

    def hypergeom_enrichments(
        self, sublist, background, gmt_file, padj_method="fdr_bh", verbose=0
    ):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        ssgsea_geneset = []
        for gset in geneset:
            if verbose > 0:
                logger.log(logging.INFO, f"Gene-set={gset}")

            p_value, intersection = self.hypergeom_test(
                signature=geneset[gset], background=background, sublist=sublist
            )

            ssgsea_geneset.append(
                dict(
                    gset=gset,
                    p_value=p_value,
                    len_sig=len(geneset[gset]),
                    len_intersection=intersection,
                )
            )

        ssgsea_geneset = (
            pd.DataFrame(ssgsea_geneset).set_index("gset").sort_values("p_value")
        )
        ssgsea_geneset["fdr"] = multipletests(
            ssgsea_geneset["p_value"], method=padj_method
        )[1]

        return ssgsea_geneset

    @staticmethod
    def one_sided_pvalue(escore, escores):
        count = np.sum((escores >= escore) if escore >= 0 else (escores <= escore))

        p_value = 1 / len(escores) if count == 0 else count / len(escores)

        return p_value
