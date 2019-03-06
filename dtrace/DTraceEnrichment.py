#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import argparse
import numpy as np
import pandas as pd
from crispy import SSGSEA, GSEAplot
from dtrace.Associations import Association
from dtrace.DTraceUtils import dpath, logger
from scipy.stats.distributions import hypergeom
from statsmodels.stats.multitest import multipletests


class DTraceEnrichment:
    """
    Gene enrichment analysis class.

    """

    def __init__(
        self, gmts, sig_min_len=5, verbose=0, padj_method="fdr_bh", permutations=0
    ):

        self.verbose = verbose
        self.padj_method = padj_method
        self.permutations = permutations

        self.sig_min_len = sig_min_len

        self.gmts = {f: self.read_gmt(f"{dpath}/pathways/{f}") for f in gmts}

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

    def gsea(self, values, signature):
        return SSGSEA.gsea(values.to_dict(), signature, permutations=self.permutations)

    def gsea_enrichments(self, values, gmt_file):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        if self.verbose > 0 and type(values) == pd.Series:
            logger.log(logging.INFO, f"Values={values.name}")

        ssgsea = []
        for gset in geneset:
            if self.verbose > 1:
                logger.log(logging.INFO, f"Gene-set={gset}")

            gset_len = len({i for i in geneset[gset] if i in values.index})

            e_score, p_value, _, _ = self.gsea(values, geneset[gset])

            ssgsea.append(
                dict(gset=gset, e_score=e_score, p_value=p_value, len=gset_len)
            )

        ssgsea = pd.DataFrame(ssgsea).set_index("gset").sort_values("e_score")

        if self.sig_min_len is not None:
            ssgsea = ssgsea.query(f"len >= {self.sig_min_len}")

        if self.permutations > 0:
            ssgsea["adj.p_value"] = multipletests(
                ssgsea["p_value"], method=self.padj_method
            )[1]

        return ssgsea

    def get_signature(self, gmt_file, signature):
        self.__assert_signature(gmt_file, signature)
        return self.gmts[gmt_file][signature]

    def plot(self, values, gmt_file, signature, vertical_lines=False, shade=False):
        if type(signature) == str:
            signature = self.get_signature(gmt_file, signature)

        e_score, p_value, hits, running_hit = self.gsea(values, signature)

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

    def hypergeom_enrichments(self, sublist, background, gmt_file):
        self.__assert_gmt_file(gmt_file)

        geneset = self.gmts[gmt_file]

        ssgsea_geneset = []
        for gset in geneset:
            if self.verbose > 0:
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
        ssgsea_geneset["adj.p_value"] = multipletests(
            ssgsea_geneset["p_value"], method=self.padj_method
        )[1]

        return ssgsea_geneset

    @staticmethod
    def one_sided_pvalue(escore, escores):
        count = np.sum((escores >= escore) if escore >= 0 else (escores <= escore))

        p_value = 1 / len(escores) if count == 0 else count / len(escores)

        return p_value


class DTraceEnrichmentBSUB(DTraceEnrichment):
    def __init__(
        self,
        dindex,
        dtype,
        gmt,
        sig_min_len=5,
        verbose=0,
        padj_method="fdr_bh",
        permutations=0,
    ):
        self.assoc = Association(dtype="ic50", load_associations=True)

        self.gmt = gmt

        self.dindex = dindex
        self.dtype = dtype
        self.gene_values = self.load_gene_values()

        DTraceEnrichment.__init__(
            self,
            gmts=[gmt],
            sig_min_len=sig_min_len,
            verbose=verbose,
            padj_method=padj_method,
            permutations=permutations,
        )

    def load_gene_values(self):
        if self.dtype == "GExp":
            return self.assoc.gexp.T.loc[self.dindex]

        elif self.dtype == "CRISPR":
            return self.assoc.crispr.T.loc[self.dindex]

        elif self.dtype == "Drug-CRISPR":
            return self.assoc.build_association_matrix(self.assoc.lmm_drug_crispr).loc[
                self.dindex
            ]

        else:
            assert False, f"{self.dtype} type not supported"


if __name__ == "__main__":
    # Command line args parser
    parser = argparse.ArgumentParser(description="Run ssGSEA with bsub")

    parser.add_argument("-dtype", nargs="?")
    parser.add_argument("-dindex", nargs="?")
    parser.add_argument("-gmt", nargs="?")
    parser.add_argument("-permutations", nargs="?", default="0")
    parser.add_argument("-len", nargs="?", default="5")
    parser.add_argument("-padj", nargs="?", default="fdr_bh")

    args = parser.parse_args()

    logger.log(logging.INFO, args)

    # Execute enrichment with specified arguments
    ssgsea = DTraceEnrichmentBSUB(
        dtype=args.dtype,
        dindex=args.dindex,
        gmt=args.gmt,
        permutations=int(args.permutations),
        sig_min_len=int(args.len),
        padj_method=args.padj,
    )
    ssgsea_gmt_index = ssgsea.gsea_enrichments(ssgsea.gene_values, ssgsea.gmt)
    ssgsea_gmt_index.to_csv(
        f"{dpath}/ssgsea/{ssgsea.dtype}_{ssgsea.gmt}_{ssgsea.dindex}.csv.gz",
        compression="gzip",
    )
