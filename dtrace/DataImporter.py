#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import pydot
import igraph
import logging
import warnings
import numpy as np
import pandas as pd
import crispy as cy
import pkg_resources
from sklearn.decomposition import PCA
from dtrace.DTracePlot import DTracePlot


dpath = pkg_resources.resource_filename("dtrace", "data/")


class DataPCA:
    @staticmethod
    def perform_pca(dataframe, n_components=10):
        df = dataframe.T.fillna(dataframe.T.mean()).T

        pca = dict()

        for by in ["row", "column"]:
            pca[by] = dict()

            df_by = df.T.copy() if by != "row" else df.copy()

            df_by = df_by.subtract(df_by.mean())

            pcs_labels = list(map(lambda v: f"PC{v + 1}", range(n_components)))

            pca[by]["pca"] = PCA(n_components=n_components).fit(df_by)
            pca[by]["vex"] = pd.Series(
                pca[by]["pca"].explained_variance_ratio_, index=pcs_labels
            )
            pca[by]["pcs"] = pd.DataFrame(
                pca[by]["pca"].transform(df_by), index=df_by.index, columns=pcs_labels
            )

        return pca


class Sample:
    """
    Import module that handles the sample list (i.e. list of cell lines) and their descriptive information.

    """

    def __init__(
        self,
        index="model_id",
        samplesheet_file="meta/ModelList_20191106.csv",
        growthrate_file="meta/GrowthRates_v1.3.0_20190222.csv",
        samples_origin="meta/SamplesOrigin_20191106.csv",
    ):
        self.index = index

        # Import samplesheet
        self.samplesheet = (
            pd.read_csv(f"{dpath}/{samplesheet_file}")
            .dropna(subset=[self.index])
            .set_index(self.index)
        )

        # Add growth information
        self.growth = pd.read_csv(f"{dpath}/{growthrate_file}")
        self.samplesheet["growth"] = (
            self.growth.groupby(self.index)["GROWTH_RATE"]
            .mean()
            .reindex(self.samplesheet.index)
        )

        # Add institute of origin
        self.institute = pd.read_csv(
            f"{dpath}/{samples_origin}", header=None, index_col=0
        ).iloc[:, 0]
        self.samplesheet["institute"] = self.institute.reindex(self.samplesheet.index)

    def growth_corr(self, df):
        growth = self.samplesheet["growth"].dropna()
        samples = list(set(growth.index).intersection(df.columns))

        logging.getLogger("DTrace").info(
            f"Correlation with growth using {len(samples)} cell lines"
        )

        corr = df[samples].T.corrwith(growth[samples])
        corr = corr.sort_values().rename("pearson").reset_index()

        return corr

    @staticmethod
    def load_coread_info(info_file="meta/meta_coread.csv"):
        return pd.read_csv(f"{dpath}/{info_file}", index_col=0)

    @staticmethod
    def load_brca_info(info_file="meta/meta_brca.csv"):
        return pd.read_csv(f"{dpath}/{info_file}", index_col=0)


class DrugResponse:
    """
    Importer module for drug-response measurements acquired at Sanger Institute GDSC (https://cancerrxgene.org).

    """

    SAMPLE_COLUMNS = ["model_id"]
    DRUG_COLUMNS = ["DRUG_ID", "DRUG_NAME", "VERSION"]

    def __init__(
        self,
        drugresponse_file="drug/DrugResponse_IC50_v1.5.1_20191108.csv",
        drugmaxconcentration_file="drug/DrugResponse_MaxC_v1.5.1_20191108.csv",
    ):
        self.drugresponse_file = drugresponse_file
        self.drugmaxconcentration_file = drugmaxconcentration_file

        self.drugsheet = self.get_drugsheet()

        # Import and Merge drug response matrix (IC50)
        self.drugresponse = pd.read_csv(f"{dpath}/{self.drugresponse_file}", index_col=[0, 1, 2])

        # Drug max concentration
        self.maxconcentration = pd.read_csv(f"{dpath}/{self.drugmaxconcentration_file}", index_col=[0, 1, 2]).iloc[:, 0]

    def perform_pca(self, n_components=10, subset=None):
        df = DataPCA.perform_pca(
            self.filter(subset=subset), n_components=n_components
        )

        for by in df:
            df[by]["pcs"].round(5).to_csv(f"{dpath}/PCA_drug_{by}_pcs.csv")
            df[by]["vex"].round(5).to_csv(f"{dpath}/PCA_drug_{by}_vex.csv")

    @staticmethod
    def import_pca():
        pca = {}

        for by in ["row", "column"]:
            pca[by] = {}

            pca[by]["pcs"] = pd.read_csv(
                f"{dpath}/PCA_drug_{by}_pcs.csv",
                index_col=[0, 1, 2] if by == "row" else 0,
            )
            pca[by]["vex"] = pd.read_csv(
                f"{dpath}/PCA_drug_{by}_vex.csv", index_col=0, header=None
            ).iloc[:, 0]

        return pca

    def perform_growth_corr(self, subset=None):
        ss = Sample()

        corr = ss.growth_corr(self.filter(subset=subset))
        corr.round(5).to_csv(
            f"{dpath}/growth_drug_correlation.csv", index=False
        )

        return corr

    def perform_number_responses(self, resp_thres=0.5, subset=None):
        df = self.filter(subset=subset)

        num_resp = {
            d: np.sum(
                df.loc[d].dropna() < np.log(self.maxconcentration[d] * resp_thres)
            )
            for d in df.index
        }
        num_resp = pd.Series(num_resp).reset_index()
        num_resp.columns = ["DRUG_ID", "DRUG_NAME", "VERSION", "n_resp"]

        num_resp.to_csv(f"{dpath}/number_responses_drug.csv", index=False)

        return num_resp

    @staticmethod
    def get_drugsheet(drugsheet_file="meta/DrugSheet_20191106.csv"):
        return pd.read_csv(f"{dpath}/{drugsheet_file}", index_col=0)

    @classmethod
    def get_drugtargets(cls, by="id"):
        if by == "id":
            d_targets = cls.get_drugsheet()["Gene Target"].dropna().to_dict()

        else:
            d_targets = (
                cls.get_drugsheet()
                .groupby("Name")["Gene Target"]
                .first()
                .dropna()
                .to_dict()
            )

        d_targets = {k: {t.strip() for t in d_targets[k].split(";")} for k in d_targets}

        return d_targets

    def get_data(self):
        return self.drugresponse.copy()

    def filter(
        self,
        subset=None,
        min_events=3,
        min_meas=0.75,
        max_c=0.5,
        filter_max_concentration=True,
        filter_combinations=True,
    ):
        # Drug max screened concentration
        df = self.get_data()
        d_maxc = np.log(self.maxconcentration * max_c)

        # - Filters
        # Subset samples
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by mininum number of observations
        df = df[df.count(1) > (df.shape[1] * min_meas)]

        # Filter by max screened concentration
        if filter_max_concentration:
            df = df[[sum(df.loc[i] < d_maxc.loc[i]) >= min_events for i in df.index]]

        # Filter combinations
        if filter_combinations:
            df = df[[" + " not in i[1] for i in df.index]]

        return self.get_data().loc[df.index, df.columns]

    def is_in_druglist(self, drug_ids):
        return np.all([d in self.drugsheet.index for d in drug_ids])

    def is_same_drug(self, drug_id_1, drug_id_2):
        """
        Check if 2 Drug IDs are represent the same drug by checking if Name or Synonyms are the same.

        :param drug_id_1:
        :param drug_id_2:
        :return: Bool
        """

        if drug_id_1 not in self.drugsheet:
            warnings.warn("Drug ID {} not in drug list".format(drug_id_1))
            return False

        if drug_id_2 not in self.drugsheet:
            warnings.warn("Drug ID {} not in drug list".format(drug_id_2))
            return False

        drug_names = {d: self.get_drug_names(d) for d in [drug_id_1, drug_id_2]}

        return len(drug_names[drug_id_1].intersection(drug_names[drug_id_2])) > 0

    def get_drug_names(self, drug_id):
        """
        From a Drug ID get drug Name and Synonyms.

        :param drug_id:
        :return:
        """

        if drug_id not in self.drugsheet.index:
            logging.getLogger("DTrace").info(f"{drug_id} Drug ID not in drug list")
            return None

        drug_name = [self.drugsheet.loc[drug_id, "Name"]]

        drug_synonyms = self.drugsheet.loc[drug_id, "Synonyms"]
        drug_synonyms = (
            [] if str(drug_synonyms).lower() == "nan" else drug_synonyms.split(", ")
        )

        return set(drug_name + drug_synonyms)


class CRISPR:
    """
    Importer module for CRISPR-Cas9 screens acquired at Sanger and Broad Institutes.

    """

    def __init__(
        self,
        fc_file="crispr/CRISPR_corrected_qnorm_20191108.csv",
        institute_file="crispr/CRISPR_Institute_Origin_20191108.csv",
    ):
        self.crispr = pd.read_csv(f"{dpath}/{fc_file}", index_col=0)
        self.institute = pd.read_csv(f"{dpath}/{institute_file}", index_col=0, header=None).iloc[:, 0]

    def perform_pca(self, n_components=10, subset=None):
        df = DataPCA.perform_pca(self.filter(subset=subset), n_components=n_components)

        for by in df:
            df[by]["pcs"].round(5).to_csv(f"{dpath}/PCA_CRISPR_{by}_pcs.csv")
            df[by]["vex"].round(5).to_csv(f"{dpath}/PCA_CRISPR_{by}_vex.csv")

    @staticmethod
    def import_pca():
        pca = {}

        for by in ["row", "column"]:
            pca[by] = {}

            pca[by]["pcs"] = pd.read_csv(f"{dpath}/PCA_CRISPR_{by}_pcs.csv", index_col=0)
            pca[by]["vex"] = pd.read_csv(f"{dpath}/PCA_CRISPR_{by}_vex.csv", index_col=0, header=None).iloc[:, 0]

        return pca

    def perform_growth_corr(self, subset=None):
        ss = Sample()

        corr = ss.growth_corr(self.filter(subset=subset))
        corr.round(5).to_csv(f"{dpath}/growth_CRISPR_correlation.csv", index=False)

        return corr

    def perform_number_responses(self, thres=-0.5, subset=None):
        df = self.filter(subset=subset)

        num_resp = (df < thres).sum(1).reset_index()
        num_resp.columns = ["GeneSymbol", "n_resp"]

        num_resp.to_csv(f"{dpath}/number_responses_CRISPR.csv", index=False)

        return num_resp

    def get_data(self, scale=True):
        df = self.crispr.copy()

        if scale:
            df = self.scale(df)

        return df

    def filter(
        self,
        subset=None,
        scale=True,
        abs_thres=None,
        drop_core_essential=False,
        min_events=5,
        drop_core_essential_broad=False,
    ):
        df = self.get_data(scale=True)

        # - Filters
        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by scaled scores
        if abs_thres is not None:
            df = df[(df.abs() > abs_thres).sum(1) >= min_events]

        # Filter out core essential genes
        if drop_core_essential:
            df = df[~df.index.isin(cy.Utils.get_adam_core_essential())]

        if drop_core_essential_broad:
            df = df[~df.index.isin(cy.Utils.get_broad_core_essential())]

        # - Subset matrices
        return self.get_data(scale=scale).loc[df.index].reindex(columns=df.columns)

    @staticmethod
    def scale(df, essential=None, non_essential=None, metric=np.median):
        if essential is None:
            essential = cy.Utils.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = cy.Utils.get_non_essential_genes(return_series=False)

        assert (
            len(essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with essential list"

        assert (
            len(non_essential.intersection(df.index)) != 0
        ), "DataFrame has no index overlapping with non essential list"

        essential_metric = metric(df.reindex(essential).dropna(), axis=0)
        non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

        df = df.subtract(non_essential_metric).divide(
            non_essential_metric - essential_metric
        )

        return df


class Genomic:
    """
    Import module for Genomic binary feature table (containing mutations and copy-number calls)
    Iorio et al., Cell, 2016.

    """

    def __init__(
        self, mobem_file="genomic/PANCAN_mobem.csv", drop_factors=True, add_msi=True
    ):
        self.sample = Sample()

        idmap = (
            self.sample.samplesheet.reset_index()
            .dropna(subset=["COSMIC_ID", "model_id"])
            .set_index("COSMIC_ID")["model_id"]
        )

        mobem = pd.read_csv(f"{dpath}/{mobem_file}", index_col=0)
        mobem = mobem[mobem.index.astype(str).isin(idmap.index)]
        mobem = mobem.set_index(idmap[mobem.index.astype(str)].values)

        if drop_factors is not None:
            mobem = mobem.drop(columns={"TISSUE_FACTOR", "MSI_FACTOR", "MEDIA_FACTOR"})

        if add_msi:
            self.msi = self.sample.samplesheet.loc[mobem.index, "msi_status"]
            mobem["msi_status"] = (self.msi == "MSI-H").astype(int)[mobem.index].values

        self.mobem = mobem.astype(int).T

    def get_data(self):
        return self.mobem.copy()

    def filter(self, subset=None, min_events=5):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Minimum number of events
        df = df[df.sum(1) >= min_events]

        return df

    @staticmethod
    def mobem_feature_to_gene(f):
        if f.endswith("_mut"):
            genes = {f.split("_")[0]}

        elif f.startswith("gain.") or f.startswith("loss."):
            genes = {
                g
                for fs in f.split("..")
                if not (fs.startswith("gain.") or fs.startswith("loss."))
                for g in fs.split(".")
                if g != ""
            }

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")

        return genes

    @staticmethod
    def mobem_feature_type(f):
        if f.endswith("_mut"):
            return "Mutation"

        elif f.startswith("gain."):
            return "CN gain"

        elif f.startswith("loss."):
            return "CN loss"

        elif f == "msi_status":
            return f

        else:
            raise ValueError(f"{f} is not a valid MOBEM feature.")


class PPI:
    """
    Module used to import protein-protein interaction networks from multiple resources (e.g. STRING, BioGRID).

    """

    def __init__(
        self,
        string_file="ppi/9606.protein.links.full.v10.5.txt",
        string_alias_file="ppi/9606.protein.aliases.v10.5.txt",
        biogrid_file="ppi/BIOGRID-ORGANISM-Homo_sapiens-3.4.157.tab2.txt",
        drug_targets=None,
    ):
        self.string_file = string_file
        self.string_alias_file = string_alias_file
        self.biogrid_file = biogrid_file

        self.drug_targets = (
            DrugResponse.get_drugtargets() if drug_targets is None else drug_targets
        )

    def ppi_annotation(self, df, ppi_type, ppi_kws, target_thres=5):
        df_genes, df_drugs = set(df["GeneSymbol"]), set(df["DRUG_ID"])

        # PPI annotation
        if ppi_type == "string":
            ppi = self.build_string_ppi(**ppi_kws)

        elif ppi_type == "biogrid":
            ppi = self.build_biogrid_ppi(**ppi_kws)

        else:
            raise Exception("ppi_type not supported, choose from: string or biogrid")

        # Drug target
        d_targets = {
            k: self.drug_targets[k] for k in df_drugs if k in self.drug_targets
        }

        # Drug targets not in the screen
        d_targets_not_screened = {
            k for k in d_targets if len(d_targets[k].intersection(df_genes)) == 0
        }

        # Calculate distance between drugs and genes in PPI
        dist_d_g = self.dist_drugtarget_genes(d_targets, df_genes, ppi)

        # Annotate drug regressions
        def drug_gene_annot(d, g):
            if d not in d_targets:
                res = "No link; No drug target information"

            elif d in d_targets_not_screened:
                res = "No link; Drug target(s) not in CRISPR screen"

            elif d not in dist_d_g:
                res = "No link; Drug target(s) not in network"

            elif g not in dist_d_g[d]:
                res = "No link; Gene not in network"

            elif g in d_targets[d]:
                res = "T"

            else:
                res = self.ppi_dist_to_string(dist_d_g[d][g], target_thres)

            return res

        df = df.assign(
            target_detailed=[
                drug_gene_annot(d, g) for d, g in df[["DRUG_ID", "GeneSymbol"]].values
            ]
        )

        df = df.assign(
            target=[
                "-" if t.startswith("No link;") else t for t in df["target_detailed"]
            ]
        )

        return df

    @staticmethod
    def dist_drugtarget_genes(drug_targets, genes, ppi):
        genes = genes.intersection(set(ppi.vs["name"]))
        assert len(genes) != 0, "No genes overlapping with PPI provided"

        dmatrix = {}

        for drug in drug_targets:
            drug_genes = drug_targets[drug].intersection(genes)

            if len(drug_genes) != 0:
                dmatrix[drug] = dict(
                    zip(
                        *(
                            genes,
                            np.min(
                                ppi.shortest_paths(source=drug_genes, target=genes),
                                axis=0,
                            ),
                        )
                    )
                )

        return dmatrix

    @staticmethod
    def ppi_dist_to_string(d, target_thres):
        if d == 0:
            res = "T"

        elif d == np.inf:
            res = "No link; No connection"

        elif d < target_thres:
            res = f"{int(d)}"

        else:
            res = f"{int(target_thres)}+"

        return res

    def build_biogrid_ppi(
        self, exp_type=None, int_type=None, organism=9606, export_pickle=None
    ):
        # 'Affinity Capture-MS', 'Affinity Capture-Western'
        # 'Reconstituted Complex', 'PCA', 'Two-hybrid', 'Co-crystal Structure', 'Co-purification'

        # Import
        biogrid = pd.read_csv(f"{dpath}/{self.biogrid_file}", sep="\t")

        # Filter organism
        biogrid = biogrid[
            (biogrid["Organism Interactor A"] == organism)
            & (biogrid["Organism Interactor B"] == organism)
        ]

        # Filter non matching genes
        biogrid = biogrid[
            (biogrid["Official Symbol Interactor A"] != "-")
            & (biogrid["Official Symbol Interactor B"] != "-")
        ]

        # Physical interactions only
        if int_type is not None:
            biogrid = biogrid[
                [i in int_type for i in biogrid["Experimental System Type"]]
            ]

        logging.getLogger("DTrace").info(
            f"Experimental System Type considered: {'; '.join(set(biogrid['Experimental System Type']))}"
        )

        # Filter by experimental type
        if exp_type is not None:
            biogrid = biogrid[[i in exp_type for i in biogrid["Experimental System"]]]

        logging.getLogger("DTrace").info(
            f"Experimental System considered: {'; '.join(set(biogrid['Experimental System']))}"
        )

        # Interaction source map
        biogrid["interaction"] = (
            biogrid["Official Symbol Interactor A"]
            + "<->"
            + biogrid["Official Symbol Interactor B"]
        )

        # Unfold associations
        biogrid = {
            (s, t)
            for p1, p2 in biogrid[
                ["Official Symbol Interactor A", "Official Symbol Interactor B"]
            ].values
            for s, t in [(p1, p2), (p2, p1)]
            if s != t
        }

        # Build igraph network
        # igraph network
        net_i = igraph.Graph(directed=False)

        # Initialise network lists
        edges = [(px, py) for px, py in biogrid]
        vertices = list({p for p1, p2 in biogrid for p in [p1, p2]})

        # Add nodes
        net_i.add_vertices(vertices)

        # Add edges
        net_i.add_edges(edges)

        # Simplify
        net_i = net_i.simplify()
        logging.getLogger("DTrace").info(net_i.summary())

        # Export
        if export_pickle is not None:
            net_i.write_pickle(export_pickle)

        return net_i

    def build_string_ppi(self, score_thres=900, export_pickle=None):
        # ENSP map to gene symbol
        gmap = pd.read_csv(f"{dpath}/{self.string_alias_file}", sep="\t")
        gmap = gmap[["BioMart_HUGO" in i.split(" ") for i in gmap["source"]]]
        gmap = (
            gmap.groupby("string_protein_id")["alias"].agg(lambda x: set(x)).to_dict()
        )
        gmap = {k: list(gmap[k])[0] for k in gmap if len(gmap[k]) == 1}
        logging.getLogger("DTrace").info(f"ENSP gene map: {len(gmap)}")

        # Load String network
        net = pd.read_csv(f"{dpath}/{self.string_file}", sep=" ")

        # Filter by moderate confidence
        net = net[net["combined_score"] > score_thres]

        # Filter and map to gene symbol
        net = net[
            [
                p1 in gmap and p2 in gmap
                for p1, p2 in net[["protein1", "protein2"]].values
            ]
        ]
        net["protein1"] = [gmap[p1] for p1 in net["protein1"]]
        net["protein2"] = [gmap[p2] for p2 in net["protein2"]]
        logging.getLogger("DTrace").info(f"String: {len(net)}")

        #  String network
        net_i = igraph.Graph(directed=False)

        # Initialise network lists
        edges = [(px, py) for px, py in net[["protein1", "protein2"]].values]
        vertices = list(set(net["protein1"]).union(net["protein2"]))

        # Add nodes
        net_i.add_vertices(vertices)

        # Add edges
        net_i.add_edges(edges)

        # Add edge attribute score
        net_i.es["score"] = list(net["combined_score"])

        # Simplify
        net_i = net_i.simplify(combine_edges="max")
        logging.getLogger("DTrace").info(net_i.summary())

        # Export
        if export_pickle is not None:
            net_i.write_pickle(export_pickle)

        return net_i

    @staticmethod
    def ppi_corr(ppi, m_corr, m_corr_thres=None):
        """
        Annotate PPI network based on Pearson correlation between the vertices of each edge using
        m_corr data-frame and m_corr_thres (Pearson > m_corr_thress).

        :param ppi:
        :param m_corr:
        :param m_corr_thres:
        :return:
        """
        # Subset PPI network
        ppi = ppi.subgraph([i.index for i in ppi.vs if i["name"] in m_corr.index])

        # Edge correlation
        crispr_pcc = np.corrcoef(m_corr.loc[ppi.vs["name"]].values)
        ppi.es["corr"] = [crispr_pcc[i.source, i.target] for i in ppi.es]

        # Sub-set by correlation between vertices of each edge
        if m_corr_thres is not None:
            ppi = ppi.subgraph_edges(
                [i.index for i in ppi.es if abs(i["corr"]) > m_corr_thres]
            )

        logging.getLogger("DTrace").info(ppi.summary())

        return ppi

    @classmethod
    def plot_ppi(
        cls,
        drug_name,
        d_associations,
        ppi,
        corr_thres=0.2,
        fdr=0.1,
        norder=1,
        exclude_nodes=None,
    ):
        d_targets = DrugResponse.get_drugtargets(by="Name")

        # Build data-set
        d_signif = d_associations.query(f"(DRUG_NAME == '{drug_name}') & (fdr < {fdr})")

        if exclude_nodes is not None:
            d_signif = d_signif[~d_signif["GeneSymbol"].isin(exclude_nodes)]

        d_ppi_df = cls.get_edges(ppi, list(d_signif["GeneSymbol"]), corr_thres, norder)

        # Build graph
        graph = pydot.Dot(graph_type="graph", pagedir="TR")

        kws_nodes = dict(
            style='"rounded,filled"',
            shape="rect",
            color=DTracePlot.PAL_DTRACE[1],
            penwidth=2,
            fontcolor="white",
        )

        kws_edges = dict(
            fontsize=9,
            fontcolor=DTracePlot.PAL_DTRACE[2],
            color=DTracePlot.PAL_DTRACE[2],
        )

        for s, t, r in d_ppi_df[["source", "target", "r"]].values:
            # Add source node
            fs = 15 if s in d_signif["GeneSymbol"].values else 9
            fc = DTracePlot.PAL_DTRACE[
                0 if drug_name in d_targets and s in d_targets[drug_name] else 2
            ]

            source = pydot.Node(s, fillcolor=fc, fontsize=fs, **kws_nodes)
            graph.add_node(source)

            # Add target node
            fc = DTracePlot.PAL_DTRACE[
                0 if drug_name in d_targets and t in d_targets[drug_name] else 2
            ]
            fs = 15 if t in d_signif["GeneSymbol"].values else 9

            target = pydot.Node(t, fillcolor=fc, fontsize=fs, **kws_nodes)
            graph.add_node(target)

            # Add edge
            edge = pydot.Edge(source, target, label="{:.2f}".format(r), **kws_edges)
            graph.add_edge(edge)

        return graph

    @classmethod
    def get_edges(cls, ppi, nodes, corr_thres, norder):
        # Subset network
        ppi_sub = ppi.copy().subgraph_edges(
            [e for e in ppi.es if abs(e["corr"]) >= corr_thres]
        )

        # Nodes that are contained in the network
        nodes = {v for v in nodes if v in ppi_sub.vs["name"]}
        assert len(nodes) > 0, "None of the nodes is contained in the PPI"

        # Nodes neighborhood
        neighbor_nodes = {
            v for n in nodes for v in ppi_sub.neighborhood(n, order=norder)
        }

        # Build subgraph
        subgraph = ppi_sub.subgraph(neighbor_nodes)

        # Build data-frame
        nodes_df = pd.DataFrame(
            [
                {
                    "source": subgraph.vs[e.source]["name"],
                    "target": subgraph.vs[e.target]["name"],
                    "r": e["corr"],
                }
                for e in subgraph.es
            ]
        ).sort_values("r")

        return nodes_df


class GeneExpression:
    """
    Import module of gene-expression data-set.

    """

    def __init__(
        self,
        voom_file="genomic/rnaseq_voom.csv.gz",
        rpkm_file="genomic/rnaseq_rpkm.csv.gz",
    ):
        self.voom = pd.read_csv(f"{dpath}/{voom_file}", index_col=0)
        self.rpkm = pd.read_csv(f"{dpath}/{rpkm_file}", index_col=0)

    def get_data(self, dtype="voom"):
        if dtype.lower() == "rpkm":
            return self.rpkm.copy()

        else:
            return self.voom.copy()

    def filter(self, dtype="voom", subset=None):
        df = self.get_data(dtype=dtype)

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    def is_not_expressed(self, rpkm_threshold=1, subset=None):
        rpkm = self.filter(dtype="rpkm", subset=subset)
        rpkm = (rpkm < rpkm_threshold).astype(int)
        return rpkm

    def perform_pca(self, n_components=10, subset=None):
        df = DataPCA.perform_pca(self.filter(subset=subset), n_components=n_components)

        for by in df:
            df[by]["pcs"].round(5).to_csv(f"{dpath}/PCA_GExp_{by}_pcs.csv")
            df[by]["vex"].round(5).to_csv(f"{dpath}/PCA_GExp_{by}_vex.csv")

    @staticmethod
    def import_pca():
        pca = {}

        for by in ["row", "column"]:
            pca[by] = {}

            pca[by]["pcs"] = pd.read_csv(f"{dpath}/PCA_GExp_{by}_pcs.csv", index_col=0)
            pca[by]["vex"] = pd.read_csv(
                f"{dpath}/PCA_GExp_{by}_vex.csv", index_col=0, header=None
            ).iloc[:, 0]

        return pca

class CopyNumber:
    def __init__(self, cnv_file="genomic/copynumber_total_new_map.csv.gz"):
        self.copynumber = pd.read_csv(f"{dpath}/{cnv_file}", index_col=0)

    def get_data(self):
        return self.copynumber.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df

    @staticmethod
    def is_amplified(
        cn, ploidy, cn_threshold_low=5, cn_thresholds_high=9, ploidy_threshold=2.7
    ):
        if (ploidy <= ploidy_threshold) and (cn >= cn_threshold_low):
            return 1

        elif (ploidy > ploidy_threshold) and (cn >= cn_thresholds_high):
            return 1

        else:
            return 0


class WES:
    def __init__(self, wes_file="genomic/WES_variants.csv.gz"):
        self.wes = pd.read_csv(f"{dpath}/{wes_file}")

    def get_data(self, as_matrix=False):
        df = self.wes.copy()

        if as_matrix:
            df["value"] = 1

            df = pd.pivot_table(
                df,
                index="Gene",
                columns="model_id",
                values="value",
                aggfunc="first",
                fill_value=0,
            )

        return df

    def filter(self, subset=None, min_events=5, as_matrix=False):
        df = self.get_data(as_matrix=as_matrix)

        if subset is not None:
            if as_matrix:
                df = df.loc[:, df.columns.isin(subset)]

            else:
                df = df[df["model_id"].isin(subset)]

            assert df.shape[1] != 0, "No columns after filter by subset"

        # Minimum number of events
        df = df[df.sum(1) >= min_events]

        return df


class RNAi:
    def __init__(
        self,
        rnai_file="rnai/D2_combined_gene_dep_scores.csv",
        samplesheet_file="rnai/DepMap-2018q4-celllines.csv",
    ):
        self.sinfo = pd.read_csv(f"{dpath}/{samplesheet_file}")
        self.rnai = self.read_data(f"{dpath}/{rnai_file}")

    def read_data(self, rnai_file):
        rnai = pd.read_csv(rnai_file, index_col=0)
        rnai.index = [i.split(" ")[0] for i in rnai.index]
        rnai = rnai[["&" not in i for i in rnai.index]]

        sinfo = self.sinfo[self.sinfo["CCLE_Name"].isin(rnai.columns)].set_index(
            "CCLE_Name"
        )["DepMap_ID"]

        rnai = rnai.loc[:, rnai.columns.isin(sinfo.index)]
        rnai = rnai.rename(columns=sinfo)

        cpass = (
            Sample()
            .samplesheet.dropna(subset=["BROAD_ID"])
            .reset_index()
            .set_index("BROAD_ID")["model_id"]
        )

        rnai = rnai.rename(columns=cpass)

        return rnai

    def get_data(self):
        return self.rnai.copy()

    def filter(self, subset=None):
        df = self.get_data()

        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]
            assert df.shape[1] != 0, "No columns after filter by subset"

        return df


class KinobeadCATDS:
    def __init__(
        self,
        catds_most_potent_file="klaeger_et_al_catds_most_potent.csv",
        catds_matrix_file="klaeger_et_al_catds.csv",
        assoc=None,
        fdr_thres=0.1,
        unstack=True,
    ):
        self.ppi_order = ["T", "1", "2", "3", "4", "5+", "-"]

        self.catds_most_potent_file = catds_most_potent_file
        self.catds_matrix_file = catds_matrix_file

        self.catds = self.import_matrix(
            unstack=unstack, assoc=assoc, fdr_thres=fdr_thres
        )

    def import_matrix(self, unstack, assoc, fdr_thres):
        """
        Imports Kinobeads CATDS from:

        Klaeger S, Heinzlmeir S, Wilhelm M, Polzer H, Vick B, Koenig P-A, Reinecke M, Ruprecht B, Petzoldt S, Meng C,
        Zecha J, Reiter K, Qiao H, Helm D, Koch H, Schoof M, Canevari G, Casale E, Depaolini SR, Feuchtinger A, et al.
        (2017) The target landscape of clinical kinase drugs. Science 358: eaan4368

        Merge information from the Drug ~ CRISPR LMM associations can only be done if the unstack is True.

        :param unstack:
        :param assoc:
        :param fdr_thres:
        :return:
        """
        dmap = self.import_drug_names()

        catds = pd.read_csv(f"{dpath}/{self.catds_matrix_file}", index_col=0)
        logging.getLogger("DTrace").info(f"Kinobeads drugs all={catds.shape[0]}")

        catds = catds[catds.index.isin(dmap.index)]
        catds = catds[catds.index.isin([i[1] for i in assoc.drespo.index])]
        logging.getLogger("DTrace").info(f"Kinobeads drugs overlap={catds.shape[0]}")

        catds.index = dmap[catds.index].values

        if unstack:
            catds = catds.unstack().dropna().reset_index()
            catds.columns = ["GeneSymbol", "DRUG_NAME", "catds"]

        if unstack and (assoc is not None):
            catds = self.merge_lmm_info(catds, assoc=assoc, fdr_thres=fdr_thres)

        return catds

    def import_drug_names(self):
        dmap = pd.read_csv(f"{dpath}/{self.catds_most_potent_file}")
        dmap = dmap.set_index("Drug")["name"].dropna()
        return dmap

    def merge_lmm_info(self, catds, assoc, fdr_thres):
        assoc_df = assoc.lmm_drug_crispr.copy()
        assoc_df = assoc_df[assoc_df["DRUG_NAME"].isin(catds["DRUG_NAME"])]
        assoc_df = assoc_df[assoc_df["GeneSymbol"].isin(catds["GeneSymbol"])]
        assoc_df["target"] = pd.Categorical(
            assoc_df["target"], self.ppi_order, ordered=True
        )

        catds_index = catds.set_index(["DRUG_NAME", "GeneSymbol"]).index

        # Is drug-target
        d_targets = assoc.drespo_obj.get_drugtargets(by="Name")
        catds["is_target"] = [
            int(t in d_targets[d]) for d, t in catds[["DRUG_NAME", "GeneSymbol"]].values
        ]

        # Annotate target distance to the drug targets
        catds["target"] = (
            assoc_df.groupby(["DRUG_NAME", "GeneSymbol"])["target"]
            .min()[catds_index]
            .values
        )

        # Annotate with p-value and FDR
        for f in ["pval", "fdr"]:
            catds[f] = (
                assoc_df.groupby(["DRUG_NAME", "GeneSymbol"])[f]
                .min()[catds_index]
                .values
            )

        # Annotate if is significant
        catds["signif"] = catds["fdr"].apply(lambda v: "Yes" if v < fdr_thres else "No")

        return catds

    def get_data(self):
        return self.catds.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df


if __name__ == "__main__":
    crispr = CRISPR()
    drug_response = DrugResponse()

    samples = list(
        set.intersection(
            set(drug_response.get_data().columns), set(crispr.get_data().columns)
        )
    )

    drug_respo = drug_response.filter(subset=samples, min_meas=0.75)

    logging.getLogger("DTrace").info(f"Samples={len(samples)}")
    logging.getLogger("DTrace").info(
        f"Spaseness={(1 - drug_respo.count().sum() / np.prod(drug_respo.shape)) * 100:.1f}%"
    )
