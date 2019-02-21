#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import pydot
import igraph
import logging
import warnings
import numpy as np
import pandas as pd
import crispy as cy
from dtrace import logger, dpath
from dtrace.DTracePlot import DTracePlot


class DrugResponse:
    """
    Importer module for drug-response measurements acquired at Sanger Institute GDSC (https://cancerrxgene.org).

    """

    SAMPLE_COLUMNS = ["model_id"]
    DRUG_COLUMNS = ["DRUG_ID", "DRUG_NAME", "VERSION"]
    DRUG_OWNERS = ["AZ", "GDSC", "MGH", "Nathaneal.Gray"]

    def __init__(
        self,
        drugresponse_file_v17="drug/screening_set_384_all_owners_fitted_data_20180308_updated.csv",
        drugresponse_file_rs="drug/fitted_rapid_screen_1536_v1.2.1_20181026_updated.csv",
    ):
        """
        Two experimental versions were used to acquire drug-response measurements, i.e. v17 and RS (chronologically
        ordered), hence raw intentisity measurements are processed and exported to different files.

        :param drugresponse_file_v17:
        :param drugresponse_file_rs:
        """
        self.drugsheet = self.get_drugsheet()

        # Import and Merge drug response matrices
        self.d_v17 = pd.read_csv(f"{dpath}/{drugresponse_file_v17}").assign(
            VERSION="v17"
        )
        self.d_rs = pd.read_csv(f"{dpath}/{drugresponse_file_rs}").assign(VERSION="RS")

        self.drugresponse = dict()
        for index_value, n in [("ln_IC50", "ic50"), ("AUC", "auc"), ("RMSE", "rmse")]:
            d_v17_matrix = pd.pivot_table(
                self.d_v17,
                index=self.DRUG_COLUMNS,
                columns=self.SAMPLE_COLUMNS,
                values=index_value,
            )

            d_vrs_matrix = pd.pivot_table(
                self.d_rs,
                index=self.DRUG_COLUMNS,
                columns=self.SAMPLE_COLUMNS,
                values=index_value,
            )

            df = pd.concat([d_v17_matrix, d_vrs_matrix], axis=0, sort=False)

            self.drugresponse[n] = df.copy()

        # Read drug max concentration
        self.maxconcentration = pd.concat(
            [
                self.d_rs.groupby(self.DRUG_COLUMNS)["maxc"].min(),
                self.d_v17.groupby(self.DRUG_COLUMNS)["maxc"].min(),
            ],
            sort=False,
        ).sort_values()

    @staticmethod
    def get_drugsheet(drugsheet_file="meta/drugsheet_20190219.xlsx"):
        return pd.read_excel(f"{dpath}/{drugsheet_file}", index_col=0)

    @classmethod
    def get_drugtargets(cls, by="id"):
        if by == "id":
            d_targets = cls.get_drugsheet()["Target Curated"].dropna().to_dict()

        else:
            d_targets = (
                cls.get_drugsheet()
                .groupby("Name")["Target Curated"]
                .first()
                .dropna()
                .to_dict()
            )

        d_targets = {k: {t.strip() for t in d_targets[k].split(";")} for k in d_targets}

        return d_targets

    def get_data(self, dtype="ic50"):
        return self.drugresponse[dtype].copy()

    def filter(
        self,
        dtype="ic50",
        subset=None,
        min_events=3,
        min_meas=0.75,
        max_c=0.5,
        filter_max_concentration=True,
        filter_owner=True,
        filter_combinations=True,
    ):
        # Drug max screened concentration
        df = self.get_data(dtype="ic50")
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

        # Filter by owners
        if filter_owner:
            ds = self.drugsheet[self.drugsheet["Owner"].isin(self.DRUG_OWNERS)]
            df = df[[i[0] in ds.index for i in df.index]]

        # Filter combinations
        if filter_combinations:
            df = df[[" + " not in i[1] for i in df.index]]

        return self.get_data(dtype).loc[df.index, df.columns]

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
            logger.log(logging.INFO, f"{drug_id} Drug ID not in drug list")
            return None

        drug_name = [self.drugsheet.loc[drug_id, "Name"]]

        drug_synonyms = self.drugsheet.loc[drug_id, "Synonyms"]
        drug_synonyms = (
            [] if str(drug_synonyms).lower() == "nan" else drug_synonyms.split(", ")
        )

        return set(drug_name + drug_synonyms)

    @staticmethod
    def growth_corr(df, growth):
        samples = list(set(growth.dropna().index).intersection(df.columns))

        g_corr = (
            df[samples]
            .T.corrwith(growth[samples])
            .sort_values()
            .rename("corr")
            .reset_index()
        )

        return g_corr


class CRISPR:
    """
    Importer module for CRISPR-Cas9 screens acquired at Sanger and Broad Institutes.

    """

    LOW_QUALITY_SAMPLES = ["SIDM00096", "SIDM01085", "SIDM00958"]

    def __init__(
        self,
        sanger_fc_file="crispr/sanger_depmap18_fc_corrected.csv",
        sanger_qc_file="crispr/sanger_depmap18_fc_ess_aucs.csv",
        broad_fc_file="crispr/broad_depmap18q4_fc_corrected.csv",
        broad_qc_file="crispr/broad_depmap18q4_fc_ess_aucs.csv",
        ess_broad_file="crispr/depmap19Q1_essential_genes.txt",
        ess_sanger_file="crispr/projectscore_essential_genes.txt",
    ):
        self.SANGER_FC_FILE = sanger_fc_file
        self.SANGER_QC_FILE = sanger_qc_file
        self.SANGER_ESS_FILE = ess_sanger_file

        self.BROAD_FC_FILE = broad_fc_file
        self.BROAD_QC_FILE = broad_qc_file
        self.BROAD_ESS_FILE = ess_broad_file

        self.crispr, self.institute = self.__merge_matricies()

        self.crispr = self.crispr.drop(columns=self.LOW_QUALITY_SAMPLES)

        self.qc_ess = self.__merge_qc_arrays()

    def import_broad_essential_genes(self):
        broad_ess = pd.read_csv(f"{dpath}/{self.BROAD_ESS_FILE}")["gene"]
        broad_ess = list(set(broad_ess.apply(lambda v: v.split(" ")[0])))
        return broad_ess

    def import_sanger_essential_genes(self):
        sanger_ess = pd.read_csv(f"{dpath}/{self.SANGER_ESS_FILE}")
        sanger_ess = list(set(sanger_ess[sanger_ess["CoreFitness"]]["GeneSymbol"]))
        return sanger_ess

    def __merge_qc_arrays(self):
        gdsc_qc = pd.read_csv(
            f"{dpath}/{self.SANGER_QC_FILE}", header=None, index_col=0
        ).iloc[:, 0]
        broad_qc = pd.read_csv(
            f"{dpath}/{self.BROAD_QC_FILE}", header=None, index_col=0
        ).iloc[:, 0]

        qcs = pd.concat(
            [
                gdsc_qc[self.institute[self.institute == "Sanger"].index],
                broad_qc[self.institute[self.institute == "Broad"].index],
            ]
        )

        return qcs

    def __merge_matricies(self):
        gdsc_fc = pd.read_csv(f"{dpath}/{self.SANGER_FC_FILE}", index_col=0).dropna()
        broad_fc = pd.read_csv(f"{dpath}/{self.BROAD_FC_FILE}", index_col=0).dropna()

        genes = list(set(gdsc_fc.index).intersection(broad_fc.index))

        merged_matrix = pd.concat(
            [
                gdsc_fc.loc[genes],
                broad_fc.loc[genes, [i for i in broad_fc if i not in gdsc_fc.columns]],
            ],
            axis=1,
            sort=False,
        )

        institute = pd.Series(
            {s: "Sanger" if s in gdsc_fc.columns else "Broad" for s in merged_matrix}
        )

        return merged_matrix, institute

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

    @staticmethod
    def growth_corr(df, growth):
        samples = list(set(growth.dropna().index).intersection(df.columns))

        g_corr = (
            df[samples]
            .T.corrwith(growth[samples])
            .sort_values()
            .rename("corr")
            .reset_index()
        )

        return g_corr


class Sample:
    """
    Import module that handles the sample list (i.e. list of cell lines) and their descriptive information.

    """

    def __init__(
        self,
        index="model_id",
        samplesheet_file="meta/model_list_2018-09-28_1452.csv",
        growthrate_file="meta/growth_rates_rapid_screen_1536_v1.2.2_20181113.csv",
        samples_origin="meta/samples_origin.csv",
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

    def __assemble_growth_rates(self, dfile):
        # Import
        dratio = pd.read_csv(dfile, index_col=0)

        # Convert to date
        dratio["DATE_CREATED"] = pd.to_datetime(dratio["DATE_CREATED"])

        # Group growth ratios per seeding
        d_nc1 = (
            dratio.groupby([self.index, "SEEDING_DENSITY"])
            .agg({"growth_rate": [np.median, "count"], "DATE_CREATED": [np.max]})
            .reset_index()
        )

        d_nc1.columns = ["_".join(filter(lambda x: x != "", i)) for i in d_nc1]

        # Pick most recent measurements per cell line
        d_nc1 = d_nc1.iloc[
            d_nc1.groupby(self.index)["DATE_CREATED_amax"].idxmax()
        ].set_index(self.index)

        return d_nc1

    def build_covariates(
        self, samples=None, discrete_vars=None, continuos_vars=None, extra_vars=None
    ):
        covariates = []

        if discrete_vars is not None:
            covariates.append(
                pd.concat(
                    [
                        pd.get_dummies(self.samplesheet[v].dropna())
                        for v in discrete_vars
                    ],
                    axis=1,
                    sort=False,
                )
            )

        if continuos_vars is not None:
            covariates.append(self.samplesheet.reindex(columns=continuos_vars))

        if extra_vars is not None:
            covariates.append(extra_vars.copy())

        if len(covariates) == 0:
            return None

        covariates = pd.concat(covariates, axis=1, sort=False)

        if samples is not None:
            covariates = covariates.loc[samples]

        return covariates


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
            raise ValueError("{} is not a valid MOBEM feature.".format(f))

        return genes

    @staticmethod
    def mobem_feature_type(f):
        if f.endswith("_mut"):
            return "Mutation"

        elif f.startswith("gain."):
            return "CN gain"

        elif f.startswith("loss."):
            return "CN loss"

        else:
            raise ValueError("{} is not a valid MOBEM feature.".format(f))


class PPI:
    """
    Module used to import protein-protein interaction networks from multiple resources (e.g. STRING, BioGRID).

    """

    def __init__(
        self,
        string_file="ppi/9606.protein.links.full.v10.5.txt",
        string_alias_file="ppi/9606.protein.aliases.v10.5.txt",
        biogrid_file="ppi/BIOGRID-ORGANISM-Homo_sapiens-3.4.157.tab2.txt",
    ):
        self.string_file = string_file
        self.string_alias_file = string_alias_file
        self.biogrid_file = biogrid_file

        self.drug_targets = DrugResponse.get_drugtargets()

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

        logger.log(
            logger.INFO,
            f"Experimental System Type considered: {'; '.join(set(biogrid['Experimental System Type']))}",
        )

        # Filter by experimental type
        if exp_type is not None:
            biogrid = biogrid[[i in exp_type for i in biogrid["Experimental System"]]]

        logger.log(
            logger.INFO,
            f"Experimental System considered: {'; '.join(set(biogrid['Experimental System']))}",
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
        logger.log(logging.INFO, net_i.summary())

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
        logger.log(logging.INFO, f"ENSP gene map: {len(gmap)}")

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
        logger.log(logging.INFO, f"String: {len(net)}")

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
        logger.log(logging.INFO, net_i.summary())

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

        logger.log(logging.INFO, ppi.summary())

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


class Proteomics:
    def __init__(self, proteomics_file="genomic/proteomics_coread.csv.gz"):
        self.proteomics = pd.read_csv(f"{dpath}/{proteomics_file}", index_col=0)

    def get_data(self):
        return self.proteomics.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df


class PhosphoProteomics:
    def __init__(
        self, phosphoproteomics_file="genomic/phosphoproteomics_coread.csv.gz"
    ):
        self.phosphoproteomics = pd.read_csv(
            f"{dpath}/{phosphoproteomics_file}", index_col=0
        )

    def get_data(self):
        return self.phosphoproteomics.copy()

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df


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


class Apoptosis:
    def __init__(self, dfile="apoptosis/AUC_data.csv"):
        crename = dict(
            HT29="HT-29",
            CW2="CW-2",
            HCT15="HCT-15",
            DIFI="DiFi",
            U2OS="U-2-OS",
            HUTU80="HuTu-80",
            LS1034="LS-1034",
            OUMS23="OUMS-23",
            SNU407="SNU-407",
            COLO205="COLO-205",
            SNU81="SNU-81",
            CCK81="CCK-81",
            NCIH747="NCI-H747",
            COLO320HSR="COLO-320-HSR",
            SNUC2B="SNU-C2B",
            SKCO1="SK-CO-1",
            CaR1="CaR-1",
            COLO678="COLO-678",
            LS411N="LS-411N",
            CL40="CL-40",
            SNUC5="SNU-C5",
            HT115="HT-115",
            LS180="LS-180",
            RCM1="RCM-1",
            NCIH630="NCI-H630",
            HCT116="HCT-116",
            NCIH508="NCI-H508",
            HCC56="HCC-56",
            SNU175="SNU-175",
            CL34="CL-34",
            LS123="LS-123",
            COLO741="COLO-741",
            CL11="CL-11",
            NCIH716="NCI-H716",
            SNUC1="SNU-C1",
        )

        self.samplesheet = Sample().samplesheet

        self.screen = pd.read_csv(f"{dpath}/{dfile}")
        self.screen = self.screen.replace(dict(CELL_LINE=crename))
        self.screen = self.screen[
            self.screen["CELL_LINE"].isin(self.samplesheet["model_name"])
        ]

        model_id = self.samplesheet.reset_index().set_index("model_name")["model_id"]
        self.screen = self.screen.assign(
            model_id=model_id.loc[self.screen["CELL_LINE"]].values
        )

    def get_data(self, drug="DMSO", values="AUC"):
        smatrix = self.screen[self.screen["DRUG"] == drug]
        smatrix = pd.pivot_table(
            smatrix, index="PEPTIDE", columns="model_id", values=values
        )
        return smatrix

    def filter(self, subset=None):
        df = self.get_data()

        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        return df


class CTDR2:
    def __init__(self, data_dir="CTRPv2.2/"):
        self.data_dir = data_dir
        self.samplesheet = self.import_samplesheet()
        self.drugsheet = self.import_compound_sheet()
        self.drespo = self.import_ctrp_aucs()

    def import_samplesheet(self):
        return pd.read_csv(
            f"{dpath}/{self.data_dir}/v22.meta.per_cell_line.txt", sep="\t"
        )

    def import_compound_sheet(self):
        return pd.read_csv(
            f"{dpath}/{self.data_dir}/v22.meta.per_compound.txt", sep="\t", index_col=0
        )

    def import_depmap18q4_samplesheet(self):
        ss = pd.read_csv(f"{dpath}/{self.data_dir}/sample_info.csv")
        ss["CCLE_ID"] = ss["CCLE_name"].apply(lambda v: v.split("_")[0])
        return ss

    def import_ctrp_aucs(self):
        ctrp_samples = self.import_samplesheet()

        ctrp_aucs = pd.read_csv(
            f"{dpath}/{self.data_dir}/v22.data.auc_sensitivities.txt", sep="\t"
        )
        ctrp_aucs = pd.pivot_table(
            ctrp_aucs, index="index_cpd", columns="index_ccl", values="area_under_curve"
        )
        ctrp_aucs = ctrp_aucs.rename(
            columns=ctrp_samples.set_index("index_ccl")["ccl_name"]
        )

        return ctrp_aucs

    def import_ceres(self):
        ss = self.import_depmap18q4_samplesheet().set_index("Broad_ID")

        ceres = pd.read_csv(f"{dpath}/{self.data_dir}/gene_effect.csv", index_col=0).T
        ceres = ceres.rename(columns=ss["CCLE_ID"])
        ceres.index = [i.split(" ")[0] for i in ceres.index]

        return ceres

    def get_compound_by_target(
        self, target, target_field="gene_symbol_of_protein_target"
    ):
        ss = self.drugsheet.dropna(subset=[target_field])
        ss = ss[[target in t.split(";") for t in ss[target_field]]]
        return ss

    def get_data(self):
        return self.drespo.copy()

    def filter(self, subset=None):
        df = self.get_data()

        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]
            assert df.shape[1] != 0, "No columns after filter by subset"

        return df


class RPPA:
    def __init__(self, rppa_file="genomic/CCLE_MDAnderson_RPPA_combined.csv"):
        self.info = [
            "model_id",
            "model_name",
            "synonyms",
            "model_type",
            "growth_properties",
            "doi",
            "pmed",
            "model_treatment",
            "model_comments",
            "msi_status",
            "mutational_burden",
            "ploidy",
            "parent_id",
            "mutation_data",
            "methylation_data",
            "expression_data",
            "cnv_data",
            "drug_data",
            "sample_id",
            "tissue",
            "cancer_type",
            "cancer_type_detail",
            "age_at_sampling",
            "sampling_day",
            "sampling_month",
            "sampling_year",
            "sample_treatment",
            "sample_treatment_details",
            "sample_site",
            "tnm_t",
            "tnm_n",
            "tnm_m",
            "tnm_integrated",
            "tumour_grade",
            "patient_id",
            "species",
            "gender",
            "ethnicity",
            "smoking_status",
            "model_relations_comment",
            "COSMIC_ID",
            "BROAD_ID",
            "RRID",
            "suppliers",
            "Cell.Line.Name",
            "Order",
            "Sample.Source",
            "Category_1",
            "Category_2",
            "Category_3",
            "Sample",
            "Sample.Name",
            "Sample.description",
            "Sample.Number",
            "CCLE_ID",
        ]

        self.rppa_matrix = (
            pd.read_csv(f"{dpath}/{rppa_file}")
            .groupby("model_id")
            .mean()
            .drop(columns=self.info, errors="ignore")
            .T
        )

    def get_data(self):
        return self.rppa_matrix.copy()

    def filter(self, subset=None):
        df = self.get_data()

        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]
            assert df.shape[1] != 0, "No columns after filter by subset"

        return df


class WES:
    def __init__(self, wes_file="genomic/WES_variants.csv.gz"):
        self.wes = pd.read_csv(f"{dpath}/{wes_file}")

    def get_data(self):
        return self.wes.copy()

    def filter(self, subset=None):
        df = self.get_data()

        if subset is not None:
            df = df[df["model_id"].isin(subset)]
            assert df.shape[1] != 0, "No columns after filter by subset"

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


class CRISPRComBat:
    LOW_QUALITY_SAMPLES = ["SIDM00096", "SIDM01085", "SIDM00958"]

    def __init__(self, dmatrix_file="InitialCombat_BroadSanger_Matrix.csv"):
        self.ss = Sample()
        self.datadir = f"{dpath}/crispr/"
        self.dmatrix_file = dmatrix_file

        self.crispr = pd.read_csv(f"{self.datadir}/{self.dmatrix_file}", index_col=0)

    def __generate_merged_matrix(self, dmatrix="InitialCombat_BroadSanger.csv"):
        df = pd.read_csv(f"{self.datadir}/{dmatrix}")

        # Split Sanger matrix
        idmap_sanger = (
            self.ss.samplesheet.reset_index()
            .dropna(subset=["model_name"])
            .set_index("model_name")
        )
        crispr_sanger = df[
            [i for i in df if i in self.ss.samplesheet["model_name"].values]
        ]
        crispr_sanger = crispr_sanger.rename(columns=idmap_sanger["model_id"])

        # Split Broad matrix
        idmap_broad = (
            self.ss.samplesheet.reset_index()
            .dropna(subset=["model_name"])
            .set_index("BROAD_ID")
        )
        crispr_broad = df[
            [i for i in df if i in self.ss.samplesheet["BROAD_ID"].values]
        ]
        crispr_broad = crispr_broad.rename(columns=idmap_broad["model_id"])

        # Merge matrices
        crispr = pd.concat(
            [
                crispr_sanger,
                crispr_broad[[i for i in crispr_broad if i not in crispr_sanger]],
            ],
            axis=1,
            sort=False,
        ).dropna()
        crispr.to_csv(f"{self.datadir}/InitialCombat_BroadSanger_Matrix.csv")

        # Store isntitute sample origin
        institute = pd.Series(
            {s: "Sanger" if s in crispr_sanger else "Broad" for s in crispr}
        )
        institute.to_csv(f"{self.datadir}/InitialCombat_BroadSanger_Institute.csv")

    def __qc_recall_curves(self):
        qc_ess = pd.Series(
            {
                i: cy.QCplot.recall_curve(
                    self.crispr[i], cy.Utils.get_essential_genes()
                )[2]
                for i in self.crispr
            }
        )
        qc_ess.to_csv(f"{self.datadir}/InitialCombat_BroadSanger_Essential_AURC.csv")

        qc_ness = pd.Series(
            {
                i: cy.QCplot.recall_curve(
                    self.crispr[i], cy.Utils.get_non_essential_genes()
                )[2]
                for i in self.crispr
            }
        )
        qc_ness.to_csv(
            f"{self.datadir}/InitialCombat_BroadSanger_NonEssential_AURC.csv"
        )

    def get_data(self, scale=True, drop_lowquality=True):
        df = self.crispr.copy()

        if drop_lowquality:
            df = df.drop(columns=self.LOW_QUALITY_SAMPLES)

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


class KinobeadCATDS:
    def __init__(
            self,
            catds_most_potent_file="klaeger_et_al_catds_most_potent.csv",
            catds_matrix_file="klaeger_et_al_catds.csv",
    ):
        self.catds_most_potent_file = catds_most_potent_file
        self.catds_matrix_file = catds_matrix_file

    def import_matrix(self):
        dmap = self.import_drug_names()

        catds_m = pd.read_csv(f"{dpath}/{self.catds_matrix_file}", index_col=0)
        catds_m = catds_m[catds_m.index.isin(dmap.index)]
        catds_m.index = dmap[catds_m.index].values

        return catds_m

    def import_drug_names(self):
        dmap = pd.read_csv(f"{dpath}/{self.catds_most_potent_file}")
        dmap = dmap.set_index("Drug")["name"].dropna()
        return dmap

    def import_catds(self):
        catds = self.import_matrix()
        catds = catds.unstack().dropna().reset_index()
        catds.columns = ["target", "drug", "catds"]
        return catds


if __name__ == "__main__":
    crispr = CRISPR()
    drug_response = DrugResponse()

    samples = list(
        set.intersection(
            set(drug_response.get_data().columns), set(crispr.get_data().columns)
        )
    )

    drug_respo = drug_response.filter(subset=samples, min_meas=0.75)

    logger.log(logging.INFO, f"Samples={len(samples)}")
    logger.log(
        logging.INFO,
        f"Spaseness={(1 - drug_respo.count().sum() / np.prod(drug_respo.shape)) * 100:.1f}%",
    )
