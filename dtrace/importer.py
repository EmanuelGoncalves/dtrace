#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import warnings
import numpy as np
import pandas as pd
from crispy.utils import Utils


class DrugResponse(object):
    SAMPLE_INFO_COLUMNS = ('COSMIC_ID', 'CELL_LINE_NAME')
    DRUG_INFO_COLUMNS = ['DRUG_ID_lib', 'DRUG_NAME', 'VERSION']

    DRUG_OWNERS = ['AZ', 'GDSC', 'MGH', 'NCI.Pommier', 'Nathaneal.Gray']

    def __init__(
            self,
            drugsheet_file='data/meta/drug_samplesheet_august_2018.txt',
            drugresponse_file_v17='data/drug/screening_set_384_all_owners_fitted_data_20180308.csv',
            drugresponse_file_rs='data/drug/rapid_screen_1536_all_owners_fitted_data_20180308.csv',
    ):
        self.drugsheet = pd.read_csv(drugsheet_file, sep='\t', index_col=0)

        # Import and Merge drug response matrices
        self.d_v17 = pd.read_csv(drugresponse_file_v17).assign(VERSION='v17')
        self.d_rs = pd.read_csv(drugresponse_file_rs).assign(VERSION='RS')

        self.drugresponse = dict()
        for index_value, n in [('IC50_nat_log', 'ic50'), ('AUC', 'auc')]:
            d_v17_matrix = pd.pivot_table(
                self.d_v17, index=self.DRUG_INFO_COLUMNS, columns=self.SAMPLE_INFO_COLUMNS, values=index_value
            )

            d_vrs_matrix = pd.pivot_table(
                self.d_rs, index=self.DRUG_INFO_COLUMNS, columns=self.SAMPLE_INFO_COLUMNS, values=index_value
            )

            df = pd.concat([d_v17_matrix, d_vrs_matrix], axis=0)
            df.columns = df.columns.droplevel(0)

            self.drugresponse[n] = df

        # Read drug max concentration
        self.maxconcentration = pd.concat([
            self.d_rs.groupby(self.DRUG_INFO_COLUMNS)['max_conc_micromolar'].min(),
            self.d_v17.groupby(self.DRUG_INFO_COLUMNS)['max_conc_micromolar'].min()
        ]).sort_values()

    def get_drugtargets(self):
        d_targets = self.drugsheet['Target Curated'].dropna().to_dict()
        d_targets = {k: {t.strip() for t in d_targets[k].split(';')} for k in d_targets}
        return d_targets

    def get_data(self, dtype='ic50'):
        return self.drugresponse[dtype].copy()

    def filter(
            self, subset=None, min_events=3, min_meas=0.85, max_c=0.5, filter_max_concentration=True, filter_owner=True,
            filter_combinations=True
    ):
        """
        Filter Drug-response (ln IC50) data-set to consider only drugs with measurements across
        at least min_meas (deafult=0.85 (85%)) of the total cell lines measured and drugs have an IC50
        lower than the maximum screened concentration (offeseted by max_c (default = 0.5 (50%)) in at least
        min_events (default = 3) cell lines.

        :param subset:
        :param min_events:
        :param min_meas:
        :param max_c:
        :param filter_max_concentration:
        :param filter_owner:
        :param filter_combinations:
        :return:
        """
        # Drug max screened concentration
        df = self.get_data(dtype='ic50')
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
            ds = self.drugsheet[self.drugsheet['Owner'].isin(self.DRUG_OWNERS)]
            df = df[[i[0] in ds.index for i in df.index]]

        # Filter combinations
        if filter_combinations:
            df = df[[' + ' not in i[1] for i in df.index]]

        # - Subset matrices
        for k in self.drugresponse:
            self.drugresponse[k] = self.drugresponse[k].loc[df.index, df.columns]

        return self

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
            warnings.warn('Drug ID {} not in drug list'.format(drug_id_1))
            return False

        if drug_id_2 not in self.drugsheet:
            warnings.warn('Drug ID {} not in drug list'.format(drug_id_2))
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
            print('{} Drug ID not in drug list'.format(drug_id))
            return None

        drug_name = [self.drugsheet.loc[drug_id, 'Name']]

        drug_synonyms = self.drugsheet.loc[drug_id, 'Synonyms']
        drug_synonyms = [] if str(drug_synonyms).lower() == 'nan' else drug_synonyms.split(', ')

        return set(drug_name + drug_synonyms)


class CRISPR(object):
    def __init__(
            self,
            datadir='data/crispr/',
            foldchanges_file='CRISPRcleaned_logFCs.tsv',
            binarydep_file='binaryDepScores.tsv',
            mageckdep_file='MAGeCK_depFDRs.tsv',
            mageckenr_file='MAGeCK_enrFDRs.tsv'
    ):
        self.crispr = dict()

        self.crispr['mageck_dep'] = pd.read_csv(f'{datadir}/{mageckdep_file}', index_col=0, sep='\t').dropna()
        self.crispr['mageck_enr'] = pd.read_csv(f'{datadir}/{mageckenr_file}', index_col=0, sep='\t').dropna()
        self.crispr['binary_dep'] = pd.read_csv(f'{datadir}/{binarydep_file}', sep='\t', index_col=0).dropna()
        self.crispr['logFC'] = pd.read_csv(f'{datadir}/{foldchanges_file}', index_col=0, sep='\t').dropna()

    def get_data(self, dtype='logFC', fdr_thres=0.05, scale=True):
        """
        CRISPR-Cas9 scores as log fold-changes (CN corrected) or binary matrices marking (1) significant
        depletions (dtype = 'depletions'), enrichments (dtype = 'enrichments') or both (dtype = 'both').

        :param dtype: String (default = 'logFC')
        :param fdr_thres: Float (default = 0.1)
        :param scale: Boolean (default = False)
        :return: pandas.DataFrame
        """

        df = self.crispr[dtype]

        if dtype in ['mageck_dep', 'mageck_enr']:
            df = (df < fdr_thres).astype(int)

        if dtype == 'logFC' and scale:
            df = self.scale(df)

        return df

    def filter(self, subset=None, min_events=3, fdr_thres=0.05, abs_fc_thres=0.5, broad_paness=False):
        """
        Filter CRISPR-Cas9 data-set to consider only genes that show a significant depletion or
        enrichment, MAGeCK depletion/enrichment FDR < fdr_thres (default = 0.05), in at least
        min_events (default = 3) cell lines.

        :param subset:
        :param min_events:
        :param fdr_thres:
        :param abs_fc_thres:
        :param broad_paness:
        :return:
        """

        df = self.get_data(scale=True)

        # - Filters
        # Subset matrices
        if subset is not None:
            df = df.loc[:, df.columns.isin(subset)]

        # Filter by genes significantly depleted or enriched
        enriched_genes = self.get_data(dtype='mageck_enr', fdr_thres=fdr_thres)[df.columns]
        enriched_genes = enriched_genes[enriched_genes.sum(1) >= min_events]

        depleted_genes = self.get_data(dtype='binary_dep')[df.columns]
        depleted_genes = depleted_genes[depleted_genes.sum(1) >= min_events]

        df = df.loc[list(set(enriched_genes.index).union(depleted_genes.index))]

        # Filter by scaled fold-changes
        df = df[(df.abs() > abs_fc_thres).sum(1) >= min_events]

        # Filter by core-essential genes
        pancore_score = Utils.get_adam_core_essential()
        df = df[~df.index.isin(pancore_score)]

        # Filter by Broad core-essential genes
        if broad_paness:
            pancore_ceres = Utils.get_broad_core_essential()
            df = df[~df.index.isin(pancore_ceres)]

        # - Subset matrices
        for k in self.crispr:
            self.crispr[k] = self.crispr[k].loc[df.index, df.columns]

        return self

    @staticmethod
    def scale(df, essential=None, non_essential=None, metric=np.median):
        """
        Min/Max scaling of CRISPR-Cas9 log-FC by median (default) Essential and Non-Essential.

        :param df: Float pandas.DataFrame
        :param essential: set(String)
        :param non_essential: set(String)
        :param metric: np.Median (default)
        :return: Float pandas.DataFrame
        """

        if essential is None:
            essential = Utils.get_essential_genes(return_series=False)

        if non_essential is None:
            non_essential = Utils.get_non_essential_genes(return_series=False)

        assert len(essential.intersection(df.index)) != 0, \
            'DataFrame has no index overlapping with essential list'

        assert len(non_essential.intersection(df.index)) != 0, \
            'DataFrame has no index overlapping with non essential list'

        essential_metric = metric(df.reindex(essential).dropna(), axis=0)
        non_essential_metric = metric(df.reindex(non_essential).dropna(), axis=0)

        df = df.subtract(non_essential_metric).divide(non_essential_metric - essential_metric)

        return df


class Sample(object):
    def __init__(
            self,
            samplesheet_file='data/meta/samplesheet.csv',
            growthrate_file='data/gdsc/growth/growth_rates_screening_set_1536_180119.csv',
            ploidy_file='data/ploidy.csv',
            index='Cell Line Name'
    ):
        self.samplesheet = pd.read_csv(samplesheet_file).dropna(subset=[index]).set_index(index)
        self.ploidy = pd.read_csv(ploidy_file, index_col=0)['ploidy']
        self.growth = self.__assemble_growth_rates(growthrate_file)

    @staticmethod
    def __assemble_growth_rates(dfile):
        # Import
        dratio = pd.read_csv(dfile, index_col=0)

        # Convert to date
        dratio['DATE_CREATED'] = pd.to_datetime(dratio['DATE_CREATED'])

        # Group growth ratios per seeding
        d_nc1 = dratio.groupby(['CELL_LINE_NAME', 'SEEDING_DENSITY']).agg(
            {'growth_rate': [np.median, 'count'], 'DATE_CREATED': [np.max]}).reset_index()
        d_nc1.columns = ['_'.join(filter(lambda x: x != '', i)) for i in d_nc1]

        # Pick most recent measurements per cell line
        d_nc1 = d_nc1.iloc[d_nc1.groupby('CELL_LINE_NAME')['DATE_CREATED_amax'].idxmax()].set_index('CELL_LINE_NAME')

        return d_nc1

    def build_covariates(self, variables=None, add_growth=True, samples=None):
        variables = ['Cancer Type'] if variables is None else variables

        covariates = pd.get_dummies(self.samplesheet[variables])

        if add_growth:
            covariates = pd.concat([covariates, self.growth['growth_rate_median']], axis=1)

        if samples is not None:
            covariates = covariates.loc[samples]

        covariates = covariates.loc[:, covariates.sum() != 0]

        return covariates


class MOBEM(object):
    def __init__(
            self,
            drop_factors=True,
            mobem_file='data/PANCAN_mobem.csv'
    ):
        self.sample = Sample(index='COSMIC ID')

        mobem = pd.read_csv(mobem_file, index_col=0)
        mobem = mobem.set_index(self.sample.samplesheet.loc[mobem.index, 'Cell Line Name'], drop=True)
        mobem = mobem.T

        if drop_factors:
            mobem = mobem.drop(['TISSUE_FACTOR', 'MSI_FACTOR', 'MEDIA_FACTOR'])

        self.mobem = mobem.astype(int)

    def get_data(self):
        return self.mobem.copy()

    def filter(self, min_events=3):
        df = self.get_data()
        df = df[df.sum(1) >= min_events]
        self.mobem = self.mobem.loc[df.index, df.columns]
        return self

    @staticmethod
    def mobem_feature_to_gene(f):
        if f.endswith('_mut'):
            genes = {f.split('_')[0]}

        elif f.startswith('gain.') or f.startswith('loss.'):
            genes = {
                g for fs in f.split('..')
                if not (fs.startswith('gain.') or fs.startswith('loss.')) for g in fs.split('.') if g != ''
            }

        else:
            raise ValueError('{} is not a valid MOBEM feature.'.format(f))

        return genes

    @staticmethod
    def mobem_feature_type(f):
        if f.endswith('_mut'):
            return 'Mutation'

        elif f.startswith('gain.'):
            return 'CN gain'

        elif f.startswith('loss.'):
            return 'CN loss'

        else:
            raise ValueError('{} is not a valid MOBEM feature.'.format(f))