#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import dtrace
import numpy as np
import pandas as pd


def assemble_growth_rates(seed_intensities_file):
    # Import
    dratio = pd.read_csv(seed_intensities_file, index_col=0)

    # Convert to date
    dratio['DATE_CREATED'] = pd.to_datetime(dratio['DATE_CREATED'])

    # Group growth ratios per seeding
    d_nc1 = dratio.groupby(['CELL_LINE_NAME', 'SEEDING_DENSITY']).agg({'growth_rate': [np.median, 'count'], 'DATE_CREATED': [np.max]}).reset_index()
    d_nc1.columns = ['_'.join(filter(lambda x: x != '', i)) for i in d_nc1]

    # Pick most recent measurements per cell line
    d_nc1 = d_nc1.iloc[d_nc1.groupby('CELL_LINE_NAME')['DATE_CREATED_amax'].idxmax()].set_index('CELL_LINE_NAME')

    return d_nc1


if __name__ == '__main__':
    # Import/assemble growth rates
    grate = assemble_growth_rates('data/gdsc/growth/growth_rates_screening_set_1536_180119.csv')

    # Export
    grate.to_csv(dtrace.GROWTHRATE_FILE)
