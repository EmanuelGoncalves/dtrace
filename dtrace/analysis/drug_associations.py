#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dtrace import DRUG_LMM


if __name__ == '__main__':
    # Linear regressions
    lm_df_crispr = pd.read_csv(trace.GROWTHRATE_FILE)
