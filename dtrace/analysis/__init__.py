#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import seaborn as sns
import matplotlib.pyplot as plt


# - PALETTES
PAL_SET2 = sns.color_palette('Set2', n_colors=8).as_hex()
PAL_DTRACE = [PAL_SET2[1], '#E1E1E1', '#656565']

# - DEFAULT AESTHETICS
SNS_RC = {
    'axes.linewidth': .3,
    'xtick.major.width': .3,
    'ytick.major.width': .3,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
}

sns.set(style='ticks', context='paper', rc=SNS_RC, font_scale=.75)
