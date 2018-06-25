#!/usr/bin/env python
# Copyright (C) 2018 Emanuel Goncalves

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
