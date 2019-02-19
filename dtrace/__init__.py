#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import logging
import seaborn as sns
from DTracePlot import DTracePlot

__version__ = "0.5.0"

sns.set(
    style="ticks",
    context="paper",
    font_scale=0.75,
    font="sans-serif",
    rc=DTracePlot.SNS_RC,
)

__all__ = ["DTracePlot"]
