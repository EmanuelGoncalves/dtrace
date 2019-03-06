#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import seaborn as sns
from dtrace.DTracePlot import DTracePlot
from dtrace.DTraceUtils import dpath, rpath, logger
from dtrace.DTraceEnrichment import DTraceEnrichment

# - Version
__version__ = "0.5.0"

# - Package paths

# - Plot main default aesthetics
sns.set(
    style="ticks",
    context="paper",
    font_scale=0.75,
    font="sans-serif",
    rc=DTracePlot.SNS_RC,
)

# - DTrace handlers
__all__ = ["DTracePlot", "DTraceEnrichment", "dpath", "rpath", "logger"]
