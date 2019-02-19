#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import sys
import logging
import pkg_resources
import seaborn as sns
from DTracePlot import DTracePlot

# - Version
__version__ = "0.5.0"

# - Package data path
dpath = pkg_resources.resource_filename("dtrace", "data/")

# - Plot main default aesthetics
sns.set(
    style="ticks",
    context="paper",
    font_scale=0.75,
    font="sans-serif",
    rc=DTracePlot.SNS_RC,
)

# - DTrace handlers
__all__ = ["DTracePlot", "logger", "dpath"]

# - Logging
logger = logging.getLogger()

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s - %(levelname)s]: %(message)s"))
logger.addHandler(ch)

logger.setLevel(logging.INFO)
