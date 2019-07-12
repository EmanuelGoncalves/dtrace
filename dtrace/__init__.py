#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import sys
import logging
import seaborn as sns
from dtrace.DTracePlot import DTracePlot

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

# - Logger
__name__ = "DTrace"

logger = logging.getLogger(__name__)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("[%(asctime)s - %(levelname)s]: %(message)s"))
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# - DTrace handlers
__all__ = ["__version__", "__name__"]
