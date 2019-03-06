#!/usr/bin/env python
# Copyright (C) 2019 Emanuel Goncalves

import sys
import logging
import pkg_resources


# - Paths
dpath = pkg_resources.resource_filename("dtrace", "data/")
rpath = pkg_resources.resource_filename("notebooks", "reports/")


# - Logging
logger = logging.getLogger()

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("[%(asctime)s - %(levelname)s]: %(message)s"))
logger.addHandler(ch)

logger.setLevel(logging.INFO)