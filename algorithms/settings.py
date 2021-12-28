import os

import pandas as pd
import pkg_resources

from algorithms.io.metadata_definition import set_bayesian_optimization_config, get_app_config

globals().update(set_bayesian_optimization_config())
globals().update(get_app_config())

MODE = "package" # package/io

if MODE == 'package':
    home_dir = 'algorithms'
else:
    home_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)


def set_directory(MODE):

    # change this via a parameter control file

    if MODE == 'package':
        DIR_MODEL_ROOT = pkg_resources.resource_filename(home_dir, f"{DATA_ROOT}/app/models")
        DIR_DATA_ROOT = pkg_resources.resource_filename(home_dir, f"{DATA_ROOT}/app/benchmarks")
    else:
        DIR_MODEL_ROOT = os.path.join(home_dir, DATA_ROOT, 'app/models')
        DIR_DATA_ROOT = os.path.join(home_dir, DATA_ROOT, 'app/benchmarks')

    globals().update({"DIR_MODEL": DIR_MODEL_ROOT,
                      "DIR_DATA": DIR_DATA_ROOT})