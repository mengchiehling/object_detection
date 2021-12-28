import pandas as pd

from algorithms.io.metadata_definition import set_bayesian_optimization_config, get_app_config

globals().update(set_bayesian_optimization_config())
globals().update(get_app_config())