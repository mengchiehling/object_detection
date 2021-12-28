import os
from glob import glob
from datetime import datetime
from typing import Tuple, Dict

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from algorithms.io.path_definition import get_project_dir
from algorithms.settings import bayesianOptimization


def optimization_process(fn, pbounds: Dict) -> Tuple[Dict, np.ndarray]:

    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.

    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn

    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    export_form = datetime.now().strftime("%Y%m%d-%H:%M")

    optimization_file_dir = f"{get_project_dir()}/data/optimization"

    if not os.path.isdir(optimization_file_dir):
        os.makedirs(optimization_file_dir)

    logs = f"{optimization_file_dir}/logs_{export_form}.json"

    previous_logs = glob(f"{optimization_file_dir}/logs_*.json")

    if previous_logs:
        load_logs(optimizer, logs=previous_logs)
        bayesianOptimization['init_points'] = 0

    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        **bayesianOptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters