import configparser
from typing import Dict

import yaml

from visual_search.io.path_definition import get_file


def _load_yaml(file) -> Dict:

    with open(file, 'r') as f:
        loaded_yaml = yaml.full_load(f)
    return loaded_yaml


def get_app_config() -> Dict:

    """
    Get all the parameters as a dictionary for the app

    """

    app_config = _load_yaml(get_file("config/app_config.yml"))
    return app_config
