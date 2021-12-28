import os
import yaml
from setuptools import setup, find_packages


def _load_yaml(file):
    with open(os.path.join(os.path.dirname(__file__), file), 'r') as f:
        loaded_yaml = yaml.full_load(f)
    return loaded_yaml


def readme():
    with open('README.md') as f:
        return f.read()


def package_files(directory):
    # recusively find all the data we need
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths


extra_files = package_files('config') + package_files('data')

meta = _load_yaml(os.path.join('build', "conda_recipe", "meta.yaml"))
PACKAGE = meta['package']

setup(
    name=PACKAGE['name'],
    version=PACKAGE['version'],
    description='DeLF(Deep Local Feature) v1 from tensorflow hub, trained on google landscape dataset;'
                'YOLOv3 for brand detections',
    long_description=readme(),
    url='https://datascience-conda-packages.fashion.check24.de/',
    packages=find_packages(),
    package_data={"": extra_files}
)