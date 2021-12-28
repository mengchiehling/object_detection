'''
Load the model weights
'''

import importlib

if __name__ == "__main__":

    modules = [importlib.import_module(f"algorithm.io.{model}") for model in ['delf', 'mask_rcnn', 'yolo']]

    for module in modules:
        module.load()