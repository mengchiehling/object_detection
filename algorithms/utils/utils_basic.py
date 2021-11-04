import os

from visual_search.settings import INFERENCE_MODEL


def load_model_key(model_name: str):

    inference_model = INFERENCE_MODEL

    model_key = inference_model[model_name]

    return os.path.join(model_name, model_key)