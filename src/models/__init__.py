from .standard import *

def get_model(model_name: str):
    models = {
        "ESPCN4x": ESPCN4x,
    }
    try:
        return models[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}")