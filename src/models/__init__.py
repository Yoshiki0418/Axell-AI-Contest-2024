from .standard import *
from .original import *

def get_model(model_name: str):
    models = {
        "ESPCN4x": ESPCN4x,
        "SRResNet4x": SRResNet4x,
        "FusionSR": FusionSR,
        "LiteFusionSR": LiteFusionSR,
        "DeepESPCN4x": DeepESPCN4x,
        "DeepESPCN4x_v2": DeepESPCN4x_v2,
        "StepShuffleSR": StepShuffleSR,
    }
    try:
        return models[model_name]
    except KeyError:
        raise ValueError(f"Unknown model name: {model_name}")