"""Architecture factory: get_model(name, num_classes)."""

from .resnet20 import ResNet20
from .mlpmixer import MLPMixer
from .convnext import ConvNeXt

_MODELS = {"resnet20": ResNet20, "mlpmixer": MLPMixer, "convnext": ConvNeXt}


def get_model(name, num_classes=10):
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f"unknown model '{name}'; choose from {list(_MODELS)}")
    return _MODELS[name](num_classes=num_classes)
