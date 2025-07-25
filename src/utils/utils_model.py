from math import sqrt

import torch
from torch import nn

from src.utils.utils_general import load_model


def default_init(m):
    """
    Initializes layers in the model with default strategies based on their type.

    - **Conv2d**: Kaiming normal for weights, zeros for bias.
    - **BatchNorm2d / GroupNorm**: Ones for weights, zeros for bias.
    - **Linear**: Uniform for weights, zeros for bias.

    Args:
        m (nn.Module): Layer to initialize.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        init_range = 1.0 / sqrt(m.out_features)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def prepare_model(model_params, init=None):
    from src.utils.mapping_new import MODEL_NAME_MAP
    """
    Prepares and initializes the model.

    - Loads a model architecture and optionally a checkpoint.
    - Applies default initialization if no checkpoint is provided.
    - Optionally freezes the backbone (encoder) parameters.

    Args:
        model_params (dict): 
            Parameters for model construction, must contain:
                - 'model_name': str, key for MODEL_NAME_MAP
                - 'model_params': dict, parameters for the model constructor
                - 'checkpoint_path': str or None, path to checkpoint
                - 'freeze_backbone': bool, whether to freeze encoder parameters
        init (callable, optional): 
            Initialization function to apply (default: None, uses default_init).

    Returns:
        nn.Module: Prepared model, ready for training or inference.
    """
    model = MODEL_NAME_MAP[model_params['model_name']](**model_params['model_params'])
    if model_params['checkpoint_path'] is not None:
        model = load_model(model, model_params['checkpoint_path'])
    else:
        model.apply(default_init)

    if model_params['freeze_backbone']:
        for param in model.encoder.parameters():
            param.requires_grad = False
    return model



def infer_dims_from_blocks(blocks, x):
    x = blocks(x)
    _, channels_out, height, width = x.shape
    return channels_out, height, width