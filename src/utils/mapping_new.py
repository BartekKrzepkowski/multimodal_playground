import torch

from src.data.get_datasets import get_cifar100, cifar100_pair_dataset, imagenette_pair_dataset, cifar10_pair_dataset, get_mmimdb_pair_dataset, get_fhmd
from src.modules.cnn import FlexibleCNN
from src.modules.resnets import BimodalResNet
from src.modules.models import FusionMLP_BN, FusionMLP_EXE
from src.modules.models_pretrained import ResnetEncoder, TextEncoder

DATASET_NAME_MAP = {
    'cifar100': get_cifar100,
    'cifar10_pair': cifar10_pair_dataset,
    'cifar100_pair': cifar100_pair_dataset,
    'imagenette_pair': imagenette_pair_dataset,
    'mmimdb_pair': get_mmimdb_pair_dataset,
    'fhmd': get_fhmd
    }

MODEL_NAME_MAP = {
    # 'simple_cnn': SimpleCNN,
    'flexible_cnn': FlexibleCNN,
    'resnet': BimodalResNet,
    'fusion_mlp_bn': FusionMLP_BN,
    'fusion_mlp_exe': FusionMLP_EXE,
    'resnet_encoder_pretrained': ResnetEncoder,
    'text_encoder_pretrained': TextEncoder
}

OPTIMIZER_NAME_MAP = {
    'sgd': torch.optim.SGD,
    'adamw': torch.optim.AdamW
}

SCHEDULER_NAME_MAP = {
    'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'multiplicative': torch.optim.lr_scheduler.MultiplicativeLR,
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

CRITERION_NAME_MAP = {
    'cross_entropy': torch.nn.CrossEntropyLoss,
    'bce_with_logits': torch.nn.BCEWithLogitsLoss,
    'mse': torch.nn.MSELoss
}
