import torch
from .unet2d import LocalEquivUNet2DCondModel
from .kamb import MinimalResNet, MinimalUNet
from .mlp import PatchMLP
from .precond import EDMPrecond, CMPrecond
from putils.config import MainConfig
from training.utils import ema_avg_fn
from torch.optim.swa_utils import AveragedModel

import sys 

__all__ = [
    "LocalEquivUNet2DCondModel",
    "MinimalResNet",
    "MinimalUNet",
    "PatchMLP",
    "EDMPrecond",
    "CMPrecond",
]


MODEL_TYPE = dict(
    unet2d=LocalEquivUNet2DCondModel,
    resnet=MinimalResNet,
    mlp=PatchMLP,
)

MODEL_TYPE_TO_CONFIG_KEY = dict(
    unet2d="LocalEquivUNet2DCondModel",
    resnet="MinimalResNet",
    mlp="PatchMLP",
)

HF_HUB = "https://huggingface.co/mhnguyen712/analytical_mmse/resolve/main/"

def load_pretrain_model(
    model_type: str = "unet2d",
    patch_size: int = 5,
    dataset: str = "FashionMNIST",
    color: bool = True,
    operator: str = "denoising",
    preinverse_type: str = "identity",
):
    """Load a pre-trained model for a specific dataset and operator.

    Args:
        model_type (str): Type of the model architecture. Options are 'unet2d', 'resnet', 'mlp'.
        patch_size (int): Size of the input patches.
        dataset (str): Name of the dataset. Options are 'FashionMNIST', 'CIFAR10' or 'FFHQ-32' or 'FFHQ-64'.
        color (bool): Whether the input images are color (True) or grayscale (False).
        preinverse_type (str): Type of the pre-inverse operator used. Options are 'identity', 'inverse'.
    """
    
    patch_size = int(patch_size)
    
    assert model_type in MODEL_TYPE.keys(), f"Model type {model_type} not recognized."
    
    model_class = MODEL_TYPE[model_type]
    if not '64' in dataset:  # For 32x32 images
        model_kwargs = MainConfig.local_equivariant_model_kwargs[
            f"{MODEL_TYPE_TO_CONFIG_KEY[model_type]}_{patch_size}"
        ]
    else:   # For 64x64 images
        model_kwargs = MainConfig.local_equivariant_model_64_kwargs[
            f"{MODEL_TYPE_TO_CONFIG_KEY[model_type]}_{patch_size}"
        ]
        
    n_channels = 3 if color else 1
    model = model_class(
        in_channels=n_channels, out_channels=n_channels, **model_kwargs
    )
    
    # Load pre-trained weights
    # TODO
    ckpt_path = f"{MODEL_TYPE_TO_CONFIG_KEY[model_type].lower()}_patch_{patch_size}/{operator.lower()}_{dataset}_subset_10000"
    
    if preinverse_type != "identity":
        ckpt_path += "_inform"
        
    ckpt_path += "/checkpoints/checkpoint.pth"
    ckpt = torch.hub.load_state_dict_from_url(
        HF_HUB + ckpt_path, map_location="cpu"
    )
    model = AveragedModel(model, avg_fn=ema_avg_fn)
    model.load_state_dict(ckpt)
    return model
