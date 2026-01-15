## Configuration for numerical experiments
# This file contains the configuration for numerical experiments, including:
# - Datasets
# - Operators
# - Neural networks architectures

class MainConfig:
    datasets = ("FashionMNIST", "MNIST", "CIFAR10", "FFHQ/images32x32", "FFHQ/images64x64", "FFHQ/images32x32full70k")  # dataset names
    colors = (False, False, True, True, True, True)  # Whether the dataset is color or not

    operators = (
        "denoising",
        "inpainting_center_15",
        "inpainting_random_30",
        "convolution_gaussian_1.0",
        "convolution_defocus_2.5",
    )  # operator names
    noise_levels = (
        0.001,
        0.01,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0,
    )  # noise levels for evaluation
    noise_training = (0.0, 1.0)  # noise levels for training

    num_samples = 100  # number of samples for reporting metrics for each dataset
   
    local_equivariant_model = (
        "LocalEquivUNet2DCondModel",
        "MinimalResNet",
        "PatchMLP",
    )
    local_equivariant_model_kwargs = dict(
        LocalEquivUNet2DCondModel_11={
            "block_out_channels": (32, 64, 128, 256),
            "kernel_size_down_blocks": (3, 1, 1, 1, 3),
            "kernel_size_mid_blocks": 1,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        LocalEquivUNet2DCondModel_9={
            "block_out_channels": (64 + 32, 128 + 64, 256 + 32),
            "kernel_size_down_blocks": (3, 1, 1, 1),
            "kernel_size_mid_blocks": 5,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        LocalEquivUNet2DCondModel_7={
            "block_out_channels": (128 - 64, 256 - 64, 512 - 64),
            "kernel_size_down_blocks": (3, 1, 1, 1),
            "kernel_size_mid_blocks": 3,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        LocalEquivUNet2DCondModel_5={
            "block_out_channels": (128 - 32, 256 - 32, 512 - 32),
            "kernel_size_down_blocks": (3, 1, 1, 1),
            "kernel_size_mid_blocks": 1,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        MinimalResNet_11={
            "emb_dim": 328,
            "mode": "circular",
            "conditional": None,
            "num_classes": None,
            "kernel_size": 3,
            "num_layers": 4,
            "normalization": "BatchNorm",
            "lastksize": 1,
        },  # ~4M parameters
        MinimalResNet_9={
            "emb_dim": 128 * 3,
            "mode": "circular",
            "conditional": None,
            "num_classes": None,
            "kernel_size": 3,
            "num_layers": 3,
            "normalization": "BatchNorm",
            "lastksize": 1,
        },  # ~4M parameters
        MinimalResNet_7={
            "emb_dim": int(128 * 3.5),
            "mode": "circular",
            "conditional": None,
            "num_classes": None,
            "kernel_size": 3,
            "num_layers": 2,
            "normalization": "BatchNorm",
            "lastksize": 1,
        },  # ~4M parameters
        MinimalResNet_5={
            "emb_dim": 128 * 5,
            "mode": "circular",
            "conditional": None,
            "num_classes": None,
            "kernel_size": 3,
            "num_layers": 1,
            "normalization": "BatchNorm",
            "lastksize": 1,
        },  # ~4M parameters
        PatchMLPMix_11={
            "patch_size": 11,
            "depth": 5,
            "hidden_dim": 4096 * 2,
            "mid_channels": 4,
            "chunk_size": 4096,
        },
        PatchMLP_11={
            "patch_size": 11,
            "depth": 5,
            "hidden_dim": 3072,
            "mid_channels": 8,
            "chunk_size": 4096,
        },
        PatchMLP_9={
            "patch_size": 9,
            "depth": 5,
            "hidden_dim": 3072 + 2 * 1024,
            "mid_channels": 8,
            "chunk_size": 4096,
        },
        PatchMLP_7={
            "patch_size": 7,
            "depth": 5,
            "hidden_dim": 3072 + 3 * 1024,
            "mid_channels": 8,
            "chunk_size": 4096,
        },
        PatchMLP_5={
            "patch_size": 5,
            "depth": 5,
            "hidden_dim": 3072 + 4 * 1024,
            "mid_channels": 8,
            "chunk_size": 4096,
        },
        LocalEquivUNet2DCondModel_17={
            "block_out_channels": (32, 64, 128, 256),
            "kernel_size_down_blocks": (3, 1, 3, 1, 3),
            "kernel_size_mid_blocks": 3,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        MinimalResNet_17={
            "emb_dim": 256,
            "mode": "circular",
            "conditional": None,
            "num_classes": None,
            "kernel_size": 3,
            "num_layers": 7,
            "normalization": "BatchNorm",
            "lastksize": 1,
        },  # ~4M parameters
        PatchMLP_17={
            "patch_size": 17,
            "depth": 5,
            "hidden_dim": 3072,
            "mid_channels": 8,
            "chunk_size": None,
        },
    )
    local_equivariant_model_64_kwargs = dict(
        # THIS IS FOR 64x64 IMAGES
        LocalEquivUNet2DCondModel_25={
            "block_out_channels": (64, 128, 256, 512),
            "kernel_size_down_blocks": (3, 3, 3, 3, 3),
            "kernel_size_mid_blocks": 3,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
        LocalEquivUNet2DCondModel_11={
            "block_out_channels": (64, 128, 256, 512),
            "kernel_size_down_blocks": (3, 1, 1, 1, 3),
            "kernel_size_mid_blocks": 1,
            "layers_per_block": 1,
            "norm_num_groups": 16,
            "norm_eps": 1e-5,
            "center_input_sample": True,
            "act_fn": "silu",
            "dropout": 0.0,
        },
    )