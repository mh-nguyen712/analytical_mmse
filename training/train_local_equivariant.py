################################################################################
#           TRAINING SCRIPT FOR LOCAL AND TRANSLATION EQUIVARIANT MODEL
################################################################################

import torch
import torch.nn as nn
from tqdm import tqdm
from utils import OptimizationConfig, TrainingConfig, LoggingConfig, OnlineMovingAverage, ema_avg_fn
from typing import Callable
from analytic.operator import Operator, get_operator, get_preinverse_operator, ConvolutionOperatorFFT
import deepinv as dinv
from torch.optim.swa_utils import AveragedModel
import os
import sys
import h5py
import numpy as np

def training_loop(
    model: nn.Module,
    operator: Operator,
    pre_inverse: Operator,
    generator: dinv.physics.generator.SigmaGenerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: torch.utils.data.DataLoader,
    criterion: Callable,
    config: TrainingConfig,
    logger: LoggingConfig = None,
    analytical_results: dict = None,
):

    # Initialize EMA for model weights using PyTorch's AveragedModel
    swa_start = 1000  # Start using SWA after 1000 iterations

    model.to(config.device)
    ema_model = AveragedModel(model, avg_fn=ema_avg_fn, use_buffers=True)
    ema_model = ema_model.to(config.device)

    state = logger.load_checkpoint()
    if state is not None:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        lr_scheduler.load_state_dict(state["scheduler_state_dict"])
        ema_model.load_state_dict(state["ema_model_state_dict"])
        global_step = state["global_step"]
        start_epoch = state["epoch"]
    else:
        print("State dict is NONE")
        global_step = 0
        start_epoch = 0

    operator.to(config.device)
    avg_loss = OnlineMovingAverage(size=5000)
    model.train()
    psnr_fn = dinv.metric.PSNR(max_pixel=None, min_pixel=None)
    config.num_epochs = config.num_epochs + start_epoch
    for epoch in range(start_epoch, config.num_epochs):
        pb = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", mininterval=10
        )
        for x in pb:
            if isinstance(x, (tuple, list)):
                x = x[0]
            x = x.to(config.device)
            y = operator.forward(x)
            sigma = generator.step(x.size(0))["sigma"]
            y = y + sigma.view(-1, 1, 1, 1) * torch.randn_like(y)

            optimizer.zero_grad()
            output = model(pre_inverse.forward(y), sigma)
            loss = criterion(output, x)
            if torch.isnan(loss):
                # Skip if loss is nan
                continue
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            if global_step > swa_start and global_step % 5 == 0:
                ema_model.update_parameters(model)

            avg_loss.update(loss.item())
            pb.set_description(
                f"Step: {global_step:.2e}, avg_loss: {avg_loss.mean:.3e}"
            )

            if global_step % logger.log_loss_freq == 0 or global_step == 1:
                metrics = {
                    "avg_loss": avg_loss.mean,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_grad_norm": grad_norm.max(),
                }
                logger.log_metrics(metrics, step=global_step)
                logger.log_histogram(grad_norm, "grad_norm", step=global_step)

            if global_step % logger.log_images_freq == 0 or global_step == 1:
                pinv_y = pre_inverse.forward(y)
                logger.log_images(
                    {"x": x, "x_hat": output, "y": y, "pinv": pinv_y}, global_step
                )

        # Validation loop
        if (epoch + 1) % logger.val_epoch_freq == 0:
            model.eval()
            ema_model.eval()
            with torch.no_grad():
                for val_key, analytical_path in analytical_results.items():
                    try:
                        with h5py.File(analytical_path, "r") as f:
                            group = f[args.operator]
                            for sigma in MainConfig.noise_levels:
                                sigma_key = f"sigma_{sigma:g}"
                                if sigma_key not in group:
                                    raise KeyError(
                                        f"Sigma {sigma_key} not found in the analytical results file."
                                    )
                                sigma_group = group[sigma_key]

                                x = torch.from_numpy(sigma_group["x"][:]).to(
                                    device=device, dtype=torch.float32
                                )
                                y = torch.from_numpy(sigma_group["y"][:]).to(
                                    device=device, dtype=torch.float32
                                )
                                analytical_output = torch.from_numpy(
                                    sigma_group["x_hat"][:]
                                ).to(device=device, dtype=torch.float32)
                                analytical_psnr = sigma_group["psnr"][:].mean()

                                pinv_y = pre_inverse.forward(y)
                                sigma_tensor = (
                                    torch.ones(
                                        x.size(0), device=device, dtype=torch.float32
                                    )
                                    * sigma
                                )
                                ema_output = ema_model(pinv_y, sigma_tensor)
                                output = model(pinv_y, sigma_tensor)

                                model_psnr = psnr_fn(output, x).mean()
                                ema_psnr = psnr_fn(ema_output, x).mean()
                                analytical_vs_net_psnr = psnr_fn(
                                    output, analytical_output
                                ).mean()
                                analytical_vs_ema_net_psnr = psnr_fn(
                                    ema_output, analytical_output
                                ).mean()

                                metrics = {
                                    f"psnr": model_psnr,
                                    f"ema_psnr": ema_psnr,
                                    f"analytical_psnr": analytical_psnr,
                                    f"analytical_vs_net_psnr": analytical_vs_net_psnr,
                                    f"analytical_vs_ema_net_psnr": analytical_vs_ema_net_psnr,
                                }
                                logger.log_metrics(
                                    metrics, step=epoch, prefix=f"{val_key}_{sigma_key}"
                                )
                                index = np.random.randint(0, x.size(0), size=(3,))
                                fig = dinv.utils.plot(
                                    [
                                        x[index],
                                        y[index],
                                        output[index],
                                        ema_output[index],
                                        pinv_y[index],
                                        analytical_output[index],
                                    ],
                                    titles=[
                                        "x",
                                        "y",
                                        "output",
                                        "ema_output",
                                        "pinv_y",
                                        "analytical_output",
                                    ],
                                    return_fig=True,
                                    show=False,
                                )
                                logger.log_figure(
                                    fig, f"{val_key}_{sigma_key}_images", step=epoch
                                )
                    except:
                        print("FILE NOT FOUND! SKIP VALIDATION!")
                        pass
            
            model.train()
            ema_model.train()

        # Save checkpoint
        if (epoch + 1) == config.num_epochs or (epoch + 1) % logger.save_freq == 0:
            state = {
                "model_state_dict": model.state_dict(),
                "ema_model_state_dict": ema_model.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler else None
                ),
            }

            if (epoch + 1) % logger.save_freq == 0:
                logger.clean_old_checkpoints(top_k=2)

            logger.save_checkpoint(state, epoch, metric_value=avg_loss.mean)

if __name__ == "__main__":
    from model import (
        LocalEquivUNet2DCondModel,
        EDMPrecond,
        CMPrecond,
        MinimalResNet,
        PatchMLP,
    )
    from putils.datasets import get_dataset
    from putils.config import MainConfig
    import torch.utils.data as data
    import argparse

    torch.set_float32_matmul_precision("high")  # To use TensorCore

    parser = argparse.ArgumentParser(description="Training script for local-equivariant model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="FashionMNIST",
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10000,
        help="Number of images to use for training",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=600, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LocalEquivUNet2DCondModel",
        choices=MainConfig.local_equivariant_model,
        help="Model architecture to use",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=5,
        choices=[5, 7, 9, 11, 17, 25],
        help="Patch size to use",
    )
    parser.add_argument(
        "--preinverse",
        type=str,
        default="identity",
        choices=["identity", "inverse", "adjoint", "epsilon_inverse"],
        help="Preinverse method to use",
    )

    parser.add_argument(
        "--precond",
        type=str,
        default=None,
        help="Preconditioning method to use (edm or cm) or None",
    )
    parser.add_argument(
        "--operator",
        type=str,
        default="denoising",
        choices=MainConfig.operators,
        help="Operator",
    )

    args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset and loader
    dataset_name = args.dataset
    assert (
        dataset_name in MainConfig.datasets
    ), f"Dataset {dataset_name} not found in config"
    color = MainConfig.colors[MainConfig.datasets.index(dataset_name)]
    dataset = get_dataset(dataset_name, train=True, color=color)

    if args.dataset_size is not None:
        print(
            f"Using only the first {min(args.dataset_size, len(dataset))} images from the dataset"
        )
        dataset = data.Subset(
            dataset, range(min(args.dataset_size, len(dataset)))
        )

    if len(dataset) < args.batch_size:
        sampler = data.RandomSampler(
            dataset, replacement=True, num_samples=args.batch_size
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
        num_workers=8,
    )

    # Create the model
    if args.model not in MainConfig.local_equivariant_model:
        raise ValueError(
            f"Model {args.model} is not supported. Supported models: {MainConfig.local_equivariant_model}"
        )
    patch_size = int(args.patch_size)
    model_class = getattr(
        sys.modules[__name__], args.model
    )  # Dynamically get the model class
    if not '64' in dataset_name:  # For 32x32 images
        model_kwargs = MainConfig.local_equivariant_model_kwargs[
            f"{args.model}_{patch_size}"
        ]
    else:   # For 64x64 images
        model_kwargs = MainConfig.local_equivariant_model_64_kwargs[
            f"{args.model}_{patch_size}"
        ]
        
    n_channels = 3 if color else 1
    model = model_class(
        in_channels=n_channels, out_channels=n_channels, **model_kwargs
    ).to(device=device, dtype=dtype)

    print("Model architecture:", model)
    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Add preconditioning if specified
    if args.precond == "edm":
        model = EDMPrecond(model)
    elif args.precond == "cm":
        model = CMPrecond(model)

    model.compile()

    # Create the operator
    operator_name = args.operator
    batch = next(iter(train_loader))
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    input_size = batch[0].shape
    print(input_size)
    if 'convolution' in operator_name:
        kernel = get_operator(
                    operator_name, input_size=input_size, device=device, dtype=dtype, get_operator_param=True,
                )
        # For faster convolution using FFT
        operator = ConvolutionOperatorFFT(filter=kernel, input_size=input_size, device=device, dtype=dtype)
    else:
        operator = get_operator(
            operator_name, input_size=input_size, device=device, dtype=dtype
        )

    pre_inverse = get_preinverse_operator(
        operator_name,
        pre_inverse_type=args.preinverse,
        input_size=input_size,
        device=device,
        dtype=dtype,
    )

    generator = dinv.physics.generator.SigmaGenerator(
        sigma_min=MainConfig.noise_training[0],
        sigma_max=MainConfig.noise_training[1],
        device=device,
        dtype=dtype,
    )

    optim_config = OptimizationConfig()

    optimizer = optim_config.get_optimizer(model)
    lr_scheduler = optim_config.get_scheduler(optimizer)
    criterion = nn.MSELoss(reduction="mean")

    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    save_dir = os.path.join(
        abs_path, "neural_results", f"{args.model.lower()}_patch_{patch_size}"
    )
    os.makedirs(save_dir, exist_ok=True)

    exp_name = f"{operator_name.lower()}_{dataset_name.replace('/', '_')}"
    if args.dataset_size is not None:
        exp_name += f"_subset_{int(args.dataset_size)}"
    if args.precond is not None:
        exp_name += f"_precond_{args.precond.lower()}"
    if args.preinverse != "identity":
        exp_name += "_inform"

    logger = LoggingConfig(project_dir=save_dir, exp_name=exp_name)
    logger.monitor_metric = "avg_loss"
    logger.monitor_mode = "min"
    logger.initialize()
    logger._save_metadata(optional_info=vars(args))

    training_config = TrainingConfig()
    training_config.device = device
    training_config.update(**vars(args))

    logger.log_hyperparameters(vars(args), main_key="training_config")
    logger.log_hyperparameters(model_kwargs, main_key="model")

    # Save checkpoint every N steps
    logger.save_freq = 10
    logger.val_epoch_freq = 10
    logger.log_loss_freq = 10
    logger.log_images_freq = 200

    # Load the pre-computed analytical results
    analytical_exp_name = dataset_name.replace("/", "_")
    if args.dataset_size is not None:
        analytical_exp_name += f"_subset_{int(args.dataset_size)}"

    estimator_name = f"loc_equiv_mmse_{patch_size}"
    ext_file_path = ".h5" if args.preinverse == "identity" else "_inform.h5"
    analytical_result_train_path = os.path.join(
        abs_path,
        "analytical_results",
        analytical_exp_name + f"_train_{estimator_name}{ext_file_path}",
    )
    analytical_result_test_path = os.path.join(
        abs_path,
        "analytical_results",
        analytical_exp_name + f"_test_{estimator_name}{ext_file_path}",
    )

    if not os.path.exists(analytical_result_train_path):
        print(
            f"Analytical training result file not found: {analytical_result_train_path}"
        )
        print("Ignoring validation...")
        logger.val_epoch_freq = 10000000000000000000000000
    if not os.path.exists(analytical_result_test_path):
        print(f"Analytical test result file not found: {analytical_result_test_path}")
        print("Ignoring validation...")
        logger.val_epoch_freq = 10000000000000000000000000

    analytical_results = dict(
        on_train_set=analytical_result_train_path,
        on_test_set=analytical_result_test_path,
    )

    training_loop(
        model,
        operator,
        pre_inverse,
        generator=generator,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        criterion=criterion,
        config=training_config,
        logger=logger,
        analytical_results=analytical_results,
    )

# python3 training/train_local_equivariant.py --batch_size 256 --dataset_size 10000 --operator denoising --preinverse identity --num_epochs 600
