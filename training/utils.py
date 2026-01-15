import torch 
import torch.optim as optim
import warnings
from torch.utils.tensorboard import SummaryWriter
import psutil
import json
import glob
import re
import os


class OptimizationConfig:
    # Optimizer settings
    optimizer = "adamw"  # Options: adamw, adam, sgd, rmsprop
    initial_lr = 1e-3
    weight_decay = 1e-4

    # AdamW and Adam specific parameters
    betas = (0.9, 0.999)
    eps = 1e-8

    # SGD specific parameters
    momentum = 0.9
    nesterov = True

    # RMSprop specific parameters
    alpha = 0.99
    centered = False

    # Learning rate scheduler settings
    lr_scheduler = "cosine"  # Options: cosine, step, plateau, onecycle
    min_lr = 1e-6

    # CosineAnnealingLR specific parameters
    epochs = 60  # T_max for cosine scheduler

    # StepLR specific parameters
    step_size = 30
    gamma = 0.1

    # ReduceLROnPlateau specific parameters
    factor = 0.1
    patience = 10
    mode = "min"

    # OneCycleLR specific parameters
    max_lr = 1e-2
    pct_start = 0.3
    div_factor = 25.0
    final_div_factor = 1e4

    def get_optimizer(self, model, **kwargs):
        """
        Get the optimizer based on the configuration.

        Args:
            model: The model whose parameters need to be optimized
            **kwargs: Additional arguments to pass to the optimizer
        """
        params = model.parameters()

        if self.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.initial_lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                **kwargs,
            )
        elif self.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.initial_lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
                **kwargs,
            )
        elif self.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.initial_lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
                **kwargs,
            )
        elif self.optimizer == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.initial_lr,
                alpha=self.alpha,
                eps=self.eps,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                centered=self.centered,
                **kwargs,
            )
        else:
            warnings.warn(f"Optimizer {self.optimizer} not found. Using AdamW instead.")
            return optim.AdamW(params, lr=self.initial_lr, **kwargs)

    def get_scheduler(self, optimizer, **kwargs):
        """
        Get the learning rate scheduler based on the configuration.

        Args:
            optimizer: The optimizer whose learning rate needs to be scheduled
            **kwargs: Additional arguments to pass to the scheduler
        """
        if self.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.min_lr, **kwargs
            )
        elif self.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=self.step_size, gamma=self.gamma, **kwargs
            )
        elif self.lr_scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.mode,
                factor=self.factor,
                patience=self.patience,
                min_lr=self.min_lr,
                **kwargs,
            )
        elif self.lr_scheduler == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_lr,
                epochs=self.epochs,
                pct_start=self.pct_start,
                div_factor=self.div_factor,
                final_div_factor=self.final_div_factor,
                **kwargs,
            )
        else:
            warnings.warn(
                f"Scheduler {self.lr_scheduler} not found. Using constant lr instead."
            )
            return optim.lr_scheduler.ConstantLR(optimizer, **kwargs)


class TrainingConfig:
    num_epochs: int = 600
    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def update(self, **kwargs):
        """
        Update the configuration with new values.

        Args:
            **kwargs: New values for the configuration parameters
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"New argument: {key}")
                setattr(self, key, value)


class LoggingConfig:
    # Logging frequencies
    log_freq = 10  # Log metrics every N iterations
    val_freq = 1  # Run validation every N epochs
    save_freq = 1  # Save checkpoint every N epochs

    # TensorBoard settings
    tensorboard = True
    tensorboard_dir = "runs"  # Directory for tensorboard logs
    exp_name = "default"  # Experiment name for the run

    # Checkpoint settings
    checkpoint_dir = "checkpoints"
    save_best_only = True  # Only save when model achieves best validation score
    save_last = True  # Always save the last checkpoint
    max_checkpoints = 5  # Maximum number of checkpoints to keep

    # Metrics to monitor
    monitor_metric = "val_loss"  # Metric to monitor for best model
    monitor_mode = "min"  # 'min' for loss, 'max' for metrics like PSNR

    # Logging content settings
    log_images = True  # Log sample images to tensorboard
    log_images_freq = 200  # Frequency of image logging (iterations)
    num_log_images = 4  # Number of images to log
    log_loss_freq: int = 10
    val_epoch_freq: int = 5

    # System monitoring
    log_gpu_stats = False  # Log GPU utilization
    log_memory_stats = False  # Log memory usage

    def __init__(self, project_dir=None, exp_name="default", **kwargs):
        self.best_metric = float("inf") if self.monitor_mode == "min" else float("-inf")
        self.global_step = 0
        self.epoch = 0
        self.writer = None
        self.metadata_file = None

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Unknown argument: {key}")

        self.project_dir = "" if project_dir is None else project_dir
        # Create unique experiment directory with timestamp

        self.exp_dir = os.path.join(self.project_dir, f"{exp_name}")
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.tensorboard_dir = os.path.join(self.exp_dir, "runs")

    def initialize(self):
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Initialize tensorboard writer
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        else:
            self.writer = None

        # Initialize best metric
        self.best_metric = float("inf") if self.monitor_mode == "min" else float("-inf")

        # Initialize global step
        self.global_step = 0

        # Initialize start epoch
        self.epoch = 0

        self.metadata_file = os.path.join(self.exp_dir, "metadata.json")
        self._save_metadata()

    def _save_metadata(self, optional_info: dict = None):
        """Save metadata about the training progress"""
        metadata = {
            "exp_name": self.exp_name,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "monitor_metric": self.monitor_metric,
            "monitor_mode": self.monitor_mode,
            "tensorboard_logdir": self.tensorboard_dir if self.tensorboard else None,
        }
        if optional_info is not None:
            self._optional_meta_data = optional_info
        else:
            self._optional_meta_data = {}

        metadata.update(self._optional_meta_data)

        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=4)

    def get_checkpoint_path(self, epoch, metric_value=None):
        """
        Generate checkpoint path based on epoch and metric value.
        """
        if metric_value is not None:
            return os.path.join(
                self.checkpoint_dir,
                f"epoch_{epoch:03d}_{self.monitor_metric}_{metric_value:.4f}.pth",
            )
        return os.path.join(self.checkpoint_dir, f"epoch_{epoch:03d}.pth")

    def save_checkpoint(self, state, epoch, metric_value=None):
        """
        Save a checkpoint with training state.

        Args:
            state (dict): State dictionary containing model, optimizer state etc.
            epoch (int): Current epoch
            metric_value (float): Current value of the monitored metric
        """
        # Update best metric if applicable
        if metric_value is not None:
            is_best = (
                self.monitor_mode == "min" and metric_value < self.best_metric
            ) or (self.monitor_mode == "max" and metric_value > self.best_metric)
            if is_best:
                self.best_metric = metric_value

        # Save checkpoint
        checkpoint_path = self.get_checkpoint_path(epoch, metric_value)
        state.update(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            }
        )
        torch.save(state, checkpoint_path)

        # Update metadata
        self._save_metadata()

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path=None):
        """
        Load a checkpoint and return the state. If no checkpoint_path is provided,
        load the most recent checkpoint. If no checkpoint exists, return None.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. If None, load latest checkpoint.

        Returns:
            dict: The loaded state dictionary, or None if no checkpoint exists
        """
        if checkpoint_path is None:
            # Find all checkpoints in the experiment directory
            checkpoints = glob.glob(
                os.path.join(self.exp_dir, f"checkpoints/epoch*.pth")
            )
            if not checkpoints:
                warnings.warn("No checkpoints found in the experiment directory.")
                return None

            # Sort by epoch number and get the latest
            def get_epoch_num(checkpoint_path):
                match = re.search(r"epoch_(\d+)", checkpoint_path)
                return int(match.group(1)) if match else -1

            checkpoint_path = max(checkpoints, key=get_epoch_num)

        if not os.path.exists(checkpoint_path):
            warnings.warn(f"Checkpoint {checkpoint_path} does not exist")
            return None

        state = torch.load(checkpoint_path)
        # Update logging state
        self.global_step = state.get("global_step", 0)
        self.epoch = state.get("epoch", 0) + 1  # Start from next epoch
        self.best_metric = state.get("best_metric", self.best_metric)

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")

        return state

    def log_metrics(self, metrics, step=None, prefix="train"):
        """
        Log metrics to TensorBoard.

        Args:
            metrics (dict): Dictionary of metric names and values
            step (int, optional): Current step within the epoch
            prefix (str): Prefix for metric names (e.g., 'train' or 'val')
        """
        if self.writer is None:
            return

        # Calculate global step if not provided
        if step is None:
            step = self.global_step + 1

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                self.writer.add_scalar(f"{prefix}/{name}", value.item(), step)

        self.global_step = step

    def log_histogram(self, values, name, step=None):
        """
        Log a histogram of values to TensorBoard.

        Args:
            values (torch.Tensor): Tensor of values to log
            name (str): Name for the histogram
            step (int, optional): Current step within the epoch
        """
        if self.writer is None:
            return

        if step is None:
            step = self.global_step + 1

        self.writer.add_histogram(name, values, step)

    def clean_old_checkpoints(self, top_k=5):
        """
        Keep only the top k best checkpoints based on the monitored metric.

        Args:
            top_k (int): Number of best checkpoints to keep
        """
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, f"epoch*.pth"))

        if len(checkpoints) <= top_k:
            return

        # Extract metric values from checkpoint names
        checkpoint_metrics = []
        # Use raw string for regex pattern to properly handle escape sequences
        metric_pattern = re.compile(rf"{self.monitor_metric}_([0-9.]+)(?=\.pth)")

        for checkpoint in checkpoints:
            match = metric_pattern.search(checkpoint)
            if match:
                metric_value = float(match.group(1))
                checkpoint_metrics.append((checkpoint, metric_value))
            else:
                # Use creation time as fallback
                creation_time = os.path.getctime(checkpoint)
                checkpoint_metrics.append((checkpoint, creation_time))

        # Sort checkpoints based on metric value
        if self.monitor_mode == "min":
            checkpoint_metrics.sort(key=lambda x: x[1])
        else:
            checkpoint_metrics.sort(key=lambda x: x[1], reverse=True)

        # Keep top k checkpoints, delete the rest
        checkpoints_to_keep = set(item[0] for item in checkpoint_metrics[:top_k])

        for checkpoint, _ in checkpoint_metrics[top_k:]:
            if checkpoint not in checkpoints_to_keep:
                try:
                    os.remove(checkpoint)
                except OSError as e:
                    warnings.warn(f"Error deleting checkpoint {checkpoint}: {e}")

    def log_images(self, images_dict, step):
        """
        Log images to TensorBoard.

        Args:
            writer: TensorBoard writer instance
            images_dict (dict): Dictionary of image names and tensors
            step (int): Current step/iteration
        """
        if self.writer is None or not self.log_images:
            return

        for name, images in images_dict.items():
            if isinstance(images, torch.Tensor):
                # Ensure images are in the right format [B, C, H, W]
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                # Only log up to num_log_images
                images = images[: self.num_log_images]
                self.writer.add_images(name, images, step)
                self.writer.flush()

    def log_figure(self, figure, name, step):
        """
        Log a matplotlib figure to TensorBoard.

        Args:
            figure: Matplotlib figure object
            name (str): Name for the figure
            step (int): Current step/iteration
        """
        if self.writer is None:
            return

        self.writer.add_figure(name, figure, global_step=step)

    def log_system_stats(self, step):
        """
        Log system statistics to TensorBoard.

        Args:
            writer: TensorBoard writer instance
            step (int): Current step/iteration
        """
        if self.writer is None:
            return

        if self.log_gpu_stats and torch.cuda.is_available():
            self.writer.add_scalar(
                "system/gpu_utilization", torch.cuda.utilization(), step
            )
            self.writer.add_scalar(
                "system/gpu_memory_allocated", torch.cuda.memory_allocated(), step
            )

        if self.log_memory_stats:
            self.writer.add_scalar(
                "system/ram_usage_percent", psutil.virtual_memory().percent, step
            )

        self.writer.flush()

    def log_hyperparameters(self, hparams, main_key: str = "hyperparameters"):
        """
        Log hyperparameters to TensorBoard.

        Args:
            hparams (dict): Dictionary of hyperparameters
        """
        if self.writer is None:
            return

        # Log hyperparameters as text
        for key, value in hparams.items():
            self.writer.add_text(f"{main_key}/{key}", str(value))

        # Log hyperparameters as a dictionary
        self.writer.flush()


from collections import deque
class OnlineMovingAverage:
    def __init__(self, size=5000):
        self.size = size
        self.queue = deque(maxlen=size)
        self.sum = 0.0
        self.mean = 1.0

    def update(self, value):
        if len(self.queue) == self.size:
            self.sum -= self.queue[0]
        self.queue.append(value)
        self.sum += value
        self.mean = self.sum / len(self.queue)


# Use a simple averaging function for EMA
def ema_avg_fn(averaged_model_parameter, model_parameter, n_averaged):
    decay = 0.99
    return decay * averaged_model_parameter + (1 - decay) * model_parameter
