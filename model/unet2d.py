# Translation Equivariance version of UNet2DModel with time / noise conditioning from diffusers
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    Timesteps,
    TimestepEmbedding,
)

ACT2CLS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(
            f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}"
        )


def get_norm(
    norm_type: str, num_channel: int, groups: Optional[int] = None, eps: float = 1e-5
) -> nn.Module:
    """Helper function to get normalization layer from string.

    Args:
        norm_type (str): Name of normalization layer.

    Returns:
        nn.Module: normalization layer.
    """

    if norm_type == "bn":
        return torch.nn.BatchNorm2d(num_features=num_channel, eps=eps, affine=True)
    elif norm_type == "lrn":
        return torch.nn.LocalResponseNorm(size=num_channel)
    elif norm_type == "grn":
        return torch.nn.GroupNorm(
            num_channels=num_channel, num_groups=groups, eps=eps, affine=True
        )
    elif norm_type == "layer":
        return torch.nn.LayerNorm(normalized_shape=[num_channel, 1, 1], eps=eps)
    else:
        print("Norm type " + norm_type + " not found, use LocalResponseNorm instead.")
        return torch.nn.LocalResponseNorm(size=num_channel)


class ResnetBlock2D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.0,
        temb_channels: int = 512,
        time_embedding_norm: str = "default",  # default, scale_shift,
        norm_type: str = "bn",
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        padding: str = "same",
        padding_mode: str = "circular",
    ):
        super().__init__()

        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.use_in_shortcut = (
            self.in_channels != conv_2d_out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(temb_channels, 2 * out_channels)
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )
        else:
            self.time_emb_proj = None

        if groups_out is None:
            groups_out = groups

        self.norm1 = get_norm(
            norm_type=norm_type, num_channel=in_channels, groups=groups, eps=eps
        )
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.norm2 = get_norm(norm_type, out_channels, groups_out, eps)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            out_channels,
            conv_2d_out_channels,
            kernel_size=1,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

        self.nonlinearity = get_activation(non_linearity)
        self.upsample = self.downsample = None

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        res_hidden_states: Tuple[torch.Tensor, ...] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if res_hidden_states is not None:
            input_tensor = torch.cat([input_tensor, res_hidden_states], dim=1)

        hidden_states = self.norm1(input_tensor)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states.contiguous())

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if self.time_embedding_norm == "default":
            if temb is not None:
                hidden_states = hidden_states + temb
            hidden_states = self.norm2(hidden_states)
        elif self.time_embedding_norm == "scale_shift":
            if temb is None:
                raise ValueError(
                    f" `temb` should not be None when `time_embedding_norm` is {self.time_embedding_norm}"
                )
            time_scale, time_shift = torch.chunk(temb, 2, dim=1)
            hidden_states = self.norm2(hidden_states)
            hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor.contiguous())

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class LocalEquivUNet2DCondModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        kernel_size_down_blocks: Tuple[int, ...] = (3, 3, 3, 3, 3),
        kernel_size_mid_blocks: int = 3,
        kernel_size_up_blocks: Optional[Tuple[int, ...]] = None,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        layers_per_block: int = 2,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_type: str = "bn",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        center_input_sample: bool = True,
    ):
        super().__init__()
        assert len(block_out_channels) + 1 == len(kernel_size_down_blocks)
        self.center_input_sample = center_input_sample
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_embedding_type = time_embedding_type.lower()
        self.time_embedding_dim = time_embed_dim
        kernel_size_up_blocks = (
            list(reversed(kernel_size_down_blocks))
            if kernel_size_up_blocks == None
            else kernel_size_up_blocks
        )
        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], scale=16
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=kernel_size_down_blocks[0],
            padding="same",
            padding_mode="circular",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]

        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            num_layers = layers_per_block
            down_block = []

            for j in range(num_layers):
                _in_channels = input_channel if j == 0 else output_channel
                down_block.append(
                    ResnetBlock2D(
                        in_channels=_in_channels,
                        out_channels=output_channel,
                        kernel_size=kernel_size_down_blocks[i + 1],
                        temb_channels=time_embed_dim,
                        time_embedding_norm=resnet_time_scale_shift,
                        norm_type=norm_type,
                        eps=norm_eps,
                        non_linearity=act_fn,
                        groups=norm_num_groups,
                        dropout=dropout,
                        padding="same",
                        padding_mode="circular",
                    )
                )
            down_block = nn.ModuleList(down_block)
            self.down_blocks.append(down_block)

        # mid

        self.mid_block = ResnetBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            kernel_size=kernel_size_mid_blocks,
            temb_channels=time_embed_dim,
            time_embedding_norm=resnet_time_scale_shift,
            eps=norm_eps,
            norm_type=norm_type,
            groups=norm_num_groups,
            dropout=dropout,
            non_linearity=act_fn,
            padding="same",
            padding_mode="circular",
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            num_layers = layers_per_block + 1 if i == 0 else layers_per_block

            # res_skip_channels = input_channel if (i == num_layers - 1) else output_channel
            # resnet_in_channels = prev_output_channel if i == 0 else output_channel

            up_block = []
            for j in range(num_layers):
                res_skip_channels = (
                    input_channel if (j == num_layers - 1) else output_channel
                )
                resnet_in_channels = prev_output_channel if j == 0 else output_channel
                up_block.append(
                    ResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,
                        out_channels=output_channel,
                        kernel_size=kernel_size_up_blocks[i],
                        temb_channels=time_embed_dim,
                        time_embedding_norm=resnet_time_scale_shift,
                        eps=norm_eps,
                        norm_type=norm_type,
                        non_linearity=act_fn,
                        groups=norm_num_groups,
                        dropout=dropout,
                        padding="same",
                        padding_mode="circular",
                    )
                )
            up_block = nn.ModuleList(up_block)
            self.up_blocks.append(up_block)

        # out
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )

        self.conv_norm_out = get_norm(norm_type, block_out_channels[0], num_groups_out)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=kernel_size_up_blocks[-1],
            padding="same",
            padding_mode="circular",
        )

    def forward(
        self,
        sample: torch.Tensor,
        sigma: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:

        # 0. center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = sigma
        if not torch.is_tensor(timesteps):
            # timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            timesteps = torch.tensor(
                [timesteps], dtype=sample.dtype, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device=sample.device, dtype=sample.dtype)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            res_samples = ()
            for resnet in downsample_block:
                sample = resnet(sample, temb=emb)
                res_samples = res_samples + (sample,)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, temb=emb)

        # 5. up
        for upsample_block in self.up_blocks:
            for resnet in upsample_block:
                res_hidden_states = down_block_res_samples[-1]
                down_block_res_samples = down_block_res_samples[:-1]
                sample = resnet(sample, emb, res_hidden_states)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        # 7. un-center input if necessary
        if self.center_input_sample:
            sample = (sample + 1.0) / 2.0
        return sample


class LocalUNet2DCondModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        kernel_size_down_blocks: Tuple[int, ...] = (3, 3, 3, 3, 3),
        kernel_size_mid_blocks: int = 3,
        kernel_size_up_blocks: Optional[Tuple[int, ...]] = None,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        layers_per_block: int = 2,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_type: str = "bn",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        center_input_sample: bool = True,
    ):
        super().__init__()
        assert len(block_out_channels) + 1 == len(kernel_size_down_blocks)

        padding = "same"
        padding_mode = "zeros"
        self.center_input_sample = center_input_sample
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4
        self.time_embedding_type = time_embedding_type.lower()
        self.time_embedding_dim = time_embed_dim
        kernel_size_up_blocks = (
            list(reversed(kernel_size_down_blocks))
            if kernel_size_up_blocks == None
            else kernel_size_up_blocks
        )
        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=block_out_channels[0], scale=16
            )
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                block_out_channels[0], flip_sin_to_cos, freq_shift
            )
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=kernel_size_down_blocks[0],
            padding=padding,
            padding_mode=padding_mode,
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]

        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            num_layers = layers_per_block
            down_block = []

            for j in range(num_layers):
                _in_channels = input_channel if j == 0 else output_channel
                down_block.append(
                    ResnetBlock2D(
                        in_channels=_in_channels,
                        out_channels=output_channel,
                        kernel_size=kernel_size_down_blocks[i + 1],
                        temb_channels=time_embed_dim,
                        time_embedding_norm=resnet_time_scale_shift,
                        norm_type=norm_type,
                        eps=norm_eps,
                        non_linearity=act_fn,
                        groups=norm_num_groups,
                        dropout=dropout,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )
            down_block = nn.ModuleList(down_block)
            self.down_blocks.append(down_block)

        # mid

        self.mid_block = ResnetBlock2D(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            kernel_size=kernel_size_mid_blocks,
            temb_channels=time_embed_dim,
            time_embedding_norm=resnet_time_scale_shift,
            eps=norm_eps,
            norm_type=norm_type,
            groups=norm_num_groups,
            dropout=dropout,
            non_linearity=act_fn,
            padding=padding,
            padding_mode=padding_mode,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]

            num_layers = layers_per_block + 1 if i == 0 else layers_per_block

            up_block = []
            for j in range(num_layers):
                res_skip_channels = (
                    input_channel if (j == num_layers - 1) else output_channel
                )
                resnet_in_channels = prev_output_channel if j == 0 else output_channel
                up_block.append(
                    ResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,
                        out_channels=output_channel,
                        kernel_size=kernel_size_up_blocks[i],
                        temb_channels=time_embed_dim,
                        time_embedding_norm=resnet_time_scale_shift,
                        eps=norm_eps,
                        norm_type=norm_type,
                        non_linearity=act_fn,
                        groups=norm_num_groups,
                        dropout=dropout,
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                )
            up_block = nn.ModuleList(up_block)
            self.up_blocks.append(up_block)

        # out
        num_groups_out = (
            norm_num_groups
            if norm_num_groups is not None
            else min(block_out_channels[0] // 4, 32)
        )

        self.conv_norm_out = get_norm(norm_type, block_out_channels[0], num_groups_out)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=kernel_size_up_blocks[-1],
            padding=padding,
            padding_mode=padding_mode,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:

        # 0. center input if necessary
        if self.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            timesteps = torch.tensor(
                [timesteps], dtype=sample.dtype, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device=sample.device, dtype=sample.dtype)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(
            sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
        )

        t_emb = self.time_proj(timesteps)
        emb = self.time_embedding(t_emb)

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            res_samples = ()
            for resnet in downsample_block:
                sample = resnet(sample, temb=emb)
                res_samples = res_samples + (sample,)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(sample, temb=emb)

        # 5. up
        for upsample_block in self.up_blocks:
            for resnet in upsample_block:
                res_hidden_states = down_block_res_samples[-1]
                down_block_res_samples = down_block_res_samples[:-1]
                sample = resnet(sample, emb, res_hidden_states)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.time_embedding_type == "fourier":
            timesteps = timesteps.reshape(
                (sample.shape[0], *([1] * len(sample.shape[1:])))
            )
            sample = sample / timesteps

        # 7. un-center input if necessary
        if self.center_input_sample:
            sample = (sample + 1.0) / 2.0
        return sample
