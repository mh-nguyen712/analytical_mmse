# Minimal implementations of U-Net and ResNet architectures:
# Adapted from: https://github.com/Kambm/convolutional_diffusion/tree/main (https://arxiv.org/abs/2412.20292)

import torch
from torch import nn
from torch import Tensor


class EmbeddingModule(nn.Module):

    def __init__(self, fdim, channels, conditional=False, num_classes=None):
        super().__init__()
        self.fdim = fdim
        self.conditional = conditional

        if conditional:
            self.class_embeddings = nn.Embedding(num_classes, fdim)
        self.channels = channels

    def forward(self, t, label=None):

        d = self.fdim // 2
        targ = (
            t[:, None]
            / (10000 ** (torch.arange(d, device=t.device) / (d - 1)))[None, :]
        )
        emb = torch.cat((torch.sin(targ), torch.cos(targ)), dim=1)
        if self.conditional:
            emb += self.class_embeddings(label)

        return emb


def get_norm(
    norm_type: str, num_channel: int, groups: int | None = None, eps: float = 1e-5
) -> nn.Module:
    """Helper function to get normalization layer from string.

    Args:
        norm_type (str): Name of normalization layer.

    Returns:
        nn.Module: normalization layer.
    """

    if norm_type == "BatchNorm":
        return torch.nn.BatchNorm2d(num_features=num_channel, eps=eps, affine=True)
    elif norm_type == "LRN":
        return torch.nn.LocalResponseNorm(size=num_channel)
    elif norm_type == "GroupNorm":
        return torch.nn.GroupNorm(
            num_channels=num_channel, num_groups=groups, eps=eps, affine=True
        )
    elif norm_type == "LayerNorm":
        return torch.nn.LayerNorm(normalized_shape=[num_channel, 1, 1], eps=eps)
    else:
        print("Norm type " + norm_type + " not found, use LocalResponseNorm instead.")
        return torch.nn.LocalResponseNorm(size=num_channel)


class MinimalResNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        emb_dim=128,
        mode="circular",
        normalization=None,
        conditional=False,
        num_classes=None,
        kernel_size=3,
        num_layers=6,
        lastksize=1,
        add_one=True,
    ):

        super().__init__()
        assert in_channels == out_channels
        channels = in_channels

        self.channels = channels
        self.emb_dim = emb_dim
        self.mode = mode
        self.conditional = conditional
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.normalization = normalization
        self.lastksize = lastksize

        self.embedding = EmbeddingModule(
            emb_dim, channels, conditional=conditional, num_classes=num_classes
        )

        self.up_projection = nn.Conv2d(
            channels, emb_dim, kernel_size, padding="same", padding_mode=mode
        )

        if add_one:
            self.embs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(emb_dim, emb_dim), nn.GroupNorm(8, emb_dim), nn.ReLU()
                    )
                    for i in range(num_layers + 1)
                ]
            )
        else:
            self.embs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(emb_dim, emb_dim), nn.GroupNorm(8, emb_dim), nn.ReLU()
                    )
                    for i in range(num_layers)
                ]
            )

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        emb_dim,
                        emb_dim,
                        kernel_size,
                        padding="same",
                        padding_mode=mode,
                    ),
                    get_norm(normalization, emb_dim, groups=8),
                    nn.ReLU(),
                )
                for i in range(num_layers)
            ]
        )

        self.down_projection = nn.Sequential(
            get_norm(normalization, emb_dim, groups=8),
            nn.Conv2d(emb_dim, channels, lastksize, padding="same", padding_mode=mode),
        )

    def forward(self, x: Tensor, sigma: float | Tensor, label=None):

        embedding_vec = self.embedding(sigma, label=label)
        state = self.up_projection(x)

        for i in range(self.num_layers):
            delta = self.convs[i](state + self.embs[i](embedding_vec)[:, :, None, None])
            state = state + delta

        delta = (
            self.embs[-1](embedding_vec)[:, :, None, None]
            if len(self.embs) > self.num_layers
            else state
        )
        nextstate = state + delta

        return self.down_projection(nextstate)


class MinimalUNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        fsizes=[32, 64, 128, 256],
        mode="circular",
        conditional=False,
        num_classes=None,
        emb_dim=256,
        normalization=None,
        last_norm=False,
        kernel_size=3,
        lastksize=1,
    ):

        super().__init__()

        assert in_channels == out_channels
        channels = in_channels

        self.fsizes = fsizes
        self.channels = channels
        self.conditional = conditional
        self.emb_dim = emb_dim
        self.kernel_size = kernel_size
        self.lastksize = lastksize

        self.embedding = EmbeddingModule(
            emb_dim, channels, conditional=conditional, num_classes=num_classes
        )

        in_channels = channels
        self.feature_blocks = nn.ModuleList()
        for f in fsizes[:-1]:
            self.feature_blocks.append(
                UBlock(
                    in_channels,
                    f,
                    normalization=normalization,
                    kernel_size=kernel_size,
                    padding_mode=mode,
                    emb_dim=emb_dim,
                )
            )
            in_channels = f

        self.bottleneck = UBlock(
            fsizes[-2],
            fsizes[-1],
            normalization=normalization,
            kernel_size=kernel_size,
            padding_mode=mode,
            emb_dim=emb_dim,
        )
        self.output_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i in range(len(fsizes) - 1, 0, -1):
            self.upsamples.append(
                nn.ConvTranspose2d(fsizes[i], fsizes[i - 1], kernel_size=2, stride=2)
            )
            self.output_blocks.append(
                UBlock(
                    2 * fsizes[i - 1],
                    fsizes[i - 1],
                    normalization=normalization,
                    padding_mode=mode,
                    emb_dim=emb_dim,
                )
            )

        self.last_emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, fsizes[0]))
        self.output_conv = nn.Conv2d(
            fsizes[0],
            self.channels,
            kernel_size=lastksize,
            padding="same",
            padding_mode=mode,
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.last_norm = last_norm
        if last_norm:
            if normalization == "GroupNorm":
                self.last_normalizer = nn.GroupNorm(min(32, fsizes[0]), fsizes[0])
            elif normalization == "BatchNorm":
                self.last_normalizer = nn.BatchNorm2d(fsizes[0])

    def forward(self, x: Tensor, sigma: float | Tensor, label=None):

        embedding_vec = self.embedding(sigma, label=label)

        skip_connections = []

        for down in self.feature_blocks:
            x = down(x, embedding_vec)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, embedding_vec)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.upsamples)):
            upconv = self.upsamples[i](x)
            skip = skip_connections[i]
            x = torch.cat((skip, upconv), dim=1)
            x = self.output_blocks[i](x, embedding_vec)

        try:
            if self.last_norm:
                return self.output_conv(
                    self.last_normalizer(
                        x + self.last_emb(embedding_vec)[:, :, None, None]
                    )
                )
            else:
                return self.output_conv(
                    x + self.last_emb(embedding_vec)[:, :, None, None]
                )
        except:
            return self.output_conv(x + self.last_emb(embedding_vec)[:, :, None, None])


class UBlock(nn.Module):

    def __init__(
        self,
        infeatures,
        outfeatures,
        depth=2,
        kernel_size=3,
        normalization=None,
        padding_mode="circular",
        emb_dim=32,
    ):

        super().__init__()

        self.emb = nn.Sequential(nn.ReLU(), nn.Linear(emb_dim, infeatures))

        module_list = []
        for i in range(depth):
            if i == 0:
                x = infeatures
            else:
                x = outfeatures

            module_list.append(
                nn.Conv2d(
                    x,
                    outfeatures,
                    kernel_size=kernel_size,
                    padding="same",
                    padding_mode=padding_mode,
                )
            )
            if normalization == "GroupNorm":
                module_list.append(nn.GroupNorm(min(32, outfeatures), outfeatures))
            elif normalization == "BatchNorm":
                module_list.append(nn.BatchNorm2d(outfeatures))
            module_list.append(nn.ReLU())

        self.model = nn.Sequential(*module_list)

    def forward(self, x, embedding):
        return self.model(x + self.emb(embedding)[:, :, None, None])
