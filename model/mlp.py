import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def unfold_image(x: torch.Tensor, patch_size: int, step: int = 1) -> torch.Tensor:
    r"""
    Extract sliding window patches with circular boundary conditions.

    Args:
        x: Tensor of shape (B, C, H, W)
        patch_size: patch side length P
        step: stride between adjacent patches (default 1)

    Returns:
        Tensor of shape (B, H, W, C, P, P) where each (C, P, P) is the patch centered
        at the corresponding (h, w) location (circularly padded at borders).
    """
    assert x.ndim == 4, f"Expected (B, C, H, W), got {tuple(x.shape)}"
    padding = patch_size // 2
    # Left/top may be one pixel smaller for even patch sizes to keep center alignment
    pad_lrtb = (
        padding - (patch_size - 1) % 2,
        padding,
        padding - (patch_size - 1) % 2,
        padding,
    )
    x_pad = F.pad(x, pad=pad_lrtb, mode="circular")
    # Unfold height then width. After first unfold, the width dim becomes -2.
    y = x_pad.unfold(dimension=-2, size=patch_size, step=step)
    y = y.unfold(dimension=-2, size=patch_size, step=step)
    # (B, C, H, W, P, P) -> (B, H, W, C, P, P)
    return y.permute(0, 2, 3, 1, 4, 5).contiguous()


class ResidualMLPBlock(nn.Module):
    """
    PreNorm residual MLP block:
            y = x + Dropout( gamma * (FC2(Dropout(GELU(FC1(LN(x)))))) )
    Shape preserved across the block: (N, D) -> (N, D).
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        layerscale_init: float = 1e-4,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.use_layerscale = layerscale_init is not None and layerscale_init > 0
        self.gamma = (
            nn.Parameter(layerscale_init * torch.ones(dim))
            if self.use_layerscale
            else None
        )
        # Optional conditioning (AdaLN/FiLM)
        self.to_gamma_beta = (
            nn.Linear(cond_dim, 2 * dim) if cond_dim is not None else None
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        y = self.norm(x)
        if self.to_gamma_beta is not None and cond is not None:
            gb = self.to_gamma_beta(cond)
            g, b = gb.chunk(2, dim=-1)
            y = y * (1 + g) + b
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout1(y)
        y = self.fc2(y)
        if self.use_layerscale:
            y = y * self.gamma
        y = self.dropout2(y)
        return x + y


class PositionalTimeEmbedding(nn.Module):
    """
    Deterministic sinusoidal (sin/cos) time embedding, followed by an MLP.
    """

    def __init__(self, embed_dim: int = 256, *args, **kwargs):
        super().__init__()
        self.fdim = embed_dim
        self.channels = embed_dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B,1)
        t = sigma.view(-1)
        d = self.fdim // 2
        # Compute deterministic sin/cos frequencies
        targ = (
            t[:, None]
            / (10000 ** (torch.arange(d, device=t.device) / (d - 1)))[None, :]
        )
        emb = torch.cat((torch.sin(targ), torch.cos(targ)), dim=1)
        return emb


class PatchChannelMLP(nn.Module):
    """
    MLP that processes each channel's P x P patch with a shared MLP and outputs one scalar per channel.

    Input:  x of shape (B, C, P, P)
    Output: y of shape (B, C)

    Notes:
    - Parameters are shared across channels for parameter efficiency and better generalization.
    - Uses PreNorm residual MLP blocks with optional LayerScale and Dropout for training stability.
    """

    def __init__(
        self,
        patch_size: int,
        depth: int = 4,
        mlp_ratio: float = 2.0,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        layerscale_init: float = 1e-4,
        final_norm: bool = True,
        enable_time_cond: bool = True,
        time_embed_dim: int = 256,
        # fourier_dim: int = 64,
    ):
        super().__init__()
        assert patch_size > 0, "patch_size must be > 0"
        self.patch_size = int(patch_size)
        dim = self.patch_size * self.patch_size  # Flattened patch dimension

        if hidden_dim is None:
            # Reasonable default: scale with input size but cap to avoid huge layers
            hidden_dim = max(128, int(mlp_ratio * dim))

        # Time/noise conditioning
        self.enable_time_cond = enable_time_cond
        self.time_embed = (
            PositionalTimeEmbedding(embed_dim=time_embed_dim)
            if enable_time_cond
            else None
        )
        cond_dim = time_embed_dim if enable_time_cond else None

        self.blocks = nn.ModuleList(
            [
                ResidualMLPBlock(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    layerscale_init=layerscale_init,
                    cond_dim=cond_dim,
                )
                for _ in range(depth)
            ]
        )

        self.out_norm = nn.LayerNorm(dim) if final_norm else nn.Identity()
        self.head = nn.Linear(dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
          x: Tensor of shape (B, C, P, P) where P == patch_size used at init.
          sigma: Optional (B,) noise levels for conditioning.
        Returns:
          Tensor of shape (B, C) (one scalar per channel).
        """
        assert x.ndim == 4, f"Expected (B, C, P, P), got {tuple(x.shape)}"
        B, C, P, Q = x.shape
        assert (
            P == self.patch_size and Q == self.patch_size
        ), f"Input patch size {(P, Q)} != model patch_size ({self.patch_size}, {self.patch_size})"

        cond = None
        if self.enable_time_cond and sigma is not None:
            t = self.time_embed(sigma)  # (B, E)
            cond = t.repeat_interleave(C, dim=0)  # (B*C, E)

        # Flatten per-channel patch and process with shared MLP
        y = x.reshape(B * C, P * Q)  # (B*C, dim)
        for block in self.blocks:
            y = block(y, cond=cond)
        y = self.out_norm(y)  # (B*C, dim)
        y = self.head(y).reshape(B, C)  # (B, C)
        return y


class PatchMLP(nn.Module):
    """
    Apply the channel-shared per-patch MLP densely over an image.

    For each pixel location (h, w), we extract a circularly padded P x P patch per channel
    and feed it to the per-patch MLP to produce one scalar per channel. The output is
    an image of the same shape as the input: (B, C, H, W).
    """

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        out_channels: int,
        depth: int = 4,
        mlp_ratio: int = 16,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        layerscale_init: float = 1e-4,
        final_norm: bool = True,
        chunk_size: Optional[int] = None,
        enable_time_cond: bool = True,
        time_embed_dim: int = 256,
        mid_channels: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = (
            mid_channels if mid_channels is not None else max(in_channels, 32)
        )

        # Channel mixing before per-patch MLP
        self.mix_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.GroupNorm(num_groups=1, num_channels=self.mid_channels),
        )

        self.per_patch = PatchChannelMLP(
            patch_size=patch_size,
            depth=depth,
            mlp_ratio=mlp_ratio,
            hidden_dim=hidden_dim,
            dropout=dropout,
            layerscale_init=layerscale_init,
            final_norm=final_norm,
            enable_time_cond=enable_time_cond,
            time_embed_dim=time_embed_dim,
        )

        # Unmix back to out_channels
        self.mix_out = nn.Conv2d(
            self.mid_channels, self.out_channels, kernel_size=1, bias=True
        )
        self.chunk_size = chunk_size

    @property
    def patch_size(self) -> int:
        return self.per_patch.patch_size

    def forward(
        self, x: torch.Tensor, sigma: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.ndim == 4, f"Expected (B, C, H, W), got {tuple(x.shape)}"
        B, C, H, W = x.shape
        assert (
            C == self.in_channels
        ), f"Expected in_channels={self.in_channels}, got {C}"
        P = self.patch_size

        # 1) Mix/expand channels: (B, C, H, W) -> (B, Cmid, H, W)
        x_feat = self.mix_in(x)
        Cmid = x_feat.shape[1]

        # 2) Extract patches on expanded features: (B, H, W, Cmid, P, P)
        patches = unfold_image(x_feat, patch_size=P, step=1)
        # Collapse spatial into batch for per-patch processing
        patches = patches.reshape(B * H * W, Cmid, P, P)

        # Build sigma repeated per spatial location if provided
        sigma_rep: Optional[torch.Tensor]
        if sigma is None:
            sigma_rep = None
        else:
            sigma_rep = sigma.view(B).repeat_interleave(H * W, dim=0)

        # Possibly process in chunks to avoid large allocations
        if self.chunk_size is None:
            out = self.per_patch(patches, sigma=sigma_rep)  # (B*H*W, Cmid)
        else:
            chunks = []
            for i in range(0, patches.shape[0], self.chunk_size):
                sigma_chunk = (
                    None if sigma_rep is None else sigma_rep[i : i + self.chunk_size]
                )
                chunks.append(
                    self.per_patch(patches[i : i + self.chunk_size], sigma=sigma_chunk)
                )
            out = torch.cat(chunks, dim=0)

        # 3) Reshape back to feature map: (B, H, W, Cmid) -> (B, Cmid, H, W)
        out_feat = out.view(B, H, W, Cmid).permute(0, 3, 1, 2).contiguous()
        # 4) Unmix to out_channels
        out_img = self.mix_out(out_feat)
        return out_img


if __name__ == "__main__":
    # Quick self-test for PatchChannelMLP: forward, backward, tiny train loop, and edge cases
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Image-level quick test with channel mixing
    B, Cin, Cout, H, W = 1, 3, 3, 32, 32
    model = PatchMLP(
        patch_size=11,
        in_channels=Cin,
        out_channels=Cout,
        depth=5,
        hidden_dim=3096,
        mid_channels=64,
        chunk_size=512,
    ).to(device)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))

    x = torch.randn(B, Cin, H, W, device=device)
    sigma = torch.rand(B, device=device)

    # Forward shape check
    out = model(x, sigma=sigma)
    print("Output shape:", out.shape)
    assert out.shape == (B, Cout, H, W)

    # Tiny train loop
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(x, sigma=sigma)
        loss = F.mse_loss(
            out, x[:, :Cout]
        )  # supervise on first Cout channels for the quick test
        loss.backward()
        opt.step()
        print(f"step {step+1} loss {loss.item():.6f}")
