import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from .utils import unfold_image, batched_cdist
from tqdm import tqdm
from .operator import (
    Identity,
    Operator,
    ConvolutionOperatorFFT,
    ConvolutionOperatorMatrix,
)
import numpy as np

from dataclasses import dataclass
@dataclass
class Output:
    estimate: torch.Tensor
    distances: torch.Tensor = None
    source_indices: torch.Tensor = None
    weights: torch.Tensor = None
    log_density: torch.Tensor = None

class MMSE(nn.Module):
    r"""
    MMSE (Minimum Mean Square Error) estimator.
    This class implements the MMSE estimator for inverse problem.
    """

    def __init__(self, device="cpu", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.factory_kwargs = dict(device=device, dtype=dtype)

    def forward(
        self,
        y: torch.Tensor,
        A: Operator,
        data_loader: DataLoader,
        sigma: float,
        return_distance=False,
        verbose=True,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of the MMSE estimator.
        Args:
            y (torch.Tensor): Input tensor of shape (1, C, H, W).
            data_loader (DataLoader): DataLoader for the dataset.
            sigma (float): Standard deviation for the noise level.
            return_distance (bool): Whether to return the distance.
        Returns:
            torch.Tensor: Output tensor of shape (1, C, H, W).
        """
        assert y.size(0) == 1, "y should have batch size of 1"
        y = y.to(device=self.device, dtype=self.dtype)
        numerator = 0.0
        denominator = 0.0
        distance_shift = None

        if return_distance:
            distance_list = []
        for batch in tqdm(data_loader, disable=not verbose):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device=self.device, dtype=self.dtype)
            batch_transformed = A.forward(batch)  # (B, C, H, W)

            distance = (
                (batch_transformed - y).pow(2).sum(dim=(1, 2, 3), keepdim=True)
            )  # (B, 1, 1, 1)
            distance = -0.5 * distance / (sigma**2)  # shape (B, 1, 1, 1)
            if return_distance:
                distance_list.append(distance.squeeze().cpu().numpy())
            numerator, denominator, distance_shift = self._update_sum_exp(
                batch, numerator, denominator, distance, distance_shift, dim=(0,)
            )

        if return_distance:
            return numerator / denominator, np.concatenate(distance_list, axis=0)
        return numerator / denominator

    def _update_sum_exp(
        self, batch, numerator, denominator, distance, distance_shift, dim
    ):
        """
        Update the sum exp values for the MMSE estimator with numerical stability.
        Args:
            batch (torch.Tensor): Batch of images of shape (B, 1, *).
            numerator (torch.Tensor): Current numerator value of shape (1, K, *).
            denominator (torch.Tensor): Current denominator value, a scalar
            distance (torch.Tensor): Distance tensor of shape (B, K, *).
            distance_shift (torch.Tensor): Shifted distance tensor.
            dim tuple(int): Dimension to sum over.
        Returns:
            tuple: Updated numerator and denominator tensors.
        """

        if distance_shift is None:
            distance_shift = torch.amax(distance, dim=dim, keepdim=True)  # (1, K, 1)
        else:
            new_distance_shift = torch.amax(
                distance, dim=dim, keepdim=True
            )  # (1, K, 1)
            delta_distance_shift = torch.where(
                new_distance_shift < distance_shift, distance_shift, new_distance_shift
            )  # (1, K, 1)
            diff = delta_distance_shift - distance_shift  # (1, K, 1)
            numerator /= torch.exp(diff)  # (1, K, *)
            denominator /= torch.exp(diff)  # (1, K, 1)
            distance_shift = delta_distance_shift  # (1, K, 1)

        exp_distance = torch.exp(distance - distance_shift)  # (B, K, 1)
        numerator += torch.sum(exp_distance * batch, dim=dim, keepdim=True)  # (1, K, *)
        denominator += torch.sum(exp_distance, dim=dim, keepdim=True)  # (1, K, 1)

        return numerator, denominator, distance_shift

    def _update_sum_exp_partial(
        self, batch, numerator, denominator, distance, distance_shift, dim
    ):
        """
        Update the sum exp values for the MMSE estimator with numerical stability.
        Args:
            batch (torch.Tensor): Batch of images of shape (B, 1, *).
            numerator (torch.Tensor): Current numerator value of shape (1, *).
            denominator (torch.Tensor): Current denominator value, a scalar
            distance (torch.Tensor): Distance tensor of shape (B, K, *).
            distance_shift (torch.Tensor): Shifted distance tensor.
            dim tuple(int): Dimension to sum over.
        Returns:
            tuple: Updated numerator and denominator tensors.
        """
        new_distance_shift = torch.amax(distance, dim=dim, keepdim=True)  # (1, K, 1)
        delta_distance_shift = torch.where(
            new_distance_shift < distance_shift, distance_shift, new_distance_shift
        )  # (1, K, 1)
        diff = delta_distance_shift - distance_shift  # (1, K, 1)
        numerator /= torch.exp(diff)  # (B, K, 1)
        denominator /= torch.exp(diff)  # (B, K, 1)
        distance_shift = delta_distance_shift  # (1, K, 1)

        exp_distance = torch.exp(distance - distance_shift)  # (B, K, 1)
        numerator += torch.sum(exp_distance * batch, dim=dim, keepdim=True)  # (1, K, *)
        denominator += torch.sum(exp_distance, dim=dim, keepdim=True)  # (1, K, 1)

        return numerator, denominator, distance_shift

    def _expand_dim_as(self, tensor: Tensor, other: Tensor) -> Tensor:
        r"""
        Expand a tensor to the shape of another tensor.
        """
        if tensor.dim() < other.dim():
            return tensor.view(*tensor.shape, *((1,) * (other.dim() - tensor.dim())))
        else:
            return tensor


class EquivariantMMSE(MMSE):
    r"""
    Translation Equivariant MMSE estimator.
    """

    def __init__(self, eps: float = 1e-3, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.eps = eps

    def translate(self, x: Tensor) -> Tensor:
        """
        Transform the batch of images to all translation..
        Args:
            batch (Tensor): Batch of images of shape (B, C, H, W).
        Returns:
            Tensor: Transformed batch of images of shape (B, num_translation, C, H, W).
        """
        size = x.size(-1)
        return unfold_image(x, size=size, step=1).flatten(
            1, 2
        )  # shape = (B, H * W, C, H, W)

    def inverse_translate(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Transform the input tensor x by applying all translation in reverse order.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            g (int): Group action to be applied.
        Returns:
            torch.Tensor: Transformed tensor of shape (B, H * W, C, H, W).
        """
        size = x.size(-1)
        return (
            unfold_image(x, size=size, step=1)
            .flip((1, 2))
            .roll(dims=(1, 2), shifts=(-1, -1))
            .flatten(1, 2)
        )

    @torch.no_grad()
    def forward(
        self,
        y: torch.Tensor,
        A: Operator,
        B: Operator = None,
        data_loader: DataLoader = None,
        sigma: float = None,
    ) -> torch.Tensor:
        """
        Forward pass of the equivariant MMSE estimator.
        Args:
            y (torch.Tensor): Input tensor of shape (1, C, H, W).
            A (Operator): forward operator A to be applied.
            B (Operator, optional): operator B to be applied. Defaults to None.
            data_loader (DataLoader): DataLoader for the dataset.
            sigma (float): Standard deviation for the noise level.
        Returns:
            torch.Tensor: Output tensor of shape (1, C, H, W).
        """
        assert y.size(0) == 1, "y should have batch size of 1"
        y = y.to(device=self.device, dtype=self.dtype)
        if B is None:
            B = Identity(device=self.device, dtype=self.dtype)

        if isinstance(B, Identity) or (
            isinstance(B, (ConvolutionOperatorMatrix, ConvolutionOperatorFFT))
            and B.is_invertible
        ):
            v = y
            keep_index = slice(None)
            v_transformed_measured = self.inverse_translate(v).flatten(
                -3, -1
            )  # (1, K, C H W)
        else:
            v = B.forward(y)
            v_transformed = self.inverse_translate(v).squeeze(
                0
            )  # Transform v to the shape (H * W, C, H, W)
            if isinstance(B, (ConvolutionOperatorFFT, ConvolutionOperatorFFT)):
                keep_index = slice(None)
            else:
                projected = B.proj_im(v_transformed)  # Shape (H * W, C, H, W)
                projected_distance = (
                    (v_transformed - projected)
                    .pow(2)
                    .sum(dim=tuple(range(1, v_transformed.ndim)))
                    .sqrt()
                )  # Shape (H * W, )
                y_norm = torch.linalg.norm(y.flatten())
                keep_index = torch.where(projected_distance < self.eps * y_norm)[
                    0
                ]  # Indices of the patches to keep

            v_transformed_measured = B.pinv(v_transformed)  # Shape (H * W, *)
            v_transformed_measured = (
                v_transformed_measured[keep_index].unsqueeze(0).flatten(-3, -1)
            )  # Shape (1, K, C H W)

        numerator = 0.0
        denominator = 0.0
        distance_shift = None

        for batch in tqdm(data_loader):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device=self.device, dtype=self.dtype)
            batch_transformed = self.translate(batch)[
                :, keep_index, ...
            ]  # (B, K, C, H, W)
            batch_measured = (
                A.forward(batch).flatten(-3, -1).unsqueeze(1)
            )  # (B, 1, C H W)

            # Compute the distance between the transformed batch and the measured v transformed
            distance = batched_cdist(
                v_transformed_measured,
                batch_measured,
            )

            distance = -0.5 * distance / (sigma**2)
            # Update the formula
            numerator, denominator, distance_shift = self._update_sum_exp(
                batch_transformed,
                numerator,
                denominator,
                self._expand_dim_as(distance, batch_transformed),
                distance_shift,
                dim=(0, 1),
            )

        return (numerator / denominator).view(1, *batch.shape[1:])


class LocalEquivariantMMSE(MMSE):
    r"""
    Local and translation equivariant MMSE estimator
    """

    def __init__(
        self, patch_size: int, device="cpu", dtype=torch.float32, *args, **kwargs
    ):
        super().__init__(device=device, dtype=dtype, *args, **kwargs)
        self.patch_size = patch_size

    def unfold(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Transform the input tensor x by cropping input image into patch.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Transformed tensor of shape (B, num_patch, C, size, size).
        """
        if x.ndim != 4:
            x = x.view(-1, *self.operator.input_size)
        patches = unfold_image(x, size=self.patch_size, step=1).flatten(1, 2)
        return patches

    @torch.no_grad()
    def forward(
        self,
        y: torch.Tensor,
        A: Operator,
        B: Operator = None,
        data_loader: DataLoader = None,
        sigma: float = 1.0,
        return_distance=False,
        return_log_density=False,
        topk=None,
        verbose=True,
        *args,
        **kwargs
    ) -> torch.Tensor:

        assert y.size(0) == 1, f"y should have batch size of 1, got {y.shape}"
        if return_distance: 
            assert topk is not None, "topk must be specified when return_distance is True"
            
        opts = dict(
            sigma=sigma,
            return_distance=return_distance,
            return_log_density=return_log_density,
            topk=topk,
            verbose=verbose,
        )
        if B is None:
            B = Identity(device=self.device, dtype=self.dtype)

        if isinstance(B, Identity):
            return self._forward_identity(y, A, data_loader, **opts)
        elif B.is_invertible:
            y = B.forward(y)
            return self._forward_invertible(
                y,
                A,
                B,
                data_loader,
                **opts
            )
        else:
            y = B.forward(y)
            return self._forward_general(
                y,
                A,
                B,
                data_loader,
                **opts
            )

    def _forward_identity(
        self,
        y: torch.Tensor,
        A: Operator,
        data_loader: DataLoader,
        sigma: float = 1.0,
        return_distance=False,
        return_log_density=False,
        topk=None,
        verbose=True,
    ) -> torch.Tensor:
        r"""
        Forward pass of the local equivariant estimator with B = I.
        Args:
            A (Operator): forward operator A to be applied.
            data_loader (DataLoader): DataLoader for the input data.
            y (torch.Tensor): Input tensor of shape (1, C, H, W) -- a single image.
            sigma (float): Standard deviation for the noise level.
        Returns:
            torch.Tensor: Output tensor of shape (1, C, H, W).
        """
        y = y.to(device=self.device, dtype=self.dtype)
        c, h, w = y.shape[1:]

        # Generate the patch v
        v = self.unfold(
            y.view((y.size(0), c, h, w))
        )  # .squeeze(0)  # shape (V, C, size, size)

        numerator = 0.0
        denominator = 0.0
        distance_shift = None
        distance_list = None
        true_denominator = None

        pad = self.patch_size // 2

        for i, batch in enumerate(tqdm(data_loader, disable=not verbose)):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device=self.device, dtype=self.dtype)
            batch_measured = A.forward(batch)
            patch_dataset = self.unfold(batch_measured)  # (B, num_patch, M)

            center_pixel = self.unfold(batch)[..., pad, pad]  # shape (B, H * W, C)
            # distance = batched_cdist(patch_dataset.flatten(-3, -1), v.flatten(-3, -1))
            distance = batched_cdist(patch_dataset.flatten(-3, -1).half(), v.flatten(-3, -1).half()).to(self.dtype)
            distance = -0.5 * distance / (sigma**2)  # shape (B, H * W, H * W)

            if return_distance:
                if distance_list is None:
                    distance_list = distance.flatten(0, 1)
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx = idx // (h * w)
                else:
                    distance_list = torch.cat(
                        [distance_list, distance.flatten(0, 1)], dim=0
                    )
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx[idx >= topk] = i * batch.shape[0] + (idx[idx >= topk] - topk) // (
                        h * w
                    )

            if return_log_density:
                if true_denominator is None:
                    true_denominator = torch.logsumexp(distance.flatten(0, 1), dim=0, keepdim=True)
                else:
                    true_denominator = torch.logsumexp(torch.cat((distance.flatten(0, 1), true_denominator), dim=0), dim=0, keepdim=True)

            numerator, denominator, distance_shift = self._update_sum_exp(
                center_pixel.unsqueeze(-1),
                numerator,
                denominator,
                distance.unsqueeze(-2),
                distance_shift,
                dim=(0, 1),
            )
        
        estimate = (numerator / denominator).view(1, *batch.shape[1:])
        
        if not return_distance and not return_log_density:
            return estimate
        else:
            return Output(
                estimate=estimate,
                distances=distance_list.squeeze() if return_distance else None,
                source_indices=src_idx.detach().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_distance else None,
                log_density=true_denominator.squeeze().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_log_density else None,
            )


    def _forward_invertible(
        self,
        y: torch.Tensor,
        A: Operator,
        B: Operator,
        data_loader: DataLoader,
        sigma: float = 1.0,
        return_distance=False,
        return_log_density=False,
        topk=None,
        verbose=True,
    ) -> torch.Tensor:
        r"""
        Forward pass of the local equivariant estimator with B invertible.
        Args:
            A (Operator): forward operator A to be applied.
            data_loader (DataLoader): DataLoader for the input data.
            y (torch.Tensor): Input tensor of shape (1, C, H, W) -- a single image.
            sigma (float): Standard deviation for the noise level.
        Returns:
            torch.Tensor: Output tensor of shape (1, C, H, W).
        """
        assert y.size(0) == 1, "y should have batch size of 1"

        y = y.to(device=self.device, dtype=self.dtype)
        c, h, w = y.shape[1:]

        # Generate the patch v
        v = self.unfold(y.view((y.size(0), c, h, w))).squeeze(
            0
        )  # shape (V, C, size, size)

        numerator = 0.0
        denominator = 0.0
        distance_shift = None
        distance_list = None
        true_denominator = None
        
        pad = self.patch_size // 2

        for i, batch in enumerate(tqdm(data_loader, disable=not verbose)):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device=self.device, dtype=self.dtype)
            distance = A._distance_local_equiv_fullrank(
                B, v, batch, patch_size=self.patch_size, sigma=sigma
            )

            if return_distance:
                if distance_list is None:
                    distance_list = distance.flatten(0, 1)
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx = idx // (h * w)
                else:
                    distance_list = torch.cat(
                        [distance_list, distance.flatten(0, 1)], dim=0
                    )
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx[idx >= topk] = i * batch.shape[0] + (idx[idx >= topk] - topk) // (
                        h * w
                    )

            if return_log_density:
                if true_denominator is None:
                    true_denominator = torch.logsumexp(distance.flatten(0, 1), dim=0, keepdim=True)
                else:
                    true_denominator = torch.logsumexp(torch.cat((distance.flatten(0, 1), true_denominator), dim=0), dim=0, keepdim=True)
                

            center_pixel = self.unfold(batch)[..., pad, pad]  # shape (B, H * W, C)
            numerator, denominator, distance_shift = self._update_sum_exp(
                center_pixel.unsqueeze(-1),
                numerator,
                denominator,
                distance.unsqueeze(-2),
                distance_shift,
                dim=(0, 1),
            )

        estimate = (numerator / denominator).view(1, *batch.shape[1:])
        if not return_distance and not return_log_density:
            return estimate
        
        else:
            return Output(
                estimate=estimate,
                distances=distance_list.detach() if return_distance else None,
                source_indices=src_idx.detach().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_distance else None,
                log_density=true_denominator.squeeze().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_log_density else None,
            )
            
    @torch.no_grad()
    def _forward_general(
        self,
        y: torch.Tensor,
        A: Operator,
        B: Operator,
        data_loader: DataLoader,
        sigma: float = 1.0,
        return_distance=False,
        return_log_density=False,
        topk=None,
        verbose=True,
    ) -> torch.Tensor:
        r"""
        Forward pass of the local equivariant estimator with general B .
        Args:
            y (torch.Tensor): Input tensor of shape (1, C, H, W).
            A (Operator): forward operator to be applied.
            B (Operator): pre-inverse operator to be applied.
            data_loader (DataLoader): DataLoader for the input data.
            y (torch.Tensor): Input tensor of shape (1, C, H, W) -- a single image.
            sigma (float): Standard deviation for the noise level.
        Returns:
            torch.Tensor: Output tensor of shape (1, C, H, W).
        """
        assert y.size(0) == 1, "y should have batch size of 1"

        y = y.to(**self.factory_kwargs)
        h, w = y.shape[-2:]

        # Unfold to patches
        v = self.unfold(y).squeeze(0)  # shape (s_in, C, P, P)
        ind, rank_Q_n = B._indicator_and_rank(v, patch_size=self.patch_size, eps=1e-4)

        numerator = 0.0
        denominator = 0.0
        distance_shift = None

        distance_list = None
        true_denominator = None
        
        for i, batch in enumerate(tqdm(data_loader, disable=not verbose)):
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(**self.factory_kwargs)
            distance = A._distance_local_equiv_general(
                B, v, batch, self.patch_size, sigma, ind, rank_Q_n
            )
            # (B, H*W, h*w)
            if return_distance:
                if distance_list is None:
                    distance_list = distance.flatten(0, 1)
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx = idx // (h * w)
                else:
                    distance_list = torch.cat(
                        [distance_list, distance.flatten(0, 1)], dim=0
                    )
                    distance_list, idx = torch.topk(
                        distance_list, topk, dim=0, largest=True, sorted=True
                    )
                    idx = idx[0, ...]
                    src_idx[idx >= topk] = i * batch.shape[0] + (idx[idx >= topk] - topk) // (
                        h * w
                    )
                    
            if return_log_density:
                if true_denominator is None:
                    true_denominator = torch.logsumexp(distance.flatten(0, 1), dim=0, keepdim=True)
                else:
                    true_denominator = torch.logsumexp(torch.cat((distance.flatten(0, 1), true_denominator), dim=0), dim=0, keepdim=True)

            center_pixel = batch.flatten(-2, -1).transpose(
                -2, -1
            )  # shape (B, H * W, C)
            numerator, denominator, distance_shift = self._update_sum_exp(
                center_pixel.unsqueeze(-1),
                numerator,
                denominator,
                distance.unsqueeze(-2),
                distance_shift,
                dim=(0, 1),
            )
        
        estimate = (numerator / denominator).view(1, *batch.shape[1:])
        if not return_distance and not return_log_density:
            return estimate
        else:
            return Output(
                estimate=estimate,
                distances=distance_list.detach() if return_distance else None,
                source_indices=src_idx.detach().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_distance else None,
                log_density=true_denominator.squeeze().view(1, 1, estimate.size(-2), estimate.size(-1)) if return_log_density else None,
            )
            
