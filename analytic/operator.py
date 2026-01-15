from __future__ import annotations
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from .utils import unfold_image
from tqdm import tqdm
import deepinv as dinv
from deepinv.optim.utils import least_squares
from functools import partial
from deepinv.utils.demo import load_degradation
from .utils import batched_cdist, unfold_image, build_circular_conv_matrix_fft
import itertools
import numpy as np
from einops import rearrange


def get_operator(
    operator: str = None,
    input_size: tuple = None,
    output_size: tuple = None,
    get_operator_param: bool = False,
    **kwargs,
) -> Operator:
    r"""
    Pre-defined operators for numerical experiments.
    ALL:
        - denoising: identity operator
        - inpainting_center_p: where p is the size of the inpainting mask
        - inpainting_random_p: where p is the percentage of pixels to be inpainted
        - convolution_gaussian_p: where p is the std of the Gaussian kernel
        - convolution_motion: convolution with a motion blur kernel (Levin)
    """
    if input_size is not None:
        c, h, w = input_size
    if operator.lower() == "denoising":
        return Identity(input_size=input_size, output_size=output_size, **kwargs)
    elif "inpainting" in operator.lower():
        if (
            "inpainting_center" in operator.lower()
        ):  # e.g., "inpainting_center_25" will create a mask that inpaints the center 25 pixels of the image
            mask = torch.ones(1, 1, h, w)
            p = int(operator.split("_")[-1])
            start_h, start_w = (h // 2 - p // 2, w // 2 - p // 2)
            mask[:, :, start_h : start_h + p, start_w : start_w + p] = 0

        elif (
            "inpainting_random" in operator.lower()
        ):  # e.g., "inpainting_random_25" will create a mask that inpaints 25% random pixels of the image
            rng = torch.Generator(device="cpu").manual_seed(1234)
            p = int(operator.split("_")[-1]) / 100
            mask = torch.rand(1, 1, h, w, generator=rng) > p
            mask = mask.float()

        mask = mask.expand(1, 1, h, w)  # expand to (1, C, H, W)
        if get_operator_param:
            return mask
        else:
            return Inpainting(mask=mask, **kwargs)
    elif "convolution" in operator.lower():
        if "convolution_gaussian" in operator.lower():
            std = float(operator.split("_")[-1])
            # Generate an isotropic Gaussian kernel
            size = h
            center = size // 2
            x = torch.arange(size, dtype=torch.float32) - center
            y = torch.arange(size, dtype=torch.float32) - center
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2 * std**2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(size, size).to(dtype=torch.float32)

        elif "convolution_defocus" in operator.lower():
            radius = int(operator.split("_")[-1])
            size = h
            center = size // 2
            x = torch.arange(size, dtype=torch.float32) - center
            y = torch.arange(size, dtype=torch.float32) - center
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            kernel = (xx**2 + yy**2) <= radius**2  # Disk kernel
            kernel = kernel / kernel.sum()
            kernel = kernel.view(size, size).to(dtype=torch.float32)

        elif "convolution_motion" in operator.lower():
            kernel_index = (
                1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
            )
            kernel = load_degradation("Levin09.npy", "./cache", index=kernel_index)
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel / kernel.sum(
                dim=(-2, -1), keepdim=True
            )  # Normalize the kernel
            kernel = kernel.to(dtype=torch.float32).squeeze()
        else:
            raise ValueError(f"Unknown operator: {operator}")

        if get_operator_param:
            return kernel
        else:
            return ConvolutionOperatorMatrix(
                filter=kernel, input_size=input_size, is_invertible=False, **kwargs
            )
    else:
        raise ValueError(f"Unknown operator: {operator}")


def get_preinverse_operator(
    operator: str = None,
    pre_inverse_type: str = "identity",
    input_size: tuple = None,
    output_size: tuple = None,
    **kwargs,
) -> Operator:
    r"""
    Pre-defined pre-inverse operators for numerical experiments.
    Args:
        - pre_inverse_type (str): type of pre-inverse operator: "identity" or "inverse"
    ALL:
        - denoising: identity operator
        - inpainting_center_p: where p is the size of the inpainting mask
        - inpainting_random_p: where p is the percentage of pixels to be inpainted
        - convolution_gaussian_p: where p is the std of the Gaussian kernel
        - convolution_motion: convolution with a motion blur kernel (Levin)
    """

    if pre_inverse_type.lower() == "identity" or operator.lower() == "denoising":
        return Identity(**kwargs)
    elif pre_inverse_type.lower() == "inverse":
        # Inpainting: return the same operator
        operator_param = get_operator(
            operator=operator,
            input_size=input_size,
            output_size=output_size,
            get_operator_param=True,
            **kwargs,
        )

        if "inpainting" in operator.lower():
            return Inpainting(
                mask=operator_param,
                input_size=input_size,
                output_size=output_size,
                **kwargs,
            )
        elif "convolution" in operator.lower():
            dtype = operator_param.dtype
            kernel_fft = fft.fft2(operator_param.double())
            kernel_inv = fft.ifft2(1.0 / (kernel_fft + 1e-7)).real.to(dtype=dtype)
            return ConvolutionOperatorMatrix(
                filter=kernel_inv.squeeze(),
                input_size=input_size,
                is_invertible=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown operator: {operator}")
    elif pre_inverse_type.lower() == "epsilon_inverse":
        # Inpainting: return the same operator
        operator_param = get_operator(
            operator=operator,
            input_size=input_size,
            output_size=output_size,
            get_operator_param=True,
            **kwargs,
        )

        if "inpainting" in operator.lower():
            mask = operator_param.clone()
            mask[mask == 0] = 1e-3
            return Inpainting(
                mask=mask,
                input_size=input_size,
                output_size=output_size,
                is_invertible=True,
                **kwargs,
            )
        else:
            raise NotImplementedError(
                "Epsilon inverse is only implemented for inpainting operator"
            )


class Operator(nn.Module):
    r"""
    Base class for operators in the inverse problem (for both A and B).

    Args:
        input_size (tuple): Size of the input tensor (C, H, W).
        output_size (tuple): Size of the output tensor (C, H, W).
        tol (float): Tolerance for the pseudo-inverse computation, default is 1e-6.
        device (str): Device to run the computations on, default is "cpu".
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(
        self,
        input_size=None,
        output_size=None,
        patch_size=None,
        tol: float = 1e-6,
        is_invertible=False,
        device="cpu",
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = tuple(input_size) if input_size is not None else None
        self.output_size = (
            tuple(output_size) if output_size is not None else self.input_size
        )
        self.device = device
        self.dtype = dtype
        self.tol = tol
        self.factory_kwargs = dict(device=device, dtype=dtype)
        self.patch_size = patch_size
        self.is_invertible = is_invertible
        self._adjoint_operator = None

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Apply the operator A to the input tensor x.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, M).
        """
        return x

    def adjoint(self, y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Apply the adjoint operator A^T to the input tensor y.

        Args:
            y (torch.Tensor): Input tensor of shape (B, M).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        if self._adjoint_operator is None:
            self._adjoint_operator = dinv.physics.adjoint_function(
                partial(self.forward, **kwargs),
                input_size=(y.size(0),) + self.input_size,
                **self.factory_kwargs,
            )

        return self._adjoint_operator(y)

    def pinv(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        Apply the pseudo inverse A^+ to the input tensor y.

        By default, it is computed by solving the least squares problem:
        .. math::
            A^+ y = (A^T A + \lambda I)^{-1} A^T y
            w
        where :math:`\lambda` is a small regularization parameter (1e-7).

        Args:
            y (torch.Tensor): Input tensor of shape (B, M).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        return least_squares(
            partial(self.forward, **kwargs),
            partial(self.adjoint, **kwargs),
            y,
            self.adjoint(y, **kwargs),
            gamma=1e5,
            tol=self.tol,
            solver="CG",
            max_iter=1000,
            verbose=True,
        )

    def forward_patch(
        self,
        y: torch.Tensor,
        patch_index: tuple[int],
        patch_size: int = None,
    ) -> torch.Tensor:
        r"""
        Apply the patch operator \Pi_n B to the input tensor y, extracting patches.
        This method should be only used for the operator B.
        Args:
            y (torch.Tensor): Input tensor of shape (B, M).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        if patch_size is not None:
            self.patch_size = patch_size

        assert self.patch_size is not None, "Patch size must be specified."
        pad = self.patch_size // 2
        padding = (
            pad - (self.patch_size - 1) % 2,
            pad,
            pad - (self.patch_size - 1) % 2,
            pad,
        )
        x = self.forward(y)
        x = F.pad(x, padding, mode="circular")

        out = x[
            ...,
            patch_index[0] : patch_index[0] + self.patch_size,
            patch_index[1] : patch_index[1] + self.patch_size,
        ]
        return out

    def adjoint_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: int = None,
    ) -> torch.Tensor:
        r"""
        Apply the adjoint of patch operator \Pi_n B to the input patch v.
        This method should be only used for the operator B.
        Args:
            v (torch.Tensor): Input tensor of shape (B, C, P, P).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        _adjoint_patch_operator = dinv.physics.adjoint_function(
            partial(self.forward_patch, patch_size=patch_size, patch_index=patch_index),
            input_size=(v.size(0),) + self.input_size,
            **self.factory_kwargs,
        )
        return _adjoint_patch_operator(v)

    def pinv_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: tuple[int],
        det: bool = False,
    ) -> torch.Tensor:
        r"""
        Apply the pseudo-inverse (\Pi_n forward)^+ to the input tensor v, which is a patch tensor of shape (B, C, P, P).

        Args:
            v (torch.Tensor): Input tensor of shape (B, C, P, P).
            patch_index (tuple[int]): Index of the patch (row, col) to apply the pseudo-inverse to.
            patch_size (tuple[int]): Size of the patch (height, width).
            det (bool): If True, return the 1 / log determinant of the pseudo-inverse matrix.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        if patch_size is not None:
            self.patch_size = patch_size

        sol = least_squares(
            partial(
                self.forward_patch,
                patch_index=patch_index,
                patch_size=self.patch_size,
            ),
            partial(
                self.adjoint_patch,
                patch_index=patch_index,
                patch_size=self.patch_size,
            ),
            v,
            self.adjoint_patch(v, patch_index=patch_index, patch_size=self.patch_size),
            gamma=1e5,
            tol=self.tol,
            solver="CG",
            max_iter=10,
            verbose=True,
        )
        if det:
            return sol, 0.0  # Default to 0 for now, to be implemented later
        else:
            return sol

    def adjoint_pinv_patch(
        self,
        x: torch.Tensor,
        patch_index: tuple[int],
        patch_size: int = None,
    ) -> torch.Tensor:
        r"""
        Apply the adjoint of the pseudo-inverse operator \Pi_n B to the input patch v.
        This method should be only used for the operator B.
        Args:
            v (torch.Tensor): Input tensor of shape (B, C, P, P).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        _adjoint_pinv_operator = dinv.physics.adjoint_function(
            partial(self.forward_patch, patch_size=patch_size, patch_index=patch_index),
            input_size=(x.size(0),) + self.input_size,
            **self.factory_kwargs,
        )
        return _adjoint_pinv_operator(x)

    def _distance_local_equiv_fullrank(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
    ) -> torch.Tensor:
        r"""
        Compute the Gaussian weights for the local equivariant operator.
            N(v, Q_n A x, sigma^2 Q_n Q_n^T) for all n.
        where self.forward is A. This function should be only called when self is A.
        """
        h, w = batch.shape[-2:]
        batch_measured = self.forward(batch)
        batch_measured = B.forward(batch_measured)
        patch_dataset = unfold_image(batch_measured, size=patch_size, step=1).flatten(
            1, 2
        )
        distance = []

        for i, j in itertools.product(range(h), range(w)):
            n = i * h + j
            patch_n = patch_dataset[:, n, ...]

            Q_dagger_v, log_det_inv = B.pinv_patch(
                v, patch_index=(i, j), patch_size=patch_size, det=True
            )
            Q_dagger_v = Q_dagger_v.flatten(-3, -1).unsqueeze(0)

            Q_dagger_Q_Ax = B.pinv_patch(
                patch_n, patch_index=(i, j), patch_size=patch_size, det=False
            )
            Q_dagger_Q_Ax = Q_dagger_Q_Ax.flatten(-3, -1).unsqueeze(0)

            dist = batched_cdist(Q_dagger_Q_Ax, Q_dagger_v)
            distance.append(dist)

        distance = -0.5 * torch.cat(distance, dim=0) / (sigma**2) + log_det_inv
        return distance.transpose(0, 1)

    def _distance_local_equiv_general(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
        strat_tab: torch.Tensor,
        rank_tab: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the Gaussian weights for the local equivariant operator.
            N(v, Q_n A x, sigma^2 Q_n Q_n^T) for all n.
        where self.forward is A. This function should be only called when self is A.
        """
        raise NotImplementedError("Should be implemented in subclass.")

    def proj_im(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Project the input tensor x onto the image of the operator A.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        return x

    def proj_im_patch(self, v: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the projection onto Im(Pi_n B).

        Args:
            v (torch.Tensor): Patch tensor of shape (B, C, P, P).

        Returns:
            torch.Tensor: Output tensor of shape (B, H * W, C * P * P, M).
        """
        return v
        # This is a no-op for the base class, subclasses should implement this method.

    def test_pinv(self, **kwargs):
        r"""
        Check pseudo-inverse and projection properties of the operator.
        """
        for _ in tqdm(range(5)):
            x = torch.randn((4,) + self.input_size, **self.factory_kwargs)

            torch.testing.assert_close(
                self(x, **kwargs),
                self(self.pinv(self(x, **kwargs), **kwargs)),
                rtol=self.tol,
                atol=self.tol,
            )

            y = torch.randn_like(self(x))
            torch.testing.assert_close(
                self.pinv(self(self.pinv(y, **kwargs), **kwargs)),
                self.pinv(y, **kwargs),
                rtol=self.tol,
                atol=self.tol,
            )
        print(f"{self.__class__.__name__} passed the pseudo inverse tests.\n")

    def test_adjointness(self, u=None, **kwargs):
        r"""
        Numerically check that :math:`A^{\top}` is indeed the adjoint of :math:`A`.

        :param torch.Tensor u: initialisation point of the adjointness test method

        :return: (float) a quantity that should be theoretically 0. In practice, it should be of the order of the chosen dtype precision (i.e. single or double).

        """
        if u is None:
            u = torch.randn(8, *self.input_size).to(**self.factory_kwargs)
        u_in = u.to(**self.factory_kwargs)
        Au = self(u_in, **kwargs)

        v = torch.randn_like(Au)
        Atv = self.adjoint(v, **kwargs)

        s1 = (v.conj() * Au).flatten().sum()
        s2 = (Atv * u_in.conj()).flatten().sum()

        torch.testing.assert_close(s1.conj(), s2, rtol=self.tol, atol=self.tol)
        print(f"{self.__class__.__name__} passed the adjointness tests.\n")


class Identity(Operator):
    r"""
    Identity operator that does not change the input tensor.

    Args:
        device (str): Device to run the computations on, default is "cpu".
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(self, device="cpu", dtype=torch.float32, *args, **kwargs):
        super().__init__(
            device=device, dtype=dtype, is_invertible=True, *args, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_size is None:
            self.input_size = x.shape[1:]
        return x

    def adjoint(self, y):
        return y

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        return y

    def proj_im(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _as_4d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        return tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        return tensor
    else:
        raise ValueError("Input tensor must be of shape (C, H, W) or (H, W).")


class Inpainting(Operator):
    r"""
    Inpainting operator that applies a mask to the input tensor.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        input_size: tuple = None,
        output_size: tuple = None,
        device="cpu",
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )

        mask = _as_4d_tensor(mask)
        self.register_buffer("mask", mask)

        inv_mask = torch.where(mask != 0, 1.0 / mask, 0)
        self.register_buffer("inv_mask", inv_mask)

        if input_size == None:
            self.input_size = mask.shape[1:]  # (C, H, W)
        self.output_size = self.input_size
        self.to(**self.factory_kwargs)

    def _pad_mask(self, patch_size: int = 5):
        r"""
        Pad the mask
        """
        if not hasattr(self, "mask_padded"):
            pad = patch_size // 2

            mask = self.mask  # (1, 1, H, W)
            padding = (pad - (patch_size - 1) % 2, pad, pad - (patch_size - 1) % 2, pad)
            mask_padded = F.pad(
                mask, padding, mode="circular"
            )  # (1, 1, H + patch_size, W + patch_size)
            self.register_buffer("mask_padded", mask_padded)

    def _pad_inv_mask(self, patch_size: int = 5):
        r"""
        Pad the mask
        """
        if not hasattr(self, "inv_mask_padded"):
            pad = patch_size // 2

            inv_mask = self.inv_mask  # (1, 1, H, W)
            padding = (pad - (patch_size - 1) % 2, pad, pad - (patch_size - 1) % 2, pad)
            inv_mask_padded = F.pad(
                inv_mask, padding, mode="circular"
            )  # (1, 1, H + patch_size, W + patch_size)
            self.register_buffer("inv_mask_padded", inv_mask_padded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.mask

    def proj_im(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask * self.inv_mask

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.inv_mask

    def proj_im_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: int,
        rank: bool = False,
        im_pinv_patch: bool = False,
        *arg,
        **kwarg,
    ) -> torch.Tensor:
        r"""
        Project the input tensor v, which is a patch tensor of shape (B, *, *, *), on the image space of \Sigma_n A

        Args:
            v (torch.Tensor): Input tensor of shape (B, C, patch_size, patch_size) if im_pinv_patch=False, else (B, C, H + patch_size, W + patch_size).
            patch_index (tuple[int]): Index of the patch (row, col) to apply the pseudo-inv to.
            patch_size (int): Size of the patch.
            det (bool) : Return the pseudo determinant of True
            im_pinv_patch (bool): Return the projection of v on Image of the pseudoinverse of (\Sigma_n A) if True
        Returns:
            torch.Tensor: Output tensor of shape (B, *, *, *).
        """
        assert v.dim() == 4, "Input tensor should have 4 dimensions"
        self._pad_mask(patch_size=patch_size)
        patch_mask = self.mask_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]
        mat = torch.zeros_like(patch_mask)
        mat[patch_mask != 0] = 1

        if not im_pinv_patch:
            # print(mat.shape, v.shape)
            proj_v = mat * v.to(**self.factory_kwargs)
        else:
            # v = circular_patch(v, patch_index, patch_size)
            v = v[
                ...,
                patch_index[0] : patch_index[0] + patch_size,
                patch_index[1] : patch_index[1] + patch_size,
            ]
            proj_v = mat * v.to(**self.factory_kwargs)

        if rank:
            return proj_v, torch.round(torch.sum(mat != 0)).to(torch.float)
        else:
            return proj_v

    def forward_patch(
        self, y: torch.Tensor, patch_index: tuple[int], patch_size: tuple[int]
    ) -> torch.Tensor:

        self.pad_mask(patch_size=patch_size)

        mask_padded = self.mask_padded
        mask_patch = mask_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]
        pad = patch_size // 2
        padding = (pad - (patch_size - 1) % 2, pad, pad - (patch_size - 1) % 2, pad)
        y = F.pad(y, padding, mode="circular")
        y = y[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]
        return y * mask_patch

    # For inpainting, the pseudo-inverse of Q is the same as the adjoint
    def pinv_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: int,
        rank: float,
        det: bool = False,
        pad_after: bool = False,
        *arg,
        **kwarg,
    ) -> torch.Tensor:
        assert patch_size % 2 == 1, "patch_size should be an odd number"
        v = v.to(**self.factory_kwargs)
        if v.ndim == 5:
            b, n = v.shape[:2]
            v = v.flatten(0, 1)
            output_shape = (b, n, *self.input_size)

        else:
            output_shape = (v.size(0), *self.input_size)

        self._pad_inv_mask(patch_size=patch_size)

        inv_mask_padded = self.inv_mask_padded
        inv_mask_patch = inv_mask_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]
        log_det_inv = torch.sum(inv_mask_patch[inv_mask_patch != 0].log()).item()
        prefactor_log = log_det_inv - 0.5 * rank * np.log(2 * np.pi)
        prefactor_log = prefactor_log * self.input_size[0]

        v = v * inv_mask_patch
        out = v.view(output_shape[:-3] + v.shape[-3:])

        if pad_after:
            out_padded = F.pad(
                out,
                pad=(
                    0,
                    self.input_size[-1] - patch_size,
                    0,
                    self.input_size[-2] - patch_size,
                ),
                mode="constant",
                value=0,
            )
            out_padded = torch.roll(
                out_padded,
                shifts=(
                    patch_index[0] - (patch_size // 2),
                    patch_index[1] - (patch_size // 2),
                ),
                dims=(-2, -1),
            )
            out = out_padded.view(output_shape)

        if det:
            return out, prefactor_log
        else:
            return out

    def _distance_local_equiv_general(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
        strat_tab: torch.Tensor,
        rank_tab: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the Gaussian weights for the local equivariant operator.
            N(v, Q_n A x, sigma^2 Q_n Q_n^T) for all n.
        where self.forward is A. This function should be only called when self is A.
        """
        h, w = batch.shape[-2:]
        batch_masked = self.forward(batch)

        pad = patch_size // 2
        padding = (pad - (patch_size - 1) % 2, pad, pad - (patch_size - 1) % 2, pad)
        batch_masked_padded = F.pad(batch_masked, padding, mode="circular")

        distance = -torch.inf * torch.ones(
            batch.size(0), h * w, h * w, **self.factory_kwargs
        )

        for i, j in itertools.product(range(h), range(w)):
            n = i * h + j
            v_acc_idx = strat_tab[i * h + j]
            v_acc = v[v_acc_idx]
            rank = rank_tab[i * h + j]

            Q_dagger_n_v, prefactor_log = B.pinv_patch(
                v_acc,
                patch_index=(i, j),
                patch_size=patch_size,
                det=True,
                rank=rank,
            )
            Q_dagger_n_v = Q_dagger_n_v.flatten(-3, -1)  # (K, *)

            Q_dagger_Q_A_batch = B.proj_im_patch(
                batch_masked_padded,
                patch_index=(i, j),
                patch_size=patch_size,
                im_pinv_patch=True,
                rank=False,
            )  # (B, *)
            Q_dagger_Q_A_batch = Q_dagger_Q_A_batch.flatten(-3, -1)

            dist = -batched_cdist(
                Q_dagger_Q_A_batch.unsqueeze(0), Q_dagger_n_v.unsqueeze(0)
            ) / (2 * sigma**2)
            dist = dist.squeeze(0) + prefactor_log  # shape (B, K)
            distance[:, n, v_acc_idx] = dist

        return distance

    def _distance_local_equiv_fullrank(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
    ) -> torch.Tensor:

        mask_inv = B.inv_mask
        mask_inv_patch = unfold_image(mask_inv, size=patch_size).flatten(
            0, 2
        )  # (num_patch, 1, size, size)

        mask_patch_to_compute_det = mask_inv_patch.clone()
        mask_patch_to_compute_det[mask_patch_to_compute_det == 0] = 1
        log_det_inv = torch.sum(
            mask_patch_to_compute_det.flatten(-3, -1).log(), dim=-1
        )[
            :, None, None
        ]  # (num_patch, )

        batch = self.forward(batch)
        # Qn^dagger Qn Ax = Pi_n Ax
        patch_batch = (
            unfold_image(batch, size=patch_size).flatten(1, 2).transpose(0, 1)
        )  # (num_patch, B, C, size, size)

        # Compute Qn^dagger v for all n
        v = mask_inv_patch.unsqueeze(1) * v.unsqueeze(
            0
        )  # (num_patch, V, C, size, size)
        distance = batched_cdist(
            patch_batch.flatten(-3, -1), v.flatten(-3, -1)
        )  # (num_patch, B, V)
        distance = -distance / (2 * sigma**2) + log_det_inv * self.input_size[0]
        return distance.transpose(0, 1)

    def _indicator_and_rank(self, v: torch.Tensor, patch_size: int, eps: float = 1e-4):
        r"""
        Calculate the indicator of Im(\Sigma_n A) and indicator of E_r, this method should only be call for the pre-inverse operator
        Args:
            v (torch.Tensor): Input tensor of shape (B, C, patch_size, patch_size).
            eps (float): Tolerance constant when approximate the indicator.
            patch_size : patch size.
        Returns:
            torch.Tensor: Transformed tensor of shape (B, num_patch, C, size, size).
        """
        h, w = self.output_size[-2:]
        self._pad_mask(patch_size=patch_size)
        ind_im_Q_n = torch.ones(h * w, h * w, **self.factory_kwargs) * torch.inf
        rank_Q_n = torch.zeros(h * w, **self.factory_kwargs)
        v_norm = torch.linalg.vector_norm(v, dim=(-3, -2, -1), ord=2)
        
        # Can be accelerated by vmap, but could have memory issue
        for i, j in itertools.product(range(h), range(w)):
            proj_v, rank = self.proj_im_patch(
                v=v,
                patch_index=(i, j),
                patch_size=patch_size,
                rank=True,
            )
            rank_Q_n[i * h + j] = rank
            ind_n = (
                torch.linalg.vector_norm(proj_v - v, dim=(-3, -2, -1), ord=2)
                < eps * v_norm
            )
            ind_im_Q_n[i * h + j, ind_n] = rank
        ind_Ebar_r = torch.min(ind_im_Q_n, dim=0).values
        ind = ind_im_Q_n == ind_Ebar_r.expand(h * w, -1)
        return ind, rank_Q_n


class ConvolutionOperatorFFT(Operator):
    r"""
    Convolution operator that applies a linear transformation defined by a filter
    Args:
        filter (torch.Tensor): A filter of size (H, W)
        input_size (tuple): Size of the input tensor (C, H, W)
        device (str): Device to run the computations on, default is "cpu".
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(
        self,
        filter: torch.Tensor,
        input_size: tuple = None,
        device="cpu",
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=input_size,  # Output size is the same as input size for convolution
            device=device,
            dtype=dtype,
            *args,
            **kwargs,
        )
        filter = _as_4d_tensor(filter)

        assert (
            filter.size(-2) <= input_size[-2] and filter.size(-1) <= input_size[-1]
        ), f"Filter size {filter.shape[-2:]} must be less than or equal to input size {input_size[-2:]}"

        filter = filter.to(device=device, dtype=dtype)

        mask = dinv.physics.blur.filter_fft_2d(filter, input_size)
        angle = torch.angle(mask)
        mask = torch.abs(mask).unsqueeze(-1)
        mask = torch.cat([mask, mask], dim=-1)
        self.register_buffer("filter", filter)
        self.register_buffer("angle", torch.exp(-1.0j * angle))
        self.register_buffer("mask", mask)
        self.input_size = input_size
        self.output_size = input_size

        # avoid division by singular value = 0
        mask_inv = torch.zeros_like(self.mask)
        mask_inv[self.mask > 1e-5] = 1 / self.mask[self.mask > 1e-5]
        self.register_buffer("mask_inv", mask_inv)

        self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(self.mask * self.V_adjoint(x))

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        mask = torch.conj(self.mask)
        return self.V(mask * self.U_adjoint(y))

    def V_adjoint(self, x: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(
            fft.rfft2(x, norm="ortho")
        ) 

    def U(self, x):
        return fft.irfft2(
            torch.view_as_complex(x) * self.angle,
            norm="ortho",
            s=self.input_size[-2:],
        )

    def U_adjoint(self, y):
        return torch.view_as_real(
            fft.rfft2(y, norm="ortho") * torch.conj(self.angle)
        )  

    def V(self, x):
        return fft.irfft2(
            torch.view_as_complex(x), norm="ortho", s=self.input_size[-2:]
        )

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        return self.V(self.mask_inv * self.U_adjoint(y))

    def proj_im(self, x: torch.Tensor) -> torch.Tensor:
        return self.V(self.V_adjoint(x))

    def pinv_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: tuple[int],
        det: bool = False,
    ) -> torch.Tensor:
        if v.ndim == 5:
            b, n = v.shape[:2]
            v = v.flatten(0, 1)
            out_shape = (b, n, *self.input_size)
        else:
            b = v.size(0)
            out_shape = (b, *self.input_size)
        out = super().pinv_patch(
            v, patch_index=patch_index, patch_size=patch_size, det=False
        )

        out = out.view(*out_shape)
        if det:
            return out, 1 / torch.sum(self.mask[..., 0])
        else:
            return out


class MatrixOperator(Operator):
    r"""
    Matrix operator that applies a linear transformation defined by a matrix A.
    Args:
        A (torch.Tensor): Forward operator matrix of shape (M, N).
        device (str): Device to run the computations on, default is "cpu".
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(
        self,
        matrix: torch.Tensor,
        input_size: tuple = None,
        output_size: tuple = None,
        device="cpu",
        dtype=torch.float32,
        spatial_only: bool = False,
        is_invertible: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            device=device,
            dtype=dtype,
            is_invertible=is_invertible,
            *args,
            **kwargs,
        )
        A = matrix.to(device=device, dtype=dtype)
        self.input_dim = A.shape[1]
        self.output_dim = A.shape[0]

        self.register_buffer("A", A.to(**self.factory_kwargs))
        self.spatial_only = spatial_only
        self.n_channels = input_size[0]

        if spatial_only:
            assert self.input_dim == np.prod(
                self.input_size[1:]
            ), f"Input size {self.input_size} does not match input dimension {self.input_dim}."

            self._input_matrix_shape = (self.n_channels, self.input_dim)
            self._output_matrix_shape = (self.n_channels, self.output_dim)
        else:
            assert self.input_dim == np.prod(
                self.input_size
            ), f"Input size {self.input_size} does not match input dimension {self.input_dim}."
            self._input_matrix_shape = (self.input_dim,)
            self._output_matrix_shape = (self.output_dim,)

        self.to(**self.factory_kwargs)

    def _as_input_vector(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Convert the input tensor x to the appropriate shape for matrix multiplication.
        If spatial_only is True, reshape x to (B, C, H * W, 1).
        Otherwise, reshape x to (B, C * H * W, 1).
        """
        return x.view(-1, *self._input_matrix_shape, 1)

    def _as_output_vector(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Convert the output tensor x to the appropriate shape after matrix multiplication.
        If spatial_only is True, reshape x to (B, C, H * W, 1).
        Otherwise, reshape x to (B, C * H * W, 1).
        """
        return x.view(-1, *self._output_matrix_shape, 1)

    def _set_A_padded(self, patch_size: int = 5):
        r"""
        Set the matrix for the operator, padded to match the input spatial size for the patches.
        The padded matrix has shape (M, C, H + patch_size, W + patch_size).

        Args:
            patch_size (int): Spatial size of the patches to be extracted.
        """
        if not hasattr(self, "A_padded"):
            assert (
                self.output_size[-2] == self.output_size[-1]
            ), "The output images should be square"
            pad = patch_size // 2

            A = self.A  # (M, N )
            shape = (
                (-1, 1, *self.input_size[1:])
                if self.spatial_only
                else (-1, *self.input_size)
            )
            A = A.transpose(-2, -1).view(shape)  # (N, M)
            padding = (pad - (patch_size - 1) % 2, pad, pad - (patch_size - 1) % 2, pad)
            A_padded = F.pad(
                A, padding, mode="circular"
            )  # (N, C, H + patch_size, W + patch_size)

            self.register_buffer("A_padded", A_padded)

    def _set_A_A_dagger(self):
        if (not hasattr(self, "A_A_dagger")) and (not hasattr(self, "A_dagger")):
            A = self.A
            A_dagger = torch.linalg.pinv(
                A.to(dtype=torch.float64), atol=1e-8, rtol=1e-8
            ).to(**self.factory_kwargs)
            self.register_buffer("A_dagger", A_dagger)
            A_A_dagger = torch.matmul(A, A_dagger).to(**self.factory_kwargs)
            self.register_buffer("A_A_dagger", A_A_dagger)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.A, self._as_input_vector(x)).view(
            -1, *self.output_size
        )

    def pinv(self, y: torch.Tensor) -> torch.Tensor:
        self._set_A_A_dagger()
        return torch.matmul(self.A, self._as_output_vector(y)).view(
            -1, *self.input_size
        )

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.A.t(), self._as_output_vector(y)).view(
            -1, *self.input_size
        )

    def forward_patch(
        self, y: torch.Tensor, patch_index: tuple[int], patch_size: int
    ) -> torch.Tensor:
        r"""
        Apply the operator \Sigma_n A to the input tensor y, which is a patch tensor of shape (B, C, H, W).

        Args:
            y (torch.Tensor): Input tensor of shape  (B, C, H, W).
            patch_index (tuple[int]): Index of the patch (row, col) to apply the pseudo-inverse to.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """

        if not hasattr(self, "A_padded"):
            self._set_A_padded(patch_size=patch_size)

        mat = self.A_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]  # (M, C, patch_size, patch_size)
        # mat = mat.flatten(-2, -1) if self.spatial_only else mat.flatten(-3, -1)
        mat = mat.flatten(-3, -1)
        y = y.to(**self.factory_kwargs)
        return torch.matmul(
            mat.transpose(0, 1),
            self._as_input_vector(y),
        ).view(
            -1, self.n_channels, patch_size, patch_size
        )  # (B, C, patch_size, patch_size)

    def adjoint_patch(self, v, patch_index, patch_size=None):
        if not hasattr(self, "A_padded"):
            self._set_A_padded(patch_size=patch_size)

        mat = self.A_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]  # (M, C, patch_size, patch_size)
        mat = mat.flatten(-2, -1) if self.spatial_only else mat.flatten(-3, -1)
        y = y.to(**self.factory_kwargs)
        return torch.matmul(
            mat,
            self._as_output_vector(y),
        ).view(
            -1, *self.input_size
        )  # (B, C, patch_size, patch_size)

    def proj_im(self, x):
        if self.is_invertible:
            return x
        else:
            self._set_A_A_dagger()
            return torch.matmul(self.A_A_dagger, self._as_output_vector(x)).view(
                -1, *self.output_size
            )

    def pinv_patch(
        self,
        v: torch.Tensor,
        patch_index: tuple[int],
        patch_size: tuple[int],
        rank: float = None,
        det=True,
    ) -> torch.Tensor:
        r"""
        Apply the pseudo-inverse of \Sigma_n A to the input tensor v, which is a patch tensor of shape (B, C, patch_height, patch_width).

        Args:
            v (torch.Tensor): Input tensor of shape (B, C, P, P).
            patch_index (tuple[int]): Index of the patch (row, col) to apply the pseudo-inverse to.
            patch_size (tuple[int]): Size of the patch (height, width).
            det (bool): If True, return the determinant of the pseudo-inverse matrix.

        Returns:
            torch.Tensor: Output tensor of shape (B, *output_size).
        """

        if not hasattr(self, "A_padded"):
            self._set_A_padded(patch_size=patch_size)

        b = v.size(0)
        v = (
            rearrange(v, "b c h w -> (c h w) b")
            if not self.spatial_only
            else rearrange(v, "b c h w -> (h w) (b c)")
        )

        full_rank = (
            (rank == (self.input_size[0] * patch_size**2))
            if not self.spatial_only
            else (rank == patch_size**2)
        )

        mat = self.A_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]
        # mat = mat.flatten(-2, -1) if self.spatial_only else mat.flatten(-3, -1)
        mat = mat.flatten(-3, -1).transpose(-1, -2)

        if det:
            S = torch.linalg.svdvals(mat)
            if rank is None:
                rank = (S > 1e-5).sum()
                prefactor_log = -torch.sum(S[S > 1e-5].log()) - 0.5 * rank * np.log(
                    2 * np.pi
                )

            if self.spatial_only:
                prefactor_log = prefactor_log * self.input_size[0]

        # mat = mat.view((1,) * (v.dim() - mat.dim()) + mat.shape)

        if full_rank:
            solution = torch.linalg.lstsq(mat, v)
        else:
            solution = torch.linalg.lstsq(
                mat.cpu().to(dtype=torch.float64),
                v.cpu().to(dtype=torch.float64),
                driver="gelsd",
            )
        out = solution.solution.transpose(-1, -2).view(b, *self.input_size)

        # Note: can not get the singular values when the matrix has fewer rows than columns
        if det:
            return out.to(device=self.device, dtype=self.dtype), prefactor_log.to(
                **self.factory_kwargs
            )
        else:
            return out.to(device=self.device, dtype=self.dtype)

    def proj_im_patch(
        self,
        v: torch.Tensor,
        patch_index=tuple[int],
        patch_size=int,
        im_pinv_patch: bool = False,
        rank: bool = False,
    ) -> torch.Tensor:
        r"""
        Project the input tensor v, which is a patch tensor of shape (B, *, *, *), on the image space of \Sigma_n A

        Args:
            v (torch.Tensor): Input tensor of shape (B, *, *, *).
            patch_index (tuple[int]): Index of the patch (row, col) to apply the pseudo-inverse to.
            patch_size (int): Size of the patch.
            det (bool) : Return the pseudo determinant of True
        Returns:
            torch.Tensor: Output tensor of shape (B, *, *, *).
        """

        if not hasattr(self, "A_padded"):
            self._set_A_padded(patch_size=patch_size)
        mat = self.A_padded[
            ...,
            patch_index[0] : patch_index[0] + patch_size,
            patch_index[1] : patch_index[1] + patch_size,
        ]  # (N, C, patch_size, patch_size)

        # mat = mat.flatten(-2, -1) if self.spatial_only else mat.flatten(-3, -1)
        # mat = mat.transpose(0, 1)
        mat = mat.flatten(-3, -1).transpose(-2, -1)
        mat_pinv = torch.linalg.pinv(mat)
        projector = (
            torch.matmul(mat, mat_pinv)
            if not im_pinv_patch
            else torch.matmul(mat_pinv, mat)
        )
        shape = (
            (-1, self.n_channels, patch_size, patch_size)
            if not im_pinv_patch
            else (-1, *self.input_size)
        )

        if v.ndim == 4:
            v = v.flatten(-3, -1) if not self.spatial_only else v.flatten(-2, -1)
        proj_v = (
            torch.matmul(projector, v.unsqueeze(-1)).squeeze(-1).view(shape)
        )  # (B, C, *, *)
        if rank:
            return proj_v, torch.round(
                torch.sum(torch.matmul(mat, mat_pinv).diagonal())
            )
        else:
            return proj_v

    def _indicator_and_rank(self, v: torch.Tensor, patch_size: int, eps: float = 1e-4):
        r"""
        Calculat the product of the indicatrix of Im(\Sigma_n A) and indicatrix of E_r, this method should only be call for the pre-inverse operator
        Args:
            v (torch.Tensor): Input tensor of shape (B, C, patch_size, patch_size).
            eps (float): Tolerance constant when approximate the indicatrix.
            patch_size : patch size.
        Returns:
            torch.Tensor: Transformed tensor of shape (B, num_patch, C, size, size).
        """
        assert (
            self.output_size[-2] == self.output_size[-1]
        ), "This method should be called for the pre-inverse operator"
        h, w = self.output_size[-2:]
        self._set_A_padded(patch_size=patch_size)
        ind_im_Q_n = torch.ones(h * w, h * w, **self.factory_kwargs) * torch.inf
        rank_Q_n = torch.zeros(h * w, **self.factory_kwargs)
        v_norm = torch.linalg.norm(v.flatten(-3, -1), dim=-1)
        for i, j in itertools.product(range(h), range(w)):
            proj_v, rank = self.proj_im_patch(
                v=v,
                patch_index=(i, j),
                patch_size=patch_size,
                rank=True,
            )
            rank_Q_n[i * h + j] = rank
            ind_n = (
                torch.linalg.norm((proj_v - v).flatten(1, -1), dim=-1) < eps * v_norm
            )
            ind_im_Q_n[i * h + j, ind_n] = rank
        ind_Ebar_r, _ = torch.min(ind_im_Q_n, dim=0)
        ind = ind_im_Q_n == ind_Ebar_r.expand(h * w, -1)
        return ind, rank_Q_n

    def _distance_local_equiv_general(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
        strat_tab: torch.Tensor,
        rank_tab: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Compute the Gaussian weights for the local equivariant operator.
            N(v, Q_n A x, sigma^2 Q_n Q_n^T) for all n.
        where self.forward is A. This function should be only called when self is A.
        """
        h, w = batch.shape[-2:]
        batch_mesured = self.forward(batch)

        distance = -torch.inf * torch.ones(
            batch.size(0), h * w, h * w, **self.factory_kwargs
        )

        for i, j in itertools.product(range(h), range(w)):
            n = i * h + j
            v_acc_idx = strat_tab[n]
            v_acc = v[v_acc_idx]
            rank = rank_tab[n]

            Q_dagger_n_v, prefactor_log = B.pinv_patch(
                v_acc,
                patch_index=(i, j),
                patch_size=patch_size,
                det=True,
                rank=rank,
            )
            Q_dagger_n_v = Q_dagger_n_v.flatten(-3, -1)  # (K, *)

            Q_dagger_Q_A_batch = B.proj_im_patch(
                batch_mesured,
                patch_index=(i, j),
                patch_size=patch_size,
                im_pinv_patch=True,
                rank=False,
            )  # (B, *)
            Q_dagger_Q_A_batch = Q_dagger_Q_A_batch.flatten(-3, -1)

            dist = -batched_cdist(
                Q_dagger_Q_A_batch.unsqueeze(0), Q_dagger_n_v.unsqueeze(0)
            ) / (2 * sigma**2)
            dist = dist.squeeze(0) + prefactor_log  # shape (B, K)

            distance[:, n, v_acc_idx] = dist

        return distance


class ConvolutionOperatorMatrix(MatrixOperator):
    r"""
    Convolution operator that applies a linear transformation defined by a filter using matrix multiplication.
    Args:
        filter (torch.Tensor): A filter of size (H, W)
        input_size (tuple): Size of the input tensor (C, H, W)
        device (str): Device to run the computations on, default is "cpu".
        dtype (torch.dtype): Data type for the tensors, default is torch.float32.
    """

    def __init__(
        self,
        filter: torch.Tensor,
        input_size: tuple,
        device: str = "cpu",
        dtype=torch.float32,
        is_invertible: bool = False,
        *args,
        **kwargs,
    ):
        assert input_size is not None, "input_size must be provided"
        assert (
            filter.size(-2) <= input_size[-2] and filter.size(-1) <= input_size[-1]
        ), f"Filter size {filter.shape[-2:]} must be less than or equal to input size {input_size[-2:]}"
        self.filter_size = filter.shape[-2:]

        # Create convolution matrix
        conv_matrix = build_circular_conv_matrix_fft(
            filter, input_size, device=device, dtype=dtype
        )

        output_size = tuple(input_size)
        super().__init__(
            matrix=conv_matrix,
            input_size=input_size,
            output_size=output_size,
            device=device,
            dtype=dtype,
            spatial_only=True,
            is_invertible=is_invertible,
            *args,
            **kwargs,
        )

    def test_Sn_constant(self, patch_size: int = 5):
        self._set_A_padded(patch_size=patch_size)
        mat = self.A_padded[..., 0 : 0 + patch_size, 0 : 0 + patch_size]
        mat = mat.flatten(-3, -1).transpose(-2, -1).to(**self.factory_kwargs)
        Sn = torch.matmul(mat, mat.transpose(-2, -1))
        _, h, w = self.input_size
        for i, j in itertools.product(range(h), range(w)):
            mat_n = self.A_padded[..., i : i + patch_size, j : j + patch_size]
            mat_n = mat_n.flatten(-3, -1).transpose(-2, -1).to(**self.factory_kwargs)
            torch.testing.assert_close(
                Sn, torch.matmul(mat_n, mat_n.transpose(-2, -1)), atol=1e-6, rtol=1e-3
            )

    def _set_Sn_pinv_sqrt(self, patch_size: int):
        r"This method should only be called for the pre-inverse operator B"
        if not hasattr(self, "Sn"):
            self._set_A_padded(patch_size=patch_size)
            mat = self.A_padded[..., 0 : 0 + patch_size, 0 : 0 + patch_size]
            mat = mat.flatten(-3, -1).transpose(-2, -1).to(**self.factory_kwargs)
            Sn = torch.matmul(mat, mat.transpose(-2, -1))
            # Sn_plus = torch.linalg.pinv(Sn.to(dtype=torch.float64), atol=1e-8, rtol=1e-8
            #     )
            Sn_plus = torch.linalg.inv(Sn.to(dtype=torch.float64))
            Sn_plus_sqrt = torch.linalg.cholesky(Sn_plus)
            torch.testing.assert_close(
                torch.matmul(Sn_plus_sqrt, Sn_plus_sqrt.transpose(-2, -1)), Sn_plus
            )
            self.register_buffer(
                "Sn_plus_sqrt", Sn_plus_sqrt.transpose(-2, -1).to(**self.factory_kwargs)
            )

    def _distance_local_equiv_fullrank(
        self,
        B: Operator,
        v: torch.Tensor,
        batch: torch.Tensor,
        patch_size: tuple[int],
        sigma: float,
    ) -> torch.Tensor:
        r"""
        Compute the Gaussian weights for the local equivariant operator.
            N(v, Q_n A x, sigma^2 Q_n Q_n^T) for all n.
        where self.forward is A. This function should be only called when self is A.
        """
        B._set_Sn_pinv_sqrt(patch_size=patch_size)
        Sn_plus_sqrt = B.Sn_plus_sqrt
        h, w = batch.shape[-2:]

        batch_measured = self.forward(batch)
        batch_measured = B.forward(batch_measured)
        patch_measured = (
            unfold_image(batch_measured, size=patch_size, step=1)
            .flatten(1, 2)
            .flatten(-2, -1)
        )

        Sn_plus_sqrt_v = torch.matmul(
            Sn_plus_sqrt, v.flatten(-2, -1).unsqueeze(-1)
        ).flatten(-3, -1)
        Sn_plus_sqrt_patch_measured = torch.matmul(
            Sn_plus_sqrt, patch_measured.unsqueeze(-1)
        ).flatten(-3, -1)

        distance = batched_cdist(
            Sn_plus_sqrt_patch_measured, Sn_plus_sqrt_v.unsqueeze(0)
        )

        distance = -0.5 * distance / (sigma**2)
        return distance
