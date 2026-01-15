"Preconditioning the denoiser for multiple noise levels"

import torch
import torch.nn as nn
import numpy as np


class DenoiserPrecond(nn.Module):
    r"""
    Base class for preconditioning the denoiser as follows:

        PrecondModel(x, sigma) = c_skip(sigma) * x + c_out(sigma) * Denoiser(c_in x, sigma)

    """

    def __init__(self, denoiser, eps=1e-3, *args, **kwargs):
        super().__init__()
        self.denoiser = denoiser
        self.eps = eps

    def c_skip(self, sigma):
        return 0.0

    def c_out(self, sigma):
        return 1.0

    def c_in(self, sigma):
        return 1.0

    def forward(self, x, timestep, *args, **kwargs):
        _sigma = self._handle_sigma_shape(
            timestep, x.dtype, x.device, batch_size=x.size(0), ndim=x.ndim
        )
        c_skip = self.c_skip(_sigma)
        c_out = self.c_out(_sigma)
        c_in = self.c_in(_sigma)

        x_hat = c_skip * x + c_out * self.denoiser(c_in * x, timestep)
        return x_hat

    @staticmethod
    def _handle_sigma_shape(sigma, dtype, device, batch_size=None, ndim=None):
        if isinstance(sigma, (float, int)):
            sigma = float(sigma)
        elif isinstance(sigma, torch.Tensor):
            sigma = sigma.squeeze().to(dtype=dtype, device=device)
        elif isinstance(sigma, list):
            sigma = torch.tensor(sigma, dtype=dtype, device=device).squeeze()
        elif isinstance(sigma, np.ndarray):
            sigma = torch.from_numpy(sigma, dtype=dtype, device=device).squeeze()
        else:
            raise TypeError(
                f"Sigma must be a float, int, or torch.Tensor. Got {type(sigma)}."
            )

        # Will reshape to (batch_size,) if batch_size is not None
        if batch_size is not None:
            # duplicate sigma for each sample in the batch
            if isinstance(sigma, float):
                sigma = torch.tensor([sigma] * batch_size, dtype=dtype, device=device)
            elif sigma.ndim == 0:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) == 1:
                sigma = sigma.view(1).expand(batch_size)
            elif sigma.ndim == 1 and sigma.size(0) != batch_size:
                raise ValueError(
                    f"Sigma tensor size {sigma.size(0)} does not match batch size {batch_size}."
                )

        # Will reshape to (batch_size, 1, ..., 1) if ndim is not None
        if ndim is not None:
            if isinstance(sigma, float):
                sigma = torch.tensor(sigma, dtype=dtype, device=device).view(
                    1, *([1] * (ndim - 1))
                )
            elif sigma.ndim == 0:
                sigma = sigma.view(1, *([1] * (ndim - 1)))
            elif sigma.ndim == 1:
                sigma = sigma.view(-1, *([1] * (ndim - 1)))
            else:
                raise ValueError(
                    f"Sigma tensor has {sigma.ndim} dimensions, expected 0 or 1."
                )
        return sigma


class EDMPrecond(DenoiserPrecond):
    r"""
    Implements the preconditioning for the denoiser as in EDM Diffusion
    """

    def __init__(self, denoiser, sigma_data=0.5, eps=1e-3, *args, **kwargs):
        super().__init__(denoiser=denoiser, eps=eps, *args, **kwargs)
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        self.sigma_data_sq = sigma_data**2
        self.eps = eps

    def c_skip(self, sigma):
        return self.sigma_data_sq / (self.sigma_data_sq + (sigma - self.eps) ** 2)

    def c_out(self, sigma):
        return (
            (sigma - self.eps)
            * self.sigma_data
            / ((self.sigma_data_sq + sigma**2) ** 0.5)
        )

    def c_in(self, sigma):
        return 1 / ((self.sigma_data_sq + sigma**2) ** 0.5)


class CMPrecond(DenoiserPrecond):
    r"""
    Implements the preconditioning for the denoiser as in EDM Diffusion
    """

    def __init__(self, denoiser, sigma_data=0.5, eps=1e-3, *args, **kwargs):
        super().__init__(denoiser=denoiser, eps=eps, *args, **kwargs)
        self.sigma_data = sigma_data
        self.sigma_data_sq = sigma_data**2

    def c_skip(self, sigma):
        return self.sigma_data_sq / (self.sigma_data_sq + (sigma - self.eps) ** 2)

    def c_out(self, sigma):
        return (
            (sigma - self.eps)
            * self.sigma_data
            / ((self.sigma_data_sq + sigma**2) ** 0.5)
        )

    def c_in(self, sigma):
        return 1.0


class InputPrecond(DenoiserPrecond):
    r"""
    Only normalize the input
    """

    def __init__(self, denoiser, sigma_data=0.5, eps=1e-3, *args, **kwargs):
        super().__init__(denoiser=denoiser, eps=eps, *args, **kwargs)
        self.sigma_data = sigma_data
        self.sigma_data_sq = sigma_data**2

    def c_skip(self, sigma):
        return self.sigma_data_sq / (self.sigma_data_sq + (sigma - self.eps) ** 2)

    def c_out(self, sigma):
        return (
            (sigma - self.eps)
            * self.sigma_data
            / ((self.sigma_data_sq + sigma**2) ** 0.5)
        )

    def c_in(self, sigma):
        return 1.0
