import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class DummyImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_images: int,
        image_size: int,
        channels: int = 1,
        dtype=torch.float32,
        device="cpu",
    ):
        self.num_images = num_images
        self.image_size = image_size
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.rng = torch.Generator(device=device)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        self.rng.manual_seed(idx)  # Ensure reproducibility
        img = torch.randn(
            self.channels,
            self.image_size,
            self.image_size,
            dtype=self.dtype,
            device=self.device,
            generator=self.rng,
        )
        return img


def unfold_image(x: torch.Tensor, size: int, step=1):
    r"""
    Return a view of patches of given size of a batch of images
    Args:
        x (torch.Tensor): input tensor of shape (B,C,H,W)
        size (int): size of the patches to be extracted
    Returns :
        torch.Tensor: Tensor of shape (B, H // step, W // step, C, size, size)
    """
    padding = size // 2
    output = F.pad(
        x,
        pad=(padding - (size - 1) % 2, padding, padding - (size - 1) % 2, padding),
        mode="circular",
    )
    output = output.unfold(
        dimension=-2, size=size, step=step
    )  # shape = (B, C, C_H,W, size)
    output = output.unfold(
        dimension=-2, size=size, step=step
    )  # shape = (B,C,C_H,C_W,size,size)
    return output.permute(0, 2, 3, 1, 4, 5)  # shape = (B,C_H,C_W,C,size,size)


@torch.no_grad()
def test_translation_equivariance(fn, x, patch_index: int = None):
    r"""
    Test the translation equivariance of the estimator.
    """

    if patch_index is None:
        patch_index = patch_index = torch.randint(0, np.prod(x.shape[-2:]), (1,)).item()
    fx = fn(x)
    tfx = unfold_image(fx, size=fx.shape[2]).flatten(1, 2)[:, patch_index, ...]
    tx = unfold_image(x, size=x.shape[2]).flatten(1, 2)[:, patch_index, ...]
    ftx = fn(tx)

    torch.testing.assert_close(
        tfx, ftx, rtol=1e-3, atol=1e-3, msg=f"Translation equivariance test failed"
    )
    print("Translation equivariance test passed")



##################################################################################
# BATCHED CDIST IMPLEMENTATION WITH TRITON
##################################################################################
import triton
import triton.language as tl
@triton.jit
def _batched_cdist_kernel(
    X_ptr,
    Y_ptr,
    C_ptr,
    B,
    M,
    N,
    D,
    stride_xb,
    stride_xm,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yd,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    m_tile = tl.program_id(1)
    n_tile = tl.program_id(2)

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # pointers to the beginning of batch b
    X_b_ptr = X_ptr + b * stride_xb
    Y_b_ptr = Y_ptr + b * stride_yb
    C_b_ptr = C_ptr + b * stride_cb

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d in range(0, D, BLOCK_D):
        d_idx = offs_d + d
        mask_d = d_idx < D

        # load X[b, offs_m, d_idx]
        x = tl.load(
            X_b_ptr + offs_m[:, None] * stride_xm + d_idx[None, :] * stride_xd,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # load Y[b, offs_n, d_idx]
        y = tl.load(
            Y_b_ptr + offs_n[:, None] * stride_yn + d_idx[None, :] * stride_yd,
            mask=mask_n[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        # compute squared differences and accumulate
        diff = x[:, None, :] - y[None, :, :]
        acc += tl.sum(diff * diff, axis=2)
    # store result
    tl.store(
        C_b_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _select_cdist_backend(x: torch.Tensor, y: torch.Tensor, backend: str) -> str:
    """
    Decide which backend to use based on tensor device and shape heuristics
    learned from benchmark patterns.

    Heuristic summary:
    - CPU: always use 'torch'.
    - Very large problems with huge D and both N & M large and sizable batch -> 'torch'.
    - Specific small case where Triton underperformed (B>=8, N,M<=128, D<=64) -> 'torch'.
    - Otherwise on CUDA prefer 'triton'.
    """
    if backend in ("torch", "triton"):
        return backend
    if backend not in (None, "auto"):
        # Unknown option, be safe
        return "torch"

    # Auto selection
    if not x.is_cuda or not y.is_cuda:
        return "torch"

    B, M, D = x.shape
    _, N, _ = y.shape

    # Large-D, large-(N,M), sizable batch => Torch was consistently better
    if (D >= 2048) and (M >= 1024 and N >= 1024) and (B >= 16):
        return "torch"

    # Guard a known small-shape regime where Triton lagged in benchmarks
    if (B >= 8) and (M <= 128 and N <= 128) and (D <= 64):
        return "torch"

    # Default to Triton on CUDA
    return "triton"


def batched_cdist(x: torch.Tensor, y: torch.Tensor, backend="torch") -> torch.Tensor:
    """
    Compute batched pairwise distances between x and y.
    x: (B, M, D)
    y: (B, N, D)
    returns: (B, M, N)

    backend: 'auto' | 'torch' | 'triton'
    - 'auto' chooses based on heuristics from observed benchmarks.
    """
    assert x.ndim == 3 and y.ndim == 3

    x = x.contiguous()
    y = y.contiguous()
    if x.size(0) == 1:
        x = x.expand(y.size(0), -1, -1)
    elif y.size(0) == 1:
        y = y.expand(x.size(0), -1, -1)

    # Decide backend automatically when requested
    backend = _select_cdist_backend(x, y, backend)

    B, M, D = x.shape
    _, N, _ = y.shape
    if not x.is_cuda or backend == "torch":
        c = torch.empty((B, M, N), device=x.device, dtype=torch.float32)
        chunk_size_n = (
            1024 if ((M >= 1024) and (D >= 1024)) else N
        )  # To avoid OOM errors
        chunk_size_m = (
            1024 if ((N >= 1024) and (D >= 1024)) else M
        )  # To avoid OOM errors
        c = (
            chunked_cdist(
                x, y, p=2, chunk_size_n=chunk_size_n, chunk_size_m=chunk_size_m
            )
            ** 2
        )
        return c

    elif backend == "triton":
        c = torch.empty((B, M, N), device=x.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 32
        BLOCK_D = 32

        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        _batched_cdist_kernel[grid](
            x,
            y,
            c,
            B,
            M,
            N,
            D,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            y.stride(0),
            y.stride(1),
            y.stride(2),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return c.to(x.dtype)


from typing import Optional


def chunked_cdist(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 2.0,
    chunk_size_n: Optional[int] = None,
    chunk_size_m: Optional[int] = None,
) -> torch.Tensor:
    """
    Chunked version of torch.cdist to prevent out-of-memory errors.

    Args:
        x: Tensor of shape (B, N, D)
        y: Tensor of shape (B, M, D)
        p: p-norm for distance computation (default: 2.0 for Euclidean)
        chunk_size_n: Chunk size for N dimension (default: auto)
        chunk_size_m: Chunk size for M dimension (default: auto)
        chunk_size_d: Chunk size for D dimension (default: auto)
        compute_mode: 'auto', 'use_mm_for_euclid_dist_if_necessary', or 'donot_use_mm_for_euclid_dist'

    Returns:
        Distance matrix of shape (B, N, M)
    """
    B, N, D = x.shape
    B2, M, D2 = y.shape

    assert B == B2 and D == D2, f"Incompatible shapes: x{x.shape}, y{y.shape}"

    # Auto-determine chunk sizes if not provided
    if chunk_size_n is None:
        chunk_size_n = min(N, 1024)  # Reasonable default
    if chunk_size_m is None:
        chunk_size_m = min(M, 1024)

    # Initialize output tensor
    output = torch.empty(B, N, M, dtype=x.dtype, device=x.device)

    # Chunk along N dimension
    for n_start in range(0, N, chunk_size_n):
        n_end = min(n_start + chunk_size_n, N)
        x_chunk = x[:, n_start:n_end]  # Shape: (B, chunk_n, D)

        # Chunk along M dimension
        for m_start in range(0, M, chunk_size_m):
            m_end = min(m_start + chunk_size_m, M)
            y_chunk = y[:, m_start:m_end]  # Shape: (B, chunk_m, D)
            dist_chunk = torch.cdist(x_chunk, y_chunk, p=p)
            output[:, n_start:n_end, m_start:m_end] = dist_chunk

    return output


#################################################################################################
#  UTILITY FUNCTION TO COMPUTE DISTANCE TO DATASET
#################################################################################################
def dist_to_dataset(
    samples,
    data_loader,
    operator: callable = None,
    metric="mse",
    reduction=None,
    verbose=True,
):
    r"""
    Compute the distance from samples to a dataset.

    Args:
        samples (torch.Tensor): Tensor of shape (B, *) representing the samples.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        metric (str): Metric to use for distance computation ('mse' or 'psnr').
        reduction (str): Reduction method ('mean' or 'min').

    Returns:
        dist torch.Tensor: Distance values computed for each sample against the dataset.
                of shape (B, ) if reduction is 'mean' or 'min' or 'max', or (B, dataset_size) if reduction is None.
    """
    B = samples.size(0)
    samples = samples.view(B, -1)  # Flatten the samples
    min_val = torch.amin(samples, dim=1, keepdim=True)
    max_val = torch.amax(samples, dim=1, keepdim=True)
    diff = (max_val - min_val) ** 2 + 1e-8

    device, dtype = samples.device, samples.dtype
    distances = []
    data_loader = tqdm(data_loader, disable=not verbose)
    for batch in data_loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device=device, dtype=dtype)
        if operator is not None:
            batch = operator(batch)
        batch = batch.view(batch.size(0), -1)  # Flatten the batch
        mse = torch.sum(
            (samples.unsqueeze(1) - batch.unsqueeze(0)) ** 2, dim=-1
        )  # (B, b)
        # mse = torch.linalg.vector_norm(samples.unsqueeze(1) - batch.unsqueeze(0), dim = -1).pow(2)
        if metric.lower() == "mse":
            distances.append(mse)
        elif metric.lower() == "psnr":
            mse = torch.clamp(mse, min=1e-8)  # Avoid log(0)
            psnr = -10 * torch.log10(mse / diff)
            distances.append(psnr)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'mse' or 'psnr'.")

    distances = torch.cat(distances, dim=1)  # (B, dataset_size)

    if reduction is None:
        return distances
    elif reduction == "mean":
        return torch.mean(distances, dim=1)  # (B, )
    elif reduction == "min":
        return torch.amin(distances, dim=1)  # (B, )
    elif reduction == "max":
        return torch.amax(distances, dim=1)  # (B, )
    else:
        raise ValueError(
            f"Unsupported reduction: {reduction}. Use 'mean', 'min', 'max', or None."
        )


def topk_nearest_neighbors(
    samples, data_loader, operator=None, k=5, metric="mse", return_distances=False
):
    r"""
    Compute the top-k nearest neighbors for each sample in a dataset.

    Args:
        samples (torch.Tensor): Tensor of shape (B, *) representing the samples.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        k (int): Number of nearest neighbors to return.
        metric (str): Metric to use for distance computation ('mse' or 'psnr').
        return_distances (bool): If True, also return distances to the nearest neighbors.

    Returns:
        torch.Tensor: Indices of the top-k nearest neighbors for each sample.
    """
    B = samples.size(0)
    shape = samples.shape
    samples = samples.view(B, -1)  # Flatten the samples
    D = samples.size(1)
    min_val = torch.amin(samples, dim=1, keepdim=True)
    max_val = torch.amax(samples, dim=1, keepdim=True)
    diff = (max_val - min_val) ** 2 + 1e-8

    device, dtype = samples.device, samples.dtype
    distances = torch.ones(B, k, device=device, dtype=dtype)
    if metric.lower() == "mse":
        distances = distances * torch.inf
    elif metric.lower() == "psnr":
        distances = distances * -torch.inf
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'mse' or 'psnr'.")

    nearest_samples = torch.zeros(B, k, D, device=device, dtype=dtype)

    for batch in tqdm(data_loader):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        batch = batch.to(device=device, dtype=dtype)
        if operator is not None:
            batch = operator(batch)
        batch = batch.view(batch.size(0), -1)  # Flatten the batch  (b, D)
        mse = torch.mean(
            (samples.unsqueeze(1) - batch.unsqueeze(0)) ** 2, dim=-1
        )  # (B, b)
        # mse = torch.cdist(samples, batch, p=2.0) ** 2  # (B, b)

        nearest_samples = torch.cat(
            (nearest_samples, batch.unsqueeze(0).expand(B, -1, -1)), dim=1
        )  # (B, b + k, D)

        if metric.lower() == "mse":
            distances = torch.cat((distances, mse), dim=1)  # (B, b + k)
            distances, indices = torch.topk(distances, k=k, dim=1, largest=False)
            nearest_samples = nearest_samples.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, k, D)

        elif metric.lower() == "psnr":
            mse = torch.clamp(mse, min=1e-8)  # Avoid log(0)
            psnr = -10 * torch.log10(mse / diff)
            distances = torch.cat((distances, psnr), dim=1)  # (B, b + k)
            distances, indices = torch.topk(distances, k=k, dim=1, largest=True)
            nearest_samples = nearest_samples.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, D)
            )  # (B, k, D)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'mse' or 'psnr'.")

    nearest_samples = nearest_samples.view(B, k, *shape[1:])
    if return_distances:
        return nearest_samples, distances
    else:
        return nearest_samples


#################################################################################################
def dist_to_set(
    samples, loader, distance_type="mean", metric_type="mse", max_pixel=1.0, min_pixel=0
):
    assert samples.dim() == 4
    factory_kwargs = dict(device=samples.device, dtype=samples.dtype)
    distance_list = []
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0].to(**factory_kwargs)
        else:
            batch = batch.to(**factory_kwargs)
        distance_to_imgs = (
            (samples.unsqueeze(0) - batch.unsqueeze(1))
            .pow(2)
            .mean(dim=(-3, -2, -1), keepdim=False)
        )
        if metric_type == "psnr":
            distance_to_imgs = -10.0 * torch.log10(
                distance_to_imgs / (max_pixel - min_pixel) ** 2 + 1e-8
            )
        # distance_list = torch.cat([distance_list,distance_to_imgs], dim=0)
        distance_list.append(distance_to_imgs)
    distance_list = torch.cat(distance_list, dim=0)
    if distance_type == "min":
        if metric_type == "psnr":
            dist = torch.max(distance_list, dim=0)[0].detach().cpu().numpy()
        else:
            dist = torch.min(distance_list, dim=0)[0].detach().cpu().numpy()
    elif distance_type == "mean":
        dist = torch.mean(distance_list, dim=0).detach().cpu().numpy()
    return dist


def closeness_to_set(
    operator, y, loader, sigma, compose_pinv=True, max_pixel=1.0, min_pixel=0.0
):
    assert y.dim() == 4
    distance_list = []
    _operator = operator.to(device=operator.device, dtype=torch.float64)
    _operator.dtype = torch.float64
    datasize = 0
    size = np.prod(operator.input_size)
    log_prefactor = size * (np.log(sigma) + np.log(2 * np.pi) / 2)
    print(log_prefactor)
    if compose_pinv:
        v = _operator.pinv(y)
    v = y
    distance_shift = None
    sum_exp = 0.0
    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch = batch[0].to(**_operator.factory_kwargs)
        else:
            batch = batch.to(**_operator.factory_kwargs)
        datasize += batch.size(0)
        batch_mesure = _operator(batch)
        distance = -(v.unsqueeze(0) - batch_mesure.unsqueeze(1)).pow(2).sum(
            dim=(-1, -2, -3)
        ) / (2 * sigma**2)
        distance = distance - log_prefactor
        if distance_shift is None:
            distance_shift = torch.amax(distance, dim=0, keepdim=False)  # (K, )
        else:
            new_distance_shift = torch.amax(distance, dim=0, keepdim=False)  # (K)
            delta_distance_shift = torch.where(
                new_distance_shift < distance_shift, distance_shift, new_distance_shift
            )  # (K, )
            diff = delta_distance_shift - distance_shift  # (K, )
            sum_exp /= torch.exp(diff)  # (K, )
            distance_shift = delta_distance_shift  # (K, )

        exp_distance = torch.exp(distance - distance_shift.unsqueeze(0))  # (B, K)
        sum_exp += torch.sum(exp_distance, dim=0, keepdim=False)  # (K, )

    log_sum_exp = sum_exp.log() + distance_shift
    return log_sum_exp, distance_shift


def build_depthwise_conv2d_matrix_circular(shared_kernel, input_shape):
    """
    Build a convolution matrix for depthwise 2D convolution with circular padding.

    Args:
        shared_kernel (Tensor): shape (kh, kw), shared across all channels
        input_shape (tuple): (in_channels, in_height, in_width)

    Returns:
        conv_matrix (Tensor): shape (in_channels * out_h * out_w, in_channels * in_h * in_w)
    """
    kh, kw = shared_kernel.shape
    _, in_h, in_w = input_shape

    pad_h = kh // 2
    pad_w = kw // 2

    out_h = in_h
    out_w = in_w

    conv_matrix = torch.zeros(
        (out_h * out_w, in_h * in_w),
        dtype=shared_kernel.dtype,
        device=shared_kernel.device,
    )

    for i in range(out_h):
        for j in range(out_w):
            out_row = i * out_w + j
            for ki in range(kh):
                for kj in range(kw):
                    # Circular padding index computation
                    in_i = (i + ki - pad_h) % in_h
                    in_j = (j + kj - pad_w) % in_w
                    in_col = in_i * in_w + in_j
                    weight = shared_kernel[ki, kj]
                    conv_matrix[out_row, in_col] = weight
    return conv_matrix

def build_circular_conv_matrix_fft(
    h: torch.Tensor, input_size: tuple, device="cpu", dtype=torch.float32
):
    """
    Build dense depthwise circular convolution matrix using vectorized FFT (faster than column loop)
    - h: 2D tensor (kh, kw)œ
    - H, W: image size
    - channel_count: number of channels (depthwise)
    """
    kh, kw = h.shape
    _, H, W = input_size

    # pad h to image size and center
    h_pad = torch.zeros((H, W), dtype=dtype, device=device)
    h_pad[:kh, :kw] = h.to(dtype=dtype, device=device)
    h_pad = torch.roll(h_pad, (-(kh // 2), -(kw // 2)), (-2, -1))

    # FFT of kernel
    Hf = torch.fft.fft2(h_pad)
    Npix = H * W

    # Create all basis vectors as an identity image (Npix x Npix) in H x W flattened form
    # Each column is a delta at position j
    E = torch.eye(Npix, dtype=dtype, device=device).reshape(Npix, H, W)  # (Npix, H, W)

    # FFT of all basis images at once
    Ef = torch.fft.fft2(E)  # (Npix, H, W), complex

    # multiply by kernel in frequency domain
    Yf = Ef * Hf  # broadcasting over first dim

    # inverse FFT to get circular convolution results
    Y = torch.fft.ifft2(Yf).real  # (Npix, H, W)

    # flatten each HxW result to column
    A_single = Y.reshape(Npix, Npix)  # (H*W, H*W)

    return A_single


def circular_patch(x: torch.Tensor, patch_index: tuple, patch_size: int):
    """
    Extract a square patch of odd size from a 2D tensor X with circular wrap-around.

    Args:
        x (torch.Tensor):  tensor (B, C, H, W).
        patch_index (tuple): tuple of index of the center pixel of the wanted patch
        patch_size (int): Must be odd (e.g. 3, 5, 7, ...).

    Returns:
        torch.Tensor: patch_size x patch_size tensor.
    """
    assert patch_size % 2 == 1, "patch_size must be odd"

    H, W = x.shape[-2:]
    radius = patch_size // 2

    row_idx = torch.arange(patch_index[0] - radius, patch_index[0] + radius + 1) % H
    col_idx = torch.arange(patch_index[1] - radius, patch_index[1] + radius + 1) % W

    return x[:, :, row_idx][:, :, :, col_idx]
