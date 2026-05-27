"""Newton-Schulz helpers for spectral matrix updates."""

from __future__ import annotations

import torch


@torch.compile
def zero_power_via_quintic_ns(matrix, steps=10, eps=1e-7):
    """Approximate Psi_0(O) = U V^T with the Muon Newton-Schulz iteration."""

    if matrix.ndim != 2:
        raise ValueError("zero_power_via_quintic_ns expects a 2D tensor")

    a, b, c = (3.4445, -4.7750, 2.0315)
    x = matrix.bfloat16() / (matrix.norm() + eps)
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        x = x.T

    for _ in range(steps):
        gram = x @ x.T
        x = a * x + b * gram @ x + c * gram @ gram @ x

    if transposed:
        x = x.T
    return x.to(matrix.dtype)


def muon_update(matrix, ns_steps=5):
    """Apply the zero-power/polar transform to a matrix update."""

    update = zero_power_via_quintic_ns(matrix, steps=ns_steps)
    update *= max(1, matrix.size(-2) / matrix.size(-1)) ** 0.5
    return update


@torch.compile
@torch.no_grad()
def coupled_newton_schulz_sqrt_invsqrt(matrix, num_iters=15, eps=1e-6):
    """Compute X^(1/2) and X^(-1/2) with coupled Newton-Schulz iterations."""

    device, dtype = matrix.device, matrix.dtype
    size = matrix.shape[-1]
    eye = torch.eye(size, device=device, dtype=dtype)

    trace = torch.trace(matrix)
    regularized = matrix + (eps + 1e-4 * trace / size) * eye
    scale = torch.norm(regularized, p=2)
    y = regularized / scale
    z = eye.clone()

    sym_freq = max(1, min(num_iters // 3, 5))
    for idx in range(num_iters):
        transform = 0.5 * (3.0 * eye - z @ y)
        y = y @ transform
        z = transform @ z
        if idx > 0 and idx % sym_freq == 0:
            y = 0.5 * (y + y.mT)
            z = 0.5 * (z + z.mT)

    y = 0.5 * (y + y.mT)
    z = 0.5 * (z + z.mT)
    sqrt_scale = torch.sqrt(scale)
    return sqrt_scale * y, z / sqrt_scale


@torch.compile
def inverse_fourth_root_via_ns(matrix, ns_iters=15, eps=1e-6):
    """Compute X^(-1/4) by applying coupled NS twice."""

    _, inverse_square_root = coupled_newton_schulz_sqrt_invsqrt(
        matrix, num_iters=ns_iters, eps=eps
    )
    square_root_of_inverse_square_root, _ = coupled_newton_schulz_sqrt_invsqrt(
        inverse_square_root, num_iters=ns_iters, eps=eps
    )
    return square_root_of_inverse_square_root


@torch.compile
def spectral_half_power_via_ns(update, ns_iters=15, eps=1e-6, prefer="auto"):
    """Compute Psi_1/2(O) = U Sigma^(1/2) V^T without an explicit SVD."""

    if update.ndim != 2:
        raise ValueError("spectral_half_power_via_ns expects a 2D update matrix")

    rows, cols = update.shape
    use_right_gram = cols <= rows if prefer == "auto" else prefer == "right"

    if use_right_gram:
        gram_matrix = update.mT @ update
        inverse_fourth_root = inverse_fourth_root_via_ns(gram_matrix, ns_iters, eps)
        return update @ inverse_fourth_root, inverse_fourth_root, "right"

    gram_matrix = update @ update.mT
    inverse_fourth_root = inverse_fourth_root_via_ns(gram_matrix, ns_iters, eps)
    return inverse_fourth_root @ update, inverse_fourth_root, "left"


def apply_spectral_transform(
    update,
    spectral_exponent: float,
    *,
    ns_iters: int,
    split_qkv_updates: bool,
):
    """Apply Psi_p to a matrix for p in {1, 1/2, 1/4, 0}."""

    if update.ndim != 2:
        raise ValueError("apply_spectral_transform expects a 2D update matrix")

    is_packed_qkv = update.ndim == 2 and update.size(0) == 3 * update.size(1)
    if split_qkv_updates and is_packed_qkv:
        return torch.cat(
            [
                apply_spectral_transform(
                    part,
                    spectral_exponent,
                    ns_iters=ns_iters,
                    split_qkv_updates=False,
                )
                for part in update.split(update.size(1))
            ]
        )

    if spectral_exponent == 1.0:
        return update
    if spectral_exponent == 0.0:
        return muon_update(update, ns_steps=ns_iters)
    if spectral_exponent not in {0.5, 0.25}:
        raise ValueError(f"Unsupported spectral exponent p={spectral_exponent}")

    half_power_steps = 1 if spectral_exponent == 0.5 else 2
    for _ in range(half_power_steps):
        update, _, _ = spectral_half_power_via_ns(update, ns_iters)
    return update
