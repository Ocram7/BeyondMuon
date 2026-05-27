"""AdamW optimizer variants with spectral matrix updates."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import torch
from torch.optim.optimizer import Optimizer

from .spectral_ns import apply_spectral_transform

__all__ = ["AdamW_NS"]


class AdamW_NS(Optimizer):
    """AdamW plus Adam/AdamS/AdamQ/AdamZ on matrix-shaped parameters."""

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        ns_iters: int = 15,
        split_qkv_updates: bool = False,
        spectral_exponent: float = 0.5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.spectral_exponent = spectral_exponent
        self.ns_iters = ns_iters
        self.split_qkv_updates = split_qkv_updates

        defaults: Dict[str, Any] = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            maximize = group["maximize"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = -param.grad if maximize else param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW_NS does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["second_moment"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if amsgrad:
                        state["max_second_moment"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                else:
                    if "first_moment" not in state and "exp_avg" in state:
                        state["first_moment"] = state.pop("exp_avg")
                    if "second_moment" not in state and "exp_avg_sq" in state:
                        state["second_moment"] = state.pop("exp_avg_sq")
                    if amsgrad and "max_second_moment" not in state:
                        state["max_second_moment"] = state.pop(
                            "max_exp_avg_sq",
                            torch.zeros_like(
                                param, memory_format=torch.preserve_format
                            ),
                        )

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0.0:
                    param.add_(param, alpha=-lr * weight_decay)

                first_moment.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                corrected_first_moment = first_moment / bias_correction1

                if amsgrad:
                    max_second_moment = state["max_second_moment"]
                    torch.maximum(
                        max_second_moment, second_moment, out=max_second_moment
                    )
                    corrected_second_moment = max_second_moment / bias_correction2
                else:
                    corrected_second_moment = second_moment / bias_correction2

                rms_denominator = corrected_second_moment.sqrt().add_(eps)
                rms_normalized_update = corrected_first_moment / rms_denominator

                if param.ndim > 1:
                    update = self._apply_spectral_transform(rms_normalized_update)
                else:
                    update = rms_normalized_update

                param.add_(update, alpha=-lr)

        return loss

    def _apply_spectral_transform(self, rms_normalized_update):
        return apply_spectral_transform(
            rms_normalized_update,
            self.spectral_exponent,
            ns_iters=self.ns_iters,
            split_qkv_updates=self.split_qkv_updates,
        )
