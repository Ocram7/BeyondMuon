"""SGDW optimizer variants with spectral matrix updates in the style of timm."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .spectral_ns import apply_spectral_transform

try:
    from torch.optim.optimizer import ParamsT
except (ImportError, TypeError):
    ParamsT = Iterable

__all__ = ["SGDW_NS"]


class SGDW_NS(Optimizer):
    """Decoupled-weight-decay momentum SGD with Psi_p(M_t) on matrix params."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        *,
        caution: bool = False,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        ns_iters: int = 15,
        split_qkv_updates: bool = False,
        spectral_exponent: float = 0.5,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum and zero dampening")

        self.spectral_exponent = spectral_exponent
        self.ns_iters = ns_iters
        self.split_qkv_updates = split_qkv_updates

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            caution=caution,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = -param.grad if group["maximize"] else param.grad
                if grad.is_sparse:
                    raise RuntimeError("SGDW_NS does not support sparse gradients")

                param.mul_(1.0 - group["lr"] * group["weight_decay"])
                momentum_update = self._apply_momentum(param, grad, group)

                if param.ndim > 1:
                    update = self._apply_spectral_transform(momentum_update)
                else:
                    update = momentum_update

                param.add_(update, alpha=-group["lr"])

        return loss

    def _apply_momentum(self, param: Tensor, grad: Tensor, group):
        momentum = group["momentum"]
        if momentum == 0:
            return grad

        state = self.state[param]
        momentum_buffer = state.get("momentum_buffer")
        if momentum_buffer is None:
            momentum_buffer = torch.clone(grad).detach()
            state["momentum_buffer"] = momentum_buffer
        else:
            momentum_buffer.mul_(momentum).add_(grad, alpha=1.0 - group["dampening"])

        if group["nesterov"]:
            momentum_update = grad.add(momentum_buffer, alpha=momentum)
        else:
            momentum_update = momentum_buffer

        if group["caution"]:
            mask = (momentum_update * grad > 0).to(grad.dtype)
            mask.div_(mask.mean().clamp_(min=1e-3))
            momentum_update = momentum_update * mask

        return momentum_update

    def _apply_spectral_transform(self, momentum_update):
        return apply_spectral_transform(
            momentum_update,
            self.spectral_exponent,
            ns_iters=self.ns_iters,
            split_qkv_updates=self.split_qkv_updates,
        )
