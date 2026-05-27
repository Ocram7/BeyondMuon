"""SGD with decoupled weight decay.

Adapted from PyTorch Image Models (timm):
https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/sgdw.py

Original implementation by Ross Wightman.
Copyright 2019 Ross Wightman.
Licensed under the Apache License, Version 2.0.
Modified for this repository in 2026.
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

try:
    from torch.optim.optimizer import _default_to_fused_or_foreach

    has_recent_pt = True
except ImportError:
    has_recent_pt = False

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import torch.optim

try:
    from torch.optim.optimizer import ParamsT
except (ImportError, TypeError):
    ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]

__all__ = ["SGDW", "sgdw"]


class SGDW(Optimizer):
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
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("caution", False)
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

    def _init_group(self, group, params_with_grad, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        return has_sparse_grad

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(
                group, params_with_grad, grads, momentum_buffer_list
            )

            sgdw(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                caution=group["caution"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
            )

            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state["momentum_buffer"] = momentum_buffer

        return loss


def sgdw(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # Keep these as kwargs for torch.distributed optimizer compatibility.
    has_sparse_grad: bool = None,
    foreach: Optional[bool] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    caution: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    if has_recent_pt and hasattr(Optimizer, "_group_tensors_by_device_and_dtype"):
        if foreach is None:
            if not torch.jit.is_scripting():
                _, foreach = _default_to_fused_or_foreach(
                    params, differentiable=False, use_fused=False
                )
            else:
                foreach = False

        if foreach and torch.jit.is_scripting():
            raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    else:
        foreach = False

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgdw
    else:
        func = _single_tensor_sgdw

    func(
        params,
        grads,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        caution=caution,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
    )


def _single_tensor_sgdw(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    caution: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        param.mul_(1.0 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if caution:
                if nesterov:
                    buf = grad.add(buf, alpha=momentum)
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                mask = (buf * grad > 0).to(grad.dtype)
                mask.div_(mask.mean().clamp_(min=1e-3))
                grad = buf * mask
            else:
                if nesterov:
                    grad = grad.add(buf, alpha=momentum)
                else:
                    grad = buf

        param.add_(grad, alpha=-lr)


def _multi_tensor_sgdw(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    caution: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    if len(params) == 0:
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True
    )
    for (
        device_params,
        device_grads,
        device_momentum_buffer_list,
    ), indices in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)

        torch._foreach_mul_(params, 1.0 - lr * weight_decay)

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[
                            indices[i]
                        ] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if caution:
                if nesterov:
                    # Can't do nesterov in-place if we want to compare against orig grad for caution
                    bufs = torch._foreach_add(device_grads, bufs, alpha=momentum)
                # Apply caution as per 'Cautious Optimizers' - https://arxiv.org/abs/2411.16085
                masks = torch._foreach_mul(bufs, device_grads)
                masks = [(m > 0).to(g.dtype) for m, g in zip(masks, device_grads)]
                mask_scale = [m.mean() for m in masks]
                torch._foreach_maximum_(mask_scale, 1e-3)
                torch._foreach_div_(masks, mask_scale)
                device_grads = torch._foreach_mul(bufs, masks)
            else:
                if nesterov:
                    torch._foreach_add_(device_grads, bufs, alpha=momentum)
                else:
                    device_grads = bufs

        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            # foreach APIs don't support sparse
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)
