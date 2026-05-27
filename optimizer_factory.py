"""Optimizer construction."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer

from optimizers.adamw_ns import AdamW_NS
from optimizers.sgdw import SGDW
from optimizers.sgdw_ns import SGDW_NS


@dataclass(frozen=True)
class OptimizerSpec:
    name: str
    family: str
    spectral_exponent: float


OPTIMIZER_SPECS = {
    "adam": OptimizerSpec("Adam", "adam", 1.0),
    "adams": OptimizerSpec("AdamS", "adam", 0.5),
    "adamq": OptimizerSpec("AdamQ", "adam", 0.25),
    "adamz": OptimizerSpec("AdamZ", "adam", 0.0),
    "msgd": OptimizerSpec("mSGD", "msgd", 1.0),
    "msgds": OptimizerSpec("mSGDS", "msgd", 0.5),
    "msgdq": OptimizerSpec("mSGDQ", "msgd", 0.25),
    "msgdz": OptimizerSpec("mSGDZ", "msgd", 0.0),
}


class OptimizerBundle:
    """Thin wrapper over one or more optimizers."""

    def __init__(self, optimizers):
        self.optimizers = optimizers if isinstance(optimizers, list) else [optimizers]

    def __iter__(self):
        return iter(self.optimizers)

    def set_lrs(self, matrix_lr, vector_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_kind = param_group.get("param_kind")
                if param_kind not in {"matrix", "vector"}:
                    raise ValueError(
                        f"Optimizer param group missing param_kind: {param_kind!r}"
                    )
                param_group["lr"] = vector_lr if param_kind == "vector" else matrix_lr

    def consolidate_state_dict(self, to=0):
        for opt in self.optimizers:
            if hasattr(opt, "consolidate_state_dict"):
                opt.consolidate_state_dict(to=to)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dict):
        if not isinstance(state_dict, list):
            if len(self.optimizers) != 1:
                raise ValueError(
                    "Checkpoint has one optimizer state for multiple optimizers"
                )
            self.optimizers[0].load_state_dict(state_dict)
            return

        if len(state_dict) != len(self.optimizers):
            raise ValueError("Checkpoint optimizer count does not match current setup")
        for opt, opt_state in zip(self.optimizers, state_dict):
            opt.load_state_dict(opt_state)

    def unscale_(self, scaler):
        for opt in self.optimizers:
            scaler.unscale_(opt)

    def step(self, scaler):
        for opt in self.optimizers:
            scaler.step(opt)
        scaler.update()

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)


def _normalize_name(name):
    return (name or "").replace("_", "").replace("-", "").replace(" ", "").lower()


def resolve_optimizer_spec(optimizer_variant):
    variant_key = _normalize_name(optimizer_variant)
    if variant_key not in OPTIMIZER_SPECS:
        choices = ", ".join(spec.name for spec in OPTIMIZER_SPECS.values())
        raise ValueError(
            f"Unknown optimizer_variant={optimizer_variant!r}; choose: {choices}"
        )
    return OPTIMIZER_SPECS[variant_key]


def build_parameter_groups(model, weight_decay, lr_matrix, lr_vector):
    param_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    matrix_params = [param for _, param in param_dict.items() if param.dim() >= 2]
    matrix_param_names = [
        name for name, param in param_dict.items() if param.dim() >= 2
    ]
    vector_params = [param for _, param in param_dict.items() if param.dim() < 2]
    vector_param_names = [name for name, param in param_dict.items() if param.dim() < 2]

    num_matrix_params = sum(param.numel() for param in matrix_params)
    num_vector_params = sum(param.numel() for param in vector_params)
    print(
        f"num matrix parameter tensors: {len(matrix_params)}, "
        f"with {num_matrix_params:,} parameters"
    )
    print(
        f"num vector parameter tensors: {len(vector_params)}, "
        f"with {num_vector_params:,} parameters"
    )

    return [
        {
            "params": matrix_params,
            "param_kind": "matrix",
            "weight_decay": weight_decay,
            "lr": lr_matrix,
            "param_names": matrix_param_names,
        },
        {
            "params": vector_params,
            "param_kind": "vector",
            "weight_decay": 0.0,
            "lr": lr_vector,
            "param_names": vector_param_names,
        },
    ]


def maybe_wrap_zero(param_groups, optimizer_cls, use_zero, **kwargs):
    if use_zero and dist.is_initialized():
        return ZeroRedundancyOptimizer(
            param_groups,
            optimizer_class=optimizer_cls,
            **kwargs,
        )
    return optimizer_cls(param_groups, **kwargs)


def build_adam_family_optimizer(
    param_groups,
    spec,
    betas,
    device_type,
    use_zero,
    ns_iters,
    split_qkv_updates,
):
    optimizer_kwargs = {
        "lr": param_groups[0]["lr"],  # fallback; group lrs override this
        "betas": betas,
    }

    if spec.name == "Adam":
        optimizer_cls = torch.optim.AdamW
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if use_fused:
            optimizer_kwargs["fused"] = True
        print(f"using fused AdamW: {use_fused}")
    elif spec.name in {"AdamS", "AdamQ", "AdamZ"}:
        optimizer_cls = AdamW_NS
        optimizer_kwargs.update(
            ns_iters=ns_iters,
            split_qkv_updates=split_qkv_updates,
            spectral_exponent=spec.spectral_exponent,
        )
    else:
        raise ValueError(f"Unsupported Adam-family optimizer: {spec.name}")

    optimizer = maybe_wrap_zero(
        param_groups,
        optimizer_cls,
        use_zero,
        **optimizer_kwargs,
    )

    print(
        f"optimizer: {spec.name}, p:{spec.spectral_exponent}, "
        f"ns_iters:{ns_iters}, split_qkv_updates:{split_qkv_updates}"
    )
    return optimizer


def build_msgd_family_optimizers(
    param_groups,
    spec,
    betas,
    use_zero,
    sgd_momentum,
    ns_iters,
    split_qkv_updates,
):
    vector_optimizer = maybe_wrap_zero(  # always use adamw to optimize vector params
        [param_groups[1]],
        torch.optim.AdamW,
        use_zero,
        lr=param_groups[1]["lr"],
        betas=betas,
    )

    matrix_optimizer_cls = None
    matrix_kwargs = dict(
        lr=param_groups[0]["lr"],
        momentum=float(sgd_momentum),
        nesterov=False,
        caution=False,
        dampening=0.0,
    )

    if spec.name == "mSGD":
        matrix_optimizer_cls = SGDW
    elif spec.name in {"mSGDS", "mSGDQ", "mSGDZ"}:
        matrix_optimizer_cls = SGDW_NS
        matrix_kwargs.update(
            ns_iters=ns_iters,
            split_qkv_updates=split_qkv_updates,
            spectral_exponent=spec.spectral_exponent,
        )
    else:
        raise ValueError(f"Unsupported mSGD-family optimizer: {spec.name}")

    matrix_optimizer = maybe_wrap_zero(
        [param_groups[0]],
        matrix_optimizer_cls,
        use_zero,
        **matrix_kwargs,
    )

    print(
        f"optimizer: {spec.name} for matrices, AdamW for vectors, "
        f"p:{spec.spectral_exponent}, ns_iters:{ns_iters}, "
        f"split_qkv_updates:{split_qkv_updates}"
    )
    return [vector_optimizer, matrix_optimizer]


def build_optimizers(
    model,
    weight_decay,
    lr_matrix,
    lr_vector,
    betas,
    device_type,
    optimizer_variant="Adam",
    use_zero=True,
    ns_iters=15,
    split_qkv_updates=False,
    sgd_momentum=None,
):
    spec = resolve_optimizer_spec(optimizer_variant)
    if lr_matrix is None or lr_vector is None:
        raise ValueError("build_optimizers requires lr_matrix and lr_vector")
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("build_optimizers requires betas=(beta1, beta2)")
    if spec.family == "msgd" and sgd_momentum is None:
        raise ValueError(f"optimizer_variant={spec.name} requires sgd_momentum")
    param_groups = build_parameter_groups(model, weight_decay, lr_matrix, lr_vector)

    if spec.family == "adam":
        optimizers = build_adam_family_optimizer(
            param_groups,
            spec,
            betas,
            device_type,
            use_zero,
            ns_iters,
            split_qkv_updates,
        )
    elif spec.family == "msgd":
        optimizers = build_msgd_family_optimizers(
            param_groups,
            spec,
            betas,
            use_zero,
            sgd_momentum,
            ns_iters,
            split_qkv_updates,
        )
    else:
        raise ValueError(f"Unsupported optimizer family: {spec.family}")

    return OptimizerBundle(optimizers)
