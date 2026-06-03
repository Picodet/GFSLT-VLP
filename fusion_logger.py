import math
import os

import torch
from loguru import logger

_LOGGER_INITIALIZED = False
_LOG_RANK0_ONLY = True


def is_rank0():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def setup_fusion_logger(log_dir="logs/fusion", filename="signgraph_fusion.jsonl", level="INFO", rank0_only=True):
    """Configure a JSONL Loguru sink for GFSLT/SignGraph tensor-protocol logs."""
    global _LOGGER_INITIALIZED, _LOG_RANK0_ONLY
    _LOG_RANK0_ONLY = rank0_only
    if _LOGGER_INITIALIZED:
        return logger
    os.makedirs(log_dir, exist_ok=True)
    logger.remove()
    logger.add(
        os.path.join(log_dir, filename),
        level=level,
        serialize=True,
        enqueue=True,
        rotation="100 MB",
        retention="14 days",
        compression="zip",
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=level,
        enqueue=True,
        colorize=False,
        backtrace=False,
        diagnose=False,
    )
    _LOGGER_INITIALIZED = True
    return logger


def _safe_float(x):
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


@torch.no_grad()
def tensor_summary(x, name=None, sample_values=False):
    """Summarize tensor metadata/statistics without logging full tensor payloads."""
    if not torch.is_tensor(x):
        return {"name": name, "type": type(x).__name__, "value": str(x)[:200]}
    item = {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "requires_grad": bool(x.requires_grad),
        "numel": int(x.numel()),
    }
    if x.numel() == 0:
        return item
    y = x.detach()
    if torch.is_floating_point(y):
        yf = y.float()
        item.update(
            {
                "mean": _safe_float(yf.mean().item()),
                "std": _safe_float(yf.std(unbiased=False).item()),
                "min": _safe_float(yf.min().item()),
                "max": _safe_float(yf.max().item()),
                "nan_count": int(torch.isnan(yf).sum().item()),
                "inf_count": int(torch.isinf(yf).sum().item()),
            }
        )
        if sample_values:
            item["sample"] = [_safe_float(v.item()) for v in yf.flatten()[:8]]
    return item


def summarize_obj(obj, name=None):
    if torch.is_tensor(obj):
        return tensor_summary(obj, name=name)
    if isinstance(obj, (list, tuple)):
        return [summarize_obj(v, name=f"{name}[{i}]" if name else f"[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: summarize_obj(v, name=k) for k, v in obj.items()}
    return {"name": name, "type": type(obj).__name__, "value": str(obj)[:200]}


def log_event(event, **payload):
    if _LOG_RANK0_ONLY and not is_rank0():
        return
    logger.bind(event=event).info(payload)


def make_forward_hook(module_name, log_every=50):
    counter = {"step": 0}

    def hook(module, inputs, output):
        counter["step"] += 1
        if counter["step"] % log_every != 0:
            return
        log_event(
            "module.forward",
            module=module_name,
            module_class=module.__class__.__name__,
            step=counter["step"],
            inputs=summarize_obj(inputs, "inputs"),
            output=summarize_obj(output, "output"),
        )

    return hook


def register_selected_forward_hooks(model, target_names, log_every=50):
    """Register forward hooks by exact module name and return removable handles."""
    handles = []
    name_to_module = dict(model.named_modules())
    for name in target_names:
        if name not in name_to_module:
            log_event("hook.missing_module", module=name, available_hint=list(name_to_module.keys())[:30])
            continue
        handles.append(name_to_module[name].register_forward_hook(make_forward_hook(name, log_every=log_every)))
        log_event("hook.registered", module=name)
    return handles
