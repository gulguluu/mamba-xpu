# Copyright (c) 2024, Mamba contributors.
# Device compatibility layer for CUDA and Intel XPU support.

import torch
from contextlib import contextmanager


def is_xpu_available():
    """Check if Intel XPU support is available."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_cuda_available():
    """Check if CUDA support is available."""
    return torch.cuda.is_available()


def get_accelerator_type():
    """Return the accelerator type string: 'cuda', 'xpu', or 'cpu'."""
    if is_cuda_available():
        return "cuda"
    elif is_xpu_available():
        return "xpu"
    return "cpu"


def is_accelerator_tensor(t):
    """Check if a tensor is on a supported accelerator (CUDA or XPU)."""
    return t.is_cuda or (hasattr(t, "is_xpu") and t.is_xpu)


@contextmanager
def device_context(device):
    """Context manager that works for both CUDA and XPU devices.

    Replaces `torch.cuda.device(idx)` with a device-agnostic version.
    For XPU devices, uses `torch.xpu.device(idx)`.
    For CUDA devices, uses `torch.cuda.device(idx)`.
    For CPU or when device index is None, yields without setting device.
    """
    if device is None:
        yield
        return

    if isinstance(device, torch.device):
        device_type = device.type
        device_index = device.index
    elif isinstance(device, int):
        # Default to the current accelerator type
        device_type = get_accelerator_type()
        device_index = device
    elif isinstance(device, str):
        parsed = torch.device(device)
        device_type = parsed.type
        device_index = parsed.index
    else:
        yield
        return

    if device_type == "xpu" and is_xpu_available():
        with torch.xpu.device(device_index):
            yield
    elif device_type == "cuda" and is_cuda_available():
        with torch.cuda.device(device_index):
            yield
    else:
        yield


def get_sm_count(device):
    """Get the number of streaming multiprocessors (or equivalent) for a device.

    For CUDA: returns multi_processor_count from device properties.
    For XPU: returns the number of execution units (sub-slices).
    Falls back to a reasonable default if unavailable.
    """
    if isinstance(device, int):
        device = torch.device(get_accelerator_type(), device)
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    elif device.type == "xpu" and is_xpu_available():
        props = torch.xpu.get_device_properties(device)
        # Intel XPU reports gpu_eu_count or gpu_subslice_count
        if hasattr(props, "gpu_subslice_count"):
            return props.gpu_subslice_count
        elif hasattr(props, "gpu_eu_count"):
            # EUs are finer-grained than SMs; approximate SM-equivalence
            return props.gpu_eu_count // 8
        else:
            # Fallback: use a conservative default for Intel Data Center GPU Max
            return 64
    else:
        return 1
