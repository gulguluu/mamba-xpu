# Copyright (c) 2024, Mamba contributors.
# Pure PyTorch causal_conv1d implementations for non-CUDA devices (Intel XPU, etc.)
# These provide drop-in replacements for the causal_conv1d CUDA package.

import torch
import torch.nn.functional as F


def causal_conv1d_fn_pt(x, weight, bias=None, activation=None, seq_idx=None):
    """Pure PyTorch causal conv1d forward.

    Args:
        x: (B, D, L) input
        weight: (D, W) conv weight
        bias: (D,) optional bias
        activation: "silu" or "swish" or None
        seq_idx: not supported in PyTorch fallback
    Returns:
        y: (B, D, L) output
    """
    assert seq_idx is None, "seq_idx not supported in PyTorch causal_conv1d fallback"
    d, w = weight.shape
    # Causal padding: pad left by (w-1)
    x_padded = F.pad(x, (w - 1, 0))
    # Use grouped conv1d: each channel has its own filter
    weight_conv = weight.unsqueeze(1)  # (D, 1, W)
    y = F.conv1d(x_padded, weight_conv, bias=bias, groups=d)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y


def causal_conv1d_update_pt(x, conv_state, weight, bias=None, activation=None):
    """Pure PyTorch causal conv1d update (single step for inference).

    Args:
        x: (B, D) new input token
        conv_state: (B, D, W) rolling convolution state, updated in-place
        weight: (D, W) conv weight
        bias: (D,) optional bias
        activation: "silu" or "swish" or None
    Returns:
        y: (B, D) output
    """
    # Shift state left and insert new input
    conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
    conv_state[:, :, -1] = x
    # Compute convolution output
    y = torch.sum(conv_state * weight, dim=-1)  # (B, D)
    if bias is not None:
        y = y + bias
    if activation in ("silu", "swish"):
        y = F.silu(y)
    return y
