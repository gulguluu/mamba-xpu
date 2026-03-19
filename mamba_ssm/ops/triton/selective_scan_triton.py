# Copyright (c) 2024, Mamba contributors.
# Triton-based selective scan for non-CUDA devices (Intel XPU, etc.)
# This provides a significantly faster alternative to the pure Python reference
# implementation by using a Triton kernel for the sequential scan.

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat
from mamba_ssm.utils.device import device_context


@triton.jit
def _selective_scan_fwd_kernel(
    # Pointers
    u_ptr, delta_ptr, A_ptr, B_ptr, C_ptr, D_ptr, z_ptr, out_ptr, x_ptr,
    # Strides for u: (B, D, L)
    stride_u_b, stride_u_d, stride_u_l,
    # Strides for delta: (B, D, L)
    stride_delta_b, stride_delta_d, stride_delta_l,
    # Strides for A: (D, N)
    stride_A_d, stride_A_n,
    # Strides for B: (B, D, N, L) or broadcast
    stride_B_b, stride_B_d, stride_B_n, stride_B_l,
    # Strides for C: (B, D, N, L) or broadcast
    stride_C_b, stride_C_d, stride_C_n, stride_C_l,
    # Strides for out: (B, D, L)
    stride_out_b, stride_out_d, stride_out_l,
    # Strides for x (last state): (B, D, N)
    stride_x_b, stride_x_d, stride_x_n,
    # Dimensions
    seqlen, dstate,
    # Flags
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Selective scan forward: processes one (batch, dim) pair per program.
    The scan is sequential along the sequence dimension but parallelized
    across batch and hidden dimensions via the grid.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Load A for this dimension: (N,)
    a_ptrs = A_ptr + pid_d * stride_A_d + tl.arange(0, BLOCK_N) * stride_A_n
    a_mask = tl.arange(0, BLOCK_N) < dstate
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (N,)

    # Initialize state x: (N,)
    x = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Base pointers for this (batch, dim)
    u_base = u_ptr + pid_b * stride_u_b + pid_d * stride_u_d
    delta_base = delta_ptr + pid_b * stride_delta_b + pid_d * stride_delta_d
    B_base = B_ptr + pid_b * stride_B_b + pid_d * stride_B_d
    C_base = C_ptr + pid_b * stride_C_b + pid_d * stride_C_d
    out_base = out_ptr + pid_b * stride_out_b + pid_d * stride_out_d

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < dstate

    for l in range(seqlen):
        # Load u, delta scalars
        u_val = tl.load(u_base + l * stride_u_l).to(tl.float32)
        delta_val = tl.load(delta_base + l * stride_delta_l).to(tl.float32)

        # deltaA = exp(delta * A)
        deltaA = tl.exp(delta_val * a)  # (N,)

        # Load B: (N,)
        b = tl.load(B_base + n_offsets * stride_B_n + l * stride_B_l, mask=n_mask, other=0.0).to(tl.float32)

        # deltaB_u = delta * B * u
        deltaB_u = delta_val * b * u_val  # (N,)

        # x = deltaA * x + deltaB_u
        x = deltaA * x + deltaB_u

        # Load C: (N,)
        c = tl.load(C_base + n_offsets * stride_C_n + l * stride_C_l, mask=n_mask, other=0.0).to(tl.float32)

        # y = sum(x * C)
        y = tl.sum(x * c, axis=0)

        # Add D * u
        if HAS_D:
            d_val = tl.load(D_ptr + pid_d)
            y += d_val * u_val

        # Apply z gating: out = y * silu(z)
        if HAS_Z:
            z_val = tl.load(z_ptr + pid_b * stride_u_b + pid_d * stride_u_d + l * stride_u_l).to(tl.float32)
            y = y * z_val * tl.sigmoid(z_val)

        tl.store(out_base + l * stride_out_l, y)

    # Store last state
    x_base = x_ptr + pid_b * stride_x_b + pid_d * stride_x_d
    tl.store(x_base + n_offsets * stride_x_n, x, mask=n_mask)


def selective_scan_triton(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                          delta_softplus=False, return_last_state=False):
    """Triton-based selective scan implementation.

    Args:
        u: (B, D, L) input
        delta: (B, D, L) time step
        A: (D, N) state matrix
        B: (B, N, L) or (B, G, N, L) input-dependent B
        C: (B, N, L) or (B, G, N, L) input-dependent C
        D: (D,) skip connection (optional)
        z: (B, D, L) gating (optional)
        delta_bias: (D,) bias for delta
        delta_softplus: whether to apply softplus to delta
        return_last_state: whether to return the last state
    """
    dtype_in = u.dtype
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]

    u = u.float().contiguous()
    delta = delta.float().contiguous()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    A = A.float().contiguous()

    # Handle variable B shapes: normalize to (B, D, N, L)
    is_variable_B = B.dim() >= 3
    B = B.float()
    if not is_variable_B:
        # B: (D, N) -> broadcast to (1, D, N, 1) during kernel
        B_expanded = repeat(B, "d n -> 1 d n 1")
        B_stride = (0, B_expanded.stride(1), B_expanded.stride(2), 0)
    elif B.dim() == 3:
        # B: (B, N, L) -> (B, 1, N, L) broadcast D
        B = B.contiguous()
        B_stride = (B.stride(0), 0, B.stride(1), B.stride(2))
    else:
        # B: (B, G, N, L) -> expand groups to match D
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1]).contiguous()
        B_stride = (B.stride(0), B.stride(1), B.stride(2), B.stride(3))

    is_variable_C = C.dim() >= 3
    C = C.float()
    if not is_variable_C:
        C_expanded = repeat(C, "d n -> 1 d n 1")
        C_stride = (0, C_expanded.stride(1), C_expanded.stride(2), 0)
    elif C.dim() == 3:
        C = C.contiguous()
        C_stride = (C.stride(0), 0, C.stride(1), C.stride(2))
    else:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1]).contiguous()
        C_stride = (C.stride(0), C.stride(1), C.stride(2), C.stride(3))

    if D is not None:
        D = D.float().contiguous()
    if z is not None:
        z = z.float().contiguous()

    out = torch.empty_like(u)
    last_state = torch.empty((batch, dim, dstate), dtype=torch.float32, device=u.device)

    BLOCK_N = triton.next_power_of_2(dstate)

    grid = (batch, dim)
    with device_context(u.device):
        _selective_scan_fwd_kernel[grid](
            u, delta, A,
            B if not is_variable_B and B.dim() < 3 else B,
            C if not is_variable_C and C.dim() < 3 else C,
            D, z, out, last_state,
            u.stride(0), u.stride(1), u.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B_stride[0], B_stride[1], B_stride[2], B_stride[3],
            C_stride[0], C_stride[1], C_stride[2], C_stride[3],
            out.stride(0), out.stride(1), out.stride(2),
            last_state.stride(0), last_state.stride(1), last_state.stride(2),
            seqlen, dstate,
            HAS_D=D is not None,
            HAS_Z=z is not None,
            BLOCK_N=BLOCK_N,
        )

    out = out.to(dtype=dtype_in)
    if not return_last_state:
        return out
    return out, last_state
