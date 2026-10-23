"""
WP3 fused sedenion GEMM (Algorithm 1 of the paper).

Y[b, m, k] = sum_n sum_j  eps(k^j, j) * W[m, n, k^j] * X[b, n, j]

Loop order is n (outer, streamed) -> j (inner, static) -> k (static), with all 16
accumulator tiles live in registers, so every X tile and every W tile is loaded
from HBM exactly once per (b-block, m-block) program. The 16x16 sign table is a
compile-time constant (256 bits), never a runtime load.

CPU fallback (`fused_sedenion_linear_reference`) is bit-identical in semantics and
is what the test-suite checks; the Triton path is only compiled when CUDA is present.
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    HAS_TRITON = False


def _sign_table_as_constexpr():
    from .core import build_cayley_dickson_tables
    _, k_signs, _ = build_cayley_dickson_tables(16)
    return tuple(tuple(int(v) for v in row) for row in k_signs.tolist())


if HAS_TRITON:
    KSIGNS = _sign_table_as_constexpr()

    @triton.jit
    def fused_sedenion_linear_kernel(
        X_ptr, W_ptr, Y_ptr, Bias_ptr,
        B, M, N,
        stride_xb, stride_xn, stride_xc,
        stride_wm, stride_wn, stride_wc,
        stride_yb, stride_ym, stride_yc,
        stride_bm, stride_bc,
        HAS_BIAS: tl.constexpr,
        KSIGNS: tl.constexpr,
        BLOCK_B: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        mask_b = offs_b < B
        mask_m = offs_m < M

        # 16 live accumulators, one per output basis component e_k
        accs = [tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32) for _ in tl.static_range(16)]

        for n0 in range(0, N, BLOCK_N):
            n_idx = n0 + offs_n
            mask_n = n_idx < N
            # Stage the 16 W component tiles for this n-block once
            w_tiles = []
            for i in tl.static_range(16):
                w_ptrs = W_ptr + offs_m[None, :] * stride_wm + n_idx[:, None] * stride_wn + i * stride_wc
                w_tiles.append(tl.load(w_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0))
            for j in tl.static_range(16):
                x_ptrs = X_ptr + offs_b[:, None] * stride_xb + n_idx[None, :] * stride_xn + j * stride_xc
                x_tile = tl.load(x_ptrs, mask=mask_b[:, None] & mask_n[None, :], other=0.0)
                for k in tl.static_range(16):
                    i = k ^ j
                    s = KSIGNS[k][j]
                    accs[k] += s * tl.dot(x_tile, w_tiles[i])

        for k in tl.static_range(16):
            acc = accs[k]
            if HAS_BIAS:
                bias = tl.load(Bias_ptr + offs_m * stride_bm + k * stride_bc, mask=mask_m, other=0.0)
                acc = acc + bias[None, :]
            y_ptrs = Y_ptr + offs_b[:, None] * stride_yb + offs_m[None, :] * stride_ym + k * stride_yc
            tl.store(y_ptrs, acc, mask=mask_b[:, None] & mask_m[None, :])

    def fused_sedenion_linear(x, w, bias=None, BLOCK_B=16, BLOCK_M=16, BLOCK_N=16):
        """x: [B, N, 16], w: [M, N, 16], bias: [M, 16] or None -> [B, M, 16]

        Default 16x16x16 tiles: the kernel keeps 16 accumulator tiles (BLOCK_B x BLOCK_M) and 16 staged
        weight tiles (BLOCK_N x BLOCK_M) live, i.e. 16*(BLOCK_B+BLOCK_N)*BLOCK_M fp32 values per program.
        At 32x32x32 that is 32768 values (~256 registers/thread on 4 warps, i.e. spills); at 16x16x16 it
        is 8192 (~64/thread).  NOT yet run on a GPU."""
        assert x.is_cuda and w.is_cuda
        B, N, _ = x.shape
        M = w.shape[0]
        y = torch.empty(B, M, 16, device=x.device, dtype=torch.float32)
        grid = (triton.cdiv(B, BLOCK_B), triton.cdiv(M, BLOCK_M))
        b_ptr = bias if bias is not None else y  # dummy, unused when HAS_BIAS=False
        fused_sedenion_linear_kernel[grid](
            x, w, y, b_ptr, B, M, N,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1), w.stride(2),
            y.stride(0), y.stride(1), y.stride(2),
            b_ptr.stride(0), b_ptr.stride(-1),
            HAS_BIAS=bias is not None, KSIGNS=KSIGNS,
            BLOCK_B=BLOCK_B, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return y


def fused_sedenion_linear_reference(x, w, bias=None):
    """Pure-torch mirror of the kernel's loop structure (n -> j -> k), for CPU tests."""
    from .core import build_cayley_dickson_tables
    _, k_signs, _ = build_cayley_dickson_tables(16)
    k_signs = k_signs.to(x.dtype)
    B, N, _ = x.shape
    M = w.shape[0]
    accs = [torch.zeros(B, M, dtype=x.dtype) for _ in range(16)]
    for j in range(16):
        xj = x[:, :, j]                                  # [B, N]
        for k in range(16):
            i = k ^ j
            accs[k] = accs[k] + k_signs[k, j] * (xj @ w[:, :, i].T)
    y = torch.stack(accs, dim=-1)
    return y + bias if bias is not None else y
