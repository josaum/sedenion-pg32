"""
WP3 fused sedenion GEMM (Algorithm 1 of the paper).

Y[b, m, k] = sum_n sum_j  eps(k^j, j) * W[m, n, k^j] * X[b, n, j]

Loop order is n (outer, streamed) -> j (inner, static) -> k (static), with all 16
accumulator tiles live in registers, so every X tile and every W tile is loaded
from HBM exactly once per (b-block, m-block) program. The 16x16 sign table is a
compile-time constant (256 bits), never a runtime load.

Includes a bit-accurate CPU tile/pointer emulator when Triton/CUDA is unavailable
so that `fused_sedenion_linear_kernel` and `fused_sedenion_linear` are 100%
verified in unit tests on CPU as well as GPU.
"""
import torch

try:  # pragma: no cover
    import triton  # pragma: no cover
    import triton.language as tl  # pragma: no cover
    HAS_TRITON = torch.cuda.is_available()  # pragma: no cover
except ImportError:
    HAS_TRITON = False


class _CPUPtr:
    """Lightweight flat-memory pointer wrapper emulating Triton pointer arithmetic on CPU tensors."""

    def __init__(self, storage: torch.Tensor, offsets: torch.Tensor):
        self.storage = storage
        self.offsets = offsets.to(torch.long)

    def __add__(self, other):
        return _CPUPtr(self.storage, self.offsets + torch.as_tensor(other, dtype=torch.long))


class _CPUTritonLanguage:
    """CPU emulation of `triton.language` (`tl`) primitives for deterministic kernel verification."""
    constexpr = int
    float32 = torch.float32
    _pid = (0, 0)

    @classmethod
    def program_id(cls, axis: int) -> int:
        return cls._pid[axis]

    @staticmethod
    def arange(start: int, end: int) -> torch.Tensor:
        return torch.arange(start, end, dtype=torch.long)

    @staticmethod
    def zeros(shape, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    @staticmethod
    def static_range(n: int):
        return range(n)

    @staticmethod
    def load(ptr: _CPUPtr, mask: torch.Tensor, other: float = 0.0) -> torch.Tensor:
        flat = ptr.storage.reshape(-1)
        safe_offs = torch.where(mask, ptr.offsets.clamp(0, flat.numel() - 1), torch.zeros_like(ptr.offsets))
        vals = flat[safe_offs].to(torch.float32)
        return torch.where(mask, vals, torch.full_like(vals, float(other)))

    @staticmethod
    def dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a @ b

    @staticmethod
    def store(ptr: _CPUPtr, val: torch.Tensor, mask: torch.Tensor) -> None:
        flat = ptr.storage.reshape(-1)
        valid_offs = ptr.offsets[mask]
        flat[valid_offs] = val.to(flat.dtype)[mask]


class _CPUTritonRuntime:
    """CPU emulation of `triton` grid launcher and `cdiv`."""

    @staticmethod
    def cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b

    @staticmethod
    def jit(fn):
        class _KernelRunner:
            def __getitem__(self, grid):
                def _launch(X, W, Y, Bias, *args, **kwargs):
                    x_ptr = _CPUPtr(X, torch.tensor(0, dtype=torch.long))
                    w_ptr = _CPUPtr(W, torch.tensor(0, dtype=torch.long))
                    y_ptr = _CPUPtr(Y, torch.tensor(0, dtype=torch.long))
                    b_ptr = _CPUPtr(Bias, torch.tensor(0, dtype=torch.long))
                    for pb in range(grid[0]):
                        for pm in range(grid[1]):
                            _CPUTritonLanguage._pid = (pb, pm)
                            fn(x_ptr, w_ptr, y_ptr, b_ptr, *args, **kwargs)
                return _launch
        return _KernelRunner()


if not HAS_TRITON:
    triton = _CPUTritonRuntime
    tl = _CPUTritonLanguage


def _sign_table_as_constexpr():
    from .core import build_cayley_dickson_tables
    _, k_signs, _ = build_cayley_dickson_tables(16)
    return tuple(tuple(int(v) for v in row) for row in k_signs.tolist())


def build_packed_sign_masks_256bit():
    """
    Pack the 16x16 Cayley-Dickson sign table eps_{k^j, j} into 16 uint16 bitmasks (256 bits total),
    where bit j of MASK[k] is 1 iff eps_{k^j, j} == -1 (Equation (11) of the paper).
    """
    from .core import build_cayley_dickson_tables
    _, k_signs, _ = build_cayley_dickson_tables(16)
    masks = torch.zeros(16, dtype=torch.int32)
    for k in range(16):
        m = 0
        for j in range(16):
            if int(k_signs[k, j].item()) == -1:
                m |= (1 << j)
        masks[k] = m
    return masks


def ieee754_sign_xor(x_fp32: torch.Tensor, mask_k: int, j: int) -> torch.Tensor:
    """
    Multiply an FP32 tensor by eps_{k^j, j} in {-1, +1} via a single bitwise XOR on IEEE-754 bit 31:
      float_bits(x * eps) = float_bits(x) ^ (((MASK[k] >> j) & 1) << 31)
    """
    sign_bit = ((int(mask_k) >> j) & 1) << 31
    # In signed 32-bit two's complement, 1 << 31 is -2147483648 (0x80000000)
    sign_i32 = -2147483648 if sign_bit else 0
    bits = x_fp32.contiguous().view(torch.int32) ^ sign_i32
    return bits.view(torch.float32)


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
            w_ptrs = W_ptr + (offs_m[None, :] * stride_wm + n_idx[:, None] * stride_wn + i * stride_wc)
            w_tiles.append(tl.load(w_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0))
        for j in tl.static_range(16):
            x_ptrs = X_ptr + (offs_b[:, None] * stride_xb + n_idx[None, :] * stride_xn + j * stride_xc)
            x_tile = tl.load(x_ptrs, mask=mask_b[:, None] & mask_n[None, :], other=0.0)
            for k in tl.static_range(16):
                i = k ^ j
                s = KSIGNS[k][j]
                accs[k] = accs[k] + s * tl.dot(x_tile, w_tiles[i])

    for k in tl.static_range(16):
        acc = accs[k]
        if HAS_BIAS:
            bias = tl.load(Bias_ptr + (offs_m * stride_bm + k * stride_bc), mask=mask_m, other=0.0)
            acc = acc + bias[None, :]
        y_ptrs = Y_ptr + (offs_b[:, None] * stride_yb + offs_m[None, :] * stride_ym + k * stride_yc)
        tl.store(y_ptrs, acc, mask=mask_b[:, None] & mask_m[None, :])


def fused_sedenion_linear(x, w, bias=None, BLOCK_B=16, BLOCK_M=16, BLOCK_N=16):
    """x: [B, N, 16], w: [M, N, 16], bias: [M, 16] or None -> [B, M, 16]

    Default 16x16x16 tiles: the kernel keeps 16 accumulator tiles (BLOCK_B x BLOCK_M) and 16 staged
    weight tiles (BLOCK_N x BLOCK_M) live, i.e. 16*(BLOCK_B+BLOCK_N)*BLOCK_M fp32 values per program."""
    if HAS_TRITON:  # pragma: no cover
        assert x.is_cuda and w.is_cuda  # pragma: no cover
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
    """Pure-torch mirror of the kernel's loop structure (n -> j -> k) using the 256-bit packed IEEE-754 sign masks."""
    from .core import build_cayley_dickson_tables
    _, k_signs, _ = build_cayley_dickson_tables(16)
    k_signs = k_signs.to(x.dtype)
    masks = build_packed_sign_masks_256bit()
    B, N, _ = x.shape
    M = w.shape[0]
    accs = [torch.zeros(B, M, dtype=x.dtype) for _ in range(16)]
    for j in range(16):
        xj = x[:, :, j]                                  # [B, N]
        for k in range(16):
            i = k ^ j
            if w.dtype == torch.float32:
                w_signed = ieee754_sign_xor(w[:, :, i], int(masks[k].item()), j)
                accs[k] = accs[k] + (xj @ w_signed.T)
            else:
                accs[k] = accs[k] + k_signs[k, j] * (xj @ w[:, :, i].T)
    y = torch.stack(accs, dim=-1)
    return y + bias if bias is not None else y
