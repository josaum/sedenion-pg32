import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. EXACT CAYLEY-DICKSON & PG(3,2) INCIDENCE TABLE COMPILER
# ==============================================================================

def build_cayley_dickson_tables(dim: int = 16) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recursively constructs the exact Cayley-Dickson multiplication tables for dimension 2^n.
    Doubling rule: (a, b) * (c, d) = (a*c - d^* * b, d*a + b * c^*)
    
    Returns:
        signs: [dim, dim] float32 tensor in {-1.0, +1.0} where e_i * e_j = signs[i, j] * e_{i ^ j}
        k_signs: [dim, dim] float32 tensor where k_signs[k, j] = signs[k ^ j, j] for output channel k
        bitmasks: [dim] int32 tensor packing each row's negative signs into a 16-bit integer
    """
    assert dim > 0 and (dim & (dim - 1)) == 0, "Dimension must be a power of 2."
    signs = torch.ones((1, 1), dtype=torch.int8)
    n = 1
    while n < dim:
        next_signs = torch.zeros((2 * n, 2 * n), dtype=torch.int8)
        conj = torch.full((n,), -1, dtype=torch.int8)
        conj[0] = 1  # e_0^* = +e_0, e_i^* = -e_i for i > 0

        for i in range(n):
            for j in range(n):
                next_signs[i, j] = signs[i, j]                      # (e_i, 0)(e_j, 0) = (e_i e_j, 0)
                next_signs[i, j + n] = signs[j, i]                  # (e_i, 0)(0, e_j) = (0, e_j e_i)
                next_signs[i + n, j] = conj[j] * signs[i, j]        # (0, e_i)(e_j, 0) = (0, e_i e_j^*)
                next_signs[i + n, j + n] = -conj[j] * signs[j, i]   # (0, e_i)(0, e_j) = (-e_j^* e_i, 0)
        signs = next_signs
        n *= 2

    # k_signs[k, j] = epsilon_{(k ^ j), j} so Y[..., k] = sum_j k_signs[k, j] * W[..., k ^ j] * X[..., j]
    k_idx = torch.arange(dim).unsqueeze(1)
    j_idx = torch.arange(dim).unsqueeze(0)
    xor_idx = k_idx ^ j_idx
    k_signs = signs[xor_idx, j_idx].to(torch.float32)

    # Bit-pack the SWIZZLED table: bit j of bitmasks[k] is 1  <=>  epsilon_{k ^ j, j} = -1
    # (Eq. 15 / Algorithm 1: sign_bit = ((MASK[k] >> j) & 1) << 31).  Packing signs[i, j] instead
    # would be a different table (they differ in 240 of 256 entries).
    bitmasks = torch.zeros(dim, dtype=torch.int32)
    for k in range(dim):
        mask = 0
        for j in range(dim):
            if k_signs[k, j].item() == -1:
                mask |= (1 << j)
        bitmasks[k] = mask

    return signs.to(torch.float32), k_signs, bitmasks


def build_pg32_fano_tables() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compiles the 15 Fano hyperplanes H_omega (omega in 1..15) and their 7 internal
    projective lines L = (i, j, i ^ j) in PG(3,2).
    
    Returns:
        fano_mask: [15, 15] float32 incidence matrix (1 if point i in H_omega else 0)
        fano_lines: [15, 7, 3] int64 tensor of 0-based expert indices (in 0..14) for the 7 lines per plane
        fano_line_signs: [15, 7] float32 sign epsilon_{i,j} for each directed triad
    """
    signs, _, _ = build_cayley_dickson_tables(16)
    fano_mask = torch.zeros((15, 15), dtype=torch.float32)
    fano_lines = torch.zeros((15, 7, 3), dtype=torch.long)
    fano_line_signs = torch.zeros((15, 7), dtype=torch.float32)

    for omega in range(1, 16):
        pts = [i for i in range(1, 16) if (bin(omega & i).count("1") % 2) == 0]
        for i in pts:
            fano_mask[omega - 1, i - 1] = 1.0

        # Extract the 7 unique unordered projective lines {i, j, k = i ^ j} inside H_omega
        lines = []
        seen = set()
        for idx_a in range(len(pts)):
            for idx_b in range(idx_a + 1, len(pts)):
                i, j = pts[idx_a], pts[idx_b]
                k = i ^ j
                triad_key = tuple(sorted((i, j, k)))
                if k in pts and triad_key not in seen:
                    seen.add(triad_key)
                    lines.append(triad_key)

        assert len(lines) == 7, f"Expected 7 lines in Fano plane {omega}, found {len(lines)}"
        for l_idx, (i, j, k) in enumerate(lines):
            # Store as 0-based expert indices (0..14 corresponding to e_1..e_15)
            fano_lines[omega - 1, l_idx] = torch.tensor([i - 1, j - 1, k - 1], dtype=torch.long)
            fano_line_signs[omega - 1, l_idx] = signs[i, j]

    return fano_mask, fano_lines, fano_line_signs


# ==============================================================================
# 2. DIFFERENTIABLE SEDENION CALCULUS & ADJOINT AUTOGRAD KERNEL
# ==============================================================================

def sedenion_conj(x: torch.Tensor) -> torch.Tensor:
    """Conjugates a sedenion tensor along its last dimension (..., 16): e_0 -> +e_0, e_{1..15} -> -e_{1..15}."""
    conj_mask = torch.ones(16, device=x.device, dtype=x.dtype)
    conj_mask[1:] = -1.0
    return x * conj_mask


def sedenion_matmul_contract(
    x: torch.Tensor,
    w: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Memory-bounded sedenion linear contraction without 16x16 weight matrix expansion.
    Computes Y[..., m, k] = sum_n (W[m, n, :] * X[..., n, :])[k]
                          = sum_j k_signs[k, j] * sum_n W[m, n, k ^ j] * X[..., n, j]
    Shapes:
        x: [..., In_Features, 16]
        w: [Out_Features, In_Features, 16]
    Returns:
        y: [..., Out_Features, 16]
    """
    orig_shape = x.shape[:-2]
    in_features = x.shape[-2]
    out_features = w.shape[0]
    x_flat = x.reshape(-1, in_features, 16)  # [B, N, 16]

    # Loop over the 16 input channels j.  For each j the contribution to every output channel k is
    #   y[b, m, k] += k_signs[k, j] * sum_n W[m, n, k ^ j] * X[b, n, j]
    # which is one [B,N] x [M,N,16] contraction.  Peak extra memory is O(B*M*16 + M*N*16): no
    # [B, M, 16, 16] pair tensor and no 16M x 16N expanded weight matrix is ever materialised.
    y_flat = x_flat.new_zeros(x_flat.shape[0], out_features, 16)
    for j in range(16):
        w_perm = w[:, :, xor_idx[:, j]]                       # [M, N, 16], w_perm[..., k] = W[..., k ^ j]
        y_flat = y_flat + torch.einsum("mnk,bn->bmk", w_perm, x_flat[:, :, j]) * k_signs[:, j]
    return y_flat.reshape(*orig_shape, out_features, 16)


class SedenionLinearFunction(torch.autograd.Function):
    """
    Custom Autograd Function implementing the WP1 Universal Cayley-Dickson Adjoint Theorem:
        Forward:  Y = W * X + B
        Backward: dL/dX = W^* * G
                  dL/dW = G * X^*
                  dL/dB = sum_batch(G)
    Avoids materializing the 16M x 16N expanded weight matrix in HBM.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        k_signs: torch.Tensor,
        xor_idx: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, bias, k_signs, xor_idx)
        y = sedenion_matmul_contract(x, weight, k_signs, xor_idx)
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, bias, k_signs, xor_idx = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        in_features = x.shape[-2]
        out_features = weight.shape[0]
        x_flat = x.reshape(-1, in_features, 16)          # [B, N, 16]
        g_flat = grad_output.reshape(-1, out_features, 16)  # [B, M, 16]

        if ctx.needs_input_grad[0]:
            # Adjoint Theorem: dL/dX_n = sum_m W_{mn}^* * G_m
            w_conj_t = sedenion_conj(weight).transpose(0, 1).contiguous()  # [N, M, 16]
            grad_x_flat = sedenion_matmul_contract(g_flat, w_conj_t, k_signs, xor_idx)
            grad_x = grad_x_flat.reshape_as(x)

        if ctx.needs_input_grad[1]:
            # Adjoint Theorem: dL/dW_{mn} = sum_b G_{bm} * X_{bn}^*.  In coordinates
            #   dL/dW[m, n, i] = sum_b sum_j k_signs[i ^ j, j] * G[b, m, i ^ j] * X[b, n, j],
            # accumulated one input channel j at a time: peak extra memory is O(B*M*16), the size of
            # the gradient itself, instead of two [M, N, 16, 16] tensors (256*M*N floats).
            grad_weight = torch.zeros_like(weight)
            for j in range(16):
                perm = xor_idx[:, j]                                  # perm[i] = i ^ j
                g_perm = g_flat[:, :, perm] * k_signs[perm, j]        # [B, M, 16]
                grad_weight = grad_weight + torch.einsum("bmi,bn->mni", g_perm, x_flat[:, :, j])

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = g_flat.sum(dim=0)

        return grad_x, grad_weight, grad_bias, None, None


class SedenionLinear(nn.Module):
    """16D Sedenion Linear Layer with 16x parameter compression and Adjoint Backpropagation."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, onnx_compatible: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.onnx_compatible = onnx_compatible

        self.weight = nn.Parameter(torch.empty(out_features, in_features, 16))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, 16))
        else:
            self.register_parameter("bias", None)

        _, k_signs, _ = build_cayley_dickson_tables(16)
        xor_idx = torch.arange(16).unsqueeze(1) ^ torch.arange(16).unsqueeze(0)
        self.register_buffer("k_signs", k_signs)
        self.register_buffer("xor_idx", xor_idx)
        self.reset_parameters()

    def reset_parameters(self):
        # Glorot / He scaling adjusted for 16D hypercomplex variance (1 / sqrt(16 * in_features))
        bound = 1.0 / math.sqrt(16.0 * self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.onnx_compatible or torch.onnx.is_in_onnx_export():
            y = sedenion_matmul_contract(x, self.weight, self.k_signs, self.xor_idx)
            return y + self.bias if self.bias is not None else y
        return SedenionLinearFunction.apply(x, self.weight, self.bias, self.k_signs, self.xor_idx)


# ==============================================================================
# 3. MORENO ZERO-DIVISOR CONE PROJECTION & PG(3,2) BITMASK ROUTER
# ==============================================================================

class MorenoConeProjection(nn.Module):
    """
    Projects arbitrary 16D sedenion representations onto the Moreno Zero-Divisor Cone
    (Stiefel manifold V_2(R^7) embedded in Im(O) x Im(O)e_8):
      1. e_0 = 0, e_8 = 0
      2. <a, b> = 0 in R^7 (Gram-Schmidt orthogonalization)
      3. ||a||_2 = ||b||_2 (Balanced spherical or norm-preserving scaling)
    """

    def __init__(self, normalize_unit_sphere: bool = True, eps: float = 0.0, min_norm: float = 1e-30):
        super().__init__()
        self.normalize_unit_sphere = normalize_unit_sphere
        self.eps = eps
        self.min_norm = min_norm

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u shape: [..., 16]
        a_im = u[..., 1:8]   # Octonion imaginary part {e_1 .. e_7}
        b_im = u[..., 9:16]  # Sedenion twist imaginary part {e_9 .. e_15}

        # Gram-Schmidt orthogonalization of b_im against a_im in R^7
        a_norm_sq = (a_im * a_im).sum(dim=-1, keepdim=True) + self.eps
        proj_coeff = (a_im * b_im).sum(dim=-1, keepdim=True) / a_norm_sq
        b_ortho = b_im - proj_coeff * a_im

        a_norm = torch.sqrt(a_norm_sq).clamp_min(self.min_norm)
        b_norm = torch.sqrt((b_ortho * b_ortho).sum(dim=-1, keepdim=True) + self.eps).clamp_min(self.min_norm)

        if self.normalize_unit_sphere:
            a_out = a_im / a_norm
            b_out = b_ortho / b_norm
        else:
            # Preserve average half-norm ||a|| = ||b|| = 0.5 * (||a|| + ||b||)
            balanced_norm = 0.5 * (a_norm + b_norm)
            a_out = a_im * (balanced_norm / a_norm)
            b_out = b_ortho * (balanced_norm / b_norm)

        zero_axis = torch.zeros_like(u[..., :1])
        return torch.cat([zero_axis, a_out, zero_axis, b_out], dim=-1)


class PG32TopologicalRouter(nn.Module):
    """
    WP2 Hierarchical PG(3,2) Router over 15 Sedenion Expert Nodes (e_1..e_15) + Executive (e_0).
    Routes tokens into either:
      - Regime I (omega in 1..8): Alternative Octonion Workspaces (Norm-preserving synthesis)
      - Regime II (omega in 9..15): Zero-Divisor Quasi-Octonion Workspaces (Moreno contradiction pruning)
    """

    def __init__(self, d_model: int, temperature: float = 1.0, contradiction_pruning: bool = True):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.contradiction_pruning = contradiction_pruning

        self.workspace_gate = nn.Linear(16, 15)
        # Regime II pair-conditioning gate g in [0,1] per (line, token, directed pair): how much of the
        # receiver-side state y is replaced by its Moreno nullspace partner Pi_{ker L(x)}(y).
        self.prune_gate = nn.Linear(32, 1)
        nn.init.zeros_(self.prune_gate.weight); nn.init.constant_(self.prune_gate.bias, -2.0)
        self.moreno_proj = MorenoConeProjection(normalize_unit_sphere=False)

        signs, k_signs, _ = build_cayley_dickson_tables(16)
        fano_mask, fano_lines, fano_line_signs = build_pg32_fano_tables()
        xor_idx = torch.arange(16).unsqueeze(1) ^ torch.arange(16).unsqueeze(0)

        self.register_buffer("signs", signs)
        self.register_buffer("k_signs", k_signs)
        self.register_buffer("xor_idx", xor_idx)
        self.register_buffer("fano_mask", fano_mask)              # [15, 15]
        self.register_buffer("fano_lines", fano_lines)            # [15, 7, 3]
        self.register_buffer("fano_line_signs", fano_line_signs)  # [15, 7]

    def sedenion_elementwise_mul(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Batched elementwise 16D sedenion product u * v for tensors of shape [..., 16]."""
        u_swizzled = u[..., self.xor_idx]  # [..., 16, 16]
        prod = u_swizzled * v.unsqueeze(-2) * self.k_signs
        return prod.sum(dim=-1)

    def condition_on(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pair conditioning (Thm 5.1 / Remark 5.2): the vector v = Pi_{ker L(x)}(y) built FROM x, with the
        norm of y.  x must lie on the Moreno cone; then x * v = 0 exactly.  This is the explicit conditioning
        that makes the prohibition y land in ker L(x) (independent encoders would not)."""
        from .nullspace import moreno_nullspace_partner
        v = moreno_nullspace_partner(x, y[..., 1:8], self.k_signs, self.xor_idx)
        return v * (y.norm(dim=-1, keepdim=True) / math.sqrt(2.0))

    def pair_message(self, x, y, eps, zd):
        """message x * ((1-g) y + g Pi_{ker L(x)}(y)) * eps; conditioning only in the zero-divisor regime."""
        if self.contradiction_pruning:
            g = torch.sigmoid(self.prune_gate(torch.cat([x, y], dim=-1))) * zd.to(x.dtype)
            y = (1.0 - g) * y + g * self.condition_on(x, y)
        return self.sedenion_elementwise_mul(x, y) * eps.view(eps.shape[0], eps.shape[1], 1, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [Batch, 16, D, 16] -> 16 channels (e_0 executive + 15 PG(3,2) experts),
               each holding D sedenions of dimension 16.
        Returns:
            h_updated: [Batch, 16, D, 16]
            workspace_probs: [Batch, 15] Fano hyperplane routing distribution
        """
        B, _, D, _ = h.shape
        h0_summary = h[:, 0].mean(dim=1)  # [B, 16] executive context summary
        logits = self.workspace_gate(h0_summary) / self.temperature
        workspace_probs = F.softmax(logits, dim=-1)  # [B, 15]

        # Select top-1 Fano workspace per batch item for sparse 7-line routing
        top_omega = torch.argmax(workspace_probs, dim=-1)  # [B] in 0..14 (corresponding to omega=1..15)
        is_zero_divisor_regime = (top_omega >= 8).view(B, 1, 1, 1)  # omega in 9..15 (index 8..14)

        experts = h[:, 1:]  # [B, 15, D, 16]
        experts_zd = self.moreno_proj(experts)
        active_experts = torch.where(is_zero_divisor_regime, experts_zd, experts)

        # Gather the 7 projective lines [B, 7, 3] for each selected Fano plane
        batch_lines = self.fano_lines[top_omega]            # [B, 7, 3]
        batch_signs = self.fano_line_signs[top_omega]       # [B, 7]

        idx_i = batch_lines[:, :, 0].unsqueeze(-1).unsqueeze(-1).expand(B, 7, D, 16)
        idx_j = batch_lines[:, :, 1].unsqueeze(-1).unsqueeze(-1).expand(B, 7, D, 16)
        idx_k = batch_lines[:, :, 2].unsqueeze(-1).unsqueeze(-1).expand(B, 7, D, 16)

        h_i = torch.gather(active_experts, 1, idx_i)  # [B, 7, D, 16]
        h_j = torch.gather(active_experts, 1, idx_j)  # [B, 7, D, 16]
        h_k = torch.gather(active_experts, 1, idx_k)  # [B, 7, D, 16]

        # Cyclic triadic messages along L = {i, j, k = i ^ j}.  Because e_i e_j = eps_ij e_k,
        # e_j e_k = eps_jk e_i and e_k e_i = eps_ki e_j, every point of the line both sends and
        # receives (the earlier version only wrote into k = i ^ j, so the two lower-indexed points
        # of every line never received an update).
        pt = lambda t: t + 1                                    # 0-based expert index -> basis index e_1..e_15
        li, lj, lk = (batch_lines[:, :, c] for c in range(3))
        eps_ij = self.signs[pt(li), pt(lj)]
        eps_jk = self.signs[pt(lj), pt(lk)]
        eps_ki = self.signs[pt(lk), pt(li)]
        zd = is_zero_divisor_regime.expand(B, 1, 1, 1)
        msg_k = self.pair_message(h_i, h_j, eps_ij, zd)
        msg_i = self.pair_message(h_j, h_k, eps_jk, zd)
        msg_j = self.pair_message(h_k, h_i, eps_ki, zd)

        # Scatter-add collinear updates back into the 15 expert nodes
        routed_delta = torch.zeros_like(active_experts)
        routed_delta.scatter_add_(1, idx_k, msg_k)
        routed_delta.scatter_add_(1, idx_i, msg_i)
        routed_delta.scatter_add_(1, idx_j, msg_j)

        # Apply differentiable Fano incidence mask weighted by top-1 probability
        top_prob = workspace_probs.gather(1, top_omega.unsqueeze(1)).view(B, 1, 1, 1)
        active_mask = self.fano_mask[top_omega].view(B, 15, 1, 1)
        experts_out = experts + top_prob * active_mask * routed_delta

        h_updated = torch.cat([h[:, :1], experts_out], dim=1)
        return h_updated, workspace_probs
