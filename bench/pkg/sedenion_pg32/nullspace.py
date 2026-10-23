"""WP4: constructive 4-D nullspace partner on the Moreno cone, plus sed/oct products."""
import torch
from .core import build_cayley_dickson_tables


def tables(dtype=torch.float64):
    signs, k_signs, _ = build_cayley_dickson_tables(16)
    xor_idx = torch.arange(16).unsqueeze(1) ^ torch.arange(16).unsqueeze(0)
    return signs.to(dtype), k_signs.to(dtype), xor_idx


def sed_mul(u, v, k_signs, xor_idx):
    """Exact 16-D sedenion product for [..., 16] tensors."""
    return (u[..., xor_idx] * v.unsqueeze(-2) * k_signs).sum(dim=-1)


def oct_mul(a, b, k_signs, xor_idx):
    """Exact 8-D octonion product (top-left block of the sedenion table)."""
    return (a[..., xor_idx[:8, :8]] * b.unsqueeze(-2) * k_signs[:8, :8]).sum(dim=-1)


def left_mult_matrix(u, k_signs, xor_idx):
    """L(u) with L(u) x = u * x, shape [..., 16, 16]."""
    E = torch.eye(16, dtype=u.dtype, device=u.device)
    return torch.stack([sed_mul(u, E[j], k_signs, xor_idx) for j in range(16)], dim=-1)


def moreno_nullspace_partner(u, raw_c, k_signs, xor_idx):
    """
    Theorem (constructive nullspace). For u=(a,b) with a,b in Im(O), |a|=|b|=1, <a,b>=0,
    and any seed raw_c in R^7: project c onto span(a,b,ab)^perp in Im(O), set d=(a c) b.
    Then v=(c,d) satisfies u*v = 0 exactly.
    """
    a, b = u[..., :8], u[..., 8:16]
    ab = oct_mul(a, b, k_signs, xor_idx)
    c = torch.zeros_like(a)
    c[..., 1:8] = raw_c[..., :7]
    for basis in (a, b, ab):
        c = c - (c * basis).sum(-1, keepdim=True) / (basis * basis).sum(-1, keepdim=True) * basis
    c[..., 0] = 0.0
    c = c / torch.linalg.norm(c, dim=-1, keepdim=True).clamp_min(1e-30)
    # kernel of L(u) is scale-invariant in u; normalise so the formula holds on the whole cone
    scale = (a.norm(dim=-1, keepdim=True) * b.norm(dim=-1, keepdim=True)).clamp_min(1e-30)
    d = oct_mul(oct_mul(a, c, k_signs, xor_idx), b, k_signs, xor_idx) / scale
    return torch.cat([c, d], dim=-1)
