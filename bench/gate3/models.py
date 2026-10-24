"""
Parameter-matched 2-layer encoders differing only in the attention kernel:

  softmax        : standard multi-head softmax attention (4 heads x 16)
  bilinear       : scalar ReLU(q.k) weights, no softmax (exact zeros via orthogonality)
  octonion       : 8-D octonion division-algebra product attention (no zero divisors: ||u * k|| = ||u|| ||k|| = 1)
  sedenion_plain : 16-D sedenion product attention WITHOUT Moreno cone projection or Theorem 5.1 nullspace partner
  sedenion       : 16-D sedenion product attention on the Moreno cone V_2(R^7) with Constructive 4D Nullspace Partner
                   (Theorem 5.1 & SedenionRank.lean `square_identity`) and Orthogonal Half-Sphere Key Modulation.

Theorem (Division-Algebra Invariance vs. Sedenion Zero-Divisor Annihilation under Half-Sphere Key Modulation):
  For static unit address u_j, carrier r_j = u_j * (c_j, 0), and unit half-sphere modulated key
      k_j(phi_j) = (c_j, -cos(phi_j) p_j + sin(phi_j) p_{perp, j}) / sqrt(2),   phi_j = pi * tau_j in [0, pi],
  where ||c_j|| = ||p_j|| = ||p_{perp, j}|| = 1 and p_j perp p_{perp, j}:
    1. Exact Unit Key Norm: ||k_j(phi_j)||_2 == 1 identically for all phi_j in R.
    2. Division-Algebra Invariance (Octonion): Since left multiplication L(u_j) in O is an orthogonal isometry
       and (c_j, 0) perp (0, -cos(phi_j) p_j + sin(phi_j) p_{perp, j}),
           ||u_j *_O k_j(phi_j)||_2 == 1   and   Re(r_j^* *_O (u_j *_O k_j(phi_j))) == 1 / sqrt(2)
       identically for all phi_j in R (zero modulation capacity without zero divisors).
    3. Sedenion Zero-Divisor Annihilation on V_2(R^7): With c_j, c_{perp, j} in span(e_0, a_j, b_j, a_j*b_j)^perp
       and Theorem 5.1 partners p_j = 2 (a_j c_j) b_j, p_{perp, j} = 2 (a_j c_{perp, j}) b_j,
           ||u_j *_S k_j(phi_j)||_2 = sqrt(1 + cos(phi_j)),   Re(r_j^* *_S (u_j *_S k_j(phi_j))) = (1 + cos(phi_j)) / sqrt(2),
       which equals sqrt(2) at tau_j = 0 (LIVE, phi_j = 0) and 0.000000 at tau_j = 1 (REVOKED, phi_j = pi,
       where k_j(pi) = (c_j, +p_j) / sqrt(2) in ker L(u_j)).
"""
import math
import pathlib
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pkg"))
sys.path.insert(0, "/home/claude/bench/pkg")
sys.path.insert(0, str(HERE))
from sedenion_pg32 import tables, MorenoConeProjection
from tasks import VOCAB_TYPE, VOCAB_ARG, VOCAB_ID, T

_, KS16, XOR16 = tables(torch.float32)
KS8, XOR8 = KS16[:8, :8].contiguous(), XOR16[:8, :8].contiguous()


def _build_clifford16_tables(dtype=torch.float32):
    """Exact 16-D associative Clifford (Geometric) Algebra Cl(0, 4) sign and reversion tables."""
    signs_cl = torch.ones(16, 16, dtype=dtype)
    for a in range(16):
        for b in range(16):
            inv = 0
            for p in range(4):
                if (a >> p) & 1:
                    for q in range(p):
                        if (b >> q) & 1:
                            inv += 1
                    if (b >> p) & 1:
                        inv += 1
            signs_cl[a, b] = -1.0 if (inv % 2 == 1) else 1.0
    ks_cl = torch.zeros(16, 16, dtype=dtype)
    for k in range(16):
        for j in range(16):
            ks_cl[k, j] = signs_cl[k ^ j, j]
    conj_cl = torch.tensor([signs_cl[a, a].item() for a in range(16)], dtype=dtype)
    return ks_cl, XOR16, conj_cl


KS_CL16, XOR_CL16, CONJ_CL16 = _build_clifford16_tables(torch.float32)


def alg_tables(kind):
    base = kind[:-6] if kind.endswith("_noffn") else kind
    if base in ("sedenion", "sedenion_plain", "linear_gate", "linear_gate_moreno"):
        conj = torch.ones(16)
        conj[1:] = -1.0
        return KS16, XOR16, 16, conj
    if base == "clifford":
        return KS_CL16, XOR_CL16, 16, CONJ_CL16
    if base == "octonion":
        conj = torch.ones(8)
        conj[1:] = -1.0
        return KS8, XOR8, 8, conj
    raise ValueError(f"Unknown algebra kind: {kind}")


class ProductAttention(nn.Module):
    """Hypercomplex product attention with pure O(N) vector products and Moreno zero-divisor key modulation."""
    def __init__(self, d, n_heads, kind):
        super().__init__()
        base = kind[:-6] if kind.endswith("_noffn") else kind
        self.d, self.h, self.kind = d, n_heads, base
        self.hd = d // n_heads
        ks, xor, blk, conj_sign = alg_tables(base)
        self.register_buffer("ks", ks)
        self.register_buffer("xor", xor)
        self.register_buffer("ks8", KS8)
        self.register_buffer("xor8", XOR8)
        self.register_buffer("conj_sign", conj_sign)
        mask_e0 = torch.zeros(blk)
        mask_e0[0] = 1.0
        self.register_buffer("mask_e0", mask_e0)
        mask_c0 = torch.ones(blk // 2)
        mask_c0[0] = 0.0
        self.register_buffer("mask_c0", mask_c0)
        self.blk = blk
        self.half = blk // 2
        self.nb = self.hd // blk
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.zeros(n_heads))
        self.moreno = MorenoConeProjection(normalize_unit_sphere=True) if base in ("sedenion", "linear_gate_moreno") else None

    def mul(self, u, v):
        return (u[..., self.xor] * v.unsqueeze(-2) * self.ks).sum(-1)

    def mul8(self, u, v):
        return (u[..., self.xor8] * v.unsqueeze(-2) * self.ks8).sum(-1)

    def _unit_addr(self, x):
        if self.moreno is not None:
            return self.moreno(x) / math.sqrt(2.0)
        return F.normalize(x, dim=-1, eps=1e-12)

    def precompute_static(self, q_src, x_stat, mask, v_src=None, use_q_proj_for_k=False, x0_dyn=None):
        B, N, _ = x_stat.shape
        g, bl, hf = self.nb, self.blk, self.half
        if v_src is None:
            v_src = x_stat
        if x0_dyn is None:
            x0_dyn = x_stat
        q_raw = self.qkv(q_src).chunk(3, -1)[0]
        k_stat_raw = self.qkv(x_stat).chunk(3, -1)[0 if use_q_proj_for_k else 1]
        v_stat_raw = self.qkv(v_src).chunk(3, -1)[2]
        k_dyn0_raw = self.qkv(x0_dyn).chunk(3, -1)[1]

        q = self._unit_addr(q_raw.view(B, N, self.h, g, bl))
        u = self._unit_addr(k_stat_raw.view(B, N, self.h, g, bl))
        v_stat = v_stat_raw.view(B, N, self.h, g, bl)
        k_dyn0 = k_dyn0_raw.view(B, N, self.h, g, bl)

        k_stat_blk = k_stat_raw.view(B, N, self.h, g, bl)
        raw_c = k_stat_blk[..., :hf].roll(1, dims=-1)
        raw_d = k_stat_blk[..., hf:bl].roll(1, dims=-1)
        if self.kind == "sedenion":
            a_j, b_j = u[..., :8], u[..., 8:16]
            ab_j = self.mul8(a_j, b_j)
            c0 = raw_c * self.mask_c0
            c_perp = (
                c0
                - 2.0 * (c0 * a_j).sum(-1, keepdim=True) * a_j
                - 2.0 * (c0 * b_j).sum(-1, keepdim=True) * b_j
                - 4.0 * (c0 * ab_j).sum(-1, keepdim=True) * ab_j
            )
            c_j = F.normalize(c_perp, dim=-1, eps=1e-12)
            p_j = 2.0 * self.mul8(self.mul8(a_j, c_j), b_j)
            d0 = raw_d * self.mask_c0
            d_perp = (
                d0
                - 2.0 * (d0 * a_j).sum(-1, keepdim=True) * a_j
                - 2.0 * (d0 * b_j).sum(-1, keepdim=True) * b_j
                - 4.0 * (d0 * ab_j).sum(-1, keepdim=True) * ab_j
                - (d0 * c_j).sum(-1, keepdim=True) * c_j
            )
            c_perp2 = F.normalize(d_perp, dim=-1, eps=1e-12)
            p_perp_j = 2.0 * self.mul8(self.mul8(a_j, c_perp2), b_j)
        else:
            c_j = F.normalize(raw_c * self.mask_c0, dim=-1, eps=1e-12)
            p_j = F.normalize(raw_d * self.mask_c0, dim=-1, eps=1e-12)
            d2 = raw_d.roll(1, dims=-1) * self.mask_c0
            d2 = d2 - (d2 * p_j).sum(-1, keepdim=True) * p_j
            p_perp_j = F.normalize(d2, dim=-1, eps=1e-12)

        c_pad = torch.cat([c_j, torch.zeros_like(c_j)], dim=-1)
        r_j_conj = self.mul(u, c_pad) * self.conj_sign

        addr_dot = torch.einsum("bihgm,bjhgm->bijhg", q, u)
        th = 0.65 if use_q_proj_for_k else 0.50
        A_hard = F.relu((addr_dot - th) / (1.0 - th))
        A_soft = torch.sigmoid((addr_dot - 0.25) * 8.0)
        A_ij = A_hard + (A_soft - A_soft.detach())
        m = mask.view(B, 1, N, 1, 1).float()
        w_addr = A_ij * m
        denom = w_addr.sum(2).clamp_min(1.0).unsqueeze(-1)
        return (u, c_j, p_j, p_perp_j, r_j_conj, w_addr, denom, v_stat, k_dyn0)

    def forward(self, x, mask, q_in=None, x_stat=None, v_in=None, cached=None, x0_dyn=None):
        B, N, _ = x.shape
        g, bl, hf = self.nb, self.blk, self.half
        if cached is None:
            if x_stat is None:
                x_stat = x
            q_src = q_in if q_in is not None else x_stat
            cached = self.precompute_static(
                q_src, x_stat, mask, v_src=v_in, use_q_proj_for_k=(q_in is not None), x0_dyn=x0_dyn
            )
        u, c_j, p_j, p_perp_j, r_j_conj, w_addr, denom, v_stat, k_dyn0 = cached

        k_dyn_raw = self.qkv(x).chunk(3, -1)[1]
        k_dyn = k_dyn_raw.view(B, N, self.h, g, bl)

        delta_k = k_dyn[..., hf:bl] - k_dyn0[..., hf:bl]
        tau_raw = F.relu((delta_k.pow(2).sum(dim=-1, keepdim=True) + 1e-12).sqrt() - 1e-6) * 8.0
        tau_hard = tau_raw.clamp(0.0, 1.0)
        tau_soft = torch.tanh(tau_raw)
        tau_j = tau_hard + (tau_soft - tau_soft.detach())
        phi_j = math.pi * tau_j
        k_j = torch.cat([c_j, -torch.cos(phi_j) * p_j + torch.sin(phi_j) * p_perp_j], dim=-1) / math.sqrt(2.0)

        if self.kind in ("linear_gate", "linear_gate_moreno"):
            # LINEAR BASELINE: the closed-form scalar gate the sedenion path computes,
            # g(phi) = (1 + cos phi)/sqrt(2) on the message, sqrt(1 + cos phi) on the score.
            # No hypercomplex products anywhere.
            cosphi = torch.cos(phi_j)
            prod_norm_j = F.relu((1.0 + cosphi).clamp_min(0.0).sqrt().squeeze(-1) - 1e-6)
            msg_j = (1.0 + cosphi) / math.sqrt(2.0) * v_stat
        else:
            # Pure O(N) zero-divisor product z_j = u_j * k_j(phi_j)
            z_j = self.mul(u, k_j)
            prod_norm_j = F.relu(z_j.norm(dim=-1) - 1e-6)
            w_raw = self.mul(r_j_conj, z_j)
            w_j = w_raw * self.mask_e0

            left_j = self.mul(w_j, v_stat)
            right_j = self.mul(v_stat, w_j)
            a_op = torch.sigmoid(self.alpha).view(1, 1, self.h, 1, 1)
            msg_j = a_op * left_j + (1.0 - a_op) * right_j

        scores_blk = w_addr * prod_norm_j.unsqueeze(1)
        scores_norm = scores_blk.mean(-1)
        o = torch.einsum("bijhg,bjhgk->bihgk", w_addr, msg_j) / denom
        self.last_scores_grad = scores_norm
        self.last_scores = scores_norm.detach()
        if q_in is not None:
            return F.linear(o.reshape(B, N, self.d), self.out.weight)
        return self.out(o.reshape(B, N, self.d))


class BilinearAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.h, self.hd = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, mask, q_in=None, x_stat=None, v_in=None, cached=None, x0_dyn=None):
        B, N, _ = x.shape
        if x_stat is None:
            x_stat = x
        if v_in is None:
            v_in = x_stat
        q_src = q_in if q_in is not None else x_stat
        q = self.qkv(q_src).chunk(3, -1)[0]
        k = self.qkv(x).chunk(3, -1)[1]
        v = self.qkv(v_in).chunk(3, -1)[2]
        q = F.normalize(q.view(B, N, self.h, self.hd), dim=-1)
        k = F.normalize(k.view(B, N, self.h, self.hd), dim=-1)
        v = v.view(B, N, self.h, self.hd)
        s = F.relu(torch.einsum("bihd,bjhd->bijh", q, k))
        m = mask.view(B, 1, N, 1).float()
        o = torch.einsum("bijh,bjhd->bihd", s * m, v) / (s * m).sum(2).clamp_min(1.0).unsqueeze(-1)
        self.last_scores_grad = s
        self.last_scores = s.detach()
        return self.out(o.reshape(B, N, self.d))


class SoftmaxAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.d, self.h, self.hd = d, n_heads, d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, mask, q_in=None, x_stat=None, v_in=None, cached=None, x0_dyn=None):
        B, N, _ = x.shape
        if x_stat is None:
            x_stat = x
        if v_in is None:
            v_in = x_stat
        q_src = q_in if q_in is not None else x_stat
        q = self.qkv(q_src).chunk(3, -1)[0]
        k = self.qkv(x).chunk(3, -1)[1]
        v = self.qkv(v_in).chunk(3, -1)[2]
        q = q.view(B, N, self.h, self.hd)
        k = k.view(B, N, self.h, self.hd)
        v = v.view(B, N, self.h, self.hd)
        s = torch.einsum("bihd,bjhd->bijh", q, k) / math.sqrt(self.hd)
        s = s.masked_fill(~mask.view(B, 1, N, 1), float("-inf"))
        p = s.softmax(2)
        p = torch.nan_to_num(p, nan=0.0)
        o = torch.einsum("bijh,bjhd->bihd", p, v)
        self.last_scores_grad = p
        self.last_scores = p.detach()
        return self.out(o.reshape(B, N, self.d))


class Block(nn.Module):
    def __init__(self, d, n_heads, kind):
        super().__init__()
        no_ffn = kind.endswith("_noffn")
        base = kind[:-6] if no_ffn else kind
        self.n1 = nn.LayerNorm(d)
        self.n2 = None if no_ffn else nn.LayerNorm(d)
        if base == "softmax":
            self.attn = SoftmaxAttention(d, n_heads)
        elif base == "bilinear":
            self.attn = BilinearAttention(d, n_heads)
        else:
            self.attn = ProductAttention(d, n_heads, base)
        self.mlp = None if no_ffn else nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def ffn(self, z):
        if self.mlp is None:
            return z
        return z + self.mlp(self.n2(z))

    def forward(self, x, mask, x_stat=None, x0_dyn=None, x_dyn=None):
        x0 = self.n1(x_stat) if x_stat is not None else None
        xd0 = self.n1(x0_dyn) if x0_dyn is not None else x0
        xd = self.n1(x_dyn) if (x_dyn is not None and hasattr(self.attn, "precompute_static")) else self.n1(x)
        x = x + self.attn(xd, mask, x_stat=x0, x0_dyn=xd0)
        return self.ffn(x)


class Encoder(nn.Module):
    def __init__(self, kind, d=64, n_heads=4, n_layers=2, n_steps=4):
        super().__init__()
        self.kind = kind
        self.d = d
        self.n_steps = n_steps
        self.et = nn.Embedding(VOCAB_TYPE, d)
        self.ea = nn.Embedding(VOCAB_ARG, d)
        self.ei = nn.Embedding(VOCAB_ID, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, kind) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, xt, xa, xi):
        mask = xt != T["PAD"]
        mask0 = xt == T["REVOKE"]
        is_revoke = mask0.unsqueeze(-1)
        target_id_emb = self.ei(xa.clamp_max(VOCAB_ID - 1))

        q0 = self.blocks[0].n1(self.ei(xi))
        k0_raw = torch.where(is_revoke, target_id_emb, self.ei(torch.full_like(xi, VOCAB_ID - 1)))
        x0_stat = self.blocks[0].n1(k0_raw)

        s0 = self.et(torch.zeros_like(xt))
        v0_stat = self.blocks[0].n1(self.et(xt))
        w0 = self.blocks[0].ffn(s0)
        x0_dyn0 = self.blocks[0].n1(w0)
        w = w0
        a = torch.zeros_like(s0)
        cached0 = (
            self.blocks[0].attn.precompute_static(
                q0, x0_stat, mask0, v_src=v0_stat, use_q_proj_for_k=True, x0_dyn=x0_dyn0
            )
            if hasattr(self.blocks[0].attn, "precompute_static")
            else None
        )
        for _ in range(self.n_steps):
            a = self.blocks[0].attn(
                self.blocks[0].n1(w), mask0, q_in=q0, x_stat=x0_stat, v_in=v0_stat, cached=cached0, x0_dyn=x0_dyn0
            )
            s1 = s0 + a
            w = self.blocks[0].ffn(s1)

        arg_emb = torch.where(is_revoke, target_id_emb.roll(self.d // 2, dims=-1), self.ea(xa))
        x_base = self.et(xt) + arg_emb + self.ei(xi)
        x0_blk1 = self.blocks[0].ffn(x_base)
        x1 = x_base + a
        x = self.blocks[0].ffn(x1)
        x = self.blocks[1](x, mask, x_stat=x0_blk1, x0_dyn=w0, x_dyn=w)

        qpos = (xt == T["QUERY"]).float().argmax(1)
        xq = self.norm(x[torch.arange(x.shape[0], device=x.device), qpos])
        return self.head(xq).squeeze(-1), qpos


if __name__ == "__main__":
    for kind in (
        "softmax", "bilinear", "octonion", "clifford", "sedenion_plain", "sedenion",
        "softmax_noffn", "bilinear_noffn", "octonion_noffn", "clifford_noffn", "sedenion_plain_noffn", "sedenion_noffn"
    ):
        m = Encoder(kind)
        n = sum(p.numel() for p in m.parameters())
        xt = torch.randint(0, 4, (2, 16))
        xt[:, -1] = T["QUERY"]
        y, _ = m(xt, torch.randint(0, 12, (2, 16)), torch.randint(0, 25, (2, 16)))
        print(f"{kind:20s} params={n}  out={tuple(y.shape)}")
