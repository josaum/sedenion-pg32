"""
Track A Models: Parameter-Matched Algebra Surrogates for Yang-Mills Scattering Amplitudes
(Section 5.2 & Table 2 of Agourakis 2026, "Sedenion PG(3,2) Neural Topologies").

Architectures:
  1. Dense-R16            : Standard Softmax Attention + Dense R^16 trivalent vertex
  2. Quat-H4              : 4 parallel Quaternion (H^4) channels (Associative)
  3. Oct-O2               : 2 parallel Octonion (O^2) channels (Alternative)
  4. Clifford-Cl04        : 16D Associative Clifford Geometric Algebra Cl_{0,4} (arXiv:2601.06793)
  5. Sed-Free             : Unconstrained 16D Sedenion (S) without PG(3,2) Fano router
  6. Sed-PG32-Moreno      : 16D Sedenion (S) with WP2 PG(3,2) 8+7 Fano Router & Moreno cone
  7. Sed-PG32-Moreno-NoFFN: Parameter-halved No-FFN variant (arXiv:2601.06793)
"""
import math
import sys
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sedenion_pg32 import tables
from tasks import FANO_MASK_16

_, KS16, XOR16 = tables(torch.float32)
KS8, XOR8 = KS16[:8, :8].contiguous(), XOR16[:8, :8].contiguous()
KS4, XOR4 = KS16[:4, :4].contiguous(), XOR16[:4, :4].contiguous()


def build_clifford_cl04_tables():
    """Constructs exact 16x16 sign and XOR tables for the associative Clifford algebra Cl(0,4)."""
    signs = torch.ones((16, 16), dtype=torch.float32)
    for i in range(16):
        for j in range(16):
            swaps = 0
            for b1 in range(4):
                if (i >> b1) & 1:
                    for b2 in range(b1):
                        if (j >> b2) & 1:
                            swaps += 1
            overlap = bin(i & j).count("1")
            if (swaps + overlap) % 2 == 1:
                signs[i, j] = -1.0
    k_idx = torch.arange(16).unsqueeze(1)
    j_idx = torch.arange(16).unsqueeze(0)
    xor_idx = k_idx ^ j_idx
    k_signs = signs[xor_idx, j_idx].contiguous()
    return k_signs, xor_idx


KS_CL16, XOR_CL16 = build_clifford_cl04_tables()


class AlgebraProduct(nn.Module):
    """Computes the bilinear product u * v in R^16 for each algebra row of Table 2."""

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.register_buffer("ks16", KS16)
        self.register_buffer("xor16", XOR16)
        self.register_buffer("ks8", KS8)
        self.register_buffer("xor8", XOR8)
        self.register_buffer("ks4", KS4)
        self.register_buffer("xor4", XOR4)
        self.register_buffer("ks_cl16", KS_CL16)
        self.register_buffer("xor_cl16", XOR_CL16)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.kind in ("Sed-Free", "Sed-PG32-Moreno", "Sed-PG32-Moreno-NoFFN"):
            return (u[..., self.xor16] * v.unsqueeze(-2) * self.ks16).sum(dim=-1)
        if self.kind == "Clifford-Cl04":
            return (u[..., self.xor_cl16] * v.unsqueeze(-2) * self.ks_cl16).sum(dim=-1)
        if self.kind == "Oct-O2":
            u2 = u.view(*u.shape[:-1], 2, 8)
            v2 = v.view(*v.shape[:-1], 2, 8)
            out = (u2[..., self.xor8] * v2.unsqueeze(-2) * self.ks8).sum(dim=-1)
            return out.reshape(*u.shape[:-1], 16)
        if self.kind == "Quat-H4":
            u4 = u.view(*u.shape[:-1], 4, 4)
            v4 = v.view(*v.shape[:-1], 4, 4)
            out = (u4[..., self.xor4] * v4.unsqueeze(-2) * self.ks4).sum(dim=-1)
            return out.reshape(*u.shape[:-1], 16)
        if self.kind == "Dense-R16":
            return u * v
        raise ValueError(f"Unknown algebra kind: {self.kind}")

    def associator(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Computes the algebra associator [a, b, c]_A = (a * b) * c - a * (b * c)."""
        return self.forward(self.forward(a, b), c) - self.forward(a, self.forward(b, c))


class YangMillsSurrogate(nn.Module):
    """
    Parameter-matched neural surrogate for 4-gluon Yang-Mills scattering amplitudes:
      A_hat = n_s / s + n_t / t + n_u / u
    where (n_s, n_t, n_u) are computed by applying the same trivalent cubic vertex V_A
    to the cyclic channel orderings (1,2,3), (2,3,1), (3,1,2).
    """

    def __init__(self, kind: str):
        super().__init__()
        self.kind = kind
        self.use_ffn = not kind.endswith("-NoFFN")
        self.use_pg32 = kind.startswith("Sed-PG32-Moreno")

        self.prod = AlgebraProduct(kind)
        self.register_buffer("fano_mask_16", FANO_MASK_16)  # [15, 16]

        # WP2 PG(3,2) Executive Workspace Gate (16 -> 15 Fano planes)
        self.workspace_gate = nn.Linear(16, 15)
        with torch.no_grad():
            nn.init.zeros_(self.workspace_gate.weight)
            nn.init.zeros_(self.workspace_gate.bias)
            for w in range(15):
                self.workspace_gate.weight[w, w + 1] = 4.0

        # Learnable channel scale / coordinate refinement
        self.coord_scale = nn.Parameter(torch.ones(16))
        if self.use_ffn:
            # Executive context MLP (modulates global amplitude scale from h0 without mixing PG(3,2) strata)
            self.ffn = nn.Sequential(
                nn.Linear(16, 32),
                nn.GELU(),
                nn.Linear(32, 16),
            )
            nn.init.zeros_(self.ffn[-1].weight)
            nn.init.zeros_(self.ffn[-1].bias)
        else:
            self.ffn = None

        # Softmax attention projections for Dense-R16 vs residual bilinear projections
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 16, bias=False)
        nn.init.normal_(self.q_proj.weight, std=0.05)
        nn.init.normal_(self.k_proj.weight, std=0.05)

        # Trivalent cubic vertex couplings:
        #   alpha_ym  : coupling on the edge-adjacent associator commutator [U_c, U_a, U_b]_A - [U_a, U_b, U_c]_A
        #   beta_alt  : coupling on the symmetric/alternating associator [U_b, U_c, U_a]_A
        #   res_scale : coupling on the bilinear associative fallback (U_a * U_b) * W_res(U_c)
        self.alpha_ym = nn.Parameter(torch.tensor(0.85))
        self.beta_alt = nn.Parameter(torch.tensor(0.0 if self.kind.startswith("Sed") else 0.25))
        self.res_proj = nn.Linear(16, 16, bias=False)
        nn.init.normal_(self.res_proj.weight, std=0.05)
        self.res_scale = nn.Parameter(torch.tensor(0.0 if self.use_pg32 else 0.20))

    def encode_particles(self, x_in: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Applies WP2 PG(3,2) Fano routing (Figure 1) and algebra-specific representation filtering.
        """
        if self.use_pg32:
            # Step 1 (Figure 1): WP2 Parity-Check Router on executive channel h0 selects top-1 Fano
            # hyperplane H_omega in PG(3,2), annihilating orthogonal gauge noise in H_omega^perp.
            logits = self.workspace_gate(h0)                # [B, 15]
            probs = F.softmax(logits, dim=-1)               # [B, 15]
            top_omega = torch.argmax(probs, dim=-1)         # [B]
            mask_hard = self.fano_mask_16[top_omega]        # [B, 16]
            mask_soft = torch.matmul(probs, self.fano_mask_16)
            mask = mask_hard + (mask_soft - mask_soft.detach())
            u = x_in * mask.unsqueeze(1)
            # Uniform scalar modulation preserves L_0 uplus Q_1 stratum orthogonality
            scale = self.coord_scale.mean()
            if self.ffn is not None:
                scale = scale + self.ffn(h0).mean(dim=-1, keepdim=True).unsqueeze(-1)
            return u * scale

        u = x_in * self.coord_scale
        if self.ffn is not None:
            u = u + self.ffn(u)

        if self.kind == "Dense-R16":
            q = self.q_proj(u)
            k = self.k_proj(u)
            attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) / 4.0, dim=-1)
            return u + 0.15 * torch.matmul(attn, u)

        return u

    def cubic_vertex(self, ua: torch.Tensor, ub: torch.Tensor, uc: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the trivalent cubic vertex V_A(ua, ub, uc) for channel (a, b | c):
          V_A = alpha_ym * ([uc, ua, ub]_A - [ua, ub, uc]_A)
                + beta_alt * [ub, uc, ua]_A
                + res_scale * ((ua * ub - 0.5 * ub * ua) * W_res(uc))
        """
        a_cab = self.prod.associator(uc, ua, ub)
        a_abc = self.prod.associator(ua, ub, uc)
        a_bca = self.prod.associator(ub, uc, ua)
        assoc_term = self.alpha_ym * (a_cab - a_abc) + self.beta_alt * a_bca

        ab_comm = self.prod(ua, ub) - 0.5 * self.prod(ub, ua)
        res_term = self.prod(ab_comm, self.res_proj(uc))
        return assoc_term + self.res_scale * res_term

    def forward(self, batch: dict):
        x_in = batch["x_in"]
        h0 = batch["h0"]
        stu = batch["stu"]

        u = self.encode_particles(x_in, h0)
        u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
        s, t, u_inv = stu[:, 0:1], stu[:, 1:2], stu[:, 2:3]

        n_s = self.cubic_vertex(u1, u2, u3)
        n_t = self.cubic_vertex(u2, u3, u1)
        n_u = self.cubic_vertex(u3, u1, u2)

        n_pred = torch.stack([n_s, n_t, n_u], dim=1)  # [B, 3, 16]
        amp_pred = n_s / s + n_t / t + n_u / u_inv
        return amp_pred, n_pred


if __name__ == "__main__":
    from tasks import generate_ym_dataset

    b = generate_ym_dataset(16, seed=0)
    for k in (
        "Dense-R16",
        "Quat-H4",
        "Oct-O2",
        "Clifford-Cl04",
        "Sed-Free",
        "Sed-PG32-Moreno",
        "Sed-PG32-Moreno-NoFFN",
    ):
        m = YangMillsSurrogate(k)
        amp, n_hat = m(b)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"{k:23s} params={n_params:5d}  amp={tuple(amp.shape)}  n_hat={tuple(n_hat.shape)}")
