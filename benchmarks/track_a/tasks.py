"""
Track A Benchmark Tasks: Yang-Mills BCJ Color-Kinematics Duality & Collinear Pole Cancellation
(Section 5.2 & Table 2 of Agourakis 2026, "Sedenion PG(3,2) Neural Topologies").

Generates 4-gluon trivalent scattering configurations with massless on-shell Mandelstam
invariants s + t + u = 0:
  - In-Distribution (ID) hard scattering: |s|, |t|, |u| in [0.08, 1.20]
  - Out-of-Distribution (OOD) collinear pole regime: min(|s|, |t|, |u|) < 1e-4 (down to 1e-6)

Physical structure:
  1. Each event belongs to an active color-helicity sector omega in {9..15} (Regime II Zero-Divisor
     Fano workspace H_omega = L_0 uplus Q_1 in PG(3,2)), encoded in the executive channel h_0 in R^16.
  2. In the resonant collinear channel c_res in {s, t, u}, the collinear pair (a, b) lies in L_0
     while spectator c lies in Q_1, plus O(s_res) longitudinal displacement and orthogonal gauge
     modes in H_omega^perp (which decouple by the Ward identity once projected onto H_omega).
  3. By Theorem 5.4 (fzd_stratum_bcj_pole_cancellation, machine-checked in Lean 4), the cyclic
     associator BCJ vertex on H_omega = L_0 uplus Q_1 satisfies:
       - Exact BCJ Jacobi identity: n_s* + n_t* + n_u* = 0 (R_BCJ = 0)
       - Exact resonant pole cancellation: [X_c, X_a, X_b] = [X_a, X_b, X_c] = 0 on (L_0, L_0, Q_1),
         so n_ab* = O(s_ab) and A* = n_s*/s + n_t*/t + n_u*/u remains strictly finite as s_ab -> 0.
"""
import math
import sys
import pathlib
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sedenion_pg32 import tables, sed_mul, build_pg32_fano_tables

_, K_SIGNS, XOR_IDX = tables(torch.float32)
FANO_MASK_15, _, _ = build_pg32_fano_tables()  # [15, 15] for omega=1..15, i=1..15

# Full 16D Fano plane incidence masks [15, 16] (e_0 = 0, e_1..e_15 = FANO_MASK_15)
FANO_MASK_16 = torch.zeros(15, 16, dtype=torch.float32)
FANO_MASK_16[:, 1:16] = FANO_MASK_15

# Precompute L_0 (octonion polar line, indices 1..7) and Q_1 (sedenion quadrangle, indices 9..15)
# masks for each of the 7 Zero-Divisor Fano workspaces omega in 9..15 (0-based indices 8..14).
L0_MASKS = torch.zeros(7, 16, dtype=torch.float32)
Q1_MASKS = torch.zeros(7, 16, dtype=torch.float32)
for idx_zd, omega in enumerate(range(9, 16)):
    for i in range(1, 16):
        if FANO_MASK_15[omega - 1, i - 1] > 0.5:
            if i < 8:
                L0_MASKS[idx_zd, i] = 1.0
            elif i > 8:
                Q1_MASKS[idx_zd, i] = 1.0


def sed_associator(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Computes [a, b, c]_S = (a * b) * c - a * (b * c) in R^16."""
    ab_c = sed_mul(sed_mul(a, b, K_SIGNS, XOR_IDX), c, K_SIGNS, XOR_IDX)
    a_bc = sed_mul(a, sed_mul(b, c, K_SIGNS, XOR_IDX), K_SIGNS, XOR_IDX)
    return ab_c - a_bc


def bcj_cyclic_numerators(x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
    """
    Exact cyclic associator BCJ vertex (Theorem 5.4):
      n_s = [x3, x1, x2] - [x1, x2, x3]
      n_t = [x1, x2, x3] - [x2, x3, x1]
      n_u = [x2, x3, x1] - [x3, x1, x2]
    Identically satisfies n_s + n_t + n_u = 0.
    """
    a123 = sed_associator(x1, x2, x3)
    a231 = sed_associator(x2, x3, x1)
    a312 = sed_associator(x3, x1, x2)
    n_s = a312 - a123
    n_t = a123 - a231
    n_u = a231 - a312
    return n_s, n_t, n_u


def generate_ym_dataset(
    n_samples: int,
    seed: int = 0,
    pole_regime: bool = False,
    pole_min: float = 1e-6,
    pole_max: float = 1e-4,
    gauge_noise_std: float = 0.35,
):
    """
    Generates n_samples 4-gluon scattering events.
    Returns dict with:
      x_in:      [N, 3, 16] input particle states (physical H_omega state + orthogonal H_omega^perp gauge noise)
      h0:        [N, 16] executive context vector encoding the active color-helicity sector omega in 9..15
      stu:       [N, 3] massless Mandelstam invariants (s, t, u) with s + t + u = 0
      omega_idx: [N] active F_ZD workspace index in 8..14 (corresponding to omega in 9..15)
      n_true:    [N, 3, 16] ground-truth BCJ numerators (n_s*, n_t*, n_u*)
      amp_true:  [N, 16] ground-truth physical scattering amplitude A* = n_s*/s + n_t*/t + n_u*/u
    """
    g = torch.Generator().manual_seed(seed)

    # 1. Sample active zero-divisor Fano plane omega in {9..15} (0-based plane index 8..14)
    zd_idx = torch.randint(0, 7, (n_samples,), generator=g)
    omega_plane = zd_idx + 8  # in 8..14
    res_chan = torch.randint(0, 3, (n_samples,), generator=g)

    # Executive channel context h0 in R^16 (4-bit binary parity signature of omega = zd_idx + 9)
    h0 = torch.zeros(n_samples, 16, dtype=torch.float32)
    omega_val = zd_idx + 9  # in 9..15
    for bit in range(4):
        h0[:, bit] = ((omega_val >> bit) & 1).float() * 2.0 - 1.0
    h0[torch.arange(n_samples), omega_val] = 1.0

    # 2. Sample Mandelstam invariants (s, t, u) with s + t + u = 0
    if pole_regime:
        log_s = torch.empty(n_samples).uniform_(math.log(pole_min), math.log(pole_max), generator=g)
        s_res_mag = torch.exp(log_s)
    else:
        s_res_mag = torch.empty(n_samples).uniform_(0.08, 0.45, generator=g)

    s_res_sign = torch.where(torch.rand(n_samples, generator=g) > 0.5, 1.0, -1.0)
    s_res = s_res_mag * s_res_sign

    s_trans1_mag = torch.empty(n_samples).uniform_(0.50, 1.20, generator=g)
    s_trans1_sign = torch.where(torch.rand(n_samples, generator=g) > 0.5, 1.0, -1.0)
    s_trans1 = s_trans1_mag * s_trans1_sign
    s_trans2 = -(s_res + s_trans1)

    stu = torch.zeros(n_samples, 3, dtype=torch.float32)
    for c in range(3):
        m = (res_chan == c)
        stu[m, c] = s_res[m]
        stu[m, (c + 1) % 3] = s_trans1[m]
        stu[m, (c + 2) % 3] = s_trans2[m]

    # 3. Construct physical states on H_omega = L_0 uplus Q_1 and orthogonal gauge noise in H_omega^perp
    raw = torch.randn(n_samples, 3, 16, generator=g)
    mask_L0 = L0_MASKS[zd_idx]              # [N, 16]
    mask_Q1 = Q1_MASKS[zd_idx]              # [N, 16]
    mask_H = FANO_MASK_16[omega_plane]      # [N, 16] (= mask_L0 + mask_Q1)
    mask_perp = 1.0 - mask_H                # [N, 16] (8 orthogonal gauge coordinates outside H_omega)

    x_phys = torch.zeros(n_samples, 3, 16, dtype=torch.float32)
    for c in range(3):
        m = (res_chan == c)
        if not m.any():
            continue
        a_idx, b_idx, c_idx = c, (c + 1) % 3, (c + 2) % 3
        l0 = mask_L0[m]
        q1 = mask_Q1[m]
        sr = s_res[m].unsqueeze(-1)  # [M, 1]

        pa = raw[m, a_idx] * l0
        pa = pa / pa.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pb = raw[m, b_idx] * l0
        pb = pb / pb.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        pc = raw[m, c_idx] * q1
        pc = pc / pc.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        # Physical longitudinal component proportional to s_res (vanishes at s_res -> 0)
        x_phys[m, a_idx] = pa + 0.5 * sr * (raw[m, a_idx] * q1)
        x_phys[m, b_idx] = pb + 0.5 * sr * (raw[m, b_idx] * q1)
        x_phys[m, c_idx] = pc + 0.5 * sr * (raw[m, c_idx] * l0)

    # Add orthogonal gauge modes in H_omega^perp (which decouple under PG(3,2) Fano projection)
    gauge_noise = torch.randn(n_samples, 3, 16, generator=g) * mask_perp.unsqueeze(1)
    x_in = x_phys + gauge_noise_std * gauge_noise

    # 4. Compute exact physical BCJ numerators and tree scattering amplitude
    n_s, n_t, n_u = bcj_cyclic_numerators(x_phys[:, 0], x_phys[:, 1], x_phys[:, 2])
    n_true = torch.stack([n_s, n_t, n_u], dim=1)  # [N, 3, 16]
    amp_true = (
        n_s / stu[:, 0:1]
        + n_t / stu[:, 1:2]
        + n_u / stu[:, 2:3]
    )

    return {
        "x_in": x_in,
        "x_phys": x_phys,
        "h0": h0,
        "stu": stu,
        "omega_plane": omega_plane,
        "n_true": n_true,
        "amp_true": amp_true,
    }


if __name__ == "__main__":
    d_id = generate_ym_dataset(512, seed=0, pole_regime=False)
    d_pole = generate_ym_dataset(512, seed=1, pole_regime=True)
    r_bcj = (
        d_pole["n_true"].sum(dim=1).norm(dim=-1).mean()
        / d_pole["n_true"].norm(dim=-1).sum(dim=1).mean()
    ).item()
    print(f"ID   | |s|_min = {d_id['stu'].abs().min().item():.4e} | ||A*||_mean = {d_id['amp_true'].norm(dim=-1).mean().item():.4f}")
    print(f"Pole | |s|_min = {d_pole['stu'].abs().min().item():.4e} | ||A*||_mean = {d_pole['amp_true'].norm(dim=-1).mean().item():.4f} | True R_BCJ = {r_bcj:.3e}")
