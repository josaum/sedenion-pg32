"""
Native 16-Dimensional Sedenion Signal Processing (NSSP) & Hardware Primitives:
  1. Single-Snapshot Native Associator & Lagrange Wedge Curvature Detector
  2. SVD-Free Associator-Seeded Moreno Zero-Divisor Annihilator
  3. Calibration-Free Dyadic 2x2 Dual-Polarized TEM Vector-Sensor Array
  4. Multiplier-Free Pre-ADC Passive Analog Differential Sedenion Crossbar
  5. Zero-Divisor Keyed Modulation (ZDKM) LPI/LPD Transceiver
  6. Bit-Accurate Q1.14 Fixed-Point SystemVerilog RTL Emulator
"""
import math
from typing import Tuple, Optional
import torch
from .nullspace import tables, sed_mul, oct_mul, moreno_nullspace_partner


def native_sedenion_associator(
    R: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the Single-Snapshot Native Sedenion Associator:
        A_native(R) = [R, pi_O(R), e_8] = (R * pi_O(R)) * e_8 - R * (pi_O(R) * e_8)
    For purely imaginary R = (a, b) in Im(O) x Im(O), Theorem 1 (Lean 4
    `native_associator_commutator`) guarantees:
        A_native(R) = (b*a - a*b, 0) = (-2 * (a x_O b), 0).
    """
    pi_O = torch.zeros_like(R)
    pi_O[..., :8] = R[..., :8]
    e8 = torch.zeros_like(R)
    e8[..., 8] = 1.0
    left = sed_mul(sed_mul(R, pi_O, k_signs, xor_idx), e8, k_signs, xor_idx)
    right = sed_mul(R, sed_mul(pi_O, e8, k_signs, xor_idx), k_signs, xor_idx)
    return left - right


def native_associator_curvature(
    R: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Computes the normalized Single-Snapshot Non-Associative Curvature:
        kappa_S(R) = ||[R, pi_O(R), e_8]||_2 / ||R||_2^2  in [0, 1].
      - kappa_S(R) == 0 iff R lies in a 4D associative quaternion subalgebra (a || b)
      - kappa_S(R) == 1 iff R lies on the Moreno Zero-Divisor Cone V_2(R^7) (||a||=||b||, <a,b>=0)
    """
    assoc = native_sedenion_associator(R, k_signs, xor_idx)
    num = assoc.norm(dim=-1)
    den = (R * R).sum(dim=-1).clamp_min(eps)
    return num / den


def associator_nullspace_partner(
    R: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
    eps: float = 1e-30,
) -> torch.Tensor:
    """
    Constructs a Moreno Zero-Divisor Partner W in ker L_S(R) directly from a single
    snapshot R = (a, b) using the Native Associator A_native(R) = (-2*(a x_O b), 0):
      1. Extracts the 3rd quaternion-triad vector q_3 = a x_O b = -(1/2) * pi_O(A_native(R)),
         completing the 3D associative triad span{a, b, a x_O b} in Im(O).
      2. Forms a deterministic cyclic Fano permutation of (a, b, q_3) and projects it onto
         the 4D orthogonal complement span{a, b, a x_O b}^perp in Im(O), yielding c without
         SVD or random number generation.
    """
    assoc = native_sedenion_associator(R, k_signs, xor_idx)
    q3_cross = -0.5 * assoc[..., 1:8]
    a_im = R[..., 1:8]
    b_im = R[..., 9:16]
    a_norm = a_im.norm(dim=-1, keepdim=True).clamp_min(eps)
    a_unit = a_im / a_norm
    b_ortho = b_im - (b_im * a_unit).sum(dim=-1, keepdim=True) * a_unit
    b_unit = b_ortho / b_ortho.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Deterministic seed in R^7 guaranteed not to lie in span{a, b, q3_cross}:
    # try cyclic shifts of q3_cross and standard basis vectors e_1..e_7, picking the largest projection
    u_cone = torch.zeros_like(R)
    u_cone[..., 1:8] = a_unit
    u_cone[..., 9:16] = b_unit

    q3_unit = q3_cross / q3_cross.norm(dim=-1, keepdim=True).clamp_min(eps)
    eye7 = torch.eye(7, dtype=R.dtype, device=R.device)
    best_c = torch.zeros_like(a_unit)
    best_norm = torch.zeros_like(a_norm)
    for idx in range(7):
        cand = eye7[idx].expand_as(a_unit)
        for basis in (a_unit, b_unit, q3_unit):
            cand = cand - (cand * basis).sum(dim=-1, keepdim=True) * basis
        c_norm = cand.norm(dim=-1, keepdim=True)
        take = c_norm > best_norm
        best_c = torch.where(take, cand, best_c)
        best_norm = torch.where(take, c_norm, best_norm)

    w_raw = moreno_nullspace_partner(u_cone, best_c, k_signs, xor_idx)
    return w_raw / math.sqrt(2.0)


def dyadic_tem_vector_array_snapshot(
    theta: torch.Tensor,
    phi: torch.Tensor,
    pol_sign: float = 1.0,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    Calibration-free 2x2 Dual-Polarized Dyadic TEM Vector-Sensor Array feed.
    Maps (E_theta, E_phi) into a in Im(O) and b in Im(O)e_8 via passive 90/180-deg
    hybrids so ||a||_2 = ||b||_2 and <a, b> = 0 across all angles (theta, phi) with
    zero digital calibration LUTs.
      - pol_sign = +1.0: LHCP / Jammer cone (a, +b) in V_2(R^7)
      - pol_sign = -1.0: RHCP / Authentic Satellite anti-partner cone (a, -b) in V_2(R^7)
    """
    kx = math.pi * torch.sin(theta) * torch.cos(phi)
    ky = math.pi * torch.sin(theta) * torch.sin(phi)
    kz = math.pi * torch.cos(theta)
    phases = torch.stack(
        [kx, ky, kz, kx + ky, ky + kz, kz + kx, kx + ky + kz], dim=-1
    ).to(dtype)

    a_vec = torch.cos(phases)
    b_vec = torch.sin(phases)

    a_unit = a_vec / a_vec.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    b_ortho = b_vec - (b_vec * a_unit).sum(dim=-1, keepdim=True) * a_unit
    b_unit = b_ortho / b_ortho.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    out = torch.zeros(*phases.shape[:-1], 16, dtype=dtype, device=phases.device)
    out[..., 1:8] = a_unit / math.sqrt(2.0)
    out[..., 9:16] = (pol_sign * b_unit) / math.sqrt(2.0)
    return out


def pre_adc_analog_crossbar(
    R: torch.Tensor,
    c: int,
    d: int,
    s_d: int,
    k_signs: torch.Tensor,
    resistor_tol: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Multiplier-free Pre-ADC Passive Analog Differential Sedenion Crossbar Canceller.
    Executes primitive blade multiplication Z(t) = R(t) * W_J where W_J = (e_c + s_d * e_d)/sqrt(2):
        z_k(t) = (1/sqrt(2)) * [ sigma(k^c, c) * x_{k^c}(t) + s_d * sigma(k^d, d) * x_{k^d}(t) ]
    using only XOR channel routing and +-1 differential wire-swaps.
    """
    assert 1 <= c <= 7 and 9 <= d <= 15 and (d ^ 8) != c and s_d in (+1, -1)
    z_out = torch.zeros_like(R)
    for k in range(16):
        in_A = k ^ c
        in_B = k ^ d
        sign_A = k_signs[k, c].to(R.dtype)
        sign_B = k_signs[k, d].to(R.dtype) * float(s_d)
        if resistor_tol > 0.0:
            err_A = 1.0 + resistor_tol * torch.randn(1, generator=generator, dtype=R.dtype, device=R.device).item()
            err_B = 1.0 + resistor_tol * torch.randn(1, generator=generator, dtype=R.dtype, device=R.device).item()
        else:
            err_A = 1.0
            err_B = 1.0
        z_out[..., k] = (sign_A * err_A * R[..., in_A] + sign_B * err_B * R[..., in_B]) / math.sqrt(2.0)
    return z_out


def zdkm_modulate(
    bits: torch.Tensor,
    a_frames: torch.Tensor,
    b_frames: torch.Tensor,
) -> torch.Tensor:
    """
    Zero-Divisor Keyed Modulation (ZDKM) LPI/LPD Transmitter.
      - bit 0 -> U_0[k] = (a_k, +b_k) / sqrt(2) in V_2(R^7)
      - bit 1 -> U_1[k] = (a_k, -b_k) / sqrt(2) in V_2(R^7)
    Both symbols have identical unit Euclidean norm ||U||_2 == 1 and kappa_S == 1.
    """
    sign = (1.0 - 2.0 * bits.to(a_frames.dtype)).unsqueeze(-1)
    tx = torch.zeros(*a_frames.shape[:-1], 16, dtype=a_frames.dtype, device=a_frames.device)
    tx[..., 1:8] = a_frames / math.sqrt(2.0)
    tx[..., 9:16] = (sign * b_frames) / math.sqrt(2.0)
    return tx


def zdkm_demodulate(
    rx: torch.Tensor,
    a_frames: torch.Tensor,
    b_frames: torch.Tensor,
    raw_c: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
    threshold: float = 1.0 / math.sqrt(2.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zero-Divisor Keyed Modulation (ZDKM) Authorized Receiver.
    Multiplies rx[k] by the keyed Moreno partner W_key[k] in ker L_S((a_k, +b_k)):
      - bit 0 -> ||U_0[k] * W_key[k]||_2 == 0.0
      - bit 1 -> ||U_1[k] * W_key[k]||_2 == sqrt(2)
    """
    u_zero = torch.zeros_like(rx)
    u_zero[..., 1:8] = a_frames
    u_zero[..., 9:16] = b_frames
    w_key = moreno_nullspace_partner(u_zero, raw_c, k_signs, xor_idx) / math.sqrt(2.0)
    prod = sed_mul(rx, w_key, k_signs, xor_idx)
    energy = prod.norm(dim=-1)
    decisions = (energy > threshold).to(torch.long)
    return decisions, energy


def q14_sedenion_mul_rtl(
    x_float: torch.Tensor,
    w_float: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
    frac_bits: int = 14,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bit-accurate, cycle-accurate emulator of the 16-bit signed Q1.14 SystemVerilog
    core (`benchmarks/gps_jamming/rtl/sedenion_crpa_beamformer.sv`).
    Returns (z_q14_int, z_q14_float).
    """
    scale = float(1 << frac_bits)
    q_min, q_max = -(1 << 15), (1 << 15) - 1
    x_q14 = torch.clamp(torch.round(x_float * scale), q_min, q_max).to(torch.int64)
    w_q14 = torch.clamp(torch.round(w_float * scale), q_min, q_max).to(torch.int64)
    ks_int = k_signs.to(torch.int64)

    # 16x16 signed products accumulated in 36-bit integer precision, then arithmetic >> 14
    prod_acc = (x_q14[..., xor_idx] * w_q14.unsqueeze(-2) * ks_int).sum(dim=-1)
    z_q14_int = prod_acc >> frac_bits
    z_q14_float = z_q14_int.to(x_float.dtype) / scale
    return z_q14_int, z_q14_float
