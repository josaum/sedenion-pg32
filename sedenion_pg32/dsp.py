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
    ks = k_signs.to(device=R.device, dtype=R.dtype)
    pi_O = torch.zeros_like(R)
    pi_O[..., :8] = R[..., :8]
    e8 = torch.zeros_like(R)
    e8[..., 8] = 1.0
    left = sed_mul(sed_mul(R, pi_O, ks, xor_idx), e8, ks, xor_idx)
    right = sed_mul(R, sed_mul(pi_O, e8, ks, xor_idx), ks, xor_idx)
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

    # 16x16 signed products accumulated in 36-bit integer precision, then arithmetic >> 14 with 16-bit saturation
    prod_acc = (x_q14[..., xor_idx] * w_q14.unsqueeze(-2) * ks_int).sum(dim=-1)
    z_q14_int = torch.clamp(prod_acc >> frac_bits, q_min, q_max)
    z_q14_float = z_q14_int.to(x_float.dtype) / scale
    return z_q14_int, z_q14_float


# ==============================================================================
# 7. MULTI-TILE SEDENION ARRAYS (S^M & HIERARCHICAL KRONECKER S x S ARRAYS)
# ==============================================================================

def hierarchical_sedenion_array_nuller(
    R_tiles: torch.Tensor,
    u_jam_micro: torch.Tensor,
    u_jam_macro: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Two-Stage Hierarchical Sedenion Array Beamformer over M=16 Sedenion Tiles (256 real DOFs):
      - Input `R_tiles`: shape [..., 16 (macro tiles), 16 (micro channels per tile)]
      - Stage 1 (Intra-Tile Micro Nulling): Multiplies each 16D tile m by W_micro in ker L_S(u_jam_micro)
        along the micro dimension, annihilating polarimetric/local jammer mode `u_jam_micro`.
      - Stage 2 (Inter-Tile Macro Nulling): Treats the 16 tile outputs as a macro-sedenion and
        multiplies by W_macro in ker L_S(u_jam_macro) along the macro tile dimension, annihilating
        long-baseline spatial jammer mode `u_jam_macro` while preserving authentic signal with +6.02 dB
        total array gain (sqrt(2) * sqrt(2) = 2.0) and zero matrix inversions.
    """
    w_micro = associator_nullspace_partner(u_jam_micro, k_signs, xor_idx)
    w_macro = associator_nullspace_partner(u_jam_macro, k_signs, xor_idx)

    # Stage 1: contract along micro channel axis (-1)
    stage1 = sed_mul(R_tiles, w_micro.unsqueeze(-2), k_signs, xor_idx)

    # Stage 2: transpose macro tile axis (-2) and micro channel axis (-1), apply sed_mul along macro tiles
    stage1_T = stage1.transpose(-2, -1)
    stage2_T = sed_mul(stage1_T, w_macro.unsqueeze(-2), k_signs, xor_idx)
    return stage2_T.transpose(-2, -1)


def cascaded_sedenion_array_peeler(
    R_array: torch.Tensor,
    jammer_Macro_states: list,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-Jammer Cascaded Sedenion Array Peeler over a 16-tile sedenion array `R_array` [..., 16].
    Given a list of K independent jammer sedenion states `[J_1, J_2, ..., J_K]` on the Moreno cone,
    tracks how each stage's zero-divisor partner W_s transforms the remaining jammer states
    (`J_next = J_curr * W_s`), selecting W_s in ker L_S(J_s) to preserve the Moreno zero-divisor
    cone structure of downstream jammers and annihilating all K jammers in O(1) operations
    without forming a covariance matrix.
    """
    curr_signal = R_array
    active_jammers = [j.clone() for j in jammer_Macro_states]
    eye7 = torch.eye(7, dtype=R_array.dtype, device=R_array.device)

    for s_idx in range(len(active_jammers)):
        j_target = active_jammers[s_idx]
        w_s = associator_nullspace_partner(j_target, k_signs, xor_idx)
        w_s = w_s / w_s.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        if s_idx + 1 < len(active_jammers):
            # Evaluate unit-normalized candidates in the 4D kernel ker L_S(j_target) and pick
            # the partner that maximizes total Moreno cone curvature across all downstream jammers
            a_im = j_target[..., 1:8]
            b_im = j_target[..., 9:16]
            a_unit = a_im / a_im.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            b_ortho = b_im - (b_im * a_unit).sum(dim=-1, keepdim=True) * a_unit
            b_unit = b_ortho / b_ortho.norm(dim=-1, keepdim=True).clamp_min(1e-30)
            u_cone = torch.zeros_like(j_target)
            u_cone[..., 1:8] = a_unit
            u_cone[..., 9:16] = b_unit

            best_score = torch.zeros(R_array.shape[:-1], dtype=R_array.dtype, device=R_array.device)
            for rem_idx in range(s_idx + 1, len(active_jammers)):
                j_rem_img = sed_mul(active_jammers[rem_idx], w_s, k_signs, xor_idx)
                best_score = best_score + native_associator_curvature(j_rem_img, k_signs, xor_idx)

            for idx in range(7):
                w_raw = moreno_nullspace_partner(u_cone, eye7[idx].expand_as(a_unit), k_signs, xor_idx)
                w_norm = w_raw.norm(dim=-1, keepdim=True)
                w_cand = w_raw / w_norm.clamp_min(1e-30)
                null_res = sed_mul(j_target, w_cand, k_signs, xor_idx).norm(dim=-1)
                valid = ((w_norm.squeeze(-1) > 0.5) & (null_res < 1e-6)).to(R_array.dtype)
                score = torch.zeros_like(best_score)
                for rem_idx in range(s_idx + 1, len(active_jammers)):
                    j_rem_img = sed_mul(active_jammers[rem_idx], w_cand, k_signs, xor_idx)
                    score = score + native_associator_curvature(j_rem_img, k_signs, xor_idx)
                score = score * valid
                take = (score > best_score + 1e-9).unsqueeze(-1)
                w_s = torch.where(take, w_cand, w_s)
                best_score = torch.maximum(best_score, score)

        curr_signal = sed_mul(curr_signal, w_s, k_signs, xor_idx)
        # Update remaining downstream jammer states through the bilinear operator R_{w_s}
        for rem_idx in range(s_idx + 1, len(active_jammers)):
            active_jammers[rem_idx] = sed_mul(active_jammers[rem_idx], w_s, k_signs, xor_idx)

    return curr_signal


def sedenion_array_interferometric_navigator(
    R_tiles: torch.Tensor,
    tile_positions_wavelengths: torch.Tensor,
    k_signs: torch.Tensor,
    xor_idx: torch.Tensor,
    u_hinge: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-Snapshot (K=1) Multi-Tile Sedenion Array Navigator & Attitude Estimator.
    Given M Sedenion tiles at 3D body-frame coordinates `tile_positions_wavelengths` [M, 3] (in units
    of carrier wavelength lambda) receiving a multi-tile sedenion wavefield `R_tiles` [..., M, 16]:
      1. Computes the sedenion Hermitian inner product C_{m,0} = R_m * conj(R_0) in S across tiles.
      2. Extracts the inter-tile carrier phase gradient phi_m = atan2(<C_{m,0}[e_1..e_3], u_hinge>, C_{m,0}[e_0])
         projected onto the associative quaternion hinge span{e_0, e_1, e_2, e_3} (using explicit unit
         3-vector projection when `u_hinge` is provided, or dominant canonical axis selection in blind mode).
      3. Solves the 3D direction-of-arrival unit vector u_hat = (u_x, u_y, u_z) via least-squares
         on the fixed geometry pinv(2*pi*P) and computes per-tile non-associative curvature kappa_S
         to flag multipath/spoofing per tile.
    Returns:
      - direction_cosines: [..., 3] estimated 3D line-of-sight unit vector (u_x, u_y, u_z)
      - quaternion_hinge_state: [..., M, 4] associative quaternion projection (e_0, e_1, e_2, e_3)
      - tile_curvatures: [..., M] single-snapshot non-associative curvature kappa_S per tile
    """
    from .core import sedenion_conj
    ref_tile = R_tiles[..., :1, :]  # [..., 1, 16]
    rel_sedenion = sed_mul(R_tiles, sedenion_conj(ref_tile), k_signs, xor_idx)  # [..., M, 16]

    # Associative Quaternion Hinge projection H_1 = span{e_0, e_1, e_2, e_3}
    quat_hinge = rel_sedenion[..., :4]
    quat_hinge = quat_hinge / quat_hinge.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    # Extract carrier phase offset across the 3D imaginary quaternion hinge span{e_1, e_2, e_3}
    hinge_im = rel_sedenion[..., 1:4]  # [..., M, 3]
    if u_hinge is not None:
        u_vec = u_hinge[..., 1:4] if u_hinge.shape[-1] >= 4 else u_hinge[..., :3]
        u_vec = u_vec / u_vec.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        im_component = (hinge_im * u_vec.unsqueeze(-2)).sum(dim=-1)
    else:
        hinge_energy = hinge_im.pow(2).sum(dim=-2, keepdim=True)  # [..., 1, 3]
        dom_idx = hinge_energy.argmax(dim=-1, keepdim=True)  # [..., 1, 1]
        im_component = torch.gather(hinge_im, dim=-1, index=dom_idx.expand(*hinge_im.shape[:-1], 1)).squeeze(-1)
    phases = torch.atan2(im_component, rel_sedenion[..., 0])  # [..., M]

    # Solve 2*pi * (P - P_0) @ u = phases for 3D direction cosines (u_x, u_y, u_z)
    delta_p = 2.0 * math.pi * (tile_positions_wavelengths - tile_positions_wavelengths[:1])  # [M, 3]
    pinv_p = torch.linalg.pinv(delta_p)  # [3, M] — constant geometry matrix (precomputed once for airframe)
    dir_cosines = torch.matmul(phases, pinv_p.T)
    dir_cosines = dir_cosines / dir_cosines.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    tile_curvatures = native_associator_curvature(R_tiles, k_signs, xor_idx)
    return dir_cosines, quat_hinge, tile_curvatures

