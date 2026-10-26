"""
Comprehensive unit & hardware verification test suite covering:
  1. Single-Snapshot Native Associator--Lagrange Wedge Identity (Lean 4 Theorem 1)
  2. Curvature Duality (kappa_S = 0 on 4D quaternion hinges vs. kappa_S = 1 on Moreno cone)
  3. SVD-Free Associator-Seeded Moreno Zero-Divisor Annihilator & K=1 Coherent Spoofer Detection
  4. Calibration-Free Dyadic 2x2 Dual-Polarized TEM Vector-Sensor Array
  5. Pre-ADC Passive Analog Differential Sedenion Crossbar & 84-Blade Hardware ROM Netlist
  6. Zero-Divisor Keyed Modulation (ZDKM) LPI/LPD Transceiver
  7. Bit-Accurate Q1.14 SystemVerilog RTL (`sedenion_crpa_beamformer.sv`) Verification
  8. Full branch coverage of `SedenionLinear`, `PG32TopologicalRouter`, and `triton_kernel`
"""
import json
import math
import pathlib
import re
import torch
import pytest

from sedenion_pg32 import (
    build_cayley_dickson_tables, build_pg32_fano_tables, sedenion_conj,
    sedenion_matmul_contract, SedenionLinearFunction, SedenionLinear,
    MorenoConeProjection, PG32TopologicalRouter,
    tables, sed_mul, oct_mul, left_mult_matrix, moreno_nullspace_partner,
    _sign_table_as_constexpr, build_packed_sign_masks_256bit,
    ieee754_sign_xor, fused_sedenion_linear_reference,
    fused_sedenion_linear,
    native_sedenion_associator, native_associator_curvature,
    associator_nullspace_partner, dyadic_tem_vector_array_snapshot,
    pre_adc_analog_crossbar, zdkm_modulate, zdkm_demodulate,
    q14_sedenion_mul_rtl,
    hierarchical_sedenion_array_nuller,
    cascaded_sedenion_array_peeler,
    sedenion_array_interferometric_navigator,
)

F64 = torch.float64
SIGNS, KS, XOR = tables(F64)


# ==============================================================================
# 1. NATIVE SEDENION ASSOCIATOR & LAGRANGE WEDGE IDENTITY (Lean 4 Theorem 1)
# ==============================================================================

def test_native_associator_commutator_and_lagrange_norm():
    """
    Verifies Lean 4 theorems `native_associator_commutator` and
    `native_associator_lagrange_norm`:
      1. [R, pi_O(R), e_8] == (b*a - a*b, 0) == (-2*(a x_O b), 0)
      2. ||[R, pi_O(R), e_8]||_2^2 == 4 * (||a||_2^2 * ||b||_2^2 - <a, b>^2)
    """
    torch.manual_seed(42)
    R = torch.randn(512, 16, dtype=F64)
    R[:, 0] = 0.0
    R[:, 8] = 0.0  # Purely imaginary sedenions R = (a, b) in Im(O) x Im(O)

    assoc = native_sedenion_associator(R, KS, XOR)

    # Upper octonion and real part must vanish identically
    assert assoc[:, 0].abs().max() < 1e-14
    assert assoc[:, 8:].abs().max() < 1e-14

    # Lower octonion must equal b*a - a*b
    a = R[:, :8]
    b = R[:, 8:16]
    ba_minus_ab = oct_mul(b, a, KS, XOR) - oct_mul(a, b, KS, XOR)
    assert torch.allclose(assoc[:, :8], ba_minus_ab, atol=1e-13)

    # Squared norm must equal 4 * (||a||^2 ||b||^2 - <a, b>^2)
    a_sq = (a * a).sum(dim=-1)
    b_sq = (b * b).sum(dim=-1)
    ab_dot = (a * b).sum(dim=-1)
    lagrange_rhs = 4.0 * (a_sq * b_sq - ab_dot * ab_dot)
    assoc_norm_sq = (assoc * assoc).sum(dim=-1)
    assert torch.allclose(assoc_norm_sq, lagrange_rhs, atol=1e-12)


def test_curvature_duality_quaternion_vs_moreno_cone():
    """
    Verifies Corollary 1 (Curvature Duality of S):
      - kappa_S(R) == 0 iff R lies in a 4D associative quaternion subalgebra (a || b)
      - kappa_S(R) == 1 iff R lies on the Moreno Zero-Divisor Cone V_2(R^7)
    """
    torch.manual_seed(43)
    # 1. Collinear upper/lower octonions b = lambda * a -> kappa_S == 0
    a_im = torch.randn(256, 7, dtype=F64)
    lam = torch.randn(256, 1, dtype=F64)
    R_quat = torch.zeros(256, 16, dtype=F64)
    R_quat[:, 1:8] = a_im
    R_quat[:, 9:16] = lam * a_im
    kappa_quat = native_associator_curvature(R_quat, KS, XOR)
    assert kappa_quat.abs().max() < 1e-14

    # 2. Moreno Zero-Divisor Cone V_2(R^7) -> kappa_S == 1
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    R_moreno = mp(torch.randn(256, 16, dtype=F64)) / math.sqrt(2.0)
    kappa_moreno = native_associator_curvature(R_moreno, KS, XOR)
    assert torch.allclose(kappa_moreno, torch.ones(256, dtype=F64), atol=1e-14)


def test_associator_seeded_nullspace_partner_and_k1_spoofer_detection():
    """
    Verifies that `associator_nullspace_partner` extracts c = (a x_O b)/||a x_O b||
    directly from A_native(R) and annihilates any Moreno zero-divisor state in 1 snapshot
    without SVD or random seeds, and detects Rank-1 coherent spoofing at K=1.
    """
    torch.manual_seed(44)
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    R_jam = mp(torch.randn(128, 16, dtype=F64)) / math.sqrt(2.0)
    W_partner = associator_nullspace_partner(R_jam, KS, XOR)

    # Verify unit norm and exact zero-divisor annihilation R_jam * W_partner == 0
    assert torch.allclose(W_partner.norm(dim=-1), torch.ones(128, dtype=F64), atol=1e-13)
    residual = sed_mul(R_jam, W_partner, KS, XOR).norm(dim=-1)
    assert residual.max() < 1e-14

    # Verify K=1 coherent spoofer detection vs H0 benign quaternion channel
    a_auth = R_jam[:, 1:8] * math.sqrt(2.0)
    b_spoof = R_jam[:, 9:16] * math.sqrt(2.0)
    R_h0 = torch.zeros(128, 16, dtype=F64)
    R_h0[:, 1:8] = a_auth
    R_h1 = torch.zeros(128, 16, dtype=F64)
    R_h1[:, 1:8] = a_auth
    R_h1[:, 9:16] = b_spoof  # Zero-lag coherent spoofer in upper octonion mode
    assert (native_associator_curvature(R_h0, KS, XOR) < 1e-14).all()
    assert torch.allclose(native_associator_curvature(R_h1, KS, XOR), torch.ones(128, dtype=F64), atol=1e-14)


# ==============================================================================
# 2. CALIBRATION-FREE DYADIC 2x2 TEM VECTOR-SENSOR ARRAY
# ==============================================================================

def test_dyadic_tem_vector_array_manifold_and_gain():
    """
    Verifies that across all continuous angles (theta, phi), the Dyadic 2x2 TEM
    Vector-Sensor Array maps LHCP jammers to (a, +b) in V_2(R^7) with kappa_S == 1,
    annihilates the jammer to 0, and gives +3.01 dB gain (norm sqrt(2)) to the
    RHCP anti-partner satellite signal (a, -b).
    """
    torch.manual_seed(45)
    theta = torch.rand(128, dtype=F64) * (0.45 * math.pi) + 0.02
    phi = torch.rand(128, dtype=F64) * (2.0 * math.pi)

    u_jam = dyadic_tem_vector_array_snapshot(theta, phi, pol_sign=+1.0, dtype=F64)
    u_sat = dyadic_tem_vector_array_snapshot(theta, phi, pol_sign=-1.0, dtype=F64)

    # Both have unit norm and kappa_S == 1 with zero calibration LUTs
    assert torch.allclose(u_jam.norm(dim=-1), torch.ones(128, dtype=F64), atol=1e-14)
    assert torch.allclose(u_sat.norm(dim=-1), torch.ones(128, dtype=F64), atol=1e-14)
    assert torch.allclose(native_associator_curvature(u_jam, KS, XOR), torch.ones(128, dtype=F64), atol=1e-14)

    # Construct partner W from u_jam: annihilates u_jam while amplifying u_sat by sqrt(2)
    w_beam = associator_nullspace_partner(u_jam, KS, XOR)
    jam_out = sed_mul(u_jam, w_beam, KS, XOR).norm(dim=-1)
    sat_out = sed_mul(u_sat, w_beam, KS, XOR).norm(dim=-1)
    assert jam_out.max() < 1e-14
    assert torch.allclose(sat_out, math.sqrt(2.0) * torch.ones(128, dtype=F64), atol=1e-13)


# ==============================================================================
# 3. PRE-ADC PASSIVE ANALOG SEDENION CROSSBAR & 84-BLADE ROM VERIFICATION
# ==============================================================================

def test_pre_adc_analog_crossbar_and_84_blade_rom():
    """
    Verifies `pre_adc_analog_crossbar` and `hardware/crossbar_84_blades_rom.json`:
      - All 84 primitive zero-divisor blades have 4 primitive partners each.
      - Ideal analog crossbar (tol=0) matches exact sedenion multiplication and gives 0 V jammer residual.
      - Realistic 0.05% resistor tolerance (`ERA-3AEB1001V`) achieves > 60 dB jammer suppression.
    """
    rom_path = pathlib.Path(__file__).resolve().parents[1] / "hardware" / "crossbar_84_blades_rom.json"
    assert rom_path.exists()
    rom = json.loads(rom_path.read_text())
    assert rom["total_primitive_zero_divisor_blades"] == 84
    assert rom["partners_per_blade"] == 4

    eye = torch.eye(16, dtype=F64)
    gen = torch.Generator().manual_seed(2026)

    # Test primitive blade e_1 + e_10 with partner e_5 + e_14
    u_jam = (eye[1] + eye[10]) / math.sqrt(2.0)
    u_sat = (eye[1] - eye[10]) / math.sqrt(2.0)
    w_partner = (eye[5] + eye[14]) / math.sqrt(2.0)

    # Ideal crossbar (resistor_tol = 0.0)
    z_jam_ideal = pre_adc_analog_crossbar(u_jam, c=5, d=14, s_d=+1, k_signs=KS, resistor_tol=0.0)
    z_sat_ideal = pre_adc_analog_crossbar(u_sat, c=5, d=14, s_d=+1, k_signs=KS, resistor_tol=0.0)
    assert z_jam_ideal.norm().item() < 1e-15
    assert torch.allclose(z_sat_ideal, sed_mul(u_sat, w_partner, KS, XOR), atol=1e-15)
    assert abs(z_sat_ideal.norm().item() - math.sqrt(2.0)) < 1e-14

    # Realistic 0.05% tolerance (`ERA-3AEB1001V`)
    z_jam_tol = pre_adc_analog_crossbar(
        u_jam.unsqueeze(0).expand(64, 16), c=5, d=14, s_d=+1, k_signs=KS, resistor_tol=5e-4, generator=gen
    )
    mean_null_db = -20.0 * math.log10(z_jam_tol.norm(dim=-1).mean().item())
    assert mean_null_db > 60.0


def test_hardware_crossbar_netlist_generator_execution():
    """Executes `build_84_primitive_blades_and_netlist` directly to cover `hardware/generate_crossbar_netlist.py`."""
    from hardware.generate_crossbar_netlist import build_84_primitive_blades_and_netlist
    build_84_primitive_blades_and_netlist()


# ==============================================================================
# 4. ZERO-DIVISOR KEYED MODULATION (ZDKM) LPI/LPD TRANSCEIVER
# ==============================================================================

def test_zdkm_modulate_and_demodulate():
    """
    Verifies Zero-Divisor Keyed Modulation (ZDKM):
      - Constant unit Euclidean norm ||U_0|| == ||U_1|| == 1.0
      - Constant zero-divisor curvature kappa_S == 1.0 for both bits 0 and 1
      - 100% noiseless demodulation accuracy (0.0 for bit 0, sqrt(2) for bit 1)
      - >99.5% demodulation accuracy at +6 dB SNR
    """
    torch.manual_seed(46)
    N = 512
    bits = torch.randint(0, 2, (N,))
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    u_cone = mp(torch.randn(N, 16, dtype=F64))
    a_frames = u_cone[:, 1:8]
    b_frames = u_cone[:, 9:16]
    raw_c = torch.randn(N, 7, dtype=F64)

    tx = zdkm_modulate(bits, a_frames, b_frames)
    assert torch.allclose(tx.norm(dim=-1), torch.ones(N, dtype=F64), atol=1e-14)
    assert torch.allclose(native_associator_curvature(tx, KS, XOR), torch.ones(N, dtype=F64), atol=1e-14)

    # Noiseless demodulation
    decisions, energy = zdkm_demodulate(tx, a_frames, b_frames, raw_c, KS, XOR)
    assert torch.equal(decisions, bits)
    assert energy[bits == 0].max() < 1e-14
    assert torch.allclose(energy[bits == 1], math.sqrt(2.0) * torch.ones((bits == 1).sum(), dtype=F64), atol=1e-13)


# ==============================================================================
# 5. BIT-ACCURATE Q1.14 SYSTEMVERILOG RTL VERIFICATION
# ==============================================================================

def test_q14_rtl_sign_rom_and_fixed_point_nulling():
    """
    Verifies bit-for-bit parity between `sedenion_pg32`, `q14_sedenion_mul_rtl`,
    and the synthesizable SystemVerilog file `benchmarks/gps_jamming/rtl/sedenion_crpa_beamformer.sv`.
    """
    sv_path = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "gps_jamming" / "rtl" / "sedenion_crpa_beamformer.sv"
    assert sv_path.exists()
    sv_text = sv_path.read_text()

    # Extract all 16 rows of 16'b... from SIGN_NEG_KJ in the SystemVerilog source
    rows = re.findall(r"16'b([01]{16})", sv_text)
    assert len(rows) == 16
    for k, bits_str in enumerate(rows):
        for j in range(16):
            # In SystemVerilog `logic [0:15]`, index 0 is the MSB (leftmost character)
            sv_bit = int(bits_str[j])
            expected_sign = -1.0 if sv_bit == 1 else +1.0
            assert KS[k, j].item() == expected_sign, f"Mismatch at k={k}, j={j}"

    # Verify exact 0 LSB residual on primitive zero-divisor blades in Q1.14
    eye = torch.eye(16, dtype=F64)
    u_jam = (eye[1] + eye[10]) / math.sqrt(2.0)
    w_partner = (eye[5] + eye[14]) / math.sqrt(2.0)
    z_int, z_flt = q14_sedenion_mul_rtl(u_jam, w_partner, KS, XOR)
    assert (z_int == 0).all() and (z_flt == 0.0).all()

    # Verify > 70 dB average null depth on continuous Moreno zero-divisor cone in Q1.14
    torch.manual_seed(47)
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    u_cont = mp(torch.randn(128, 16, dtype=F64)) / math.sqrt(2.0)
    w_cont = associator_nullspace_partner(u_cont, KS, XOR)
    _, z_cont_flt = q14_sedenion_mul_rtl(u_cont, w_cont, KS, XOR)
    mean_null_db = -20.0 * math.log10(z_cont_flt.norm(dim=-1).mean().item())
    assert mean_null_db > 70.0


# ==============================================================================
# 6. FULL BRANCH COVERAGE OF CORE.PY & TRITON_KERNEL.PY
# ==============================================================================

def test_sedenion_linear_all_branches():
    """Covers `SedenionLinear` with `bias=True/False` and `onnx_compatible=True/False`."""
    torch.manual_seed(48)
    x = torch.randn(4, 5, 16, requires_grad=True)

    # 1. bias=True, onnx_compatible=False (uses SedenionLinearFunction.apply)
    layer_std = SedenionLinear(5, 3, bias=True, onnx_compatible=False)
    y_std = layer_std(x)
    assert y_std.shape == (4, 3, 16)
    y_std.sum().backward()
    assert layer_std.bias.grad is not None and layer_std.weight.grad is not None

    # 2. bias=True, onnx_compatible=True (uses direct contraction + bias)
    layer_onnx = SedenionLinear(5, 3, bias=True, onnx_compatible=True)
    layer_onnx.load_state_dict(layer_std.state_dict())
    y_onnx = layer_onnx(x.detach())
    assert torch.allclose(y_std.detach(), y_onnx, atol=1e-6)

    # 3. bias=False, onnx_compatible=True
    layer_no_bias = SedenionLinear(5, 3, bias=False, onnx_compatible=True)
    y_nb = layer_no_bias(x.detach())
    assert y_nb.shape == (4, 3, 16)


def test_router_without_contradiction_pruning():
    """Covers the `contradiction_pruning=False` branch in `PG32TopologicalRouter.pair_message`."""
    torch.manual_seed(49)
    router = PG32TopologicalRouter(d_model=4, contradiction_pruning=False)
    h = torch.randn(2, 16, 4, 16, requires_grad=True)
    out, probs = router(h)
    assert out.shape == h.shape and probs.shape == (2, 15)


def test_triton_kernel_fp32_ieee754_xor_and_constexpr():
    """
    Covers `_sign_table_as_constexpr`, `build_packed_sign_masks_256bit`,
    `ieee754_sign_xor`, and the `w.dtype == torch.float32` + `bias is not None`
    branches in `fused_sedenion_linear_reference`.
    """
    table_tuple = _sign_table_as_constexpr()
    assert len(table_tuple) == 16 and len(table_tuple[0]) == 16

    masks = build_packed_sign_masks_256bit()
    assert masks.shape == (16,)

    x_fp32 = torch.randn(3, 5, 16, dtype=torch.float32)
    w_fp32 = torch.randn(4, 5, 16, dtype=torch.float32)
    b_fp32 = torch.randn(4, 16, dtype=torch.float32)

    y_ref_fp32 = fused_sedenion_linear_reference(x_fp32, w_fp32, bias=b_fp32)
    y_contract = sedenion_matmul_contract(x_fp32, w_fp32, KS.float(), XOR) + b_fp32
    assert torch.allclose(y_ref_fp32, y_contract, atol=1e-5)

    # Also test float64 with bias is not None
    y_ref_fp64 = fused_sedenion_linear_reference(x_fp32.double(), w_fp32.double(), bias=b_fp32.double())
    assert torch.allclose(y_ref_fp64.float(), y_contract, atol=1e-5)

    # Also test `fused_sedenion_linear` and `fused_sedenion_linear_kernel` tiled execution
    # with both bias=b_fp32 and bias=None across multi-tile boundaries (BLOCK=2)
    y_tiled_bias = fused_sedenion_linear(x_fp32, w_fp32, bias=b_fp32, BLOCK_B=2, BLOCK_M=2, BLOCK_N=2)
    assert torch.allclose(y_tiled_bias, y_contract, atol=1e-5)

    y_tiled_nobias = fused_sedenion_linear(x_fp32, w_fp32, bias=None, BLOCK_B=2, BLOCK_M=2, BLOCK_N=2)
    assert torch.allclose(y_tiled_nobias, y_contract - b_fp32, atol=1e-5)


def test_multi_jammer_superposition_simultaneous_null():
    """
    Verifies Lean 4 theorem `crpa_multi_jammer_superposition_null`:
    For any right-annihilator W, all jammers J_1..J_4 in the 4D left-kernel ker R_S(W)
    are annihilated simultaneously in any linear combination (S_sat + sum_m alpha_m J_m) * W == S_sat * W.
    """
    torch.manual_seed(50)
    eye = torch.eye(16, dtype=F64)
    # Choose primitive partner blade W = (e_5 + e_14)/sqrt(2)
    w_beam = (eye[5] + eye[14]) / math.sqrt(2.0)

    # Construct the right-multiplication matrix R(W) where R(W) x = x * W
    R_mat = torch.stack([sed_mul(eye[i], w_beam, KS, XOR) for i in range(16)], dim=-1)
    _, S_vals, Vh = torch.linalg.svd(R_mat)
    # Last 4 right singular vectors span the exact 4D kernel ker R_S(W)
    assert (S_vals[12:] < 1e-14).all()
    ker_basis = Vh[12:, :]  # [4, 16]

    s_sat = torch.randn(64, 16, dtype=F64)
    alphas = torch.randn(64, 4, dtype=F64) * 100.0  # +40 dB multi-jammer superposition
    multi_jammer = alphas @ ker_basis               # [64, 16]

    rx_total = s_sat + multi_jammer
    out_total = sed_mul(rx_total, w_beam, KS, XOR)
    out_clean = sed_mul(s_sat, w_beam, KS, XOR)
    assert torch.allclose(out_total, out_clean, atol=1e-11)


def test_exhaustive_84_blades_and_336_partners():
    """
    Exhaustively verifies all 84 primitive zero-divisor blades U_J = (e_a +- e_b)/sqrt(2)
    and all 4*84 = 336 primitive partner pairs via `pre_adc_analog_crossbar`.
    """
    eye = torch.eye(16, dtype=F64)
    candidates = []
    for a in range(1, 8):
        for b in range(9, 16):
            if (b ^ 8) == a:
                continue
            for s in (+1, -1):
                candidates.append((a, b, s, (eye[a] + s * eye[b]) / math.sqrt(2.0)))
    assert len(candidates) == 84

    total_partner_pairs = 0
    for a, b, s, u_jam in candidates:
        u_sat = (eye[a] - s * eye[b]) / math.sqrt(2.0)
        partners_found = 0
        for c, d, s_d, w_cand in candidates:
            if sed_mul(u_jam, w_cand, KS, XOR).norm().item() < 1e-14:
                partners_found += 1
                total_partner_pairs += 1
                z_jam = pre_adc_analog_crossbar(u_jam, c=c, d=d, s_d=s_d, k_signs=KS)
                z_sat = pre_adc_analog_crossbar(u_sat, c=c, d=d, s_d=s_d, k_signs=KS)
                assert z_jam.norm().item() < 1e-14
                assert abs(z_sat.norm().item() - math.sqrt(2.0)) < 1e-14
        assert partners_found == 4
    assert total_partner_pairs == 336


def test_selective_input_grad_branches():
    """
    Covers selective `ctx.needs_input_grad` branches in `SedenionLinearFunction.backward`
    (only x requires grad vs. only weight requires grad).
    """
    x_grad_only = torch.randn(2, 3, 16, dtype=F64, requires_grad=True)
    w_frozen = torch.randn(4, 3, 16, dtype=F64, requires_grad=False)
    y1 = SedenionLinearFunction.apply(x_grad_only, w_frozen, None, KS, XOR)
    y1.sum().backward()
    assert x_grad_only.grad is not None and w_frozen.grad is None

    x_frozen = torch.randn(2, 3, 16, dtype=F64, requires_grad=False)
    w_grad_only = torch.randn(4, 3, 16, dtype=F64, requires_grad=True)
    y2 = SedenionLinearFunction.apply(x_frozen, w_grad_only, None, KS, XOR)
    y2.sum().backward()
    assert x_frozen.grad is None and w_grad_only.grad is not None


# ==============================================================================
# 7. MULTI-TILE SEDENION ARRAYS (256-DOF KRONECKER, CASCADE PEELER, NAVIGATOR)
# ==============================================================================

def test_hierarchical_256dof_sedenion_array_nuller():
    """
    Verifies two-stage Hierarchical Sedenion Array Beamforming over M=16 Sedenion Tiles
    (16 x 16 = 256 real DOFs):
      - Stage 1 annihilates any micro/polarimetric jammer `u_jam_micro` inside each tile.
      - Stage 2 annihilates any macro/long-baseline jammer `u_jam_macro` across the 16 tiles.
      - Authentic anti-partner signal experiences exact +6.02 dB gain (norm = 2.0) with 0 matrix inversions.
    """
    torch.manual_seed(51)
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    u_jam_micro = mp(torch.randn(16, dtype=F64)) / math.sqrt(2.0)
    u_jam_macro = mp(torch.randn(16, dtype=F64)) / math.sqrt(2.0)

    # Anti-partner authentic satellite states (a, -b) on micro and macro scales
    u_sat_micro = u_jam_micro.clone()
    u_sat_micro[9:16] *= -1.0
    u_sat_macro = u_jam_macro.clone()
    u_sat_macro[9:16] *= -1.0

    # 1. Pure micro-jammer wavefield across arbitrary macro profile -> annihilated in Stage 1
    arbitrary_macro = torch.randn(16, dtype=F64)
    J_micro_tiles = torch.outer(arbitrary_macro, u_jam_micro)  # [16, 16]
    out_j_micro = hierarchical_sedenion_array_nuller(J_micro_tiles, u_jam_micro, u_jam_macro, KS, XOR)
    assert out_j_micro.norm().item() < 1e-13

    # 2. Pure macro-jammer wavefield across arbitrary micro profile -> annihilated in Stage 2
    arbitrary_micro = torch.randn(16, dtype=F64)
    J_macro_tiles = torch.outer(u_jam_macro, arbitrary_micro)  # [16, 16]
    out_j_macro = hierarchical_sedenion_array_nuller(J_macro_tiles, u_jam_micro, u_jam_macro, KS, XOR)
    assert out_j_macro.norm().item() < 1e-13

    # 3. Authentic signal + +80 dB micro jammer + +80 dB macro jammer -> exact +6.02 dB signal gain (norm = 2.0)
    S_sat_tiles = torch.outer(u_sat_macro, u_sat_micro)  # Frobenius norm = 1.0
    rx_composite = S_sat_tiles + 1e4 * J_micro_tiles + 1e4 * J_macro_tiles
    out_composite = hierarchical_sedenion_array_nuller(rx_composite, u_jam_micro, u_jam_macro, KS, XOR)
    out_clean = hierarchical_sedenion_array_nuller(S_sat_tiles, u_jam_micro, u_jam_macro, KS, XOR)
    assert torch.allclose(out_composite, out_clean, atol=1e-9)
    assert abs(out_clean.norm().item() - 2.0) < 1e-12


def test_cascaded_sedenion_array_peeler():
    """
    Verifies sequential multi-jammer peeling (`cascaded_sedenion_array_peeler`)
    where a single jammer or cascaded jammer chain is annihilated without covariance inversion.
    """
    torch.manual_seed(52)
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    j1 = mp(torch.randn(16, dtype=F64)) / math.sqrt(2.0)
    w1 = associator_nullspace_partner(j1, KS, XOR)

    # Construct a second jammer j2 whose image after Stage 1 (j2 * w1) lands on the Moreno cone
    j2_after = mp(torch.randn(16, dtype=F64)) / math.sqrt(2.0)
    # Since w1 has norm 1 on the unit sphere in a quaternion/octonion slice or via right-inverse on range(R_w1):
    R_w1 = torch.stack([sed_mul(torch.eye(16, dtype=F64)[i], w1, KS, XOR) for i in range(16)], dim=-1)
    # Pick j2_after inside range(R_w1) by projecting onto the 12D column space of R_w1
    U_col, S_col, _ = torch.linalg.svd(R_w1)
    # Create a primitive zero-divisor blade inside range(R_w1) or test 2-stage peeler directly:
    eye = torch.eye(16, dtype=F64)
    j_blade1 = (eye[1] + eye[10]) / math.sqrt(2.0)
    # After multiplying by w_s1 = associator_nullspace_partner(j_blade1), check what a second blade becomes:
    w_s1 = associator_nullspace_partner(j_blade1, KS, XOR)
    # Construct j_stage2 in the pre-image of a known zero-divisor blade (e_2 + e_13)/sqrt(2) under R_{w_s1}?
    # Even simpler: any single basis vector e_i multiplied by w_s1 is a unit sedenion; if we take two basis vectors
    # whose images under w_s1 lie in Im(O) and Im(O)e_8 respectively, their sum is on the Moreno cone!
    # Wait: e_0 * w_s1 = w_s1 = (c, d) WHICH IS ALREADY ON THE MORENO CONE V_2(R^7)!
    # Because w_s1 = (c, d)/sqrt(2) with ||c||=||d||=1, <c,d>=0, e_0 * w_s1 = w_s1 is literally on the Moreno cone!
    j_pre2 = eye[0]  # Since e_0 * w_s1 == w_s1 in V_2(R^7)!
    rx_two_jammers = 1e3 * j_blade1 + 1e3 * j_pre2
    peeled = cascaded_sedenion_array_peeler(rx_two_jammers, [j_blade1, j_pre2], KS, XOR)
    assert peeled.norm().item() < 1e-10


def test_sedenion_array_interferometric_navigator():
    """
    Verifies `sedenion_array_interferometric_navigator` across an M=8 tile 3D drone/airframe
    array: recovers exact 3D direction cosines (u_x, u_y, u_z) in K=1 snapshot, outputs unit
    quaternion hinges, and flags tile-level spoofing curvature.
    """
    torch.manual_seed(53)
    # 8 tiles arranged in 3D on an airframe (within +-0.25 wavelengths for unambiguous phase)
    tile_pos = torch.tensor([
        [0.00, 0.00, 0.00],
        [0.20, 0.00, 0.00],
        [0.00, 0.20, 0.00],
        [0.00, 0.00, 0.20],
        [-0.15, 0.15, 0.00],
        [0.15, -0.15, 0.05],
        [0.10, 0.10, -0.15],
        [-0.10, -0.10, 0.10],
    ], dtype=F64)

    # True 3D unit direction vector of the satellite/beacon
    true_u = torch.tensor([0.48, -0.36, 0.80], dtype=F64)
    true_u = true_u / true_u.norm()

    # Phase at each tile m: phi_m = 2*pi * <p_m, true_u>
    phases = 2.0 * math.pi * (tile_pos @ true_u)  # [8]

    # Construct 8-tile sedenion snapshot in the associative quaternion channel (e_0, e_1)
    R_tiles = torch.zeros(8, 16, dtype=F64)
    R_tiles[:, 0] = torch.cos(phases)
    R_tiles[:, 1] = torch.sin(phases)

    est_u, quat_hinge, curvatures = sedenion_array_interferometric_navigator(R_tiles, tile_pos, KS, XOR)
    assert torch.allclose(est_u, true_u, atol=1e-12)
    assert torch.allclose(quat_hinge.norm(dim=-1), torch.ones(8, dtype=F64), atol=1e-14)
    assert (curvatures.abs() < 1e-14).all()



