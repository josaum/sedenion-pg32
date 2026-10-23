"""
Test-suite backing every algebraic claim in the manuscript.
Each test name cites the paper statement it verifies.
"""
import io
import math
import pytest
import torch

from sedenion_pg32 import (
    build_cayley_dickson_tables, build_pg32_fano_tables, sedenion_conj,
    sedenion_matmul_contract, SedenionLinearFunction, SedenionLinear,
    MorenoConeProjection, tables, sed_mul, oct_mul, left_mult_matrix,
    moreno_nullspace_partner,
)
from sedenion_pg32.triton_kernel import fused_sedenion_linear_reference

torch.manual_seed(0)
F64 = torch.float64
SIGNS, KS, XOR = tables(F64)
E = torch.eye(16, dtype=F64)


def mul(u, v):
    return sed_mul(u, v, KS, XOR)


# ---------------------------------------------------------------- Sec. 2.1 table
def test_identity_and_squares():
    for i in range(16):
        assert torch.equal(mul(E[i], E[0]), E[i]) and torch.equal(mul(E[0], E[i]), E[i])
    for i in range(1, 16):
        assert torch.equal(mul(E[i], E[i]), -E[0])


def test_basis_product_is_signed_xor():
    for i in range(16):
        for j in range(16):
            p = mul(E[i], E[j])
            assert p.abs().sum() == 1 and p[i ^ j].abs() == 1
            assert p[i ^ j] == SIGNS[i, j]


def test_octonion_subblock_is_composition_algebra():
    a = torch.randn(500, 16, dtype=F64); a[:, 8:] = 0
    b = torch.randn(500, 16, dtype=F64); b[:, 8:] = 0
    assert torch.allclose(mul(a, b).norm(dim=-1), a.norm(dim=-1) * b.norm(dim=-1))
    assert torch.allclose(mul(mul(a, a), b), mul(a, mul(a, b)))  # alternative


def test_sedenions_are_not_alternative_and_have_zero_divisors():
    a = torch.randn(500, 16, dtype=F64)
    b = torch.randn(500, 16, dtype=F64)
    assert not torch.allclose(mul(mul(a, a), b), mul(a, mul(a, b)))
    u = E[1] + E[10]; v = E[5] + E[14]  # (e1+e10)(e5+e14)=0 under this CD convention; with -e14 it is NOT zero
    assert mul(u, v).abs().max() < 1e-15


# ---------------------------------------------------------------- Thm 2.3 adjoint
def test_adjoint_theorem_L_and_R():
    w = torch.randn(16, dtype=F64)
    L = left_mult_matrix(w, KS, XOR)
    Lc = left_mult_matrix(sedenion_conj(w), KS, XOR)
    assert torch.allclose(L.T, Lc)
    R = torch.stack([mul(E[j], w) for j in range(16)], dim=-1)
    Rc = torch.stack([mul(E[j], sedenion_conj(w)) for j in range(16)], dim=-1)
    assert torch.allclose(R.T, Rc)


def test_gradcheck_custom_backward():
    ks = KS
    x = torch.randn(2, 3, 16, dtype=F64, requires_grad=True)
    w = torch.randn(4, 3, 16, dtype=F64, requires_grad=True)
    b = torch.randn(4, 16, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x_, w_, b_: SedenionLinearFunction.apply(x_, w_, b_, ks, XOR), (x, w, b), eps=1e-6, atol=1e-4)


# ---------------------------------------------------------------- Sec. 3 contraction / kernel
def test_contract_matches_bruteforce_and_kernel_reference():
    M, N, B = 5, 7, 3
    W = torch.randn(M, N, 16, dtype=F64); X = torch.randn(B, N, 16, dtype=F64)
    Y = sedenion_matmul_contract(X, W, KS, XOR)
    Yref = torch.stack([torch.stack([sum(mul(W[m, n], X[b, n]) for n in range(N)) for m in range(M)]) for b in range(B)])
    assert torch.allclose(Y, Yref)
    assert torch.allclose(fused_sedenion_linear_reference(X, W), Yref)


def test_parameter_count_is_one_sixteenth_of_dense():
    N, M = 64, 32
    sed = SedenionLinear(N, M, bias=False)
    dense = torch.nn.Linear(16 * N, 16 * M, bias=False)
    assert sum(p.numel() for p in sed.parameters()) * 16 == sum(p.numel() for p in dense.parameters())


# ---------------------------------------------------------------- Sec. 2.3 PG(3,2)
def test_pg32_fano_planes():
    fm, fl, fs = build_pg32_fano_tables()
    assert fm.shape == (15, 15) and fm.sum(1).eq(7).all()
    for om in range(15):
        pts = set((fm[om] > 0).nonzero().flatten().tolist())
        lines = {tuple(sorted(l)) for l in fl[om].tolist()}
        assert len(lines) == 7
        for (i, j, k) in lines:
            assert {i, j, k} <= pts and ((i + 1) ^ (j + 1)) == k + 1
    # every pair of the 15 points lies on exactly one of the 35 lines of PG(3,2)
    all_lines = {tuple(sorted(l)) for om in range(15) for l in fl[om].tolist()}
    assert len(all_lines) == 35


# ---------------------------------------------------------------- Thm 5.1 nullspace
def _moreno(n):
    mp = MorenoConeProjection(normalize_unit_sphere=True)
    return mp(torch.randn(n, 16, dtype=F64))


def test_moreno_projection_lands_on_cone():
    u = _moreno(256)
    a, b = u[:, 1:8], u[:, 9:16]
    assert (u[:, 0] == 0).all() and (u[:, 8] == 0).all()
    assert torch.allclose(a.norm(dim=-1), torch.ones(256, dtype=F64))
    assert torch.allclose(b.norm(dim=-1), torch.ones(256, dtype=F64))
    assert (a * b).sum(-1).abs().max() < 1e-14


def test_left_mult_has_rank_12_on_moreno_cone():
    u = _moreno(256)
    L = torch.stack([left_mult_matrix(r, KS, XOR) for r in u])
    sv = torch.linalg.svdvals(L)
    assert (sv[:, :12] > 1e-8).all() and (sv[:, 12:] < 1e-12).all()


def test_constructive_partner_annihilates_exactly():
    u = _moreno(1024)
    v = moreno_nullspace_partner(u, torch.randn(1024, 7, dtype=F64), KS, XOR)
    assert torch.allclose(v.norm(dim=-1), math.sqrt(2) * torch.ones(1024, dtype=F64))
    clr = mul(u, v).norm(dim=-1) / (u.norm(dim=-1) * v.norm(dim=-1))
    assert clr.max() < 1e-14


def test_constructive_partner_spans_full_4d_kernel():
    u = _moreno(1)[0]
    L = left_mult_matrix(u, KS, XOR)
    vs = torch.stack([moreno_nullspace_partner(u, torch.randn(7, dtype=F64), KS, XOR) for _ in range(40)])
    assert torch.linalg.matrix_rank(vs, tol=1e-8) == 4
    assert (L @ vs.T).abs().max() < 1e-14


# ---------------------------------------------------------------- Cor. 2.3 precedence
def test_psi_is_one_on_moreno_override_and_zero_when_associative():
    u = _moreno(512)
    v = moreno_nullspace_partner(u, torch.randn(512, 7, dtype=F64), KS, XOR)
    c = torch.randn(512, 16, dtype=F64)
    left = mul(mul(u, v), c); right = mul(u, mul(v, c))
    psi = (left - right).norm(dim=-1) / (left.norm(dim=-1) + right.norm(dim=-1))
    assert torch.allclose(psi, torch.ones(512, dtype=F64))
    # quaternion sub-algebra e0..e3: associative, PSI == 0
    q = torch.randn(3, 512, 16, dtype=F64); q[..., 4:] = 0
    l = mul(mul(q[0], q[1]), q[2]); r = mul(q[0], mul(q[1], q[2]))
    assert (l - r).abs().max() < 1e-13


# ---------------------------------------------------------------- Sec. 3.3 ONNX
def test_onnx_runtime_parity():
    ort = pytest.importorskip("onnxruntime")
    m = SedenionLinear(7, 5, bias=True, onnx_compatible=True).eval()
    x = torch.randn(3, 7, 16)
    with torch.no_grad():
        ref = m(x)
    buf = io.BytesIO()
    torch.onnx.export(m, (x,), buf, input_names=["x"], output_names=["y"], opset_version=18, dynamo=False)
    buf.seek(0)
    out = ort.InferenceSession(buf.read(), providers=["CPUExecutionProvider"]).run(None, {"x": x.numpy()})[0]
    assert abs(ref - torch.from_numpy(out)).max() < 1e-5


def test_constructive_partner_on_non_unit_cone():
    u = _moreno(128) * (torch.rand(128, 1, dtype=F64) * 3 + 0.1)  # radial scaling stays on the cone
    v = moreno_nullspace_partner(u, torch.randn(128, 7, dtype=F64), KS, XOR)
    assert (mul(u, v).norm(dim=-1) / (u.norm(dim=-1) * v.norm(dim=-1))).max() < 1e-14


# ---------------------------------------------------------------- Thm 3.1 bifurcation
import itertools

def _planes():
    pc = lambda x: bin(x).count("1")
    return {w: [i for i in range(1, 16) if pc(w & i) % 2 == 0] for w in range(1, 16)}


def test_alt_planes_are_division_algebras_zd_planes_are_not():
    H = _planes()
    for w, pts in H.items():
        idx = [0] + pts
        g = torch.Generator().manual_seed(w)
        a = torch.zeros(300, 16, dtype=F64); a[:, idx] = torch.randn(300, 8, generator=g, dtype=F64)
        b = torch.zeros(300, 16, dtype=F64); b[:, idx] = torch.randn(300, 8, generator=g, dtype=F64)
        p = mul(a, b)
        assert (p[:, [i for i in range(16) if i not in idx]].abs().max() < 1e-12)  # closed
        norm_mult = torch.allclose(p.norm(dim=-1), a.norm(dim=-1) * b.norm(dim=-1))
        assert norm_mult == (w <= 8)


def test_zd_plane_internal_annihilating_pair_and_polar_line():
    H = _planes()
    for w in range(9, 16):
        p = w - 8
        L0 = [i for i in H[w] if i < 8]
        assert L0 == [i for i in range(1, 8) if bin(p & i).count("1") % 2 == 0]      # polar line
        assert (p in L0) == (bin(p).count("1") % 2 == 0)
        found = False
        for m, n8, c, d8, s in itertools.product(L0, [i for i in H[w] if i > 8], L0, [i for i in H[w] if i > 8], (1, -1)):
            if mul(E[m] + E[n8], E[c] + s * E[d8]).abs().max() < 1e-12:
                found = True; break
        assert found, w
    assert mul(E[2] + E[9], E[4] + E[15]).abs().max() < 1e-12  # the example quoted in the paper


def test_zd_line_counts_per_plane_and_associator_census():
    H = _planes()
    def lines(pts):
        return {tuple(sorted((i, j, i ^ j))) for i, j in itertools.combinations(pts, 2) if (i ^ j) in pts}
    is_zd = lambda l: sum(1 for x in l if x < 8) == 1 and 8 not in l
    counts = {w: sum(is_zd(l) for l in lines(H[w])) for w in H}
    assert counts[8] == 0 and all(counts[w] == 3 for w in range(1, 8)) and all(counts[w] == 6 for w in range(9, 16))
    zero = total = 0
    for i, j, k in itertools.permutations(range(1, 16), 3):
        if i ^ j ^ k == 0:
            assert (mul(mul(E[i], E[j]), E[k]) - mul(E[i], mul(E[j], E[k]))).abs().max() < 1e-12  # collinear => associative
            continue
        total += 1
        zero += (mul(mul(E[i], E[j]), E[k]) - mul(E[i], mul(E[j], E[k]))).abs().max() < 1e-12
    assert (total, zero) == (2520, 672)


# ---------------------------------------------------------------- regression tests for review findings
def test_bitmask_packs_swizzled_table():
    """Eq. (15): bit j of MASK[k] is 1  iff  eps_{k^j, j} = -1  (the k_signs table, not signs)."""
    _, k_signs, masks = build_cayley_dickson_tables(16)
    for k in range(16):
        for j in range(16):
            assert ((int(masks[k]) >> j) & 1 == 1) == (k_signs[k, j].item() == -1)
    # applying the mask as a sign-bit XOR reproduces k_signs
    import struct
    for k in (0, 5, 11, 15):
        for j in range(16):
            bits = struct.unpack("I", struct.pack("f", 3.0))[0] ^ (((int(masks[k]) >> j) & 1) << 31)
            assert struct.unpack("f", struct.pack("I", bits))[0] == 3.0 * k_signs[k, j].item()


def test_linear_matches_reference_and_grads():
    from sedenion_pg32.core import sedenion_matmul_contract, SedenionLinear
    torch.manual_seed(0)
    _, ks, _ = build_cayley_dickson_tables(16)
    xi = torch.arange(16).unsqueeze(1) ^ torch.arange(16).unsqueeze(0)
    x = torch.randn(5, 7, 16, dtype=torch.float64); w = torch.randn(4, 7, 16, dtype=torch.float64)
    ks = ks.double()
    ref = torch.einsum("mni,bnj->bmij", w, x)[:, :, xi, torch.arange(16).unsqueeze(0)]
    ref = (ref * ks).sum(-1)
    assert torch.allclose(sedenion_matmul_contract(x, w, ks, xi), ref, atol=1e-12)
    # custom backward == autograd through the loop-free reference
    from sedenion_pg32.core import SedenionLinearFunction
    x1 = x.clone().requires_grad_(); w1 = w.clone().requires_grad_()
    g = torch.randn(5, 4, 16, dtype=torch.float64)
    SedenionLinearFunction.apply(x1, w1, None, ks, xi).backward(g)
    x2 = x.clone().requires_grad_(); w2 = w.clone().requires_grad_()
    r2 = torch.einsum("mni,bnj->bmij", w2, x2)[:, :, xi, torch.arange(16).unsqueeze(0)]
    (r2 * ks).sum(-1).backward(g)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-10) and torch.allclose(w1.grad, w2.grad, atol=1e-10)


def test_no_pair_sized_intermediate():
    """peak extra allocation must not scale like 256*M*N (checked via tensor shapes seen in autograd)."""
    from sedenion_pg32.core import sedenion_matmul_contract
    _, ks, _ = build_cayley_dickson_tables(16)
    xi = torch.arange(16).unsqueeze(1) ^ torch.arange(16).unsqueeze(0)
    M, N, B = 64, 64, 4
    w = torch.randn(M, N, 16); x = torch.randn(B, N, 16)
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        sedenion_matmul_contract(x, w, ks, xi)
    biggest = max(e.cpu_memory_usage for e in prof.events())      # largest single allocation (bytes)
    assert biggest <= 16 * M * N * 4, biggest                      # = size of W; a pair tensor would be 16x larger


def test_router_runs_gradients_and_symmetry():
    from sedenion_pg32 import PG32TopologicalRouter
    torch.manual_seed(0)
    r = PG32TopologicalRouter(d_model=8)
    h = torch.randn(3, 16, 4, 16, requires_grad=True)
    out, probs = r(h)
    assert out.shape == h.shape and probs.shape == (3, 15)
    assert torch.allclose(probs.sum(-1), torch.ones(3), atol=1e-6)
    assert torch.equal(out[:, 0], h[:, 0])                  # executive channel untouched
    out.pow(2).sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    assert r.workspace_gate.weight.grad.abs().sum() > 0     # gradient reaches the gate via top_prob
    # every point on every active line receives a message (no node left behind)
    top = probs.argmax(-1)
    for b in range(3):
        lines = r.fano_lines[top[b]]
        touched = set(lines.flatten().tolist())
        assert len(touched) == 7                              # the 7 points of the chosen plane
        changed = {n for n in range(15) if not torch.allclose(out[b, n + 1], h[b, n + 1])}
        assert changed == touched


def test_router_regime2_exact_annihilation():
    """Regime II: condition_on(x, y) lies in ker L(x) exactly, and a fully-gated message is zero."""
    from sedenion_pg32 import PG32TopologicalRouter, MorenoConeProjection
    torch.manual_seed(1)
    r = PG32TopologicalRouter(d_model=8).double()
    x = MorenoConeProjection(normalize_unit_sphere=False)(torch.randn(6, 5, 16, dtype=torch.float64))
    y = torch.randn(6, 5, 16, dtype=torch.float64)
    v = r.condition_on(x, y)
    assert r.sedenion_elementwise_mul(x, v).abs().max() < 1e-12
    assert (v.norm(dim=-1) > 0).all() and v.norm(dim=-1).max() < 2 * y.norm(dim=-1).max()
    # force g -> 1: every Regime-II message vanishes; g -> 0 recovers the plain product
    with torch.no_grad():
        r.prune_gate.bias.fill_(60.0)
    x4, y4 = x.view(6, 1, 5, 16)[:, :, :, :], y.view(6, 1, 5, 16)
    eps = torch.ones(6, 1, dtype=torch.float64)
    zd = torch.ones(6, 1, 1, 1, dtype=torch.bool)
    assert r.pair_message(x4, y4, eps, zd).abs().max() < 1e-10
    assert r.pair_message(x4, y4, eps, ~zd).abs().max() > 1e-3                 # Regime I: untouched
    with torch.no_grad():
        r.prune_gate.bias.fill_(-60.0)
    assert torch.allclose(r.pair_message(x4, y4, eps, zd), r.sedenion_elementwise_mul(x4, y4), atol=1e-10)


def test_router_pruning_gradients_flow_to_gate():
    from sedenion_pg32 import PG32TopologicalRouter
    torch.manual_seed(2)
    r = PG32TopologicalRouter(d_model=8)
    with torch.no_grad():                                   # force a zero-divisor workspace (omega >= 9)
        r.workspace_gate.weight.zero_(); r.workspace_gate.bias.zero_(); r.workspace_gate.bias[12] = 5.0
    h = torch.randn(2, 16, 3, 16, requires_grad=True)
    out, probs = r(h)
    assert (probs.argmax(-1) == 12).all()
    out.pow(2).sum().backward()
    assert r.prune_gate.weight.grad.abs().sum() > 0 and torch.isfinite(h.grad).all()
