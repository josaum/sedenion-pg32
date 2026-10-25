"""
Complex-aware sedenion nulling: the 2-frame gauge.

Fact (numerically exact, see check_kernels): for every Moreno cone point u, ker L(u) = ker L(Ju),
where J(a,b) = (-b, a) is multiplication by i in the C^8 ~ R^16 embedding.  So any v in the fibre
annihilates u e^{i phi} for ALL phi.  What broke the earlier benches was the gauge: a real Householder
maps u1 -> u_zd but not J u1 -> J u_zd.  Fix: an orthogonal T that maps the 2-frame {u1, J u1}
onto {u_zd, J u_zd}.  Stage 2 of a cascade does the same for the 2-plane {M1 u2, M1 J u2}.
Readout: the dual basis of {y(u_sat), y(J u_sat)}, which makes the satellite transfer exactly
complex-linear (carrier preserved), independent of the 16x16 map being complex-linear.

Evaluated under the OKComputer report protocol (okc_revised_crpa_validation.py), so results are
directly comparable with that report's table.
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("okc", HERE / "okc_revised_crpa_validation.py")
okc = importlib.util.module_from_spec(spec); spec.loader.exec_module(okc)
cd_mul, runit, r16, c8, householder, fiber_basis, I16, J16 = (
    okc.cd_mul, okc.runit, okc.r16, okc.c8, okc.householder, okc.fiber_basis, okc.I16, okc.J16)

UZD = np.zeros(16); UZD[3] = UZD[12] = 1 / math.sqrt(2)
JUZD = J16 @ UZD


def check_kernels(rng, n=20):
    """dim ker[L(u); L(Ju)] == 4 for random cone points (kernels coincide)."""
    def L(u): return np.stack([cd_mul(u, I16[k]) for k in range(16)], 1)
    dims = []
    for _ in range(n):
        a = rng.standard_normal(8); a[0] = 0; a = runit(a)
        b = rng.standard_normal(8); b[0] = 0; b -= (a @ b) * a; b = runit(b)
        u = np.concatenate([a, b])
        s = np.linalg.svd(np.vstack([L(u), L(J16 @ u)]), compute_uv=False)
        dims.append(int((s < 1e-9).sum()))
    return dims


def frame_gauge(w1, w2):
    """Orthogonal T with T w1 = UZD, T w2 = JUZD, for an orthonormal pair (w1, w2)."""
    T1 = householder(w1, UZD)
    w2p = T1 @ w2
    w2p -= (w2p @ UZD) * UZD            # numerically re-orthogonalise against the fixed vector
    T2 = householder(runit(w2p), JUZD)  # reflection vector is ⊥ UZD, so UZD is fixed
    return T2 @ T1


def frame_from(x1, x2):
    w1 = runit(x1); w2 = x2 - (x1 @ x2) / (x1 @ x1) * x1
    return w1, runit(w2)


def stage_matrix(x1, x2, sat_pair, rng, pre=None, n_cand=96):
    """16x16 real matrix M: X -> (T X) . v, with T the 2-frame gauge of (x1, x2) and v in the fibre.
       The 4 fibre DOF are spent minimising the noise gain of the final dual-basis readout
       (pre = product of earlier stages).  M is linear in v, so M(alpha) = sum_k alpha_k M_k."""
    T = frame_gauge(*frame_from(x1, x2))
    V = fiber_basis(rng)                                   # 4 x 16, each in ker L(UZD) = ker L(JUZD)
    P = np.eye(16) if pre is None else pre
    TE = T @ I16                                           # columns T e_k
    Mk = [np.stack([cd_mul(TE[:, k], vk) for k in range(16)], axis=1) for vk in V]   # 4 x (16x16)
    MP = [m @ P for m in Mk]
    sat, jsat = sat_pair
    Bk = [np.stack([cd_mul(T @ sat, vk), cd_mul(T @ jsat, vk)], 1) for vk in V]      # 4 x (16x2)
    def cost(alpha):
        B = sum(a * b for a, b in zip(alpha, Bk)); M = sum(a * m for a, m in zip(alpha, MP))
        s = np.linalg.svd(B, compute_uv=False)
        if s[-1] < 1e-9 * s[0]: return np.inf
        H = np.linalg.pinv(B) @ M
        return math.sqrt((H[0] @ H[0] + H[1] @ H[1]) / 2)
    G = np.zeros((4, 4))
    for s_ in (T @ sat, T @ jsat):
        Y = np.stack([cd_mul(s_, vk) for vk in V]); G += Y @ Y.T
    cands = [np.linalg.eigh(G)[1][:, -1]] + [runit(rng.standard_normal(4)) for _ in range(n_cand)]
    best = min(cands, key=cost); bc = cost(best); step = 0.3
    for _ in range(30):
        improved = False
        for k in range(4):
            for sgn in (+1, -1):
                a_ = best.copy(); a_[k] += sgn * step; a_ = runit(a_); c = cost(a_)
                if c < bc: best, bc, improved = a_, c, True
        if not improved: step *= 0.5
        if step < 1e-4: break
    return sum(a * m for a, m in zip(best, Mk)) / np.linalg.norm(best @ V)


def dual_readout(M, sat):
    """2x16 readout H with H @ sat = (1, 0) and H @ J sat = (0, 1): exact carrier transfer."""
    yI, yQ = M @ sat, M @ (J16 @ sat)
    B = np.stack([yI, yQ], 1)                               # 16 x 2
    return np.linalg.pinv(B) @ M                            # 2 x 16 on the INPUT snapshot; H sat = (1,0), H J sat = (0,1)


def sed_frame_single(u_hat, sat, rng):
    M = stage_matrix(u_hat, J16 @ u_hat, [sat, J16 @ sat], rng)
    return dual_readout(M, sat)


def sed_frame_cascade(u1, u2, sat, rng):
    M1 = stage_matrix(u1, J16 @ u1, [sat, J16 @ sat], rng)
    M2 = stage_matrix(M1 @ u2, M1 @ (J16 @ u2), [M1 @ sat, M1 @ (J16 @ sat)], rng, pre=M1)
    M = M2 @ M1
    return dual_readout(M, sat)


# ---------------------------------------------------------------- run the report's two experiments
def controlled(rng, eps=1e-3):
    jammers, sat, _ = okc.make_scene(rng, False)
    u = okc.estimate_with_error(jammers[0], eps, rng)
    uh, sh = r16(u), r16(sat)
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
    methods = {"sed_complex_iq (report)": okc.sed_single(uh, sh, rng, True),
               "sed_frame (ours)": sed_frame_single(uh, sh, rng),
               "lcmv_complex": okc.complex_lcmv([uh], sh)}
    rows = []
    for name, H in methods.items():
        nulls, cerr, zs = okc.null_and_carrier_metrics(H, sh, [r16(jammers[0])], phases)
        rows.append(dict(method=name, null1_db=nulls[0], carrier_relative_error=cerr,
                         noise_gain=okc.noise_gain(H), sat_amplitude=abs(zs),
                         sat_over_noise=abs(zs) / okc.noise_gain(H)))
    return rows


def heldout(rng, phases):
    estimates, sat, eval_signatures, R = okc.heldout_data(rng, True)
    sh = r16(sat); u1, u2 = r16(estimates[0]), r16(estimates[1])
    methods = {"sed_2stage (report)": okc.sed_cascade(u1, u2, sh, rng),
               "sed_frame_cascade (ours)": sed_frame_cascade(u1, u2, sh, rng),
               "lcmv_complex": okc.complex_lcmv([u1, u2], sh),
               "mvdr_complex": okc.complex_mvdr(R, sh)}
    jam_noise = okc.cnormal(rng, okc.L); thermal = okc.cnormal(rng, okc.L)
    rows = []
    for name, H in methods.items():
        ng = okc.noise_gain(H); zs = okc.h_to_z(H @ sh)
        res = [[], []]; cerr = []
        for t in range(len(eval_signatures[0])):
            for p in phases:
                for j in range(2):
                    res[j].append(abs(okc.apply_phase(H, eval_signatures[j][t], p)) / abs(zs))
                cerr.append(abs(okc.apply_phase(H, sat, p) - zs * p) / abs(zs))
        nulls = [-20 * math.log10(max(math.sqrt(np.mean(np.square(r))), 1e-16)) for r in res]
        leak = math.sqrt(np.mean(np.concatenate(res) ** 2)); sat_norm = abs(zs) / ng
        hit, p2o, sinr = okc.acquire(sat_norm, leak * sat_norm, 1e4, 10.0, jam_noise, thermal)
        rows.append(dict(method=name, null1_db=nulls[0], null2_db=nulls[1],
                         carrier_relative_error=math.sqrt(np.mean(np.square(cerr))), noise_gain=ng,
                         detection_hit=hit, postcorr_sinr_db=sinr))
    return rows


def agg(rows, keys):
    out = {}
    for r in rows:
        d = out.setdefault(r["method"], {k: [] for k in keys})
        for k in keys: d[k].append(r[k])
    return {m: {k: (round(float(np.mean(v)), 3) if k != "detection_hit" else round(float(np.mean(v)), 3))
                for k, v in d.items()} for m, d in out.items()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    rng = np.random.default_rng(2026)
    print("kernel coincidence check, dim ker[L(u);L(Ju)] over 20 cone points:", sorted(set(check_kernels(rng))))
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    C = [r for _ in range(n) for r in controlled(rng)]
    Hh = [r for _ in range(n) for r in heldout(rng, phases)]
    res = dict(controlled_single_jammer=agg(C, ["null1_db", "carrier_relative_error", "noise_gain", "sat_amplitude", "sat_over_noise"]),
               heldout_dual_jammer=agg(Hh, ["null1_db", "null2_db", "carrier_relative_error", "noise_gain", "detection_hit", "postcorr_sinr_db"]))
    (HERE / "results_complex_frame.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print("==", k)
        for m, d in v.items(): print(f"  {m:28s} {d}")
