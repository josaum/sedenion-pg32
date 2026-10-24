"""
Path B tests: does the 4D kernel fiber W_ab^perp buy anything a constrained linear beamformer lacks?

U1a  Phase-locked single-jammer mode: choose c in W_ab^perp (R^4) so the satellite output on the
     hinge quadrature pair (e_i, e_j) has zero Q component, maximise I.  Compare carrier phase jump
     across noisy weight updates against a linear LCMV beamformer with the same constraint
     (w ⊥ u1_hat, w^H u_sat real-positive), with non-zero <u_sat, u1> in [0.1, 0.3].
U1b  Single-stage dual-jammer + phase lock: 3 homogeneous constraints on c in R^4.  Compare to a
     linear beamformer with the same 3 constraints (w ⊥ u1_hat, w ⊥ u2_hat on the I/Q readout, phase lock).
U2   Co-delayed spoofer excision in the same tap, X = u_sat + beta*u_spoof, SSR +30 dB.  Compare
     sedenion excision v_beam(u_spoof_hat; c) against the 16-parameter projector P = I - u_hat u_hat^T,
     as a function of estimate error eps.  (The scalar e0 gate is not a competitor here and is not run.)
numpy only.
"""
import json, math, pathlib, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rb", HERE / "run_robustness.py")
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
cd_mul, unit, moreno_project, householder, db = rb.cd_mul, rb.unit, rb.moreno_project, rb.householder, rb.db
rng = np.random.default_rng(2026)
E = np.eye(16)


def fiber_basis(uzd):
    """orthonormal basis b_1..b_4 of W_ab^perp in Im(O) and the 4 kernel vectors v(b_k)."""
    a = uzd[:8] * math.sqrt(2); b = uzd[8:] * math.sqrt(2); ab = cd_mul(a, b)
    Bs = []
    while len(Bs) < 4:
        c = np.zeros(8); c[1:] = rng.standard_normal(7)
        for q in (a, b, ab): c -= (c @ q) / (q @ q) * q
        for q in Bs: c -= (c @ q) * q
        c[0] = 0
        if np.linalg.norm(c) > 1e-6: Bs.append(unit(c))
    V = np.stack([np.concatenate([c, cd_mul(cd_mul(a, c), b)]) for c in Bs])  # 4 x 16, each in ker L(uzd)
    return V


def c8(x):  # R^16 (Re|Im) -> C^8
    return x[:8] + 1j * x[8:]

def r16(z):
    return np.concatenate([z.real, z.imag])


def make_scene(corr_lo=0.1, corr_hi=0.3):
    """jammer-1 signature and a satellite signature with |<u_sat,u1>| in [corr_lo, corr_hi]."""
    while True:
        th1 = rng.uniform(-60, 60); th_s = rng.uniform(-60, 60)
        u1 = unit(rb.c8_to_r16(rb.steer(th1))); us = unit(rb.c8_to_r16(rb.steer(th_s)))
        if corr_lo <= abs(us @ u1) <= corr_hi:
            return u1, us


def sed_phase_locked(u1_hat, u_sat, i=1, j=2, extra=None):
    """Canonical Moreno cone uzd=(e3,e4)/sqrt(2), anti-partner uanti=(e3,-e4)/sqrt(2).
       T1 maps u1_hat -> uzd; T2 (fixing uzd) maps u_sat^perp -> uanti so (e1,e2) sits
       directly in the anti-partner passband a*W_ab^perp."""
    u1u = unit(u1_hat)
    uzd = np.zeros(16); uzd[3] = 1/math.sqrt(2); uzd[8+4] = 1/math.sqrt(2)
    uanti = np.zeros(16); uanti[3] = 1/math.sqrt(2); uanti[8+4] = -1/math.sqrt(2)
    T1 = householder(u1u, uzd)
    us_perp = u_sat - (u1u @ u_sat) * u1u
    T2 = householder(unit(T1 @ us_perp), uanti)
    T = T2 @ T1
    V = fiber_basis(uzd)
    ys = np.stack([cd_mul(T @ u_sat, Vk) for Vk in V])          # 4 x 16 satellite outputs
    rows = [ys[:, j]]                                           # phase lock: Q = 0
    if extra is not None:
        for x in extra:
            yx = np.stack([cd_mul(T @ x, Vk) for Vk in V]); rows += [yx[:, i], yx[:, j]]
    Lc = np.stack(rows)
    _, s, Vt = np.linalg.svd(Lc); null = Vt[len(rows):] if len(rows) < 4 else Vt[-1:]
    gI = ys[:, i]
    alpha = null.T @ (null @ gI)
    if np.linalg.norm(alpha) < 1e-12: alpha = null[0]
    alpha = unit(alpha)
    if alpha @ gI < 0: alpha = -alpha
    return T, unit(alpha @ V)


def lin_phase_locked(u1_hat, u_sat, extra=None):
    """LCMV in C^8: w ⊥ u1_hat (and extras), w^H u_sat = 1 (real, positive) -> distortionless, phase 0."""
    C = [c8(u1_hat)] + ([c8(x) for x in extra] if extra else [])
    A = np.stack(C + [c8(u_sat)], 1)                            # 8 x (1+J+1)
    f = np.zeros(A.shape[1], complex); f[-1] = 1.0
    w = A @ np.linalg.solve(A.conj().T @ A, f)                  # min-norm solution of A^H w = f
    return w


def u1a_phase(n_trials=256, eps=1e-3):
    sed, lin, sed_amp = [], [], []
    for _ in range(n_trials):
        u1, us = make_scene()
        ua = unit(u1 + eps * unit(rng.standard_normal(16))); ub = unit(u1 + eps * unit(rng.standard_normal(16)))
        Ta, va = sed_phase_locked(ua, us); Tb, vb = sed_phase_locked(ub, us)
        ya = cd_mul(Ta @ us, va); yb = cd_mul(Tb @ us, vb)
        pa = math.atan2(ya[2], ya[1]); pb = math.atan2(yb[2], yb[1])
        sed.append(abs(math.degrees(pa - pb))); sed_amp.append(ya[1])
        wa = lin_phase_locked(ua, us); wb = lin_phase_locked(ub, us)
        za = np.vdot(wa, c8(us)); zb = np.vdot(wb, c8(us))
        lin.append(abs(math.degrees(np.angle(za) - np.angle(zb))))
    return dict(sed_fiber_phase_jump_deg_mean=round(float(np.mean(sed)), 6), sed_fiber_phase_jump_deg_max=round(float(np.max(sed)), 6),
                lin_lcmv_phase_jump_deg_mean=round(float(np.mean(lin)), 6), lin_lcmv_phase_jump_deg_max=round(float(np.max(lin)), 6),
                sed_I_amplitude_mean=round(float(np.mean(sed_amp)), 4))


def u1b_dual(n_trials=256, eps=1e-3):
    sed1, sed2, sedph, lin1, lin2, linph = [], [], [], [], [], []
    for _ in range(n_trials):
        u1, us = make_scene()
        th2 = rng.uniform(-60, 60); x2 = unit(rb.c8_to_r16(rb.steer(th2)))
        u1h = unit(u1 + eps * unit(rng.standard_normal(16))); u2h = unit(x2 + eps * unit(rng.standard_normal(16)))
        T, v = sed_phase_locked(u1h, us, extra=[u2h])
        y1 = cd_mul(T @ u1, v); y2 = cd_mul(T @ x2, v); ys = cd_mul(T @ us, v)
        sed1.append(db(np.linalg.norm(y1))); sed2.append(db(math.hypot(y2[1], y2[2]) / math.hypot(ys[1], ys[2])))
        sedph.append(abs(math.degrees(math.atan2(ys[2], ys[1]))))
        w = lin_phase_locked(u1h, us, extra=[u2h])
        z1 = np.vdot(w, c8(u1)); z2 = np.vdot(w, c8(x2)); zs = np.vdot(w, c8(us))
        lin1.append(db(abs(z1) / abs(zs))); lin2.append(db(abs(z2) / abs(zs))); linph.append(abs(math.degrees(np.angle(zs))))
    return dict(sed_jam1_16D_null_db=round(float(np.mean(sed1)), 1), sed_jam2_IQ_null_db=round(float(np.mean(sed2)), 1),
                sed_sat_phase_err_deg=round(float(np.mean(sedph)), 6),
                lin_jam1_null_db=round(float(np.mean(lin1)), 1), lin_jam2_null_db=round(float(np.mean(lin2)), 1),
                lin_sat_phase_err_deg=round(float(np.mean(linph)), 6), lean_bound_db=round(db(math.sqrt(2) * eps), 1))


def u2_codelayed(n_trials=256, ssr_db=30.0):
    rows = []
    beta = 10 ** (ssr_db / 20)
    for eps in [1e-4, 1e-3, 1e-2]:
        sed, prj = [], []
        for _ in range(n_trials):
            u1, us = make_scene(0.0, 0.5)          # u1 plays the spoofer
            usp_hat = unit(u1 + eps * unit(rng.standard_normal(16)))
            X = us + beta * u1                      # same correlator tap
            T, _, v = rb.sed_nuller(usp_hat, us) if hasattr(rb, "sed_nuller") else (None, None, None)
            y = cd_mul(T @ X, v); ysat = cd_mul(T @ us, v); s_hat = unit(ysat)
            # recovered satellite: component along the clean-satellite output direction vs residual spoofer
            sed.append(db(abs(cd_mul(T @ (beta * u1), v) @ s_hat) / abs(ysat @ s_hat)))
            P = np.eye(16) - np.outer(usp_hat, usp_hat); ps = P @ us; w = unit(ps)
            prj.append(db(abs((beta * u1) @ w) / abs(us @ w)))
        rows.append(dict(eps=eps, sed_spoofer_residual_rel_sat_db=round(float(np.mean(sed)), 1),
                         projector16_spoofer_residual_rel_sat_db=round(float(np.mean(prj)), 1)))
    return rows


if __name__ == "__main__":
    res = dict(U1a_phase_locked=u1a_phase(), U1b_dual_null=u1b_dual(), U2_codelayed=u2_codelayed())
    (HERE / "results_pathb_fiber.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print("==", k); print("  ", v)
