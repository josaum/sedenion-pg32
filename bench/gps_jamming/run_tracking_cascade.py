"""
Tests requested after the fair-baseline rerun:
  T1  Fast-update regime: Sed/projection re-estimate u_hat every K_sub chips across a 64-chip
      window with 0.5 deg total drift; MVDR holds a K=64 block weight.  Leakage averaged over
      every chip in the window (fair for all).
  T2  Cascade (X.v1).v2 against two simultaneous jammers, v2 from Moreno(u2.v1).
  T3  Single-stage 4D-subspace steering: pick c in W_ab^perp so the satellite beam is also
      free of the second jammer (one linear constraint, 4 unknowns).
  T4  Phase continuity of the satellite beam output across successive weight updates
      (does the hinge protect the PLL when v itself changes?).
numpy only.  Writes results_tracking_cascade.json.
"""
import json, math, pathlib, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rb", HERE / "run_robustness.py")
rb = importlib.util.module_from_spec(spec); spec.loader.exec_module(rb)
cd_mul, unit, moreno_project, nullspace_partner, householder, db, steer, c8_to_r16 = (
    rb.cd_mul, rb.unit, rb.moreno_project, rb.nullspace_partner, rb.householder, rb.db, rb.steer, rb.c8_to_r16)
rng = np.random.default_rng(2026)
JAM = 10 ** (80 / 20); NOISE = 10.0 / 4   # per-element: JSR +80 dB, SNR -20 dB, as in the paper bench


def snapshots(thetas, u_sat):
    return np.stack([rng.standard_normal() * JAM * c8_to_r16(steer(t)) for t in thetas]) \
        + rng.standard_normal((len(thetas), 16)) * NOISE + np.outer(np.sign(rng.standard_normal(len(thetas))), u_sat)

def sed_weights(u_src, u_sat):
    uzd = moreno_project(u_src + 1e-3 * rng.standard_normal(16))
    T = householder(unit(u_src), uzd)
    v = nullspace_partner(uzd, rng.standard_normal(7))
    s_hat = unit(cd_mul(T @ u_sat, v))
    return T, v, s_hat

def sed_beam(T, v, s_hat, x):
    return cd_mul(T @ x, v) @ s_hat


# ------------------------------------------------------------------ T1 fast update
def t1_fast_update(n_trials=128, K=64, drift=0.5):
    rows = []
    for K_sub in [1, 2, 4, 8, 64]:
        sed, prj, mv = [], [], []
        for _ in range(n_trials):
            th0 = rng.uniform(-60, 60); th_s = th0 + (20 if th0 < 35 else -20); u_sat = unit(c8_to_r16(steer(th_s)))
            thetas = th0 + np.linspace(0, drift, K)
            X = snapshots(thetas, u_sat)
            R = X.T @ X / K; w_mv = unit(np.linalg.solve(R + 1e-2 * np.eye(16), u_sat))
            leak_mv = []; leak_sed = []; leak_prj = []
            for start in range(0, K, K_sub):
                blk = X[start:start + K_sub]
                if K_sub == 1:
                    u_hat = unit(blk[0])
                else:
                    ev, V = np.linalg.eigh(blk.T @ blk / K_sub); u_hat = unit(V[:, -1])
                T, v, s_hat = sed_weights(u_hat, u_sat)
                w_p = unit(u_sat - (u_hat @ u_sat) * u_hat)
                # apply to the NEXT block's true signatures (causal: estimate then apply)
                nxt = range(start + K_sub, min(start + 2 * K_sub, K)) if start + K_sub < K else range(start, K)
                for t in nxt:
                    x = unit(c8_to_r16(steer(thetas[t])))
                    leak_sed.append(abs(sed_beam(T, v, s_hat, x))); leak_prj.append(abs(x @ w_p))
            for t in range(K):
                leak_mv.append(abs(unit(c8_to_r16(steer(thetas[t]))) @ w_mv))
            sed.append(db(np.mean(leak_sed))); prj.append(db(np.mean(leak_prj))); mv.append(db(np.mean(leak_mv)))
        rows.append(dict(K_sub=K_sub, drift_per_update_deg=round(drift * K_sub / K, 4),
                         sed_null_db=round(float(np.mean(sed)), 1), proj_null_db=round(float(np.mean(prj)), 1),
                         mvdr_K64_block_null_db=round(float(np.mean(mv)), 1)))
    return rows


# ------------------------------------------------------------------ T2 cascade, T3 4D steering
def t2_t3_two_jammers(n_trials=256, eps=1e-3):
    one, casc, steer4, prj2 = [], [], [], []
    sat_gain_casc, sat_gain_steer = [], []
    for _ in range(n_trials):
        th1, th2 = rng.uniform(-60, 60, 2)
        while abs(th1 - th2) < 10: th2 = rng.uniform(-60, 60)
        th_s = th1 + (20 if th1 < 35 else -20); u_sat = unit(c8_to_r16(steer(th_s)))
        x1 = unit(c8_to_r16(steer(th1))); x2 = unit(c8_to_r16(steer(th2)))
        u1 = unit(x1 + eps * unit(rng.standard_normal(16))); u2 = unit(x2 + eps * unit(rng.standard_normal(16)))
        # stage 1
        T1, v1, s1 = sed_weights(u1, u_sat)
        y1 = lambda x: cd_mul(T1 @ x, v1)
        r_sat = y1(u_sat); r1 = y1(x1); r2 = y1(x2)
        one.append(db(np.sqrt(abs(r1 @ unit(r_sat))**2 + abs(r2 @ unit(r_sat))**2)))
        # T2 cascade: second stage built on the residual of jammer 2
        r2_hat = y1(u2)
        T2, v2, _ = sed_weights(r2_hat, r_sat)
        y2 = lambda r: cd_mul(T2 @ r, v2)
        s2 = unit(y2(r_sat))
        l1 = abs(y2(r1) @ s2); l2 = abs(y2(r2) @ s2)
        casc.append(db(math.sqrt(l1**2 + l2**2))); sat_gain_casc.append(20 * math.log10(np.linalg.norm(y2(r_sat))))
        # T3 single stage, choose c in W_ab^perp (4D) so that beam output of jammer 2 is zero
        uzd = moreno_project(u1 + 1e-3 * rng.standard_normal(16)); T = householder(u1, uzd)
        a = uzd[:8] * math.sqrt(2); b = uzd[8:] * math.sqrt(2); ab = cd_mul(a, b)
        # orthonormal basis of W_ab^perp in Im(O)
        Bs = []
        for _k in range(12):
            c = np.zeros(8); c[1:] = rng.standard_normal(7)
            for basis in (a, b, ab): c -= (c @ basis) / (basis @ basis) * basis
            for q in Bs: c -= (c @ q) * q
            c[0] = 0
            if np.linalg.norm(c) > 1e-6: Bs.append(unit(c))
            if len(Bs) == 4: break
        V = np.stack([np.concatenate([c, cd_mul(cd_mul(a, c), b)]) for c in Bs])  # 4 kernel vectors
        # beam = <x . v(c), s>;  choose s = direction of u_sat . v(c0) for c0 = first basis, then solve
        # for coefficients alpha: sum_i alpha_i <u2 . V_i, s> = 0  (1 eq, 4 unknowns), maximise sat
        s = unit(cd_mul(T @ u_sat, V[0]))
        g2 = np.array([cd_mul(T @ u2, Vi) @ s for Vi in V]); gs = np.array([cd_mul(T @ u_sat, Vi) @ s for Vi in V])
        P = np.eye(4) - np.outer(g2, g2) / (g2 @ g2)
        alpha = P @ gs; alpha /= np.linalg.norm(alpha)
        v = alpha @ V; v = unit(v)
        s = unit(cd_mul(T @ u_sat, v))
        l1 = abs(cd_mul(T @ x1, v) @ s); l2 = abs(cd_mul(T @ x2, v) @ s)
        steer4.append(db(math.sqrt(l1**2 + l2**2))); sat_gain_steer.append(20 * math.log10(np.linalg.norm(cd_mul(T @ u_sat, v))))
        # projection with both signatures
        Q, _ = np.linalg.qr(np.stack([u1, u2], 1)); w = unit(u_sat - Q @ (Q.T @ u_sat))
        prj2.append(db(math.sqrt(abs(x1 @ w)**2 + abs(x2 @ w)**2)))
    return dict(single_stage_two_jammers_db=round(float(np.mean(one)), 1),
                cascade_two_stage_db=round(float(np.mean(casc)), 1), cascade_sat_gain_db=round(float(np.mean(sat_gain_casc)), 2),
                single_stage_4D_steering_db=round(float(np.mean(steer4)), 1), steering_sat_gain_db=round(float(np.mean(sat_gain_steer)), 2),
                projection_rank2_db=round(float(np.mean(prj2)), 1), lean_bound_db=round(db(math.sqrt(2) * eps), 1))


# ------------------------------------------------------------------ T4 phase continuity across weight updates
def t4_phase_across_updates(n_trials=256, eps=1e-3):
    """Satellite output phase in the (e_i, e_j) quadrature pair of the beam, before/after v update."""
    jumps_generic, jumps_hinge = [], []
    for _ in range(n_trials):
        th1 = rng.uniform(-60, 60); th_s = th1 + (20 if th1 < 35 else -20); u_sat = unit(c8_to_r16(steer(th_s)))
        x1 = unit(c8_to_r16(steer(th1)))
        ua = unit(x1 + eps * unit(rng.standard_normal(16))); ub = unit(x1 + eps * unit(rng.standard_normal(16)))
        # generic cone weights, two successive estimates (different seeds, as in a real update)
        Ta, va, _ = sed_weights(ua, u_sat); Tb, vb, _ = sed_weights(ub, u_sat)
        ya = cd_mul(Ta @ u_sat, va); yb = cd_mul(Tb @ u_sat, vb)
        # phase = angle between the two outputs in the full output space (worst case), and in quadrature (e1,e2)
        cosang = np.clip(ya @ yb / (np.linalg.norm(ya) * np.linalg.norm(yb)), -1, 1)
        jumps_generic.append(math.degrees(math.acos(cosang)))
        # hinge case as in Prop 3.1: carrier in span(e1,e2), stages u1,u2 restricted to H_{L12}
        th = rng.uniform(-math.pi, math.pi); xc = np.zeros(16); xc[1] = math.cos(th); xc[2] = math.sin(th)
        h1 = np.zeros(16); h2 = np.zeros(16)
        for q in (0, 1, 2, 3): h1[q] = rng.standard_normal(); h2[q] = rng.standard_normal()
        left = cd_mul(cd_mul(xc, h1), h2); right = cd_mul(xc, cd_mul(h1, h2))
        jumps_hinge.append(math.degrees(math.atan2(left[2], left[1]) - math.atan2(right[2], right[1])))
    return dict(generic_weight_update_phase_jump_deg_mean=round(float(np.mean(jumps_generic)), 2),
                generic_weight_update_phase_jump_deg_max=round(float(np.max(jumps_generic)), 2),
                hinge_assoc_vs_nonassoc_phase_deg_max=float(np.max(np.abs(jumps_hinge))))


if __name__ == "__main__":
    res = dict(T1_fast_update=t1_fast_update(), T2_T3_two_jammers=t2_t3_two_jammers(), T4_phase=t4_phase_across_updates())
    (HERE / "results_tracking_cascade.json").write_text(json.dumps(res, indent=1))
    print("== T1 fast update (64-chip window, 0.5 deg drift)")
    for r in res["T1_fast_update"]: print("  ", r)
    print("== T2/T3 two jammers"); print("  ", res["T2_T3_two_jammers"])
    print("== T4 phase"); print("  ", res["T4_phase"])
