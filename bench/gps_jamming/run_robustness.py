"""
Robustness experiment: where does Sedenion PG(3,2) zero-divisor nulling actually win?

Measures jammer null depth as a function of
  (A) jammer-signature estimation error  eps = ||x_true - x_hat|| / ||x_true||
  (B) per-channel gain/phase calibration mismatch of the 8-element CRPA
  (C) covariance sample support K under jammer DOA drift (MVDR-SMI's failure regime)

and checks every sedenion residual against the machine-checked bound
      ||(u + delta) . v||^2  <=  2 ||delta||^2 ||v||^2      (v in ker L(u))
from SedenionRobust.lean  (theorem `leakage_bound`).

Baselines use the *same* estimated signature x_hat, so the comparison is apples-to-apples:
  - Orthogonal-projection nuller  w = (I - x_hat x_hat^T) u_sat     (signature-driven, like ours)
  - MVDR-SMI (diagonal loaded)    w = (R_hat + d I)^-1 u_sat        (data-driven)

numpy only; no torch dependency.
"""
import json, math, pathlib, sys
import numpy as np

rng = np.random.default_rng(2026)
OUT = pathlib.Path(__file__).resolve().parent / "results_robustness.json"


# ----------------------------------------------------------------------------------
# Cayley–Dickson algebra on R^16 (same doubling convention as SedenionReal.lean)
# ----------------------------------------------------------------------------------
def cd_conj(x):
    y = -x.copy(); y[..., 0] = x[..., 0]; return y

def cd_mul(x, y):
    n = x.shape[-1]
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[..., :h], x[..., h:]
    c, d = y[..., :h], y[..., h:]
    # (a,b)(c,d) = (ac - d* b, da + b c*)
    return np.concatenate([cd_mul(a, c) - cd_mul(cd_conj(d), b),
                           cd_mul(d, a) + cd_mul(b, cd_conj(c))], axis=-1)

def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def moreno_project(x):
    """Π_Moreno: R^16 -> V_2(R^7) ⊂ S, unit norm.  Drops e0,e8; Gram–Schmidt; balance norms."""
    a = x[:8].copy(); b = x[8:].copy()
    a[0] = 0.0; b[0] = 0.0
    b = b - (a @ b) / (a @ a) * a
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    return np.concatenate([a, b]) / math.sqrt(2.0)

def nullspace_partner(u, seed):
    """Theorem 5.1: for u=(a,b) on the cone (||a||=||b||=1 after rescale), c ⊥ {a,b,ab},
       v = (c, (ac)b) satisfies u.v = 0 exactly.  Returns unit v."""
    a = u[:8] * math.sqrt(2.0); b = u[8:] * math.sqrt(2.0)
    ab = cd_mul(a, b)
    c = np.zeros(8); c[1:] = seed[:7]
    for basis in (a, b, ab):
        c = c - (c @ basis) / (basis @ basis) * basis
    c[0] = 0.0
    d = cd_mul(cd_mul(a, c), b)
    return unit(np.concatenate([c, d]))

def householder(x, y):
    """Orthogonal T with T x = y for unit x, y (the 'gauge' used in run_phase2_defense.py)."""
    w = x - y
    if np.linalg.norm(w) < 1e-14:
        return np.eye(16)
    w = w / np.linalg.norm(w)
    return np.eye(16) - 2.0 * np.outer(w, w)

def db(v):
    return -20.0 * math.log10(max(v, 1e-16))


# ----------------------------------------------------------------------------------
# Array model: 8-element half-wavelength ULA, complex steering -> R^16 (Re | Im)
# ----------------------------------------------------------------------------------
def steer(theta_deg):
    idx = np.arange(8)
    a = np.exp(1j * math.pi * idx * math.sin(math.radians(theta_deg))) / math.sqrt(8.0)
    return a

def c8_to_r16(z):
    return np.concatenate([z.real, z.imag])


def sed_nuller(x_hat_r16, u_sat_r16):
    """Signature-driven sedenion nuller: gauge x_hat onto a cone point, build partner."""
    u_zd = moreno_project(x_hat_r16 + 1e-3 * rng.standard_normal(16))  # generic cone point
    T = householder(unit(x_hat_r16), u_zd)
    v = nullspace_partner(u_zd, rng.standard_normal(7))
    return T, u_zd, v

def sed_residual(T, v, x_true_r16):
    return np.linalg.norm(cd_mul(T @ unit(x_true_r16), v))

def proj_residual(x_hat_r16, u_sat_r16, x_true_r16):
    xh = unit(x_hat_r16)
    w = u_sat_r16 - (xh @ u_sat_r16) * xh
    w = unit(w)
    return abs(unit(x_true_r16) @ w)

def mvdr_residual(X_snap, u_sat_r16, x_true_r16, load=1e-2):
    K = X_snap.shape[0]
    R = X_snap.T @ X_snap / K + load * np.eye(16)
    w = unit(np.linalg.solve(R, u_sat_r16))
    return abs(unit(x_true_r16) @ w)


# ----------------------------------------------------------------------------------
# (A) Null depth vs signature-estimation error, with the Lean bound
# ----------------------------------------------------------------------------------
def exp_A(n_trials=256):
    eps_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    rows = []
    for eps in eps_grid:
        sed, proj, bound_ok, bound_gap = [], [], 0, []
        for _ in range(n_trials):
            th_j = rng.uniform(-60, 60); th_s = th_j + (20 if th_j < 35 else -20)
            x_true = c8_to_r16(steer(th_j)); u_sat = unit(c8_to_r16(steer(th_s)))
            n = unit(rng.standard_normal(16))
            x_hat = unit(unit(x_true) + eps * n)
            T, u_zd, v = sed_nuller(x_hat, u_sat)
            r = sed_residual(T, v, x_true)
            # Lean bound: delta = T x_true - T x_hat  (T orthogonal => ||delta|| = ||x_true - x_hat||)
            delta = np.linalg.norm(T @ unit(x_true) - T @ x_hat)
            bnd = math.sqrt(2.0) * delta * np.linalg.norm(v)
            bound_ok += int(r <= bnd + 1e-12)
            bound_gap.append(bnd / max(r, 1e-16))
            sed.append(db(r)); proj.append(db(proj_residual(x_hat, u_sat, x_true)))
        rows.append(dict(eps=eps, sed_null_db=round(float(np.mean(sed)), 2),
                         proj_null_db=round(float(np.mean(proj)), 2),
                         lean_bound_db=round(db(math.sqrt(2.0) * eps), 2),
                         bound_violations=n_trials - bound_ok,
                         bound_slack_x=round(float(np.median(bound_gap)), 2)))
    return rows


# ----------------------------------------------------------------------------------
# (B) Per-channel gain/phase calibration mismatch (signature-driven vs data-driven)
# ----------------------------------------------------------------------------------
def exp_B(n_trials=256, K=64, jnr_db=60.0):
    rows = []
    for gain_pct, phase_deg in [(0.1, 0.1), (0.5, 0.5), (1.0, 1.0), (2.0, 2.0), (5.0, 5.0)]:
        sed, proj, mvdr, eps_eff = [], [], [], []
        for _ in range(n_trials):
            th_j = rng.uniform(-60, 60); th_s = th_j + (20 if th_j < 35 else -20)
            a_j = steer(th_j); a_s = steer(th_s)
            g = (1 + gain_pct / 100 * rng.standard_normal(8)) * np.exp(1j * math.radians(phase_deg) * rng.standard_normal(8))
            x_true = c8_to_r16(g * a_j)        # what the array actually sees
            x_hat = c8_to_r16(a_j)             # what the calibration table says
            u_sat = unit(c8_to_r16(a_s))
            eps_eff.append(np.linalg.norm(unit(x_true) - unit(x_hat)))
            T, u_zd, v = sed_nuller(x_hat, u_sat)
            sed.append(db(sed_residual(T, v, x_true)))
            proj.append(db(proj_residual(x_hat, u_sat, x_true)))
            amp = 10 ** (jnr_db / 20)
            X = rng.standard_normal((K, 1)) * amp * unit(x_true)[None, :] + rng.standard_normal((K, 16)) / 4.0
            mvdr.append(db(mvdr_residual(X, u_sat, x_true)))
        rows.append(dict(gain_pct=gain_pct, phase_deg=phase_deg,
                         eps_effective=round(float(np.mean(eps_eff)), 4),
                         sed_null_db=round(float(np.mean(sed)), 2),
                         proj_null_db=round(float(np.mean(proj)), 2),
                         mvdr_K64_null_db=round(float(np.mean(mvdr)), 2)))
    return rows


# ----------------------------------------------------------------------------------
# (C) Sample support / DOA drift: MVDR's failure regime, with perfect calibration
# ----------------------------------------------------------------------------------
def exp_C(n_trials=256, jnr_db=60.0, slew_deg_per_snapshot=0.3, cal_eps=1e-3):
    rows = []
    for K in [2, 4, 8, 16, 64]:
        mvdr, sed, proj = [], [], []
        for _ in range(n_trials):
            th0 = rng.uniform(-60, 60); th_s = th0 + (20 if th0 < 35 else -20)
            u_sat = unit(c8_to_r16(steer(th_s)))
            amp = 10 ** (jnr_db / 20)
            thetas = th0 + slew_deg_per_snapshot * np.arange(K)
            X = np.stack([rng.standard_normal() * amp * c8_to_r16(steer(t)) for t in thetas]) + rng.standard_normal((K, 16)) / 4.0
            x_now = c8_to_r16(steer(thetas[-1]))
            mvdr.append(db(mvdr_residual(X, u_sat, x_now)))
            # signature-driven methods use the *current* snapshot with a cal_eps residual error
            x_hat = unit(unit(x_now) + cal_eps * unit(rng.standard_normal(16)))
            T, u_zd, v = sed_nuller(x_hat, u_sat)
            sed.append(db(sed_residual(T, v, x_now)))
            proj.append(db(proj_residual(x_hat, u_sat, x_now)))
        rows.append(dict(K=K, drift_deg=round(slew_deg_per_snapshot * (K - 1), 2),
                         mvdr_null_db=round(float(np.mean(mvdr)), 2),
                         sed_null_db=round(float(np.mean(sed)), 2),
                         proj_null_db=round(float(np.mean(proj)), 2)))
    return rows


# ----------------------------------------------------------------------------------
# (D) Things that are genuinely distinct: anti-partner gain and hinge-switch phase
# ----------------------------------------------------------------------------------
def exp_D(n_trials=256):
    gains, jumps = [], []
    for _ in range(n_trials):
        u = moreno_project(rng.standard_normal(16))
        v = nullspace_partner(u, rng.standard_normal(7))
        u_anti = u.copy(); u_anti[8:] = -u_anti[8:]
        gains.append(20 * math.log10(np.linalg.norm(cd_mul(u_anti, v))))
        # hinge: quaternion subalgebra span(e0,e_i,e_j,e_{i^j}) is associative
        i, j = 1, 2
        th = rng.uniform(-math.pi, math.pi)
        xc = np.zeros(16); xc[i] = math.cos(th); xc[j] = math.sin(th)
        u1 = np.zeros(16); u2 = np.zeros(16)
        for q in (0, i, j, i ^ j):
            u1[q] = rng.standard_normal(); u2[q] = rng.standard_normal()
        assoc = cd_mul(cd_mul(xc, u1), u2) - cd_mul(xc, cd_mul(u1, u2))
        jumps.append(np.linalg.norm(assoc))
    return dict(antipartner_gain_db_mean=round(float(np.mean(gains)), 3),
                antipartner_gain_db_std=round(float(np.std(gains)), 6),
                hinge_associator_max=float(np.max(jumps)))


if __name__ == "__main__":
    res = dict(A_signature_error=exp_A(), B_channel_mismatch=exp_B(), C_sample_support=exp_C(), D_structural=exp_D())
    OUT.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print("\n==", k)
        if isinstance(v, list):
            for r in v: print(r)
        else:
            print(v)
