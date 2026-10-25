"""
Block stage: one sedenion stage nulls a 4-dimensional real subspace.

For a fibre vector v (itself a zero divisor), right multiplication R(v): x -> x.v has singular values
sqrt2 (x4), 1 (x8), 0 (x4).  Let K_v = ker R(v) (4-D) and P_v = the sqrt2 passband (4-D).  A gauge T that
maps the jammer directions (<= 4 real) into K_v and the satellite pair {sat_perp, J sat_perp} isometrically
into P_v gives: exact null of all <= 4 directions, exact carrier transfer via the dual-basis readout, and
readout noise gain 1/||sat_perp|| -- identical to widely-linear LCMV.  J > 4 directions: cascade ceil(J/4).

A circular jammer is 2 real directions {u, Ju}; a rectilinear one is 1.  So one stage handles 2 circular
or 4 rectilinear interferers, 4 stages handle 15 rectilinear ones on 8 elements (the 2N-1 limit).
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cf = load("run_complex_frame"); rect = load("run_rectilinear"); okc = cf.okc
r16, c8, J16, runit, cd_mul, I16 = cf.r16, cf.c8, cf.J16, cf.runit, cf.cd_mul, cf.I16

_rng0 = np.random.default_rng(0)
V_FIBRE = runit(cf.fiber_basis(_rng0)[0])                      # one fixed fibre vector (a zero divisor)
R_V = np.stack([cd_mul(I16[i], V_FIBRE) for i in range(16)], 1)  # R(v) x = x . v
_U, _S, _Vt = np.linalg.svd(R_V)
K_V = _Vt[12:].T                                              # 16 x 4 orthonormal basis of ker R(v)
P_V = _Vt[:4].T                                               # 16 x 4 orthonormal basis of the sqrt2 passband


def gram_schmidt(vecs, tol=1e-10):
    out = []
    for x in vecs:
        y = x.copy()
        for q in out: y -= (q @ y) * q
        n = np.linalg.norm(y)
        if n > tol: out.append(y / n)
    return out


def complete(frame):
    """Extend an orthonormal frame (list of 16-vectors) to an orthonormal basis; returns 16x16 with frame first."""
    Q = list(frame)
    rng = np.random.default_rng(len(Q))
    while len(Q) < 16:
        Q = gram_schmidt(Q + [rng.standard_normal(16)])
    return np.stack(Q, 1)


def block_stage(jam_dirs, sat_pair):
    """16x16 M = R(v) T, with T jam_dirs -> K_v and T {sat_perp, J sat_perp} -> P_v (isometrically)."""
    assert len(jam_dirs) <= 4
    jam_q = gram_schmidt(jam_dirs)
    sat_q = gram_schmidt(jam_q + list(sat_pair))[len(jam_q):]   # satellite pair, orthogonalised against jammers
    in_frame = jam_q + sat_q
    out_frame = [K_V[:, k] for k in range(len(jam_q))] + [P_V[:, k] for k in range(len(sat_q))]
    Qin, Qout = complete(in_frame), complete(out_frame)
    T = Qout @ Qin.T
    return R_V @ T


def sed_block_cascade(dirs, sat):
    """dirs: list of real 16-vectors to null (circular jammer -> [u, J u]; rectilinear -> [u])."""
    M = np.eye(16); dirs = list(dirs)
    while dirs:
        chunk, dirs = dirs[:4], dirs[4:]
        M = block_stage([M @ d for d in chunk], [M @ sat, M @ (J16 @ sat)]) @ M
    return cf.dual_readout(M, sat)


# --------------------------------------------------------------- report protocol (circular jammers)
def controlled(rng, eps=1e-3):
    jam, sat, _ = okc.make_scene(rng, False)
    u = okc.estimate_with_error(jam[0], eps, rng); uh, sh = r16(u), r16(sat)
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 16, endpoint=False))
    methods = {"sed_block (1 stage)": sed_block_cascade([uh, J16 @ uh], sh),
               "lcmv_complex": okc.complex_lcmv([uh], sh),
               "wl_lcmv": rect.wl_lcmv([uh, J16 @ uh], sh)}
    rows = []
    for name, H in methods.items():
        nulls, cerr, zs = okc.null_and_carrier_metrics(H, sh, [r16(jam[0])], phases)
        rows.append(dict(method=name, null1_db=nulls[0], carrier=cerr, noise_gain=okc.noise_gain(H)))
    return rows


def heldout(rng, phases):
    est, sat, ev_sig, R = okc.heldout_data(rng, True)
    sh = r16(sat); u1, u2 = r16(est[0]), r16(est[1])
    methods = {"sed_block (1 stage, 4 dirs)": sed_block_cascade([u1, J16 @ u1, u2, J16 @ u2], sh),
               "lcmv_complex": okc.complex_lcmv([u1, u2], sh),
               "mvdr_complex": okc.complex_mvdr(R, sh),
               "wl_lcmv": rect.wl_lcmv([u1, J16 @ u1, u2, J16 @ u2], sh)}
    jn = okc.cnormal(rng, okc.L); th = okc.cnormal(rng, okc.L)
    rows = []
    for name, H in methods.items():
        ng = okc.noise_gain(H); zs = okc.h_to_z(H @ sh); res = [[], []]; cerr = []
        for t in range(len(ev_sig[0])):
            for p in phases:
                for j in range(2): res[j].append(abs(okc.apply_phase(H, ev_sig[j][t], p)) / abs(zs))
                cerr.append(abs(okc.apply_phase(H, sat, p) - zs * p) / abs(zs))
        nulls = [-20 * math.log10(max(math.sqrt(np.mean(np.square(r))), 1e-16)) for r in res]
        leak = math.sqrt(np.mean(np.concatenate(res) ** 2)); sn = abs(zs) / ng
        hit, _, sinr = okc.acquire(sn, leak * sn, 1e4, 10.0, jn, th)
        rows.append(dict(method=name, null1_db=nulls[0], null2_db=nulls[1], carrier=math.sqrt(np.mean(np.square(cerr))),
                         noise_gain=ng, p_d=hit, sinr_db=sinr))
    return rows


# --------------------------------------------------------------- rectilinear scaling
def scene_grid(rng, J):
    """J+1 angles on a jittered grid over [-60, 60] (rejection sampling is impractical for J > 10)."""
    g = np.linspace(-60, 60, J + 1) + rng.uniform(-0.3, 0.3, J + 1) * (120 / J); rng.shuffle(g)
    return [okc.steer(t) * np.exp(1j * rng.uniform(0, 2 * np.pi)) for t in g[:J]], okc.steer(g[J])


def rect_trial(rng, J, K=64, jsr_db=70.0, snr_db=-20.0):
    jam, sat = rect.scene(rng, J) if J <= 10 else scene_grid(rng, J)
    amp = 10 ** (jsr_db / 20); noise = 10 ** (-snr_db / 20)
    X = sum(np.sign(rng.standard_normal((K, 1))) * amp * z[None, :] for z in jam)
    X = X + noise * okc.cnormal(rng, (K, 8)) + np.sign(rng.standard_normal((K, 1))) * np.exp(1j * rng.uniform(0, 2 * np.pi, (K, 1))) * sat[None, :]
    Xr = np.stack([r16(x) for x in X]); Rr = Xr.T @ Xr / K
    ev, Vv = np.linalg.eigh(Rr); u_hats = [Vv[:, -j - 1] for j in range(J)]
    Rc = X.T @ X.conj() / K; sh = r16(sat)
    methods = {"complex_mvdr": okc.complex_mvdr(Rc, sh), "wl_lcmv": rect.wl_lcmv(u_hats, sh),
               "sed_block_cascade": sed_block_cascade(u_hats, sh)}
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    jn = okc.cnormal(rng, okc.L); th = okc.cnormal(rng, okc.L); rows = []
    for name, H in methods.items():
        ng = okc.noise_gain(H); zs = okc.h_to_z(H @ sh); zs = zs if abs(zs) > 1e-12 else 1e-12
        leak = [abs(okc.h_to_z(H @ r16(z))) / abs(zs) for z in jam]
        cerr = [abs(okc.apply_phase(H, sat, p) - zs * p) / abs(zs) for p in phases]
        lr = math.sqrt(np.mean(np.square(leak))); sn = abs(zs) / ng
        hit, _, sinr = okc.acquire(sn, lr * sn * math.sqrt(J), amp, 10.0, jn, th)
        rows.append(dict(method=name, null_db=-20 * math.log10(max(lr, 1e-16)), carrier=math.sqrt(np.mean(np.square(cerr))),
                         noise_gain=ng, p_d=hit, sinr_db=sinr))
    return rows


def agg(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r["method"], {})
        for k, v in r.items():
            if k != "method": d.setdefault(k, []).append(v)
    return {m: {k: round(float(np.mean(v)), 3) for k, v in d.items()} for m, d in out.items()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    rng = np.random.default_rng(2026)
    res = {}
    res["controlled_single_circular"] = agg([r for _ in range(n) for r in controlled(rng)])
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    res["heldout_dual_circular"] = agg([r for _ in range(n) for r in heldout(rng, phases)])
    for k in ["controlled_single_circular", "heldout_dual_circular"]:
        print("==", k)
        for m, d in res[k].items(): print(f"   {m:30s} {d}")
    for J in [4, 7, 8, 10, 12, 14, 15]:
        res[f"rectilinear_J={J}"] = agg([r for _ in range(max(n // 2, 32)) for r in rect_trial(rng, J)])
        print(f"== rectilinear J={J}")
        for m, d in res[f"rectilinear_J={J}"].items(): print(f"   {m:30s} {d}", flush=True)
    (HERE / "results_block_stage.json").write_text(json.dumps(res, indent=1))
