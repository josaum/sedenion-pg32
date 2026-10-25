"""
Rectilinear (non-circular) interferers: breaking the N-1 barrier.

A BPSK/rectilinear interferer is s(t) u with s(t) real, so in R^16 it spans ONE real direction.
Complex-linear beamformers (standard LCMV / MVDR) spend a complex DOF per interferer and cannot null
more than N-1 = 7 with N = 8 elements.  Any real-linear (widely-linear) map can null up to 2N-1 = 15.
The sedenion stage with the 1-frame gauge nulls exactly one real direction per stage; the satellite is
treated as circular (unknown rotating carrier) and read out in the dual basis of {y(u_sat), y(J u_sat)}.

Methods: complex LCMV, complex MVDR (SMI, K=64), widely-linear LCMV (real 16-dim, the honest ceiling),
sedenion cascade (1-frame gauge, J stages).  Signatures are estimated from the same K snapshots:
the top-J eigenvectors of the REAL 16x16 covariance (rectilinear interferers are rank-1 there).
Evaluation on the true signatures with the satellite swept over 8 carrier phases.
"""
import json, math, pathlib, sys, importlib.util
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
def load(n):
    s = importlib.util.spec_from_file_location(n, HERE / f"{n}.py"); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cf = load("run_complex_frame"); okc = cf.okc
r16, c8, J16, runit, householder, fiber_basis, cd_mul, I16 = (
    cf.r16, cf.c8, cf.J16, cf.runit, cf.householder, cf.fiber_basis, cf.cd_mul, cf.I16)
UZD = cf.UZD


def stage_1frame(x1, sat_pair, rng, pre=None, n_cand=24):
    """Sedenion stage nulling ONE real direction x1: T = Householder(x1 -> UZD); fibre DOF spent on
       minimising the dual-basis readout noise gain."""
    T = householder(runit(x1), UZD)
    V = fiber_basis(rng); P = np.eye(16) if pre is None else pre
    TE = T @ I16
    Mk = [np.stack([cd_mul(TE[:, k], vk) for k in range(16)], axis=1) for vk in V]
    MP = [m @ P for m in Mk]
    sat, jsat = sat_pair
    Bk = [np.stack([cd_mul(T @ sat, vk), cd_mul(T @ jsat, vk)], 1) for vk in V]
    def cost(al):
        B = sum(a * b for a, b in zip(al, Bk)); M = sum(a * m for a, m in zip(al, MP))
        s = np.linalg.svd(B, compute_uv=False)
        if s[-1] < 1e-9 * s[0]: return np.inf
        H = np.linalg.pinv(B) @ M
        return math.sqrt((H[0] @ H[0] + H[1] @ H[1]) / 2)
    cands = [runit(rng.standard_normal(4)) for _ in range(n_cand)]
    best = min(cands, key=cost); bc = cost(best); step = 0.3
    for _ in range(12):
        imp = False
        for k in range(4):
            for sg in (1, -1):
                a_ = best.copy(); a_[k] += sg * step; a_ = runit(a_); c = cost(a_)
                if c < bc: best, bc, imp = a_, c, True
        if not imp: step *= 0.5
        if step < 1e-4: break
    return sum(a * m for a, m in zip(best, Mk)) / np.linalg.norm(best @ V)


def sed_cascade_rect(u_hats, sat, rng):
    M = np.eye(16)
    for u in u_hats:
        M = stage_1frame(M @ u, [M @ sat, M @ (J16 @ sat)], rng, pre=M) @ M
    return cf.dual_readout(M, sat)


def wl_lcmv(u_hats, sat):
    """Widely-linear LCMV: real 2x16 H with H u_j = 0 (real directions), H sat = (1,0), H J sat = (0,1);
       minimum Frobenius norm."""
    A = np.stack(list(u_hats) + [sat, J16 @ sat], 1)          # 16 x (J+2)
    F = np.zeros((2, A.shape[1])); F[0, -2] = 1; F[1, -1] = 1
    H = F @ np.linalg.pinv(A)                                  # min-norm solution of H A = F
    return H


def complex_lcmv_ls(u_hats_c, sat_c):
    """Complex LCMV; least-squares when over-constrained (J+1 > 8)."""
    A = np.stack(list(u_hats_c) + [sat_c], 1)
    f = np.zeros(A.shape[1], complex); f[-1] = 1
    w = np.linalg.pinv(A.conj().T) @ f
    return okc.w_to_h(w)


def scene(rng, J, min_sep=6.0):
    while True:
        ths = rng.uniform(-60, 60, J + 1)
        if all(abs(ths[i] - ths[j]) >= min_sep for i in range(J + 1) for j in range(i)):
            break
    jam = [okc.steer(t) * np.exp(1j * rng.uniform(0, 2 * np.pi)) for t in ths[:J]]   # fixed carrier phase each
    return jam, okc.steer(ths[J])


def trial(rng, J, K=64, jsr_db=70.0, snr_db=-20.0):
    jam, sat = scene(rng, J)
    amp = 10 ** (jsr_db / 20); noise = 10 ** (-snr_db / 20)
    # rectilinear interferers: real +-1 chips times fixed complex signature; satellite BPSK with rotating carrier
    X = sum(np.sign(rng.standard_normal((K, 1))) * amp * z[None, :] for z in jam)
    X = X + noise * okc.cnormal(rng, (K, 8)) + np.sign(rng.standard_normal((K, 1))) * np.exp(1j * rng.uniform(0, 2 * np.pi, (K, 1))) * sat[None, :]
    Xr = np.stack([r16(x) for x in X])                          # K x 16 real
    Rr = Xr.T @ Xr / K
    ev, V = np.linalg.eigh(Rr); u_hats = [V[:, -j - 1] for j in range(J)]       # top-J real directions
    Rc = X.T @ X.conj() / K
    # complex estimates for the complex methods: use the same real eigenvectors as complex vectors
    u_hats_c = [c8(u) for u in u_hats]
    sh = r16(sat)
    methods = {
        "complex_lcmv": complex_lcmv_ls(u_hats_c, sat),
        "complex_mvdr": okc.complex_mvdr(Rc, sh),
        "wl_lcmv": wl_lcmv(u_hats, sh),
        "sed_cascade_1frame": sed_cascade_rect(u_hats, sh, rng),
    }
    phases = np.exp(1j * np.linspace(0, 2 * np.pi, 8, endpoint=False))
    jam_noise = okc.cnormal(rng, okc.L); thermal = okc.cnormal(rng, okc.L)
    rows = []
    for name, H in methods.items():
        ng = okc.noise_gain(H); zs = okc.h_to_z(H @ sh)
        if abs(zs) < 1e-12: zs = 1e-12
        leak = [abs(okc.h_to_z(H @ r16(z))) / abs(zs) for z in jam]                 # rectilinear: fixed phase
        cerr = [abs(okc.apply_phase(H, sat, p) - zs * p) / abs(zs) for p in phases]
        null_db = -20 * math.log10(max(math.sqrt(np.mean(np.square(leak))), 1e-16))
        leak_rms = math.sqrt(np.mean(np.square(leak))); sat_norm = abs(zs) / ng
        hit, _, sinr = okc.acquire(sat_norm, leak_rms * sat_norm * math.sqrt(J), amp, 10.0, jam_noise, thermal)
        rows.append(dict(J=J, method=name, null_db=null_db, carrier_err=math.sqrt(np.mean(np.square(cerr))),
                         noise_gain=ng, p_d=hit, sinr_db=sinr))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    rng = np.random.default_rng(2026)
    out = {}
    for J in [2, 4, 7, 8, 10, 12, 14]:
        rows = [r for _ in range(n) for r in trial(rng, J)]
        out[f"J={J}"] = {}
        for m in ["complex_lcmv", "complex_mvdr", "wl_lcmv", "sed_cascade_1frame"]:
            rr = [r for r in rows if r["method"] == m]
            out[f"J={J}"][m] = dict(null_db=round(float(np.mean([r["null_db"] for r in rr])), 1),
                                     carrier_err=round(float(np.mean([r["carrier_err"] for r in rr])), 3),
                                     noise_gain=round(float(np.mean([r["noise_gain"] for r in rr])), 3),
                                     p_d=round(float(np.mean([r["p_d"] for r in rr])), 3),
                                     sinr_db=round(float(np.mean([r["sinr_db"] for r in rr])), 1))
        print(f"== J={J} rectilinear interferers, 8 elements, JSR +70 dB each, K=64")
        for m, d in out[f"J={J}"].items(): print(f"   {m:20s} {d}", flush=True)
    (HERE / "results_rectilinear.json").write_text(json.dumps(out, indent=1))
